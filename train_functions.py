import torch
from models import VJEPA, HNN, LNN
from dataset import PendulumDataset
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================
# 0 · helper: train ONE epoch, return loss dict
# ================================================================
def train_one_epoch(model, theta_head, lnn, hnn,
                     loader, opt, λ_lnn, λ_hnn, λ_sup):
    
    model.train(); theta_head.train()
    if lnn : lnn.train()
    if hnn : hnn.train()

    agg = dict(jepa=0., lnn=0., hnn=0., sup=0., total=0.)
    for imgs_seq, states_seq in loader:
        imgs_seq, states_seq = imgs_seq.to(device), states_seq.to(device)
        imgs0   = imgs_seq[:,0]                         # (B,C,H,W)
        θ_true  = states_seq[:,:,0:1]
        ω_true  = states_seq[:,:,1:2]

        # --- JEPA on frame-0 ------------------------------------
        loss_jepa, _, _ = model(imgs0)

        # --- latent → θ̂, ω̂ frame-0 -----------------------------
        z0   = model.patch_embed(imgs0)+model.pos_embed
        z0   = model.context_encoder(z0).mean(1)
        θ̂0, ω̂0 = theta_head(z0).split(1,1)

        # --- HNN residual (if active) ---------------------------
        if hnn and λ_hnn>0:
            qp = torch.cat([θ̂0, ω̂0], 1)
            hnn_loss = F.mse_loss(hnn.time_derivative(qp)[:,0:1], ω̂0)
        else:
            hnn_loss = torch.tensor(0., device=device)

        # --- LNN residual (if active) ---------------------------
        if lnn and λ_lnn>0:
            lnn_loss = lnn.lagrangian_residual(θ_true, ω_true)
        else:
            lnn_loss = torch.tensor(0., device=device)

        # --- supervised θ at t=0 -------------------------------
        sup_loss = F.mse_loss(θ̂0, θ_true[:,0])

        loss = (loss_jepa + λ_lnn*lnn_loss + λ_hnn*hnn_loss + λ_sup*sup_loss)
        opt.zero_grad(); loss.backward(); opt.step()

        agg['jepa']  += loss_jepa.item()
        agg['lnn']   += lnn_loss.item()
        agg['hnn']   += hnn_loss.item()
        agg['sup']   += sup_loss.item()
        agg['total'] += loss.item()

    for k in agg: agg[k] /= len(loader)
    return agg


# ================================================================
# 1 · run one MODE and save log
# ================================================================
def run_mode(
        mode       : str,
        dataloader,
        lambda_map,
        *,
        epochs     = 10,
        batch_size = 32,
        suffix     = "_dense",
        model_dir  = "./models",     # ← NEW: where to save *.pt
        log_dir    = "./results_numpy"):   # ← NEW: where to save *.npz

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)
    print(f"\n###  {mode.upper()}  (saving to {model_dir})  ###")

    # ── hyper-parameters per mode ───────────────────────────────
    λ_map = lambda_map
    λ_h, λ_l = λ_map[mode]["λ_h"], λ_map[mode]["λ_l"]
    λ_sup    = 1e-2

    # ── networks ───────────────────────────────────────────────
    model       = VJEPA(embed_dim=384, depth=6, num_heads=6).to(device)
    theta_head  = nn.Linear(384, 2).to(device)
    hnn         = HNN(hidden_dim=256).to(device) if λ_h > 0 else None
    lnn         = LNN(input_dim=2, hidden_dim=256).to(device) if λ_l > 0 else None

    params = list(model.parameters()) + list(theta_head.parameters())
    if hnn: params += list(hnn.parameters())
    if lnn: params += list(lnn.parameters())
    opt = optim.AdamW(params, lr=1e-4)

    # ── training loop ──────────────────────────────────────────
    log = {k: [] for k in ["loss_total", "loss_jepa",
                           "loss_hnn",  "loss_lnn", "loss_sup"]}

    for ep in range(epochs):
        ep_loss = train_one_epoch(model, theta_head, lnn, hnn,
                                  dataloader, opt, λ_l, λ_h, λ_sup)
        for k_new, k_old in zip(log.keys(),
                                ["total", "jepa", "hnn", "lnn", "sup"]):
            log[k_new].append(ep_loss[k_old])

        print(f"ep {ep+1:02d}: "
              f"tot {ep_loss['total']:.3f}  "
              f"j {ep_loss['jepa']:.3f}  "
              f"h {ep_loss['hnn']:.3f}  "
              f"l {ep_loss['lnn']:.3f}")

    # ── build single combined checkpoint ───────────────────────
    full_sd = {}

    for k, v in model.state_dict().items():
        full_sd[f"vjepa.{k}"] = v.cpu()
    for k, v in theta_head.state_dict().items():
        full_sd[f"theta_head.{k}"] = v.cpu()
    if hnn:
        for k, v in hnn.state_dict().items():
            full_sd[f"hnn.{k}"] = v.cpu()
    if lnn:
        for k, v in lnn.state_dict().items():
            full_sd[f"lnn.{k}"] = v.cpu()

    ckpt_path = os.path.join(model_dir, f"model_{mode}{suffix}.pt")
    torch.save(full_sd, ckpt_path)
    print("checkpoint saved →", ckpt_path)

    # ── save training curves ───────────────────────────────────
    cfg = dict(mode=mode, epochs=epochs,
               λ_hnn=λ_h, λ_lnn=λ_l, λ_sup=λ_sup,
               batch_size=batch_size)

    log_path = os.path.join(log_dir, f"results_{mode}{suffix}.npz")
    np.savez(log_path, **log, config=cfg)
    print("training log saved →", log_path)