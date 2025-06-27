import torch
from models import VJEPA, HNN, LNN
from dataset import PendulumDataset
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================
# 0 · helper: train ONE epoch, return loss dict
# ================================================================
def _train_one_epoch(model, theta_head, lnn, hnn,
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
def run_mode(mode:str, dataset, dataloader, epochs:int=10, batch_size:int=32, suffix = '_dense'):
    print(f"\n###  {mode.upper()}  ###")

    SUFFIX = suffix
    
    # ---- hyper-params specific to run -------------------------
    λ_map = {
        "plain":    dict(λ_h=0.0,   λ_l=0.0),
        "hnn":      dict(λ_h=1e-3,  λ_l=0.0),
        "lnn":      dict(λ_h=0.0,   λ_l=1e-3),
        "hnn+lnn":  dict(λ_h=5e-4,  λ_l=1e-3),
    }
    λ_h, λ_l = λ_map[mode]["λ_h"], λ_map[mode]["λ_l"]
    λ_sup = 1e-2

    # ---- data loader ------------------------------------------
    ds = dataset
    dl = dataloader

    # ---- networks ---------------------------------------------
    model       = VJEPA(embed_dim=384, depth=6, num_heads=6).to(device)
    theta_head  = nn.Linear(384, 2).to(device)
    hnn         = HNN(hidden_dim=256).to(device) if λ_h > 0 else None
    lnn         = LNN(input_dim=2, hidden_dim=256).to(device) if λ_l > 0 else None

    params = list(model.parameters()) + list(theta_head.parameters())
    if hnn: params += list(hnn.parameters())
    if lnn: params += list(lnn.parameters())
    opt = optim.AdamW(params, lr=1e-4)

    # ---- training ---------------------------------------------
    log = {k:[] for k in ["total","jepa","hnn","lnn","sup"]}
    for ep in range(epochs):
        ep_loss = _train_one_epoch(model, theta_head, lnn, hnn,
                                   dl, opt, λ_l, λ_h, λ_sup)
        for k in log: log[k].append(ep_loss[k])
        print(f"ep {ep+1}: tot {ep_loss['total']:.3f} "
              f"j {ep_loss['jepa']:.3f} "
              f"h {ep_loss['hnn']:.3f} "
              f"l {ep_loss['lnn']:.3f}")

    # ---- save dictionary --------------------------------------
    cfg = dict(mode=mode, epochs=epochs,
               λ_hnn=λ_h, λ_lnn=λ_l, λ_sup=λ_sup,
               batch_size=batch_size)
    # ---------------------------------------------------------------
    # replace the two separate torch.save(...) calls by this block
    # ---------------------------------------------------------------
    full_sd = {}

    # 1)  V-JEPA backbone
    for k, v in model.state_dict().items():
        full_sd[f"vjepa.{k}"] = v.cpu()

    # 2)  θ-ω linear head
    for k, v in theta_head.state_dict().items():
        full_sd[f"theta_head.{k}"] = v.cpu()

    # 3)  HNN parameters  (only if this mode uses an HNN)
    if hnn is not None:
        for k, v in hnn.state_dict().items():
            full_sd[f"hnn.{k}"] = v.cpu()

    # 4)  LNN parameters  (only if this mode uses an LNN)
    if lnn is not None:
        for k, v in lnn.state_dict().items():
            full_sd[f"lnn.{k}"] = v.cpu()

    # ---- one self-contained checkpoint ----
    torch.save(full_sd, f"model_{mode}{SUFFIX}.pt")

    # --------- training log stays unchanged ----------
    np.savez(f"results_{mode}{SUFFIX}.npz", **log, config=cfg)
    print(f"saved to  model_{mode}{SUFFIX}.pt   &   results_{mode}{SUFFIX}.npz")
    
    np.savez(f"results_{mode}_dense.npz", **log, config=cfg)
    print("saved to results_%s_dense.npz" % mode)
