# train_functions.py
# --------------------------------------------------------------------
# Core training utilities for the Pendulum-VJEPA research project.
#
# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────
# • TrainConfig  – dataclass with every configurable hyper-parameter
# • train_one_epoch(...)
# • run_mode(cfg, loader)
#
# Defaults reproduce your prior experiments; edit a single TrainConfig
# field to explore new settings without touching code.
# --------------------------------------------------------------------

from __future__ import annotations

import csv, os
from dataclasses import dataclass, asdict
from typing import Dict, Literal, Optional, List
import math
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# project-local
from utils.models   import VJEPA, HNN, LNN
from utils.datasets import PendulumDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================================================================
# 1 · Config dataclass
# ====================================================================
@dataclass
class TrainConfig:
    mode       : Literal["plain","hnn","lnn","hnn+lnn"]

    epochs     : int   = 10
    batch_size : int   = 32
    lr         : float = 1e-4

    img_size   : int   = 64
    patch_size : int   = 8
    embed_dim  : int   = 384
    depth      : int   = 6
    num_heads  : int   = 6

    h_hidden   : int   = 256
    l_hidden   : int   = 256

    λ_hnn      : float = 0.0
    λ_lnn      : float = 0.0
    λ_sup      : float = 1e-2

    suffix     : str   = "_dense"
    model_dir  : str   = "./models"
    log_dir    : str   = "./results_numpy"

    # --- supervision ---
    sup_use_frames: Literal["first", "all", "N"] = "first"  # "first" | "all" | "N"
    sup_num_frames: int = 3                                 # used when "N"
    normalize_sup: bool = True
    B_OMEGA: float = 1.0

    # --- optimization ---
    weight_decay: float = 1e-4
    max_grad_norm: Optional[float] = None                   # e.g., 1.0
    # cosine schedule with warmup
    warmup_epochs: int = 1
    cosine_final_lr_ratio: float = 0.1                      # lr_min = ratio * lr

    # --- EMA ---
    use_ema: bool = True
    ema_decay: float = 0.999
    use_ema_head: bool = True

    @staticmethod
    def preset(mode: Literal["plain","hnn","lnn","hnn+lnn"]) -> "TrainConfig":
        cfg = TrainConfig(mode=mode)
        cfg.__dict__.update({
            "plain"   : dict(λ_hnn=0.0,   λ_lnn=0.0),
            "hnn"     : dict(λ_hnn=1e-3,  λ_lnn=0.0),
            "lnn"     : dict(λ_hnn=0.0,   λ_lnn=1e-3),
            "hnn+lnn" : dict(λ_hnn=5e-4,  λ_lnn=1e-3),
        }[mode])
        return cfg

class EMA:
    def __init__(self, module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in module.state_dict().items()}

    @torch.no_grad()
    def update(self, module: torch.nn.Module):
        for k, v in module.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, module: torch.nn.Module):
        module.load_state_dict(self.shadow, strict=True)

# ====================================================================
# 2 · One-epoch loop
# ====================================================================
# def train_one_epoch(
#     model: VJEPA,
#     theta_head: nn.Linear,
#     lnn: Optional[LNN],
#     hnn: Optional[HNN],
#     loader: DataLoader,
#     opt: optim.Optimizer,
#     λ_lnn: float,
#     λ_hnn: float,
#     λ_sup: float
# ) -> Dict[str,float]:

#     model.train(); theta_head.train()
#     if lnn: lnn.train()
#     if hnn: hnn.train()

#     agg = dict(jepa=0., lnn=0., hnn=0., sup=0., total=0.)

#     for imgs_seq, states_seq in loader:
#         imgs_seq, states_seq = imgs_seq.to(device), states_seq.to(device)
#         imgs0  = imgs_seq[:,0]
#         θ_true = states_seq[:,:,0:1]
#         ω_true = states_seq[:,:,1:2]

#         loss_jepa, _, _ = model(imgs0)

#         z0 = model.patch_embed(imgs0) + model.pos_embed
#         z0 = model.context_encoder(z0).mean(1)
#         θ̂0, ω̂0 = theta_head(z0).split(1,1)

#         hnn_loss = torch.tensor(0., device=device)
#         if hnn and λ_hnn>0:
#             hnn_loss = F.mse_loss(hnn.time_derivative(torch.cat([θ̂0, ω̂0],1))[:,0:1], ω̂0)

#         lnn_loss = torch.tensor(0., device=device)
#         if lnn and λ_lnn>0:
#             lnn_loss = lnn.lagrangian_residual(θ_true, ω_true)

#         sup_loss = F.mse_loss(θ̂0, θ_true[:,0])

#         loss = loss_jepa + λ_lnn*lnn_loss + λ_hnn*hnn_loss + λ_sup*sup_loss

#         opt.zero_grad(); loss.backward(); opt.step()

#         agg["jepa"]  += loss_jepa.item()
#         agg["lnn"]   += lnn_loss.item()
#         agg["hnn"]   += hnn_loss.item()
#         agg["sup"]   += sup_loss.item()
#         agg["total"] += loss.item()

#     for k in agg: agg[k] /= len(loader)
#     return agg

def _pick_supervision_frames(seq_len: int, mode: str, N: int) -> slice | list[int]:
    if mode == "first": return [0]
    if mode == "all"  : return list(range(seq_len))
    if mode == "N"    : return list(range(min(N, seq_len)))
    raise ValueError(f"sup_use_frames must be 'first' | 'all' | 'N', got {mode}")

def train_one_epoch(
    model: VJEPA,
    theta_head: nn.Linear,
    lnn: Optional[LNN],
    hnn: Optional[HNN],
    loader: DataLoader,
    opt: optim.Optimizer,
    λ_lnn: float,
    λ_hnn: float,
    λ_sup: float,
    *,
    normalize_sup: bool = True,
    B_OMEGA: float = 1.0,
    sup_use_frames: str = "first",
    sup_num_frames: int = 3,
    max_grad_norm: float | None = None,
    ema_updater: EMA | dict | None = None,
) -> Dict[str, float]:

    model.train(); theta_head.train()
    if lnn: lnn.train()
    if hnn: hnn.train()

    agg = dict(jepa=0., lnn=0., hnn=0., sup=0., sup_theta=0., sup_omega=0., total=0.)
    THETA_SCALE, OMEGA_SCALE = math.pi, 5.0

    for imgs_seq, states_seq in loader:
        imgs_seq  = imgs_seq.to(device)           # (B,T,3,H,W)
        states_seq= states_seq.to(device)         # (B,T,2)
        T = imgs_seq.size(1)
        idxs = _pick_supervision_frames(T, sup_use_frames, sup_num_frames)

        imgs0  = imgs_seq[:, 0]
        θ_true = states_seq[:, :, 0:1]
        ω_true = states_seq[:, :, 1:2]
        θ0, ω0 = θ_true[:, 0], ω_true[:, 0]

        # JEPA pretext on frame-0 (you can extend to more frames if desired)
        loss_jepa, _, _ = model(imgs0)
        z0 = model.patch_embed(imgs0) + model.pos_embed
        z0 = model.context_encoder(z0).mean(1)
        
        # --- multi-frame head supervision ---
        sup_theta = torch.tensor(0., device=device)
        sup_omega = torch.tensor(0., device=device)

        # encode each chosen frame t independently (small T, cost is fine)
        for t in idxs:
            zt = model.patch_embed(imgs_seq[:, t]) + model.pos_embed
            zt = model.context_encoder(zt).mean(1)          # (B,D)
            θ̂t, ω̂t = theta_head(zt).split(1, 1)           # (B,1)

            θt = θ_true[:, t]
            ωt = ω_true[:, t]

            if normalize_sup:
                sup_theta = sup_theta + F.mse_loss(θ̂t / THETA_SCALE, θt / THETA_SCALE)
                sup_omega = sup_omega + F.mse_loss(ω̂t / OMEGA_SCALE, ωt / OMEGA_SCALE)
            else:
                sup_theta = sup_theta + F.mse_loss(θ̂t, θt)
                sup_omega = sup_omega + F.mse_loss(ω̂t, ωt)

        sup_theta = sup_theta / len(idxs)
        sup_omega = sup_omega / len(idxs)
        sup_loss  = sup_theta + B_OMEGA * sup_omega

        # --- physics aux losses (unchanged) ---
        hnn_loss = torch.tensor(0., device=device)
        if hnn and λ_hnn > 0:
            # use frame-0 estimates for consistency term
            θ̂0, ω̂0 = theta_head(z0).split(1, 1)
            h_out = hnn.time_derivative(torch.cat([θ̂0, ω̂0], 1))
            dθdt  = h_out[:, 0:1] if h_out.ndim == 2 else h_out.reshape(-1,1)
            hnn_loss = F.mse_loss(dθdt, ω̂0)

        lnn_loss = torch.tensor(0., device=device)
        if lnn and λ_lnn > 0:
            lnn_loss = lnn.lagrangian_residual(θ_true, ω_true)

        # total
        loss = loss_jepa + λ_lnn*lnn_loss + λ_hnn*hnn_loss + λ_sup*sup_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(theta_head.parameters())
                + ([] if not hnn else list(hnn.parameters()))
                + ([] if not lnn else list(lnn.parameters())),
                max_norm=max_grad_norm
            )
        opt.step()

        # --- EMA update(s) ---
        if ema_updater is not None:
            if isinstance(ema_updater, dict):
                if "encoder" in ema_updater and ema_updater["encoder"] is not None:
                    ema_updater["encoder"].update(model)
                if "head" in ema_updater and ema_updater["head"] is not None:
                    ema_updater["head"].update(theta_head)
            else:
                # old behavior: single EMA for the encoder
                ema_updater.update(model)

        # logging
        agg["jepa"]      += loss_jepa.item()
        agg["lnn"]       += lnn_loss.item()
        agg["hnn"]       += hnn_loss.item()
        agg["sup"]       += sup_loss.item()
        agg["sup_theta"] += sup_theta.item()
        agg["sup_omega"] += sup_omega.item()
        agg["total"]     += loss.item()

    for k in agg: agg[k] /= len(loader)
    return agg

# ====================================================================
# 3 · Train-and-save
# ====================================================================
# def run_mode(cfg: TrainConfig, dataloader: DataLoader) -> None:
#     os.makedirs(cfg.model_dir, exist_ok=True)
#     os.makedirs(cfg.log_dir,   exist_ok=True)

#     print(f"\n### {cfg.mode.upper()} | {cfg.epochs} epochs ###")

#     model = VJEPA(img_size=cfg.img_size,
#                   patch_size=cfg.patch_size,
#                   embed_dim=cfg.embed_dim,
#                   depth=cfg.depth,
#                   num_heads=cfg.num_heads).to(device)

#     theta_head = nn.Linear(cfg.embed_dim, 2).to(device)
#     hnn = HNN(cfg.h_hidden).to(device) if cfg.λ_hnn>0 else None
#     lnn = LNN(2, cfg.l_hidden).to(device) if cfg.λ_lnn>0 else None

#     opt = optim.AdamW(
#         list(model.parameters())
#         + list(theta_head.parameters())
#         + ([] if hnn is None else list(hnn.parameters()))
#         + ([] if lnn is None else list(lnn.parameters())),
#         lr=cfg.lr)

#     rows: List[Dict[str,float]] = []
#     for ep in range(cfg.epochs):
#         loss_d = train_one_epoch(model, theta_head,
#                                  lnn, hnn, dataloader, opt,
#                                  cfg.λ_lnn, cfg.λ_hnn, cfg.λ_sup)
#         rows.append({"epoch": ep+1, **loss_d})
#         print(f"ep{ep+1:02d}",
#               " ".join([f"{k}={v:.3f}" for k,v in loss_d.items()]))

#     ckpt = {f"vjepa.{k}":v.cpu() for k,v in model.state_dict().items()}
#     ckpt.update({f"theta_head.{k}":v.cpu() for k,v in theta_head.state_dict().items()})
#     if hnn: ckpt.update({f"hnn.{k}":v.cpu() for k,v in hnn.state_dict().items()})
#     if lnn: ckpt.update({f"lnn.{k}":v.cpu() for k,v in lnn.state_dict().items()})

#     torch.save(ckpt, os.path.join(cfg.model_dir, f"model_{cfg.mode}{cfg.suffix}.pt"))

#     csv_path = os.path.join(cfg.log_dir, f"train_{cfg.mode}{cfg.suffix}.csv")
#     with open(csv_path, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=rows[0].keys())
#         writer.writeheader()
#         writer.writerows(rows)

#     np.savez(os.path.join(cfg.log_dir, f"results_{cfg.mode}{cfg.suffix}.npz"),
#              loss_total=np.array([r["total"] for r in rows]),
#              loss_jepa =np.array([r["jepa"]  for r in rows]),
#              loss_hnn  =np.array([r["hnn"]   for r in rows]),
#              loss_lnn  =np.array([r["lnn"]   for r in rows]),
#              loss_sup  =np.array([r["sup"]   for r in rows]),)

def run_mode(cfg: TrainConfig, dataloader: DataLoader) -> None:
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.log_dir,   exist_ok=True)

    print(f"\n### {cfg.mode.upper()} | {cfg.epochs} epochs ###")

    model = VJEPA(img_size=cfg.img_size,
                  patch_size=cfg.patch_size,
                  embed_dim=cfg.embed_dim,
                  depth=cfg.depth,
                  num_heads=cfg.num_heads).to(device)

    theta_head = nn.Linear(cfg.embed_dim, 2).to(device)
    hnn = HNN(cfg.h_hidden).to(device) if cfg.λ_hnn>0 else None
    lnn = LNN(2, cfg.l_hidden).to(device) if cfg.λ_lnn>0 else None

    # -------- optimizer (AdamW + weight decay) --------
    params = (
        list(model.parameters())
        + list(theta_head.parameters())
        + ([] if hnn is None else list(hnn.parameters()))
        + ([] if lnn is None else list(lnn.parameters()))
    )
    opt = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # -------- cosine LR with warmup --------
    total_epochs = cfg.epochs
    warmup = max(0, int(cfg.warmup_epochs))
    T_cos = max(1, total_epochs - warmup)
    lr_min = cfg.lr * cfg.cosine_final_lr_ratio
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_cos, eta_min=lr_min)

    # -------- EMA (optional) --------
    ema_encoder = EMA(model, decay=cfg.ema_decay) if cfg.use_ema else None
    ema_head    = EMA(theta_head, decay=cfg.ema_decay) if (cfg.use_ema and cfg.use_ema_head) else None

    rows: List[Dict[str,float]] = []
    for ep in range(cfg.epochs):
        # warmup: keep lr linear for first `warmup` epochs
        if ep < warmup:
            for g in opt.param_groups:
                g["lr"] = cfg.lr * (ep + 1) / warmup

        loss_d = train_one_epoch(
            model, theta_head, lnn, hnn, dataloader, opt,
            cfg.λ_lnn, cfg.λ_hnn, cfg.λ_sup,
            normalize_sup=cfg.normalize_sup,
            B_OMEGA=cfg.B_OMEGA,
            sup_use_frames=cfg.sup_use_frames,
            sup_num_frames=cfg.sup_num_frames,
            max_grad_norm=cfg.max_grad_norm,
            ema_updater={"encoder": ema_encoder, "head": ema_head},
        )
        rows.append({"epoch": ep+1, "lr": opt.param_groups[0]["lr"], **loss_d})
        print(f"ep{ep+1:02d}",
              f"lr={opt.param_groups[0]['lr']:.2e}",
              " ".join([f"{k}={v:.3f}" for k,v in loss_d.items()]))
        
        scheduler.step()  # cosine from epoch `warmup` onward

    # -------- choose which weights to save: EMA or raw --------
    model_to_save = copy.deepcopy(model).to("cpu")
    head_to_save  = copy.deepcopy(theta_head).to("cpu")   # <-- add this
    
    if cfg.use_ema:
        if ema_encoder is not None: ema_encoder.copy_to(model_to_save)
        if ema_head    is not None: ema_head.copy_to(head_to_save)
    
    ckpt = {f"vjepa.{k}": v for k, v in model_to_save.state_dict().items()}
    ckpt.update({f"theta_head.{k}": v for k, v in head_to_save.state_dict().items()}) 
    if hnn: ckpt.update({f"hnn.{k}": v.cpu() for k, v in hnn.state_dict().items()})
    if lnn: ckpt.update({f"lnn.{k}": v.cpu() for k, v in lnn.state_dict().items()})

    torch.save(ckpt, os.path.join(cfg.model_dir, f"model_{cfg.mode}{cfg.suffix}.pt"))

    # -------- logs --------
    csv_path = os.path.join(cfg.log_dir, f"train_{cfg.mode}{cfg.suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    np.savez(os.path.join(cfg.log_dir, f"results_{cfg.mode}{cfg.suffix}.npz"),
             loss_total=np.array([r["total"] for r in rows]),
             loss_jepa =np.array([r["jepa"]  for r in rows]),
             loss_hnn  =np.array([r["hnn"]   for r in rows]),
             loss_lnn  =np.array([r["lnn"]   for r in rows]),
             loss_sup  =np.array([r["sup"]   for r in rows]),)


def run_all_modes(modes: list[str],
                  dl: DataLoader,
                  *,
                  base_cfg: TrainConfig | None = None):
    """
    Train every mode in `modes` **sequentially** on the same DataLoader.

    Parameters
    ----------
    modes     : list like ['plain','hnn', …]
    dl        : DataLoader built once outside
    base_cfg  : optional template; fields are copied into every preset
    """
    for mode in modes:
        cfg = TrainConfig.preset(mode)
        if base_cfg is not None:          # copy user overrides
            cfg.__dict__.update({k: v for k, v in base_cfg.__dict__.items()
                                 if k != "mode"})
        run_mode(cfg, dl)


# ====================================================================
# 4 · Optional CLI entry-point
# ====================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="One-line trainer for V-JEPA physics variants.")
    parser.add_argument("mode", choices=["plain", "hnn", "lnn", "hnn+lnn"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch",  type=int, default=32)
    parser.add_argument("--suffix", default="_dense")
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--log-dir",   default="./results_numpy")
    args = parser.parse_args()

    # rebuild dataset (adjust if you have a central dataset module)
    train_ds = PendulumDataset(num_episodes=40, episode_length=400,
                               img_size=64, seq_len=3)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)

    # get default config and overwrite CLI overrides
    cfg = TrainConfig.preset(args.mode)
    cfg.epochs     = args.epochs
    cfg.batch_size = args.batch
    cfg.suffix     = args.suffix
    cfg.model_dir  = args.model_dir
    cfg.log_dir    = args.log_dir

    run_mode(cfg, train_dl)