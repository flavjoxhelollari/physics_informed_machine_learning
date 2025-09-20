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
from utils.models   import *
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

    # --- NEW: time step used for FD loss / rollout-consistency ---
    dt         : float = 0.05

    # --- supervision ---
    sup_use_frames: Literal["first", "all", "N"] = "first"
    sup_num_frames: int = 3
    normalize_sup: bool = True
    B_OMEGA: float = 1.0

    # --- NEW: FD-consistency & HNN target knobs (optional) ---
    λ_fd: float = 0.0           # 0 disables FD consistency
    fd_use_gt: bool = True      # use (θ_{t+1}-θ_t)/dt vs (θ̂_{t+1}-θ̂_t)/dt
    hnn_target: Literal["gt","pred"] = "gt"

    # --- optimization ---
    weight_decay: float = 1e-4
    max_grad_norm: Optional[float] = None
    warmup_epochs: int = 1
    cosine_final_lr_ratio: float = 0.1

    # --- EMA ---
    use_ema: bool = True
    ema_decay: float = 0.999
    use_ema_head: bool = True   # applies to both heads now

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

def _ctx(model: VJEPA, imgs: torch.Tensor) -> torch.Tensor:
    # (B,3,H,W) -> (B,D)
    return model.context_encoder(model.patch_embed(imgs) + model.pos_embed).mean(1)

def train_one_epoch(
    model: VJEPA,
    theta_head: nn.Module,          # ThetaHead1F(z)->(B,1)
    omega_head: nn.Module,          # OmegaHead2F(z_t, z_t1, dt)->(B,1)
    lnn: Optional[LNN],
    hnn: Optional[HNN],
    loader: DataLoader,
    opt: optim.Optimizer,
    λ_lnn: float,
    λ_hnn: float,
    λ_sup: float,
    *,
    dt: float,                      # NEW: Δt for FD target and ω head
    normalize_sup: bool = True,
    B_OMEGA: float = 1.0,
    sup_use_frames: str = "first",  # "first" | "all" | "N"
    sup_num_frames: int = 3,        # used only when "N"
    λ_fd: float = 0.0,              # NEW: weight for FD consistency
    fd_use_gt: bool = True,         # NEW: FD target uses GT vs model θ̂
    hnn_target: Literal["gt","pred"] = "gt",  # NEW: what dθ/dt is matched to
    max_grad_norm: float | None = None,
    ema_updater: EMA | dict | None = None     # support {"encoder","theta_head","omega_head"}
) -> Dict[str, float]:

    model.train(); theta_head.train(); omega_head.train()
    if lnn: lnn.train()
    if hnn: hnn.train()

    agg = dict(
        jepa=0., lnn=0., hnn=0.,
        sup=0., sup_theta=0., sup_omega=0., sup_fd=0.,
        total=0.
    )
    THETA_SCALE, OMEGA_SCALE = math.pi, 5.0

    for imgs_seq, states_seq in loader:
        imgs_seq   = imgs_seq.to(device)           # (B,T,3,H,W)
        states_seq = states_seq.to(device)         # (B,T,2)
        B, T = imgs_seq.shape[0], imgs_seq.shape[1]
        idxs = _pick_supervision_frames(T, sup_use_frames, sup_num_frames)

        θ_true = states_seq[:, :, 0:1]             # (B,T,1)
        ω_true = states_seq[:, :, 1:2]             # (B,T,1)

        # ------- JEPA pretext on frame 0 -------
        imgs0 = imgs_seq[:, 0]
        loss_jepa, _, _ = model(imgs0)

        # ------- cache latents z_t for used frames (and t+1 for ω) -------
        need = set(idxs) | {t+1 for t in idxs if t+1 < T}
        z_cache: dict[int, torch.Tensor] = {}
        for t in need:
            z_cache[t] = _ctx(model, imgs_seq[:, t])   # (B,D)

        # ------- supervised θ/ω + FD consistency -------
        sup_theta = torch.zeros((), device=device)
        sup_omega = torch.zeros((), device=device)
        sup_fd    = torch.zeros((), device=device)
        nθ, nω = 0, 0

        θhat_cache: dict[int, torch.Tensor] = {}

        for t in idxs:
            zt = z_cache[t]
            θ̂t = theta_head(zt)                    # (B,1)
            θhat_cache[t] = θ̂t

            θt = θ_true[:, t]                      # (B,1)
            if normalize_sup:
                sup_theta = sup_theta + F.mse_loss(θ̂t / THETA_SCALE, θt / THETA_SCALE)
            else:
                sup_theta = sup_theta + F.mse_loss(θ̂t, θt)
            nθ += 1

            # ω supervision needs t+1
            if t + 1 < T:
                zt1 = z_cache[t+1]
                ω̂t  = omega_head(zt, zt1, dt=dt)  # (B,1)
                ωt  = ω_true[:, t]                 # (B,1)

                if normalize_sup:
                    sup_omega = sup_omega + F.mse_loss(ω̂t / OMEGA_SCALE, ωt / OMEGA_SCALE)
                else:
                    sup_omega = sup_omega + F.mse_loss(ω̂t, ωt)
                nω += 1

                # FD consistency
                if λ_fd > 0.0:
                    if fd_use_gt:
                        ω_fd = (θ_true[:, t+1] - θ_true[:, t]) / dt      # (B,1)
                    else:
                        # need θ̂_{t+1}
                        if (t+1) not in θhat_cache:
                            θhat_cache[t+1] = theta_head(zt1)
                        ω_fd = (θhat_cache[t+1] - θ̂t) / dt
                    sup_fd = sup_fd + F.mse_loss(ω̂t, ω_fd)

        # means
        if nθ > 0: sup_theta = sup_theta / nθ
        if nω > 0:
            sup_omega = sup_omega / nω
            if λ_fd > 0.0: sup_fd = sup_fd / nω

        sup_loss = sup_theta + B_OMEGA * sup_omega + λ_fd * sup_fd

        # ------- physics aux losses -------
        # HNN: compare dθ/dt vs chosen target (GT or predicted ω̂)
        hnn_loss = torch.zeros((), device=device)
        if hnn and λ_hnn > 0:
            z0  = z_cache.get(0, _ctx(model, imgs_seq[:, 0]))
            z1  = z_cache.get(1, _ctx(model, imgs_seq[:, 1])) if T > 1 else z0
            θ̂0 = theta_head(z0)                              # (B,1)
            if hnn_target == "gt":
                ω_tar = ω_true[:, 0]                          # (B,1)
            else:
                ω_tar = omega_head(z0, z1, dt=dt).detach()    # (B,1)  stop grad into ω-head here

            x0 = torch.cat([θ̂0.detach(), ω_tar], dim=1).requires_grad_(True)  # (B,2) with grad
            td = hnn.time_derivative(x0)                      # (B,2) or (B,1)
            dθdt = td[:, 0:1] if (td.ndim == 2 and td.size(-1) >= 2) else td.reshape(-1, 1)
            hnn_loss = F.mse_loss(dθdt, ω_tar)

        # LNN: your residual over full (θ_true, ω_true) tensor
        lnn_loss = torch.zeros((), device=device)
        if lnn and λ_lnn > 0:
            lnn_loss = lnn.lagrangian_residual(θ_true, ω_true)

        # ------- total & step -------
        loss = loss_jepa + λ_lnn * lnn_loss + λ_hnn * hnn_loss + λ_sup * sup_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters())
                + list(theta_head.parameters())
                + list(omega_head.parameters())
                + ([] if not hnn else list(hnn.parameters()))
                + ([] if not lnn else list(lnn.parameters())),
                max_norm=max_grad_norm
            )
        opt.step()

        # ------- EMA updates (support dict or legacy single) -------
        if ema_updater is not None:
            if isinstance(ema_updater, dict):
                if ema_updater.get("encoder") is not None:
                    ema_updater["encoder"].update(model)
                if ema_updater.get("theta_head") is not None:
                    ema_updater["theta_head"].update(theta_head)
                if ema_updater.get("omega_head") is not None:
                    ema_updater["omega_head"].update(omega_head)
            else:
                ema_updater.update(model)  # legacy single-EMA-on-encoder

        # ------- logging -------
        agg["jepa"]      += loss_jepa.item()
        agg["lnn"]       += lnn_loss.item()
        agg["hnn"]       += hnn_loss.item()
        agg["sup"]       += sup_loss.item()
        agg["sup_theta"] += sup_theta.item()
        agg["sup_omega"] += (sup_omega.item() if nω > 0 else 0.0)
        agg["sup_fd"]    += (sup_fd.item() if (λ_fd > 0 and nω > 0) else 0.0)
        agg["total"]     += loss.item()

    for k in agg: agg[k] /= len(loader)
    return agg

# ====================================================================
# 3 · Train-and-save
# ====================================================================
def run_mode(cfg: TrainConfig, dataloader: DataLoader) -> None:
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.log_dir,   exist_ok=True)

    print(f"\n### {cfg.mode.upper()} | {cfg.epochs} epochs ###")

    # -------- models --------
    model = VJEPA(img_size=cfg.img_size,
                  patch_size=cfg.patch_size,
                  embed_dim=cfg.embed_dim,
                  depth=cfg.depth,
                  num_heads=cfg.num_heads).to(device)

    # separate heads
    theta_head = ThetaHead1F(cfg.embed_dim).to(device)        # Linear(D->1)
    omega_head = OmegaHead2F(cfg.embed_dim).to(device)        # MLP(2D->1), uses z_t, z_{t+1}, dt

    hnn = HNN(cfg.h_hidden).to(device) if cfg.λ_hnn > 0 else None
    lnn = LNN(2, cfg.l_hidden).to(device) if cfg.λ_lnn > 0 else None

    # -------- optimizer (AdamW + WD) --------
    params = (
        list(model.parameters())
        + list(theta_head.parameters())
        + list(omega_head.parameters())
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
    ema_encoder    = EMA(model,       decay=cfg.ema_decay) if cfg.use_ema else None
    ema_theta_head = EMA(theta_head,  decay=cfg.ema_decay) if (cfg.use_ema and cfg.use_ema_head) else None
    ema_omega_head = EMA(omega_head,  decay=cfg.ema_decay) if (cfg.use_ema and getattr(cfg, "use_ema_omega_head", True)) else None

    rows: List[Dict[str, float]] = []
    for ep in range(cfg.epochs):
        # warmup
        if ep < warmup:
            for g in opt.param_groups:
                g["lr"] = cfg.lr * (ep + 1) / max(1, warmup)

        loss_d = train_one_epoch(
            model=model,
            theta_head=theta_head,
            omega_head=omega_head,
            lnn=lnn,
            hnn=hnn,
            loader=dataloader,
            opt=opt,
            λ_lnn=cfg.λ_lnn,
            λ_hnn=cfg.λ_hnn,
            λ_sup=cfg.λ_sup,
            dt=cfg.dt,                             # NEW
            normalize_sup=cfg.normalize_sup,
            B_OMEGA=cfg.B_OMEGA,
            sup_use_frames=cfg.sup_use_frames,
            sup_num_frames=cfg.sup_num_frames,
            λ_fd=getattr(cfg, "λ_fd", 0.0),        # NEW
            fd_use_gt=getattr(cfg, "fd_use_gt", True),
            hnn_target=getattr(cfg, "hnn_target", "gt"),
            max_grad_norm=cfg.max_grad_norm,
            ema_updater={
                "encoder":    ema_encoder,
                "theta_head": ema_theta_head,
                "omega_head": ema_omega_head,
            },
        )

        rows.append({"epoch": ep+1, "lr": opt.param_groups[0]["lr"], **loss_d})
        print(f"ep{ep+1:02d}",
              f"lr={opt.param_groups[0]['lr']:.2e}",
              " ".join([f"{k}={v:.3f}" for k,v in loss_d.items()]))

        scheduler.step()

    # -------- choose which weights to save: EMA or raw --------
    model_to_save = copy.deepcopy(model).to("cpu")
    θhead_to_save = copy.deepcopy(theta_head).to("cpu")
    ωhead_to_save = copy.deepcopy(omega_head).to("cpu")

    if cfg.use_ema:
        if ema_encoder    is not None: ema_encoder.copy_to(model_to_save)
        if ema_theta_head is not None: ema_theta_head.copy_to(θhead_to_save)
        if ema_omega_head is not None: ema_omega_head.copy_to(ωhead_to_save)

    # pack checkpoint (names match your loaders)
    ckpt = {f"vjepa.{k}": v for k, v in model_to_save.state_dict().items()}
    ckpt.update({f"theta_head.{k}": v for k, v in θhead_to_save.state_dict().items()})
    ckpt.update({f"omega_head.{k}": v for k, v in ωhead_to_save.state_dict().items()})
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
             loss_sup  =np.array([r["sup"]   for r in rows]),
             loss_sup_theta=np.array([r["sup_theta"] for r in rows]),
             loss_sup_omega=np.array([r["sup_omega"] for r in rows]),
             loss_sup_fd=np.array([r["sup_fd"] for r in rows]),
    )

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