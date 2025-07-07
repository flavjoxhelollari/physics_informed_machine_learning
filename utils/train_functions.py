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

# ====================================================================
# 2 · One-epoch loop
# ====================================================================
def train_one_epoch(
    model: VJEPA,
    theta_head: nn.Linear,
    lnn: Optional[LNN],
    hnn: Optional[HNN],
    loader: DataLoader,
    opt: optim.Optimizer,
    λ_lnn: float,
    λ_hnn: float,
    λ_sup: float
) -> Dict[str,float]:

    model.train(); theta_head.train()
    if lnn: lnn.train()
    if hnn: hnn.train()

    agg = dict(jepa=0., lnn=0., hnn=0., sup=0., total=0.)

    for imgs_seq, states_seq in loader:
        imgs_seq, states_seq = imgs_seq.to(device), states_seq.to(device)
        imgs0  = imgs_seq[:,0]
        θ_true = states_seq[:,:,0:1]
        ω_true = states_seq[:,:,1:2]

        loss_jepa, _, _ = model(imgs0)

        z0 = model.patch_embed(imgs0) + model.pos_embed
        z0 = model.context_encoder(z0).mean(1)
        θ̂0, ω̂0 = theta_head(z0).split(1,1)

        hnn_loss = torch.tensor(0., device=device)
        if hnn and λ_hnn>0:
            hnn_loss = F.mse_loss(hnn.time_derivative(torch.cat([θ̂0, ω̂0],1))[:,0:1], ω̂0)

        lnn_loss = torch.tensor(0., device=device)
        if lnn and λ_lnn>0:
            lnn_loss = lnn.lagrangian_residual(θ_true, ω_true)

        sup_loss = F.mse_loss(θ̂0, θ_true[:,0])

        loss = loss_jepa + λ_lnn*lnn_loss + λ_hnn*hnn_loss + λ_sup*sup_loss

        opt.zero_grad(); loss.backward(); opt.step()

        agg["jepa"]  += loss_jepa.item()
        agg["lnn"]   += lnn_loss.item()
        agg["hnn"]   += hnn_loss.item()
        agg["sup"]   += sup_loss.item()
        agg["total"] += loss.item()

    for k in agg: agg[k] /= len(loader)
    return agg

# ====================================================================
# 3 · Train-and-save
# ====================================================================
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

    opt = optim.AdamW(
        list(model.parameters())
        + list(theta_head.parameters())
        + ([] if hnn is None else list(hnn.parameters()))
        + ([] if lnn is None else list(lnn.parameters())),
        lr=cfg.lr)

    rows: List[Dict[str,float]] = []
    for ep in range(cfg.epochs):
        loss_d = train_one_epoch(model, theta_head,
                                 lnn, hnn, dataloader, opt,
                                 cfg.λ_lnn, cfg.λ_hnn, cfg.λ_sup)
        rows.append({"epoch": ep+1, **loss_d})
        print(f"ep{ep+1:02d}",
              " ".join([f"{k}={v:.3f}" for k,v in loss_d.items()]))

    ckpt = {f"vjepa.{k}":v.cpu() for k,v in model.state_dict().items()}
    ckpt.update({f"theta_head.{k}":v.cpu() for k,v in theta_head.state_dict().items()})
    if hnn: ckpt.update({f"hnn.{k}":v.cpu() for k,v in hnn.state_dict().items()})
    if lnn: ckpt.update({f"lnn.{k}":v.cpu() for k,v in lnn.state_dict().items()})

    torch.save(ckpt, os.path.join(cfg.model_dir, f"model_{cfg.mode}{cfg.suffix}.pt"))

    csv_path = os.path.join(cfg.log_dir, f"train_{cfg.mode}{cfg.suffix}.csv")
    with open(csv_path,"w",newline="") as f:
        csv.DictWriter(f, fieldnames=rows[0].keys()).writeheader(); f.writerows(rows)

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