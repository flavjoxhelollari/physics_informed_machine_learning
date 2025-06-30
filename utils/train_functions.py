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
from typing      import Dict, Literal, Optional, List

import numpy as np
import torch
import torch.nn        as nn
import torch.nn.functional as F
import torch.optim     as optim
from torch.utils.data  import DataLoader
from tqdm import tqdm

# --------------------------------------------------------------------
# Project-local modules (must be import-able from your PYTHONPATH)
# --------------------------------------------------------------------
from utils.models  import VJEPA, HNN, LNN
from utils.datasets import PendulumDataset

# device selection once per module
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====================================================================
# 1 · Dataclass : all knobs live in one place
# ====================================================================
@dataclass
class TrainConfig:
    """
    Hyper-parameter container.

    Parameters
    ----------
    mode : {'plain','hnn','lnn','hnn+lnn'}
        Which loss terms are active.
    epochs, batch_size, lr
        Usual SGD knobs.
    embed_dim, depth, num_heads
        ViT backbone size.
    h_hidden, l_hidden
        Width of the HNN / LNN MLPs.
    λ_hnn, λ_lnn, λ_sup
        Loss weights (λ_sup is small supervised loss for θ̂).
    suffix
        Suffix appended to checkpoint/log filenames.
    model_dir, log_dir
        Output directories (created on the fly).
    """
    mode       : Literal["plain", "hnn", "lnn", "hnn+lnn"]
    epochs     : int   = 10
    batch_size : int   = 32
    lr         : float = 1e-4

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

    # -------- presets that replicate your original four modes -------
    @staticmethod
    def preset(mode: Literal["plain","hnn","lnn","hnn+lnn"]) -> "TrainConfig":
        cfg = TrainConfig(mode=mode)
        presets = {
            "plain"   : dict(λ_hnn=0.0,   λ_lnn=0.0),
            "hnn"     : dict(λ_hnn=1e-3,  λ_lnn=0.0),
            "lnn"     : dict(λ_hnn=0.0,   λ_lnn=1e-3),
            "hnn+lnn" : dict(λ_hnn=5e-4,  λ_lnn=1e-3),
        }
        cfg.__dict__.update(presets[mode])
        return cfg


# ====================================================================
# 2 · Single-epoch loop (unchanged math, heavily commented)
# ====================================================================
def train_one_epoch(
    model      : VJEPA,
    theta_head : nn.Linear,
    lnn        : Optional[LNN],
    hnn        : Optional[HNN],
    loader     : DataLoader,
    opt        : optim.Optimizer,
    λ_lnn      : float,
    λ_hnn      : float,
    λ_sup      : float
) -> Dict[str, float]:
    """Run one epoch and return averaged component losses."""
    # switch to training mode
    model.train(); theta_head.train()
    if lnn: lnn.train()
    if hnn: hnn.train()

    # rolling sum of per-batch losses
    agg = dict(jepa=0., lnn=0., hnn=0., sup=0., total=0.)

    for imgs_seq, states_seq in loader:
        # move batch to GPU (if any)
        imgs_seq, states_seq = imgs_seq.to(device), states_seq.to(device)

        # --- frame-0 only for JEPA & supervised head ----------------
        imgs0  = imgs_seq[:, 0]          # (B,C,H,W)
        θ_true = states_seq[:, :, 0:1]   # (B,T,1)
        ω_true = states_seq[:, :, 1:2]   # (B,T,1)

        # 1) JEPA self-supervised loss
        loss_jepa, _, _ = model(imgs0)

        # 2) latent → (θ̂, ω̂)
        z0   = model.patch_embed(imgs0) + model.pos_embed
        z0   = model.context_encoder(z0).mean(1)
        θ̂0, ω̂0 = theta_head(z0).split(1, 1)

        # 3) HNN residual (if enabled)
        hnn_loss = torch.tensor(0., device=device)
        if hnn and λ_hnn > 0:
            qp       = torch.cat([θ̂0, ω̂0], dim=1)
            hnn_loss = F.mse_loss(hnn.time_derivative(qp)[:, 0:1], ω̂0)

        # 4) LNN residual (if enabled)
        lnn_loss = torch.tensor(0., device=device)
        if lnn and λ_lnn > 0:
            lnn_loss = lnn.lagrangian_residual(θ_true, ω_true)

        # 5) small supervised loss to keep head aligned
        sup_loss = F.mse_loss(θ̂0,   θ_true[:, 0])

        # 6) total
        loss = (loss_jepa
                + λ_lnn * lnn_loss
                + λ_hnn * hnn_loss
                + λ_sup * sup_loss)

        # backward + step
        opt.zero_grad(); loss.backward(); opt.step()

        # accumulate
        agg["jepa"]  += loss_jepa.item()
        agg["lnn"]   += lnn_loss.item()
        agg["hnn"]   += hnn_loss.item()
        agg["sup"]   += sup_loss.item()
        agg["total"] += loss.item()

    # average across batches
    for k in agg:
        agg[k] /= len(loader)
    return agg


# ====================================================================
# 3 · High-level run helper
# ====================================================================
def run_mode(cfg: TrainConfig, dataloader: DataLoader) -> None:
    """
    Train a model according to `cfg` on `dataloader`, then save:

    * `model_<mode><suffix>.pt`  (single file with all sub-nets)
    * `train_<mode><suffix>.csv` (per-epoch losses)
    * `results_<mode><suffix>.npz` (legacy arrays, +config)
    """
    # ensure directories exist
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.log_dir,   exist_ok=True)

    print(f"\n###  {cfg.mode.upper()}  |  {cfg.epochs} epochs  ###")

    # --- build networks ----------------------------------------------
    model = VJEPA(cfg.embed_dim, depth=cfg.depth,
                  num_heads=cfg.num_heads).to(device)
    theta_head = nn.Linear(cfg.embed_dim, 2).to(device)
    hnn = HNN(cfg.h_hidden).to(device) if cfg.λ_hnn > 0 else None
    lnn = LNN(2, cfg.l_hidden).to(device) if cfg.λ_lnn > 0 else None

    # unified optimiser
    opt = optim.AdamW(
        list(model.parameters())
        + list(theta_head.parameters())
        + ([] if hnn is None else list(hnn.parameters()))
        + ([] if lnn is None else list(lnn.parameters())),
        lr=cfg.lr)

    # --- training loop ------------------------------------------------
    log_rows: List[Dict[str, float]] = []

    for ep in range(cfg.epochs):
        ep_loss = train_one_epoch(model, theta_head,
                                  lnn, hnn, dataloader, opt,
                                  cfg.λ_lnn, cfg.λ_hnn, cfg.λ_sup)
        log_rows.append({"epoch": ep+1, **ep_loss})

        print(f"ep {ep+1:02d}  "
              f"tot {ep_loss['total']:.3f} | "
              f"j {ep_loss['jepa'] :.3f} | "
              f"h {ep_loss['hnn']  :.3f} | "
              f"l {ep_loss['lnn']  :.3f}")

    # --- checkpoint ---------------------------------------------------
    ckpt = {}
    ckpt.update({f"vjepa.{k}"      : v.cpu() for k,v in model.state_dict().items()})
    ckpt.update({f"theta_head.{k}" : v.cpu() for k,v in theta_head.state_dict().items()})
    if hnn: ckpt.update({f"hnn.{k}" : v.cpu() for k,v in hnn.state_dict().items()})
    if lnn: ckpt.update({f"lnn.{k}" : v.cpu() for k,v in lnn.state_dict().items()})

    ckpt_file = os.path.join(cfg.model_dir, f"model_{cfg.mode}{cfg.suffix}.pt")
    torch.save(ckpt, ckpt_file)
    print("✓ checkpoint →", ckpt_file)

    # --- CSV log ------------------------------------------------------
    csv_path = os.path.join(cfg.log_dir, f"train_{cfg.mode}{cfg.suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader(); writer.writerows(log_rows)
    print("✓ CSV log     →", csv_path)

    # --- NPZ legacy log ----------------------------------------------
    npz_path = os.path.join(cfg.log_dir, f"results_{cfg.mode}{cfg.suffix}.npz")
    np.savez(npz_path,
             loss_total = np.array([r["total"] for r in log_rows]),
             loss_jepa  = np.array([r["jepa"]  for r in log_rows]),
             loss_hnn   = np.array([r["hnn"]   for r in log_rows]),
             loss_lnn   = np.array([r["lnn"]   for r in log_rows]),
             loss_sup   = np.array([r["sup"]   for r in log_rows]),
             config     = asdict(cfg))
    print("✓ NPZ log     →", npz_path)


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