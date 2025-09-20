"""
evaluation_functions.py
───────────────────────
Inference-side utilities for the Pendulum-VJEPA study.

This module is intentionally **stand-alone**: the only external
dependencies are `models.py`, `loading_functions.py`,
`evaluation_metrics.py` and `helper_functions.py`.

Public API
----------
EvalConfig            – central container for *all* evaluation knobs  
evaluate_mode         – run a single checkpoint, compute metrics, dump JSON  
evaluate_all_modes    – loop `evaluate_mode` over a list of modes  
collect_head_scatter  – gather (θ_true, ω_true, θ̂, ω̂) arrays  
head_regression_metrics – tiny helper for (R², MSE) of the latent head  
normalise_keys        – legacy →   loss_*  key renamer  
plot_loss             – quick multi-run loss visualiser
"""

from __future__ import annotations  # ← forward-ref typing (Py < 3.11)

import os, json                                   # file I/O helpers
from dataclasses import dataclass, asdict         # lightweight settings
from typing      import Dict, Iterable, List, Literal      # static typing

import numpy as np                                # numeric arrays
import torch                                      # tensors & autograd
import matplotlib.pyplot as plt                   # plotting
from torch.utils.data import DataLoader           # batching
from tqdm       import tqdm                  # nice progress-bars
from sklearn.metrics import r2_score, mean_squared_error  # regression
from sklearn.linear_model import LinearRegression          # linear-fit

# --------------------------------------------------------------------
# Project-local imports (paths must be on PYTHONPATH when importing)
# --------------------------------------------------------------------
from utils.models              import VJEPA, HNN, LNN                # NN classes
from utils.loading_functions   import load_components                # ckpt splitter
from utils.evaluation_metrics  import *
from utils.helper_functions    import rollout                        # latent rollout

# CUDA if available, otherwise CPU
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====================================================================
# 1 · Dataclass holding every evaluation hyper-parameter
# ====================================================================
# evaluation_config.py

@dataclass
class EvalConfig:
    """
    All evaluation knobs for a single run.

    Paths / I/O
    -----------
    suffix     : checkpoint suffix used when loading (e.g., "_dense")
    model_dir  : directory containing model_*.pt
    out_dir    : directory to write metrics_*.json (+ optional curves .npz)

    Rollout
    -------
    horizon    : number of rollout time steps
    dt         : integration time step
    rollout_init : how to initialise (θ,ω) from the sequence frames.
                   "first" → frame-0 only (current rollout default)
                   "mean3" → average encodings from first 3 frames
                   (requires the tiny patch in `rollout` you added)

    Physics constants
    -----------------
    m, g, l    : mass, gravity, length (for energy & analytic α)

    Batching
    --------
    batch_size : eval DataLoader batch size

    Latent-head diagnostics (scatter)
    ---------------------------------
    scatter_samples : number of examples to probe (t=0)
    scatter_batch   : batch size for the probe loader
    scatter_shuffle : shuffle the probe loader
    scatter_seed    : optional RNG seed (applied only if shuffle=True)

    Neighbour-divergence (Δ_div) – dispatcher & reduction
    -----------------------------------------------------
    ndiv_mode    : {"pairwise","grid","window","knn"}
      • "pairwise": O(N^2) within ε
      • "grid"   : only neighbouring ICs on a rectangular grid (stencil = "4" or "8")
      • "window" : fixed small deltas in (θ,ω) space (|Δθ|≤dθ, |Δω|≤dω)
      • "knn"    : k nearest neighbours in initial phase-space
    ndiv_stencil : {"4","8"}    (used when ndiv_mode="grid")
    ndiv_k       : int          (used when ndiv_mode="knn")
    ndiv_dtheta  : float | None (used when ndiv_mode="window")
    ndiv_domega  : float | None (used when ndiv_mode="window")
    ndiv_eps     : float        (radius used by "pairwise" — legacy)
    ndiv_reduce  : {"step","mean"}   how to reduce the Δ(t) curve → scalar
    ndiv_step    : int               step index if reduce="step" (supports negatives)

    ▼ Replicate your *old* Δ_div (pairwise at final step):
        ndiv_mode="pairwise", ndiv_eps=0.1, ndiv_reduce="step", ndiv_step=-1

    Euler–Lagrange residual (EL) – throttles & reduction
    ----------------------------------------------------
    el_t_max   : None|int   → use only first `t_max` steps (after stride)
    el_stride  : int        → temporal downsampling factor
    el_max_ic  : None|int   → limit number of trajectories (ICs)
    el_reduce  : {"mean","step"}   curve → scalar
    el_step    : int               index if reduce="step" (supports negatives)

    ▼ Replicate your *old* EL metric (mean over first 3 frames):
        el_t_max=3, el_stride=1, el_max_ic=None, el_reduce="mean"

    Curves persistence
    ------------------
    save_curves : if True, evaluate_mode will write a compressed .npz
                  with Δ_curve / E_curve / (EL_curve if available),
                  and place a pointer path into the JSON.
    """

    # ---------- Paths ----------
    suffix:    str = "_dense"
    model_dir: str = "./models"
    out_dir:   str = "./metrics"

    # ---------- Rollout ----------
    horizon: int = 60
    dt:      float = 0.05
    rollout_init: Literal["first","mean3"] = "first"  # needs rollout patch for "mean3"

    # ---------- Physics ----------
    m: float = 1.0
    g: float = 9.81
    l: float = 1.0

    # ---------- Batching ----------
    batch_size: int = 64

    # ---------- Scatter diagnostics ----------
    scatter_samples: int = 500
    scatter_batch:   int = 64
    scatter_shuffle: bool = True
    scatter_seed:    int | None = None

    # ---------- Neighbour-divergence ----------
    ndiv_mode:    Literal["pairwise","grid","window","knn"] = "grid"
    ndiv_stencil: Literal["4","8"] = "8"     # only used for mode="grid"
    ndiv_k:       int = 8                    # only used for mode="knn"
    ndiv_dtheta:  float | None = None        # only used for mode="window"
    ndiv_domega:  float | None = None        # only used for mode="window"
    ndiv_eps:     float = 0.1                # only used for mode="pairwise"
    ndiv_reduce:  Literal["step","mean"] = "step"
    ndiv_step:    int = -1

    theta_axis: Optional[np.ndarray] = None
    omega_axis: Optional[np.ndarray] = None

    # ---------- Euler–Lagrange residual ----------
    el_t_max:  int | None = None
    el_stride: int = 1
    el_max_ic: int | None = None
    el_reduce: Literal["mean","step"] = "mean"
    el_step:   int = -1

    # ---------- Curves persistence ----------
    save_curves: bool = False


def collect_head_scatter(
    model: VJEPA,
    head:  torch.nn.Module | None,
    dataset,
    *,
    n_samples: int,
    batch:     int,
    shuffle:   bool = True,
    seed:      int | None = None,
    theta_head_1f: torch.nn.Module | None = None,
    omega_head_2f: torch.nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    if head is not None: head.eval()
    if theta_head_1f is not None: theta_head_1f.eval()
    if omega_head_2f is not None: omega_head_2f.eval()

    mdev = next(model.parameters()).device
    n_total = len(dataset)
    n_take  = min(n_samples, n_total)
    if n_take <= 0:
        return (np.empty(0),)*4

    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    loader = DataLoader(dataset, batch_size=batch, shuffle=shuffle,
                        generator=g if shuffle and seed is not None else None)

    θ_true, ω_true, θ_pred, ω_pred = [], [], [], []
    with torch.no_grad():
        for seq, states in loader:
            imgs0 = seq[:, 0].to(mdev)
            z0 = model.context_encoder(model.patch_embed(imgs0) + model.pos_embed).mean(1)

            if (theta_head_1f is not None) and (omega_head_2f is not None) and (seq.size(1) >= 2):
                imgs1 = seq[:, 1].to(mdev)
                z1 = model.context_encoder(model.patch_embed(imgs1) + model.pos_embed).mean(1)
                θ_hat = theta_head_1f(z0).squeeze(1)
                ω_hat = omega_head_2f(torch.cat([z0, z1], dim=1)).squeeze(1)
            else:
                assert head is not None, "need legacy head or both theta/omega heads"
                θ_hat, ω_hat = head(z0).split(1, dim=1)
                θ_hat, ω_hat = θ_hat.squeeze(1), ω_hat.squeeze(1)

            θ_true.extend(states[:, 0, 0].cpu().numpy())
            ω_true.extend(states[:, 0, 1].cpu().numpy())
            θ_pred.extend(θ_hat.detach().cpu().numpy())
            ω_pred.extend(ω_hat.detach().cpu().numpy())

            if len(θ_true) >= n_take:
                break

    return (np.asarray(θ_true)[:n_take],
            np.asarray(ω_true)[:n_take],
            np.asarray(θ_pred)[:n_take],
            np.asarray(ω_pred)[:n_take])

# ====================================================================
# 3 · Tiny regression helper (latent-head)
# ====================================================================
def head_regression_metrics(
    θ_t: np.ndarray, ω_t: np.ndarray,
    θ_p: np.ndarray, ω_p: np.ndarray
) -> Dict[str, float]:
    """Return R² and MSE for θ and ω."""
    # Optional: drop NaNs/Infs to avoid sklearn errors
    def _clean(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return a[m], b[m]

    θ_t, θ_p = _clean(θ_t, θ_p)
    ω_t, ω_p = _clean(ω_t, ω_p)

    return dict(
        r2_theta  = float(r2_score(θ_t, θ_p))  if θ_t.size else 0.0,
        mse_theta = float(mean_squared_error(θ_t, θ_p)) if θ_t.size else 0.0,
        r2_omega  = float(r2_score(ω_t, ω_p))  if ω_t.size else 0.0,
        mse_omega = float(mean_squared_error(ω_t, ω_p)) if ω_t.size else 0.0,
    )

# ====================================================================
# 4 · Evaluate ONE mode  ––  **now aware of `combine`**
# ====================================================================
# ===========================================
# IC/grid helpers (place near top of file)
# ===========================================
def _initial_conditions(theta: torch.Tensor, omega: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    θ0 = theta[:, 0].detach().cpu().numpy()
    ω0 = omega[:, 0].detach().cpu().numpy()
    return θ0, ω0

def _reorder_to_grid(theta: torch.Tensor, omega: torch.Tensor,
                     theta0: np.ndarray, omega0: np.ndarray,
                     theta_axis: np.ndarray, omega_axis: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reorder (θ, ω) so index = iθ * Om + jω (θ outer, ω inner).
    Each IC is mapped to the NEAREST (i,j) on supplied axes.
    """
    Th, Om = theta_axis.size, omega_axis.size

    # nearest index along each axis
    ii = np.abs(theta_axis[None, :] - theta0[:, None]).argmin(axis=1)  # (N,)
    jj = np.abs(omega_axis[None, :] - omega0[:, None]).argmin(axis=1)  # (N,)

    order = ii * Om + jj                                               # (N,)
    inv_perm = np.argsort(order).astype(np.int64)
    inv_perm_t = torch.from_numpy(inv_perm).to(theta.device)

    return theta.index_select(0, inv_perm_t), omega.index_select(0, inv_perm_t)

import numpy as np
import torch
from typing import Tuple, Optional

def collapse_to_grid_ics(
    theta: torch.Tensor,   # (N, T) rollout θ (CPU or CUDA)
    omega: torch.Tensor,   # (N, T) rollout ω
    *,
    theta0: np.ndarray,    # (N,) TRUE initial θ for each window (from dataset labels at t=0)
    omega0: np.ndarray,    # (N,) TRUE initial ω for each window (from dataset labels at t=0)
    theta_axis: np.ndarray,  # (Th,)
    omega_axis: np.ndarray,  # (Om,)
    require_full_cover: bool = True,   # error if any grid cell has no representative
    tol: Optional[float] = None        # optional max distance to accept a representative
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Collapse many windows per IC into exactly one trajectory per (θ_axis[i], ω_axis[j]) cell.

    Returns
    -------
    theta_c : (Th*Om, T)
    omega_c : (Th*Om, T)
    chosen  : (Th*Om,) int array of source indices used for each grid cell (row-major θ-outer, ω-inner)

    Notes
    -----
    • Uses *true* θ0/ω0 to pick the representative (closest window to the grid point).
    • Reorders outputs to row-major (iθ outer, jω inner).
    • If `require_full_cover=False`, cells with no candidate are skipped; outputs shrink accordingly.
    """
    assert theta.ndim == 2 and omega.ndim == 2, "θ/ω must be (N,T)"
    N, T = theta.shape
    assert omega.shape == (N, T)
    assert theta0.shape == (N,) and omega0.shape == (N,), "theta0/omega0 must be (N,) arrays"

    # Work on CPU for indexing
    dev = theta.device
    θ = theta.detach().cpu()
    ω = omega.detach().cpu()

    Th, Om = int(theta_axis.size), int(omega_axis.size)

    # Build (Th*Om, 2) grid of target ICs
    grid_θ, grid_ω = np.meshgrid(theta_axis, omega_axis, indexing="ij")  # θ-outer, ω-inner
    targets = np.stack([grid_θ.reshape(-1), grid_ω.reshape(-1)], axis=1) # (Th*Om, 2)

    # All window ICs (N, 2)
    src = np.stack([theta0, omega0], axis=1)

    chosen = np.full((Th * Om,), fill_value=-1, dtype=np.int64)
    dmin   = np.full((Th * Om,), fill_value=np.inf, dtype=np.float64)

    # For each grid cell, find the closest window (in IC space)
    # Vectorised: compute distances from all windows to each target in chunks (memory-safe)
    # But for typical sizes, a simple loop is fine and clearer:
    for idx, (th_t, om_t) in enumerate(targets):
        d = (src[:, 0] - th_t)**2 + (src[:, 1] - om_t)**2  # squared distance
        j = int(np.argmin(d))
        dm = float(d[j])
        if tol is not None and dm > tol**2:
            # No acceptable representative for this cell
            continue
        chosen[idx] = j
        dmin[idx]   = dm

    if require_full_cover and np.any(chosen < 0):
        missing = np.nonzero(chosen < 0)[0]
        # give a small, actionable message
        raise ValueError(
            f"collapse_to_grid_ics: {missing.size} grid cells have no representative "
            f"(consider lowering `tol` or regenerating windows to cover the grid)."
        )

    # If skipping missing cells, filter them out
    valid_mask = (chosen >= 0)
    chosen_valid = chosen[valid_mask]
    if chosen_valid.size == 0:
        raise ValueError("collapse_to_grid_ics: no grid cell received a representative window.")

    # Gather and keep row-major θ-outer, ω-inner order
    idx_t = torch.from_numpy(chosen_valid).to(torch.long)
    θ_c = θ.index_select(0, idx_t).to(dev)
    ω_c = ω.index_select(0, idx_t).to(dev)

    return θ_c, ω_c, chosen
    
# ====================================================================
# Evaluate ONE mode — wired to dispatchers + helper rollout
# ====================================================================
def evaluate_mode(
    mode: str,
    dataset,
    cfg: EvalConfig,
    *,
    combine: str | tuple[str, float] | None = None,
    plot_head_scatter: bool = False,
    save_curves: bool | None = None,
) -> Dict[str, float]:
    # 1) load components (5-tuple)
    v_sd, theta_sd, omega_sd, hnn_sd, lnn_sd = load_components(
        mode, suffix=cfg.suffix, base_dir=cfg.model_dir)

    # 2) rebuild backbone
    model = VJEPA(embed_dim=384, depth=6, num_heads=6).to(device)
    model.load_state_dict(v_sd, strict=True)

    # Heads (match training classes)
    theta_head_1f: torch.nn.Module | None = None
    omega_head_2f: torch.nn.Module | None = None
    head_combined: torch.nn.Module | None = None
    
    if (theta_sd is not None) and (omega_sd is not None):
        # separate heads
        from utils.models import ThetaHead1F, OmegaHead2F
        theta_head_1f = ThetaHead1F(384).to(device)
        omega_head_2f = OmegaHead2F(384).to(device)
        theta_head_1f.load_state_dict(theta_sd, strict=True)
        omega_head_2f.load_state_dict(omega_sd, strict=True)
    else:
        # legacy combined head (Linear(D→2))
        head_combined = torch.nn.Linear(384, 2).to(device)
        if theta_sd is not None:
            head_combined.load_state_dict(theta_sd, strict=True)
        else:
            raise RuntimeError("No head weights found (neither separate nor combined).")

    # Physics nets
    hnn = lnn = None
    if hnn_sd:
        hnn = HNN(hidden_dim=256).to(device); hnn.load_state_dict(hnn_sd, strict=True)
    if lnn_sd:
        lnn = LNN(input_dim=2, hidden_dim=256).to(device); lnn.load_state_dict(lnn_sd, strict=True)

    # 3) eval loader
    eval_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    # 4) rollout
    if (theta_head_1f is not None) and (omega_head_2f is not None):
        θ, ω, α = rollout(
            model, None,
            eval_loader=eval_loader,
            horizon=cfg.horizon, dt=cfg.dt,
            hnn=hnn, lnn=lnn, combine=combine,
            theta_head_1f=theta_head_1f,
            omega_head_2f=omega_head_2f,
        )
        head_for_scatter = (theta_head_1f, omega_head_2f)
    else:
        θ, ω, α = rollout(
            model, head_combined,
            eval_loader=eval_loader,
            horizon=cfg.horizon, dt=cfg.dt,
            hnn=hnn, lnn=lnn, combine=combine
        )
        head_for_scatter = head_combined

    θc, ωc = θ.detach().cpu(), ω.detach().cpu()

    # 5) ICs / axes (unchanged from your latest)
    θ0_np, ω0_np = _initial_conditions(θc, ωc)
    θ_axis = getattr(cfg, "theta_axis", None)
    ω_axis = getattr(cfg, "omega_axis", None)
    if θ_axis is not None: θ_axis = np.asarray(θ_axis).reshape(-1)
    if ω_axis is not None: ω_axis = np.asarray(ω_axis).reshape(-1)
    if cfg.ndiv_mode in ("grid", "window") and (θ_axis is None or ω_axis is None):
        θi, ωi = _infer_grid_axes(θ0_np, ω0_np)
        θ_axis = θ_axis if θ_axis is not None else θi
        ω_axis = ω_axis if ω_axis is not None else ωi
    if cfg.ndiv_mode == "grid" and (θ_axis is not None) and (ω_axis is not None):
        θc, ωc = _reorder_to_grid(θc, ωc, θ0_np, ω0_np, θ_axis, ω_axis)
        θ0_np, ω0_np = _initial_conditions(θc, ωc)
        N, Th, Om = θc.shape[0], len(θ_axis), len(ω_axis)
        if N != Th*Om:
            raise ValueError(f"Grid size mismatch: N={N} vs {Th}*{Om}={Th*Om}")

    # 6) metrics (unchanged)
    Δ_div = neighbour_divergence_scalar_dispatch(
        θc, ωc, cfg, theta_axis=θ_axis, omega_axis=ω_axis,
        theta0=θ0_np, omega0=ω0_np
    )
    metrics: Dict[str, float] = {
        "Δ_div": float(Δ_div),
        "E_drift": float(energy_drift(θc, ωc, m=cfg.m, g=cfg.g, l=cfg.l)),
    }
    if α is not None:
        metrics["accel_mse"] = float(accel_mse(θc, α.detach().cpu(), g=cfg.g, l=cfg.l))
    if lnn is not None:
        metrics["EL_res"] = float(el_residual_metric(
            lnn, θc, ωc, dt=cfg.dt,
            t_max=cfg.el_t_max, stride=cfg.el_stride, max_ic=cfg.el_max_ic,
            reduce=cfg.el_reduce, step=cfg.el_step
        ))

    Δ_curve, Δ_rate = neighbour_divergence_curve_dispatch(
        θc, ωc, cfg, theta_axis=θ_axis, omega_axis=ω_axis,
        theta0=θ0_np, omega0=ω0_np, return_curve=True
    )
    E_curve, E_rate = energy_drift_curve(θc, ωc, m=cfg.m, g=cfg.g, l=cfg.l)
    metrics.update(Δ_rate=float(Δ_rate), E_rate=float(E_rate))

    EL_curve = None
    if lnn is not None:
        EL_curve, EL_rate = el_residual_curve(
            lnn, θc, ωc, dt=cfg.dt, return_curve=True,
            t_max=cfg.el_t_max, stride=cfg.el_stride, max_ic=cfg.el_max_ic
        )
        metrics["EL_rate"] = float(EL_rate)

    # 8) save curves (unchanged)
    if save_curves is None:
        save_curves = cfg.save_curves
    if save_curves:
        metrics["Δ_curve_head"] = Δ_curve[:10].tolist() if Δ_curve is not None else []
        metrics["E_curve_head"] = E_curve[:10].tolist() if E_curve is not None else []
        if EL_curve is not None:
            metrics["EL_curve_head"] = EL_curve[:10].tolist()
        os.makedirs(cfg.out_dir, exist_ok=True)
        curves_path = os.path.join(cfg.out_dir, f"curves_{mode}{cfg.suffix}.npz")
        np.savez_compressed(
            curves_path,
            delta_curve=np.asarray(Δ_curve, dtype=np.float32) if Δ_curve is not None else np.zeros(0, np.float32),
            energy_curve=np.asarray(E_curve, dtype=np.float32) if E_curve is not None else np.zeros(0, np.float32),
            **({"el_curve": np.asarray(EL_curve, dtype=np.float32)} if EL_curve is not None else {})
        )
        metrics["curves_npz"] = curves_path

    # 9) head scatter (works with combined or (θ,ω) heads)
    if (theta_head_1f is not None) and (omega_head_2f is not None):
        θ_t, ω_t, θ_p, ω_p = collect_head_scatter(
            model, None, dataset,
            n_samples=cfg.scatter_samples,
            batch=cfg.scatter_batch,
            shuffle=cfg.scatter_shuffle,
            seed=cfg.scatter_seed,
            theta_head_1f=theta_head_1f,
            omega_head_2f=omega_head_2f,
        )
    else:
        assert head_combined is not None
        θ_t, ω_t, θ_p, ω_p = collect_head_scatter(
            model, head_combined, dataset,
            n_samples=cfg.scatter_samples,
            batch=cfg.scatter_batch,
            shuffle=cfg.scatter_shuffle,
            seed=cfg.scatter_seed,
        )
    metrics.update(head_regression_metrics(θ_t, ω_t, θ_p, ω_p))

    # 10) optional plot (unchanged)
    if plot_head_scatter:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(θ_t, θ_p, s=8, alpha=.6)
        plt.xlabel("true θ"); plt.ylabel("pred θ")
        plt.grid(True); plt.xlim(-4, 4); plt.ylim(-4, 4)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.subplot(1, 2, 2)
        plt.scatter(ω_t, ω_p, s=8, alpha=.6)
        plt.xlabel("true ω"); plt.ylabel("pred ω")
        plt.grid(True); plt.xlim(-10, 10); plt.ylim(-10, 10)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.suptitle(f"Head predictions vs truth – {mode}")
        plt.tight_layout(); plt.show()

    # 11) write JSON
    os.makedirs(cfg.out_dir, exist_ok=True)
    out_path = os.path.join(cfg.out_dir, f"metrics_{mode}{cfg.suffix}.json")
    with open(out_path, "w") as fp:
        json.dump(metrics, fp, indent=2)

    print(f"{mode:8} →",
          {k: (round(v, 4) if isinstance(v, float) else '…') for k, v in metrics.items()},
          f"(saved → {out_path})")
    return metrics

# ====================================================================
# 5. Evaluate MANY modes — auto-detect fusion, pass cfg/switches
# ====================================================================
def evaluate_all_modes(
    modes: Iterable[str],
    dataset,
    cfg: EvalConfig,
    *,
    combine: str | tuple[str, float] | None = "mean",
    plot_scatter: bool = False,
    save_curves: bool | None = None,
) -> Dict[str, Dict[str, float]]:
    all_metrics: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        try:
            # returns: vjepa_sd, theta_sd, omega_sd, hnn_sd, lnn_sd
            _, _, _, hnn_sd, lnn_sd = load_components(
                mode, suffix=cfg.suffix, base_dir=cfg.model_dir, map_location="cpu"
            )
            comb_arg = combine if (hnn_sd and lnn_sd) else None

            all_metrics[mode] = evaluate_mode(
                mode, dataset, cfg,
                combine=comb_arg,
                plot_head_scatter=plot_scatter,
                save_curves=save_curves
            )
        except FileNotFoundError as err:
            print(f"[{mode}] skipped – checkpoint not found: {err}")
        except ValueError as err:
            print(f"[{mode}] skipped – {err}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    out_file = os.path.join(cfg.out_dir, f"metrics_all{cfg.suffix}.json")
    with open(out_file, "w") as fh:
        json.dump(all_metrics, fh, indent=2)
    print(f"\n✓ combined metrics saved → {out_file}")
    return all_metrics

# ====================================================================
# 6 · Legacy plotting helpers
# ====================================================================
_REMAP = {"total":"loss_total","jepa":"loss_jepa",
          "hnn":"loss_hnn","lnn":"loss_lnn","sup":"loss_sup"}

metrics_to_plot: list[tuple[str, str]] = [
    ("loss_total", "total objective"),
    ("loss_hnn",   "HNN residual"),
    ("loss_lnn",   "LNN residual"),
    ("loss_jepa",  "JEPA reconstruction"),
    ("loss_sup",   "θ supervision"),
]

def normalise_keys(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Map legacy keys (total/jepa/…) → new loss_* names."""
    return { _REMAP.get(k, k): v for k,v in arrays.items() }

def plot_loss(
    component: str,
    logs: Dict[str, Dict[str, np.ndarray]],
    *,
    ylabel: str | None = None
) -> None:
    """
    Overlay loss curves from multiple runs.

    Parameters
    ----------
    component : e.g. 'loss_total', 'loss_hnn', …
    logs      : mode → arrays (already `normalise_keys`-processed)
    """
    plt.figure(figsize=(7,4))
    for mode, rec in logs.items():
        if component in rec:
            plt.plot(rec[component], label=mode)
    plt.xlabel("epoch")
    plt.ylabel(ylabel or component)
    plt.title(component)
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
    
def plot_selected_losses(
    logs: Dict[str, Dict[str, np.ndarray]],
    *,
    items: list[tuple[str, str]] = metrics_to_plot
) -> None:
    """
    Iterate over `items` and draw a loss curve **only if** at least one
    run in `logs` contains the given key.

    Parameters
    ----------
    logs
        Dict mapping *mode* → dict of numpy arrays
        (already passed through `normalise_keys` if needed).
    items
        Sequence of ``(loss_key, y-label)`` tuples.  Defaults to the
        canonical list above.
    """
    for key, label in items:
        if any(key in rec for rec in logs.values()):  # at least one run has it?
            plot_loss(key, logs, ylabel=label)        # use the existing helper
        else:
            print(f"[skip] no log contains '{key}'")

