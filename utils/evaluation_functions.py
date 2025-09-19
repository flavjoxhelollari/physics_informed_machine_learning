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
from typing      import Dict, Iterable, List      # static typing

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
from utils.evaluation_metrics  import ( neighbour_divergence,
                                   energy_drift,
                                   el_residual_metric,
                                   neighbour_divergence_curve,
                                   energy_drift_curve,
                                   el_residual_curve)        # metric fns
from utils.helper_functions    import rollout                        # latent rollout

# CUDA if available, otherwise CPU
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====================================================================
# 1 · Dataclass holding every evaluation hyper-parameter
# ====================================================================
@dataclass
class EvalConfig:
    """All knobs for a single evaluation run (defaults = your current)."""
    suffix           : str   = "_dense"      # checkpoint suffix
    model_dir        : str   = "./models"    # where model_*.pt live
    out_dir          : str   = "./metrics"   # where metrics_*.json are written

    # roll-out parameters (used by `rollout`)
    horizon          : int   = 60            # time-steps (3 s if dt=0.05)
    dt               : float = 0.05          # simulation Δt

    # physical constants for energy / accel ground-truth
    m                : float = 1.0
    g                : float = 9.81
    l                : float = 1.0

    # latent-head scatter settings
    scatter_samples  : int   = 500
    scatter_batch    : int   = 64

    #batch-size
    batch_size       : int   = 64

    # --- NEW: neighbour divergence options ---
    # mode: "pairwise" (original O(N^2)), "grid" (adjacent ICs), "window" (Δθ/Δω thresholds), "knn"
    ndiv_mode: str = "grid"          # good default for gridded eval
    ndiv_stencil: str = "8"          # used when ndiv_mode="grid" ("4" or "8")
    ndiv_k: int = 8                  # used when ndiv_mode="knn"
    ndiv_dtheta: float | None = None # used when ndiv_mode="window"
    ndiv_domega: float | None = None # used when ndiv_mode="window"
    ndiv_eps: float = 0.1

    # NEW: how to reduce across time for the scalar Δ_div
    # "step" → use ndiv_step index; "mean" → average over all T
    ndiv_reduce: Literal["step","mean"] = "step"
    ndiv_step: int = -1  # used only when ndiv_reduce == "step"


# # ====================================================================
# # 2 · Head scatter collector
# # ====================================================================
# def collect_head_scatter(
#     model: VJEPA,
#     head:  torch.nn.Module,
#     dataset,
#     *,
#     n_samples: int,
#     batch:     int
# ) -> tuple[np.ndarray, ...]:
#     """
#     Sample up to *n_samples* frames (t=0) and return:

#     Returns
#     -------
#     θ_true, ω_true, θ_pred, ω_pred : np.ndarray
#         All shape (n_samples,)
#     """
#     # put sub-nets in eval mode
#     model.eval(); head.eval()

#     # random batching
#     loader = DataLoader(dataset, batch_size=batch, shuffle=True)

#     # pre-allocate Python lists, convert later
#     θ_true, ω_true, θ_pred, ω_pred = [], [], [], []
#     collected = 0                                                     # counter

#     for seq, states in loader:                                        # loop batches
#         imgs = seq[:, 0].to(device)                                   # (B,C,H,W) first frame
#         with torch.no_grad():                                         # no grad needed
#             z_lat = model.patch_embed(imgs) + model.pos_embed         # patch + pos
#             z_lat = model.context_encoder(z_lat).mean(1)              # (B,D) pooled
#             θ_hat, ω_hat = head(z_lat).split(1, 1)                    # (B,1) each

#         # store CPU numpy copies
#         θ_true.extend(states[:, 0, 0].cpu().numpy())                  # real θ
#         ω_true.extend(states[:, 0, 1].cpu().numpy())                  # real ω
#         θ_pred.extend(θ_hat.squeeze().cpu().numpy())                  # predicted θ
#         ω_pred.extend(ω_hat.squeeze().cpu().numpy())                  # predicted ω

#         collected += imgs.size(0)                                     # update counter
#         if collected >= n_samples:                                    # stop if enough
#             break                                                     # … exit loop

#     # cast lists → numpy of exact length n_samples
#     return (np.array(θ_true)[:n_samples],
#             np.array(ω_true)[:n_samples],
#             np.array(θ_pred)[:n_samples],
#             np.array(ω_pred)[:n_samples])


# # ====================================================================
# # 3 · Tiny regression helper (latent-head)
# # ====================================================================
# def head_regression_metrics(
#     θ_t: np.ndarray, ω_t: np.ndarray,
#     θ_p: np.ndarray, ω_p: np.ndarray
# ) -> Dict[str, float]:
#     """Return R² and MSE for θ and ω."""
#     return dict(
#         r2_theta  = float(r2_score(θ_t, θ_p)),
#         mse_theta = float(mean_squared_error(θ_t, θ_p)),
#         r2_omega  = float(r2_score(ω_t, ω_p)),
#         mse_omega = float(mean_squared_error(ω_t, ω_p)),
#     )

# ====================================================================
# 2 · Head scatter collector (robust)
# ====================================================================
def collect_head_scatter(
    model: VJEPA,
    head:  torch.nn.Module,
    dataset,
    *,
    n_samples: int,
    batch:     int,
    shuffle:   bool = True,
    seed:      int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample up to *n_samples* first-frame (t=0) examples and return:
        θ_true, ω_true, θ_pred, ω_pred  (all shape (n_samples,))

    Notes
    -----
    • Uses model/head's own device.
    • Uses eval mode + no grad.
    • `shuffle=False` if you want a uniform sweep over IC order.
    """
    # put sub-nets in eval mode
    model.eval(); head.eval()

    # device from model params (safer than global `device`)
    mdev = next(model.parameters()).device

    # cap samples
    n_total = len(dataset)
    n_take  = min(n_samples, n_total)
    if n_take <= 0:
        return (np.empty(0),)*4

    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size = batch,
        shuffle    = shuffle,
        generator  = g if shuffle and seed is not None else None,
        num_workers= 0,
        pin_memory = False,
    )

    θ_true, ω_true, θ_pred, ω_pred = [], [], [], []
    collected = 0

    with torch.no_grad():
        for seq, states in loader:
            imgs = seq[:, 0].to(mdev, non_blocking=False)  # (B,C,H,W)

            z_lat = model.patch_embed(imgs) + model.pos_embed
            z_lat = model.context_encoder(z_lat).mean(1)   # (B,D)
            θ_hat, ω_hat = head(z_lat).split(1, dim=1)     # (B,1) each

            # store CPU numpy copies (use squeeze(1) to keep batch dim safe)
            θ_true.extend(states[:, 0, 0].cpu().numpy())
            ω_true.extend(states[:, 0, 1].cpu().numpy())
            θ_pred.extend(θ_hat.squeeze(1).detach().cpu().numpy())
            ω_pred.extend(ω_hat.squeeze(1).detach().cpu().numpy())

            collected += imgs.size(0)
            if collected >= n_take:
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
def evaluate_mode(
    mode       : str,
    dataset,
    cfg        : EvalConfig,
    *,
    combine           : str | tuple[str, float] | None = None,
    plot_head_scatter : bool = False,
    save_curves       : bool = False,          # ← NEW FLAG
) -> Dict[str, float]:
    """
    Inference + physics metrics for one checkpoint.

    Parameters
    ----------
    …
    save_curves
        If **True**, also dump the full per-step curves
        ``Δ_curve``, ``E_curve`` and (when LNN present) ``EL_curve`` to
        the output JSON.  Beware: curves are length ≈ `cfg.horizon` and
        stored as full-precision floats, so the file grows linearly with
        horizon.

    Returns
    -------
    Dict[str, float]
        Keys:
            Δ_div, E_drift, EL_res (if LNN),
            Δ_rate, E_rate, EL_rate (if LNN),
            (optionally the three *_curve lists)
            + latent-head regression stats.
    """
    # 1) -------- load state-dicts -----------------------------------
    v_sd, t_sd, hnn_sd, lnn_sd = load_components(
        mode, suffix=cfg.suffix, base_dir=cfg.model_dir)

    # 2) -------- rebuild modules & weights --------------------------
    model = VJEPA(embed_dim=384, depth=6, num_heads=6).to(device)
    head  = torch.nn.Linear(384, 2).to(device)
    model.load_state_dict(v_sd, strict=True)
    head .load_state_dict(t_sd, strict=True)

    hnn = lnn = None
    if hnn_sd:
        hnn = HNN(hidden_dim=256).to(device); hnn.load_state_dict(hnn_sd, strict=True)
    if lnn_sd:
        lnn = LNN(input_dim=2, hidden_dim=256).to(device); lnn.load_state_dict(lnn_sd, strict=True)

    # 3) -------- fixed evaluation DataLoader ------------------------
    eval_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    # 4) -------- latent rollout -------------------------------------
    θ, ω, _ = rollout(
        model, head,
        hnn         = hnn,
        lnn         = lnn,
        combine     = combine,
        horizon     = cfg.horizon,
        dt          = cfg.dt,
        eval_loader = eval_loader
    )

    # 5) -------- base physics metrics -------------------------------
    metrics: Dict[str, float] = {
        "Δ_div"  : neighbour_divergence(θ, ω),
        "E_drift": energy_drift(θ, ω, m=cfg.m, g=cfg.g, l=cfg.l),
    }
    if lnn is not None:
        metrics["EL_res"] = el_residual_metric(lnn, θ, ω, dt=cfg.dt)

    # 6) -------- per-step curves + rates ----------------------------
    Δ_curve, Δ_rate = neighbour_divergence_curve(θ, ω)
    E_curve, E_rate = energy_drift_curve(θ, ω, m=cfg.m, g=cfg.g, l=cfg.l)
    metrics.update(Δ_rate=Δ_rate, E_rate=E_rate)

    if lnn is not None:
        EL_curve, EL_rate = el_residual_curve(lnn, θ, ω, dt=cfg.dt)
        metrics["EL_rate"] = EL_rate

    if save_curves:
        # keep JSON (back-compat)
        metrics["Δ_curve"] = Δ_curve.tolist()
        metrics["E_curve"] = E_curve.tolist()
        if lnn is not None:
            metrics["EL_curve"] = EL_curve.tolist()
    
        # add compact sidecar
        curves_path = os.path.join(cfg.out_dir, f"curves_{mode}{cfg.suffix}.npz")
        np.savez_compressed(
            curves_path,
            delta_curve=np.asarray(Δ_curve, dtype=np.float32),
            energy_curve=np.asarray(E_curve, dtype=np.float32),
            **({"el_curve": np.asarray(EL_curve, dtype=np.float32)} if lnn is not None else {})
        )
        metrics["curves_npz"] = curves_path  # tiny pointer in JSON

    # 7) -------- latent-head diagnostics ---------------------------
    θ_t, ω_t, θ_p, ω_p = collect_head_scatter(
        model, head, dataset,
        n_samples=cfg.scatter_samples,
        batch    =cfg.scatter_batch)
    metrics.update(head_regression_metrics(θ_t, ω_t, θ_p, ω_p))

    # 8) -------- optional scatter plot -----------------------------
    if plot_head_scatter:
        plt.figure(figsize=(10, 5))
        # θ
        plt.subplot(1, 2, 1)
        plt.scatter(θ_t, θ_p, s=8, alpha=.6)
        plt.xlabel("true θ"); plt.ylabel("pred θ")
        plt.grid(True); plt.xlim(-4, 4); plt.ylim(-4, 4)
        plt.gca().set_aspect("equal", adjustable="box")
        # ω
        plt.subplot(1, 2, 2)
        plt.scatter(ω_t, ω_p, s=8, alpha=.6)
        plt.xlabel("true ω"); plt.ylabel("pred ω")
        plt.grid(True); plt.xlim(-10, 10); plt.ylim(-10, 10)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.suptitle(f"Head predictions vs truth – {mode}")
        plt.tight_layout(); plt.show()

    # 9) -------- write JSON ----------------------------------------
    os.makedirs(cfg.out_dir, exist_ok=True)
    out_path = os.path.join(cfg.out_dir, f"metrics_{mode}{cfg.suffix}.json")
    with open(out_path, "w") as fp:
        json.dump(metrics, fp, indent=2)

    print(f"{mode:8} →", {k: round(v, 4) if isinstance(v, float) else '…' for k, v in metrics.items()},
          f"(saved → {out_path})")
    return metrics
# ====================================================================
# 5 · Evaluate *many* modes (auto-detects whether fusion is possible)
# ====================================================================
def evaluate_all_modes(
    modes        : Iterable[str],
    dataset,
    cfg          : EvalConfig,
    *,
    combine      : str | tuple[str, float] | None = "mean",
    plot_scatter : bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Run :func:`evaluate_mode` for every string in *modes*.

    The helper **checks each checkpoint first** and forwards the *combine*
    argument **only when *both* physics nets are present** (otherwise the
    call falls back to ``combine=None`` so that the run is *not* skipped).

    Parameters
    ----------
    modes
        Iterable with any subset of ``{"plain","hnn","lnn","hnn+lnn"}``.
    dataset
        Evaluation dataset (e.g. deterministic Pendulum grid).
    cfg
        Shared :class:`EvalConfig` (horizon, dt, paths…).
    combine
        Fusion rule when *both* HNN **and** LNN exist in the checkpoint
        (ignored otherwise).  Supported values:

        ================  ============================================
        ``None``          do **not** fuse – use HNN alone  
        ``"mean"``        α = ½ (α_HNN + α_LNN)  ← **default**  
        ``"sum"``         α = α_HNN + α_LNN  
        ``("blend", w)``  α = w·α_HNN + (1-w)·α_LNN  (0 ≤ w ≤ 1)  
        ================  ============================================

    plot_scatter
        Forwarded to :func:`evaluate_mode(plot_head_scatter=…)`.

    Returns
    -------
    dict
        ``{mode: metric-dict}`` for every *successfully* evaluated mode.
        The same object is also written as JSON to
        ``cfg.out_dir / metrics_all<suffix>.json``.
    """
    all_metrics: Dict[str, Dict[str, float]] = {}

    for mode in modes:
        try:
            # --- lightweight peek: are both physics nets stored? ----
            _, _, hnn_sd, lnn_sd = load_components(
                mode,
                suffix       = cfg.suffix,
                base_dir     = cfg.model_dir,
                map_location = "cpu")               # tiny, no GPU alloc

            # decide on the fusion rule to pass downstream
            comb_arg = combine if (hnn_sd and lnn_sd) else None

            # ------------------ full evaluation --------------------
            all_metrics[mode] = evaluate_mode(
                mode,
                dataset,
                cfg,
                combine           = comb_arg,
                plot_head_scatter = plot_scatter
            )
        except FileNotFoundError as err:
            print(f"[{mode}] skipped – checkpoint not found: {err}")
        except ValueError as err:
            # propagate unexpected errors but keep loop alive
            print(f"[{mode}] skipped – {err}")

    # ------------------ persist combined dict -----------------------
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