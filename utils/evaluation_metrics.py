"""
evaluation_metrics.py
─────────────────────
Metric utilities for *any* single–pendulum data-set produced by vision
pipelines (e.g. V-JEPA) **independent** of the training code.

Why a separate module?
----------------------
* **Re-use** — the exact same metrics are consumed by notebooks, unit-
  tests, model-selection scripts and final plots.  
* **Transparency** — every formula is visible in *one* canonical place
  (energy, divergence, Euler–Lagrange residual …).  
* **Pure functions** — no hidden globals, no file I/O: every function is
  a deterministic map **tensor → float** that can run on *any* device.

Default physical constants
--------------------------
We expose module-level defaults so 99 % of experiments need *zero*
extra kwargs:

>>> from evaluation_metrics import energy_drift  
>>> drift = energy_drift(theta, omega)            # uses MASS, GRAV, LENGTH

Override per-call if you study a different pendulum:

>>> drift = energy_drift(theta, omega, m=2.0, l=0.75)

Notation
~~~~~~~~
* ``θ`` (theta) / ``ω`` (omega) ∶ tensors of shape **(N, T)**  
  – *N* trajectories, *T* time-steps.  
* All outputs are Python *floats* (ready for ``json.dump``).  
* All autograd calls respect devices; CUDA tensors stay on the GPU.  
"""

from __future__ import annotations

import torch                                # tensors & autograd
import torch.nn.functional as F             # loss helpers
import numpy as np                          # reference math
from sklearn.linear_model  import LinearRegression
from sklearn.metrics       import r2_score, mean_squared_error
from typing                 import Dict, Tuple
import torch.linalg as LA
from utils.helper_functions import *
# ---------------------------------------------------------------------
# Physical defaults (override via function kwargs for other setups)
# ---------------------------------------------------------------------
MASS   : float = 1.0     # kg
GRAV   : float = 9.81    # m s⁻²
LENGTH : float = 1.0     # m


# =====================================================================
# 1 · Neighbour-divergence
# =====================================================================
def neighbour_divergence(
    theta  : torch.Tensor,                           # (N,T)  angles  [rad]
    omega  : torch.Tensor,                           # (N,T)  angular rates
    *,
    epsilon: float = 0.1,                            # max initial distance
    step   : int   = -1                              # time-index for eval
) -> float:
    """
    Average *phase-space* distance after ``step`` **between every pair of
    trajectories whose initial distance is < ε**.

    Parameters
    ----------
    theta, omega : (N, T) ``torch.Tensor``
        Angles and angular rates for *N* roll-outs.
    epsilon      : float, default = ``0.1``
        Threshold in phase-space at *t = 0* to decide who is a “neighbour”.
    step         : int, default = ``-1`` (final frame)
        Index along the *T* dimension at which the separation is measured.

    Returns
    -------
    float
        Mean Euclidean separation in phase-space.

    Notes
    -----
    •  A *lower* value ⇒ stronger trajectory coherence.  
    •  Uses ``torch.cdist`` so it runs on CPU **or** GPU tensors.
    """
    start = torch.stack([theta[:, 0], omega[:, 0]], dim=1)        # (N,2)
    dist0 = torch.cdist(start, start, p=2)                        # (N,N)
    mask  = (dist0 > 0) & (dist0 < epsilon)                       # neighbours?
    i, j  = torch.nonzero(mask, as_tuple=True)                    # index pairs
    diff  = torch.stack([theta[i, step] - theta[j, step],         # Δθ
                         omega[i, step] - omega[j, step]], dim=1) # Δω
    return diff.norm(dim=1).mean().item()                         # scalar

def _reduce_over_time(dist_pairs_T: torch.Tensor, *,   # shape (P, T)
                      reduce: str, step: int) -> float:
    if dist_pairs_T.numel() == 0:
        return 0.0
    if reduce == "mean":
        return dist_pairs_T.mean().item()
    # "step": pick a single time index (supports negative indices)
    t_idx = step if step >= 0 else dist_pairs_T.size(1) + step
    t_idx = max(0, min(t_idx, dist_pairs_T.size(1)-1))
    return dist_pairs_T[:, t_idx].mean().item()

def neighbour_divergence_scalar_pairwise(
    θ: torch.Tensor, ω: torch.Tensor, *,
    eps: float = 0.1, reduce: str = "step", step: int = -1
) -> float:
    start = torch.stack([θ[:, 0], ω[:, 0]], dim=1)  # (N,2)
    dist0 = torch.cdist(start, start, p=2)          # (N,N)
    mask  = (dist0 > 0) & (dist0 < eps)
    if not mask.any():
        return 0.0
    # gather pairs once
    I, J = torch.nonzero(mask, as_tuple=True)
    dθ = θ[I] - θ[J]                                 # (P,T)
    dω = ω[I] - ω[J]
    dist_pairs_T = torch.linalg.vector_norm(
        torch.stack([dθ, dω], dim=0), dim=0)         # (P,T)
    return _reduce_over_time(dist_pairs_T, reduce=reduce, step=step)

def neighbour_divergence_scalar_grid(
    θ: torch.Tensor, ω: torch.Tensor, *,
    theta_axis: np.ndarray, omega_axis: np.ndarray,
    stencil: str = "8", reduce: str = "step", step: int = -1
) -> float:
    N, T = θ.shape
    Th, Om = len(theta_axis), len(omega_axis)
    assert N == Th * Om, "Grid size mismatch"
    offs4 = [(+1,0), (-1,0), (0,+1), (0,-1)]
    offs8 = offs4 + [(+1,+1), (+1,-1), (-1,+1), (-1,-1)]
    offs  = offs8 if stencil == "8" else offs4
    pairs = []
    for i in range(Th):
        for j in range(Om):
            src = i*Om + j
            for di, dj in offs:
                ii, jj = i+di, j+dj
                if 0 <= ii < Th and 0 <= jj < Om:
                    pairs.append((src, ii*Om + jj))
    if not pairs:
        return 0.0
    I = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    J = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    dθ = θ[I] - θ[J]                                  # (P,T)
    dω = ω[I] - ω[J]
    dist_pairs_T = torch.linalg.vector_norm(
        torch.stack([dθ, dω], dim=0), dim=0)          # (P,T)
    return _reduce_over_time(dist_pairs_T, reduce=reduce, step=step)

def neighbour_divergence_scalar_window(
    θ: torch.Tensor, ω: torch.Tensor, *,
    theta0: np.ndarray, omega0: np.ndarray,
    dθ_max: float, dω_max: float,
    max_neighbours: int | None = None,
    reduce: str = "step", step: int = -1
) -> float:
    N, _ = θ.shape
    I_list, J_list = [], []
    for i in range(N):
        mask = (np.abs(theta0 - theta0[i]) <= dθ_max) & (np.abs(omega0 - omega0[i]) <= dω_max)
        idx = np.nonzero(mask)[0]
        idx = idx[idx != i]
        if max_neighbours is not None and idx.size > max_neighbours:
            idx = idx[:max_neighbours]
        for j in idx:
            I_list.append(i); J_list.append(int(j))
    if not I_list:
        return 0.0
    I = torch.tensor(I_list, dtype=torch.long)
    J = torch.tensor(J_list, dtype=torch.long)
    dθ = θ[I] - θ[J]                                   # (P,T)
    dω = ω[I] - ω[J]
    dist_pairs_T = torch.linalg.vector_norm(
        torch.stack([dθ, dω], dim=0), dim=0)           # (P,T)
    return _reduce_over_time(dist_pairs_T, reduce=reduce, step=step)

def neighbour_divergence_scalar_knn(
    θ: torch.Tensor, ω: torch.Tensor, *,
    k: int = 8, reduce: str = "step", step: int = -1
) -> float:
    start = torch.stack([θ[:, 0], ω[:, 0]], dim=1)     # (N,2)
    d0 = torch.cdist(start, start, p=2)                # (N,N)
    d0.fill_diagonal_(float("inf"))
    k_eff = min(k, d0.size(1)-1)
    idx = torch.topk(d0, k=k_eff, largest=False).indices  # (N,k)
    # build P = N*k pairs
    I = torch.arange(idx.size(0)).repeat_interleave(idx.size(1))
    J = idx.reshape(-1)
    dθ = θ[I] - θ[J]                                   # (P,T)
    dω = ω[I] - ω[J]
    dist_pairs_T = torch.linalg.vector_norm(
        torch.stack([dθ, dω], dim=0), dim=0)           # (P,T)
    return _reduce_over_time(dist_pairs_T, reduce=reduce, step=step)

def neighbour_divergence_scalar_dispatch(
    θ: torch.Tensor, ω: torch.Tensor, cfg: EvalConfig, *,
    theta_axis: np.ndarray | None = None,
    omega_axis: np.ndarray | None = None,
    theta0: np.ndarray | None = None,
    omega0: np.ndarray | None = None,
) -> float:
    if cfg.ndiv_mode == "pairwise":
        return neighbour_divergence_scalar_pairwise(
            θ, ω, eps=cfg.ndiv_eps, reduce=cfg.ndiv_reduce, step=cfg.ndiv_step
        )
    elif cfg.ndiv_mode == "grid":
        if theta_axis is None or omega_axis is None:
            raise ValueError("grid mode needs theta_axis and omega_axis")
        return neighbour_divergence_scalar_grid(
            θ, ω, theta_axis=theta_axis, omega_axis=omega_axis,
            stencil=cfg.ndiv_stencil, reduce=cfg.ndiv_reduce, step=cfg.ndiv_step
        )
    elif cfg.ndiv_mode == "window":
        if theta0 is None or omega0 is None:
            raise ValueError("window mode needs theta0/omega0 arrays")
        dθ = cfg.ndiv_dtheta; dω = cfg.ndiv_domega
        if dθ is None or dω is None:
            raise ValueError("window mode needs ndiv_dtheta and ndiv_domega")
        return neighbour_divergence_scalar_window(
            θ, ω, theta0=theta0, omega0=omega0,
            dθ_max=dθ, dω_max=dω, max_neighbours=None,
            reduce=cfg.ndiv_reduce, step=cfg.ndiv_step
        )
    elif cfg.ndiv_mode == "knn":
        return neighbour_divergence_scalar_knn(
            θ, ω, k=cfg.ndiv_k, reduce=cfg.ndiv_reduce, step=cfg.ndiv_step
        )
    else:
        raise ValueError(f"Unknown ndiv_mode: {cfg.ndiv_mode}")

# =====================================================================
# 2 · Energy drift
# =====================================================================
def energy_drift(
    theta : torch.Tensor,            # (N,T)
    omega : torch.Tensor,            # (N,T)
    *,
    m: float = MASS,
    g: float = GRAV,
    l: float = LENGTH
) -> float:
    """
    Absolute mean drift of total mechanical energy **between t=0 and t=T-1**.

    Energy model
    ------------
    *Kinetic* = ½ m l² ω²  
    *Potential* = m g l (1 − cos θ)

    Returns
    -------
    float – |E<sub>T</sub> − E<sub>0</sub>| averaged over all trajectories.
    """
    E0 = 0.5 * m * l**2 * omega[:, 0]**2 + m * g * l * (1 - torch.cos(theta[:, 0]))
    Et = 0.5 * m * l**2 * omega[:, -1]**2 + m * g * l * (1 - torch.cos(theta[:, -1]))
    return (Et - E0).abs().mean().item()


# =====================================================================
# 3 · Acceleration MSE
# =====================================================================
def accel_mse(
    theta      : torch.Tensor,       # (N,T)
    alpha_pred : torch.Tensor,       # (N,T)  learned ω̇
    *,
    g: float = GRAV,
    l: float = LENGTH
) -> float:
    """
    Mean-squared error between *predicted* and *analytic* angular
    acceleration:

    α_true = −(g/l) sin θ
    """
    alpha_true = -g / l * torch.sin(theta[:, :-1])      # ignore final step
    return F.mse_loss(alpha_pred[:, :-1], alpha_true).item()


# # =====================================================================
# # 4 · Euler–Lagrange residual
# # =====================================================================
# def el_residual_metric(
#     lnn,
#     theta: torch.Tensor,             # (N,T)
#     omega: torch.Tensor,             # (N,T)
#     *,
#     dt: float
# ) -> float:
#     """
#     Mean squared Euler–Lagrange residual R<sub>EL</sub> over the first
#     three frames *(T ≥ 3 required)*.

#     Implementation details
#     ----------------------
#     •  `lnn.lagrangian_residual` expects (B,T,d) on **the same device**
#        as the LNN’s parameters — tensors are moved automatically.  
#     •  Only *θ* and *ω* are fed; momentum terms are not needed for a
#        single-pendulum.
#     """
#     q = theta[:, :3, None]                          # (N,3,1)   θ
#     v = omega[:, :3, None]                          # (N,3,1)   ω
#     lnn_device = next(lnn.parameters()).device      # LNN’s own device
#     return lnn.lagrangian_residual(q.to(lnn_device),
#                                    v.to(lnn_device),
#                                    dt).item()

# =====================================================================
# 4 · Euler–Lagrange residual (scalar, built from the curve)
# =====================================================================
def el_residual_metric(
    lnn,
    theta: torch.Tensor,             # (N,T)  CPU or CUDA
    omega: torch.Tensor,             # (N,T)
    *,
    dt: float,
    # Scalability knobs (apply identically to curve & scalar)
    t_max: int | None = None,        # e.g., 64 → use first t_max steps (after stride)
    stride: int = 1,                 # e.g., 2  → time downsampling
    max_ic: int | None = None,       # e.g., 50 → evaluate on subset of trajectories
    # How to reduce curve → scalar
    reduce: Literal["mean","step"] = "mean",
    step: int = -1                   # used only when reduce="step" (supports negatives)
) -> float:
    """
    Scalar EL residual obtained by computing the per-step EL residual curve
    (see `el_residual_curve`) and then reducing it in time.

    ▼ Replicating *your original behavior* (mean over the first three frames):
        el_residual_metric(..., dt=..., t_max=3, stride=1,
                           max_ic=None, reduce="mean")

    Parameters
    ----------
    lnn : trained LNN module (expects concat [q, v] as input)
    theta, omega : (N,T) rollouts
    dt : float    step size used in the rollout
    t_max, stride, max_ic : optional throttles for long horizons / many ICs
    reduce : "mean" → average over time; "step" → pick a single time index
    step : index when reduce="step" (e.g., -1 for final step)

    Returns
    -------
    float
        Scalar residual summary after reduction.
    """
    curve, _ = el_residual_curve(
        lnn, theta, omega, dt=dt,
        return_curve=True,
        t_max=t_max, stride=stride, max_ic=max_ic
    )
    if curve is None or len(curve) == 0:
        return 0.0

    if reduce == "mean":
        return float(np.mean(curve))

    # reduce == "step"
    t_idx = step if step >= 0 else len(curve) + step
    t_idx = max(0, min(t_idx, len(curve) - 1))
    return float(curve[t_idx])


# =====================================================================
# 5 · Latent-to-θ regression diagnostics
# =====================================================================
def latent_r2(
    model,                                          # VJEPA-style encoder
    head,                                           # linear θ,ω head
    *,
    eval_loader,                                    # DataLoader of sequences
    n_samples: int = 500
) -> Dict[str, float]:
    """
    Linear probe: fit *one* `sklearn.LinearRegression` on frozen latent
    vectors → θ (frame-0).  Reports R² **and** MSE.

    Returns
    -------
    dict  { "r2": …, "mse": … }
    """
    model.eval(); head.eval()

    Z, θ_list = [], []                                       # buffers
    collected = 0

    for seq, states in eval_loader:                          # iterate batches
        imgs0 = seq[:, 0].to(next(model.parameters()).device)

        with torch.no_grad():                                # no autograd
            latent = model.context_encoder(
                         model.patch_embed(imgs0) + model.pos_embed).mean(1)
        Z.append(latent.cpu())                               # detach GPU → CPU
        θ_list.extend(states[:, 0, 0].tolist())              # ground-truth θ

        collected += imgs0.size(0)
        if collected >= n_samples:                           # stop early
            break

    Z_arr = torch.cat(Z, 0)[:n_samples].numpy()             # (n_samples,D)
    θ_arr = np.array(θ_list)[:n_samples]                    # (n_samples,)

    θ_pred = LinearRegression().fit(Z_arr, θ_arr).predict(Z_arr)
    return {
        "r2" : float(r2_score(θ_arr, θ_pred)),
        "mse": float(mean_squared_error(θ_arr, θ_pred))
    }
    
# ─────────────────────────────────────────────────────────────
# 7 · Time–series diagnostics (divergence, energy, EL-residual)
# ─────────────────────────────────────────────────────────────

_norm = torch.linalg.vector_norm

# ---------- a tiny utility -----------------------------------------
def _linear_slope(y: np.ndarray, dt: float = 1.0) -> float:
    """
    Fit y(t) ≈ a·t + b and return the slope a.
    """
    t = np.arange(len(y), dtype=np.float32).reshape(-1, 1) * dt
    return float(LinearRegression().fit(t, y).coef_[0])

# -------------------------------------------------------------------
# 7a · Neighbour-divergence curve + rate
# -------------------------------------------------------------------
def neighbour_divergence_curve(
    θ: torch.Tensor,
    ω: torch.Tensor,
    *,
    ε: float = 0.1,
    return_curve: bool = False          # ← NEW
) -> Tuple[np.ndarray | None, float]:
    """
    Parameters
    ----------
    θ, ω : (N, T)  – rollout tensors on **CPU**
    ε    : float   – neighbourhood radius at t=0
    return_curve : bool
        If False the first element of the tuple is ``None``.

    Returns
    -------
    curve | None ,  slope
        • curve shape (T,) – mean pair-wise distance per step (or None)  
        • slope – d⟨Δ⟩/dt, same units as θ
    """
    phase = torch.stack([θ, ω], dim=1)                # (N,2,T)

    dist0 = _norm(phase[:, :, 0][:, None]             # pairwise at t=0
                  - phase[:, :, 0][None], dim=2)
    mask  = (dist0 > 0) & (dist0 < ε)                 # neighbours

    dist  = _norm(phase[:, None] - phase[None], dim=2)  # (N,N,T)
    curve = dist[mask].mean(0).cpu().numpy()          # (T,)

    slope = _linear_slope(curve)                      # growth-rate
    return (curve if return_curve else None), slope

def neighbour_divergence_curve_window(
    θ: torch.Tensor, ω: torch.Tensor,                      # (N,T) CPU
    theta0: np.ndarray, omega0: np.ndarray,                # (N,) true ICs
    *, dθ_max: float, dω_max: float,
    max_neighbours: int | None = None,                     # cap per i, optional
    return_curve: bool = False
):
    N, T = θ.shape
    # sort by theta0 to reduce search (optional micro-opt)
    order = np.lexsort((omega0, theta0))
    θ0_sorted, ω0_sorted = theta0[order], omega0[order]
    θ_sorted, ω_sorted   = θ[order], ω[order]

    pairs_i, pairs_j = [], []
    for idx_i in range(N):
        th_i, om_i = θ0_sorted[idx_i], ω0_sorted[idx_i]

        # find candidates within dθ_max along theta axis
        # (since we sorted primarily by theta, we can scan a small band)
        # naive safe approach (still fine for N≈100–1000):
        dθ = np.abs(θ0_sorted - th_i) <= dθ_max
        dω = np.abs(ω0_sorted - om_i) <= dω_max
        cand = np.nonzero(dθ & dω)[0]
        cand = cand[cand != idx_i]
        if max_neighbours is not None and cand.size > max_neighbours:
            cand = cand[:max_neighbours]
        for j in cand:
            pairs_i.append(idx_i); pairs_j.append(j)

    if len(pairs_i) == 0:
        curve = np.zeros(T, dtype=np.float32)
        return (curve if return_curve else None), 0.0

    idx_i_t = torch.tensor(pairs_i, dtype=torch.long)
    idx_j_t = torch.tensor(pairs_j, dtype=torch.long)

    dθ = θ_sorted[idx_i_t] - θ_sorted[idx_j_t]       # (P,T)
    dω = ω_sorted[idx_i_t] - ω_sorted[idx_j_t]
    dist = torch.linalg.vector_norm(torch.stack([dθ, dω], dim=0), dim=0)  # (P,T)

    curve = dist.mean(dim=0).numpy()
    slope = _linear_slope(curve)
    return (curve if return_curve else None), slope

import numpy as np, torch

def neighbour_divergence_curve_grid(
    θ: torch.Tensor, ω: torch.Tensor,                  # (N,T) CPU tensors
    *, theta_axis: np.ndarray, omega_axis: np.ndarray, # 1D arrays used to build the grid
    stencil: str = "4",                                # "4" or "8"
    return_curve: bool = False
):
    """
    Neighbour divergence only to grid-adjacent ICs.
    Assumes dataset order is theta-outer, omega-inner (no shuffle) so that
    index = iθ * Om + jω.
    """
    N, T = θ.shape
    Th, Om = len(theta_axis), len(omega_axis)
    assert N == Th * Om, "θ/ω count does not match theta×omega grid size"

    # build (iθ, jω) for each trajectory
    ij = np.array([(i, j) for i in range(Th) for j in range(Om)], dtype=int)

    # neighbour offsets
    offs4 = [(+1,0), (-1,0), (0,+1), (0,-1)]
    offs8 = offs4 + [(+1,+1), (+1,-1), (-1,+1), (-1,-1)]
    offs  = offs8 if stencil == "8" else offs4

    # collect all valid neighbour pairs (i -> j) once
    pairs = []
    for idx, (i, j) in enumerate(ij):
        for di, dj in offs:
            ii, jj = i+di, j+dj
            if 0 <= ii < Th and 0 <= jj < Om:
                nbr = ii*Om + jj
                pairs.append((idx, nbr))
    if len(pairs) == 0:
        curve = np.zeros(T, dtype=np.float32)
        return (curve if return_curve else None), 0.0

    # stack distances over time
    idx_i = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    idx_j = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    dθ = θ[idx_i] - θ[idx_j]           # (P, T)
    dω = ω[idx_i] - ω[idx_j]           # (P, T)
    dist = torch.linalg.vector_norm(torch.stack([dθ, dω], dim=0), dim=0)   # (P, T)

    curve = dist.mean(dim=0).numpy()   # (T,)
    slope = _linear_slope(curve)
    return (curve if return_curve else None), slope

def neighbour_divergence_curve_k(
    θ: torch.Tensor, ω: torch.Tensor,
    *, k: int = 8, return_curve: bool = False
) -> tuple[np.ndarray|None, float]:
    """
    k-NN version: for each trajectory, average distance to its k nearest
    neighbours (in phase space at t=0), then average across trajectories.
    """
    # (N,T) CPU tensors expected
    start = torch.stack([θ[:, 0], ω[:, 0]], dim=1)   # (N,2)
    d0 = torch.cdist(start, start, p=2)              # (N,N)
    d0.fill_diagonal_(float("inf"))
    idx = torch.topk(d0, k=min(k, θ.size(0)-1), largest=False).indices  # (N,k)

    N, T = θ.shape
    curve_acc = torch.zeros(T, dtype=torch.float32)
    for i in range(N):
        j = idx[i]                                  # (k,)
        dθ = θ[i].unsqueeze(0) - θ[j]               # (k,T)
        dω = ω[i].unsqueeze(0) - ω[j]               # (k,T)
        dij = torch.linalg.vector_norm(
            torch.stack([dθ, dω], dim=0), dim=0)    # (k,T)
        curve_acc += dij.mean(dim=0)

    curve = (curve_acc / N).numpy()
    slope = _linear_slope(curve)
    return (curve if return_curve else None), slope

from typing import Tuple

def neighbour_divergence_curve_dispatch(
    θ: torch.Tensor, ω: torch.Tensor, cfg, *,
    # supply these if your mode needs them:
    theta_axis: np.ndarray | None = None,
    omega_axis: np.ndarray | None = None,
    # for "window" mode:
    theta0: np.ndarray | None = None,
    omega0: np.ndarray | None = None,
    return_curve: bool = True
) -> Tuple[np.ndarray | None, float]:
    """
    Dispatch the neighbour-divergence *curve* based on cfg.ndiv_mode.

    Modes
    -----
    - "pairwise": original O(N^2) neighbours (within radius `cfg.ndiv_eps`)
    - "grid"    : grid-adjacent neighbours (4- or 8-connected via `cfg.ndiv_stencil`)
    - "window"  : neighbours whose ICs satisfy |Δθ0|<=dθ and |Δω0|<=dω
                  (`cfg.ndiv_dtheta`, `cfg.ndiv_domega`; defaults to one grid step)
    - "knn"     : k nearest neighbours at t=0 (`cfg.ndiv_k`)

    Returns
    -------
    curve | None, rate
        curve shape (T,) if `return_curve=True` else None
        rate = linear slope d⟨Δ⟩/dt (via _linear_slope)
    """
    mode = getattr(cfg, "ndiv_mode", "grid")

    if mode == "pairwise":
        # uses epsilon at t=0
        eps = getattr(cfg, "ndiv_eps", 0.1)
        return neighbour_divergence_curve(
            θ, ω, ε=eps, return_curve=return_curve
        )

    elif mode == "grid":
        if theta_axis is None or omega_axis is None:
            raise ValueError("grid mode requires theta_axis and omega_axis")
        stencil = getattr(cfg, "ndiv_stencil", "8")
        return neighbour_divergence_curve_grid(
            θ, ω,
            theta_axis=theta_axis, omega_axis=omega_axis,
            stencil=stencil, return_curve=return_curve
        )

    elif mode == "window":
        if theta0 is None or omega0 is None:
            raise ValueError("window mode requires theta0 and omega0 arrays")
        # default thresholds to one grid step if not provided
        dθ = getattr(cfg, "ndiv_dtheta", None)
        dω = getattr(cfg, "ndiv_domega", None)
        if dθ is None:
            if theta_axis is None:
                raise ValueError("window mode: ndiv_dtheta not set and theta_axis is None")
            dθ = float(abs(theta_axis[1] - theta_axis[0]))
            cfg.ndiv_dtheta = dθ
        if dω is None:
            if omega_axis is None:
                raise ValueError("window mode: ndiv_domega not set and omega_axis is None")
            dω = float(abs(omega_axis[1] - omega_axis[0]))
            cfg.ndiv_domega = dω

        # optional cap per-i (keeps it linear in N). Leave None to use all in-window
        max_nb = getattr(cfg, "ndiv_window_max_neighbours", None)

        return neighbour_divergence_curve_window(
            θ, ω,
            theta0=theta0, omega0=omega0,
            dθ_max=dθ, dω_max=dω,
            max_neighbours=max_nb,
            return_curve=return_curve
        )

    elif mode == "knn":
        k = getattr(cfg, "ndiv_k", 8)
        return neighbour_divergence_curve_k(
            θ, ω, k=k, return_curve=return_curve
        )

    else:
        raise ValueError(f"Unknown ndiv_mode: {mode}")
        
# -------------------------------------------------------------------
# 7b · Energy-drift curve + rate
# -------------------------------------------------------------------
def energy_drift_curve(
    θ: torch.Tensor, ω: torch.Tensor,
    *,
    m: float = 1.0, g: float = 9.81, l: float = 1.0,
    return_curve: bool = False
) -> Tuple[np.ndarray | None, float]:
    """
    |E(t) – E(0)| averaged over rollout.
    """
    θ_np, ω_np = θ.numpy(), ω.numpy()                 # (N,T)

    E = 0.5 * m * l**2 * ω_np**2 + m * g * l * (1 - np.cos(θ_np))
    drift = np.abs(E - E[:, [0]])                     # (N,T)

    curve = drift.mean(0)                             # (T,)
    slope = _linear_slope(curve)
    return (curve if return_curve else None), slope

# # -------------------------------------------------------------------
# # 7c · Euler–Lagrange residual curve + rate
# # -------------------------------------------------------------------
# def el_residual_curve(
#     lnn,
#     θ: torch.Tensor, ω: torch.Tensor,
#     *,
#     dt: float,
#     return_curve: bool = False
# ) -> Tuple[np.ndarray | None, float]:
#     """
#     Per-time-step Euler–Lagrange residual ‖d/dt ∂L/∂v – ∂L/∂q‖².

#     Parameters
#     ----------
#     lnn
#         Trained :class:`~utils.models.LNN` (expects concat [q, v]).
#     θ, ω
#         Trajectories, each shape ``(N, T)`` **on CPU**.
#     dt
#         Time increment between frames (must match roll-out).
#     return_curve
#         • **True**  → return *(curve, slope)*  
#         • **False** → return *(None,  slope)*  (saves memory)

#     Returns
#     -------
#     curve : np.ndarray | None   (length **T-1**)
#     slope : float               linear drift-rate  d(residual)/dt
#     """
#     # -------- move inputs to the same device as LNN -----------------
#     dev = next(lnn.parameters()).device
#     q = θ.to(dev)[:, :, None]          # (N,T,1)
#     v = ω.to(dev)[:, :, None]

#     # -------- split t   and  t+1  -----------------------------------
#     q0, q1 = q[:, :-1], q[:, 1:]       # (N,T-1,1)
#     v0, v1 = v[:, :-1], v[:, 1:]

#     # Need grads on q0, v0, v1 for autograd
#     q0, v0, v1 = q0.requires_grad_(True), v0.requires_grad_(True), v1.requires_grad_(True)

#     # -------- ∂L/∂q  and  ∂L/∂v  at t -------------------------------
#     L0   = lnn(torch.cat([q0, v0], dim=-1)).sum()
#     dLdq0, dLdv0 = torch.autograd.grad(L0, (q0, v0), create_graph=True)

#     # -------- ∂L/∂v  at t+1  (needed for finite difference) ---------
#     L1   = lnn(torch.cat([q1, v1], dim=-1)).sum()
#     dLdv1 = torch.autograd.grad(L1, v1, create_graph=True)[0]

#     # -------- residual  --------------------------------------------
#     dLdv_dt = (dLdv1 - dLdv0) / dt                     # finite difference
#     res     = (dLdv_dt - dLdq0).square().squeeze(-1)   # (N,T-1)
#     curve_t = res.mean(0)                              # (T-1,)

#     curve   = curve_t.detach().cpu().numpy()           # ← detach fixes crash
#     slope   = _linear_slope(curve, dt)

#     return (curve if return_curve else None), slope

# -------------------------------------------------------------------
# 7c · Euler–Lagrange residual curve + rate  (scalable)
# -------------------------------------------------------------------
def el_residual_curve(
    lnn,
    θ: torch.Tensor, ω: torch.Tensor,    # (N,T)  CPU tensors OK
    *,
    dt: float,
    return_curve: bool = False,
    # Scalability knobs
    t_max: int | None = None,            # use only first t_max steps (after stride)
    stride: int = 1,                     # temporal downsample
    max_ic: int | None = None            # evaluate on a subset of trajectories
) -> Tuple[np.ndarray | None, float]:
    """
    Per-time-step Euler–Lagrange residual: ‖ d/dt (∂L/∂v) − ∂L/∂q ‖²
    computed over time (length T-1 after slicing). Also returns a linear
    slope (drift-rate) fitted to the curve.

    ▼ Replicating *your original* “first three frames” setting:
        el_residual_curve(..., dt=..., return_curve=True,
                          t_max=3, stride=1, max_ic=None)

    Notes
    -----
    • Inputs are automatically moved to the LNN's device.
    • `t_max`, `stride`, and `max_ic` make evaluation stable for large N,T.
    """
    # ---- subset trajectories if requested --------------------------
    if max_ic is not None and θ.size(0) > max_ic:
        sel = torch.arange(max_ic, dtype=torch.long)
        θ = θ.index_select(0, sel)
        ω = ω.index_select(0, sel)

    # ---- slice time (and then downsample) --------------------------
    if t_max is not None:
        θ = θ[:, :t_max]
        ω = ω[:, :t_max]
    if stride > 1:
        θ = θ[:, ::stride]
        ω = ω[:, ::stride]

    # Need at least 2 steps to form T-1 residuals
    if θ.size(1) < 2:
        curve = np.zeros(0, dtype=np.float32)
        return (curve if return_curve else None), 0.0

    # -------- move inputs to the same device as LNN -----------------
    dev = next(lnn.parameters()).device
    q = θ.to(dev)[:, :, None]          # (N,T,1)
    v = ω.to(dev)[:, :, None]

    # -------- split t   and  t+1  -----------------------------------
    q0, q1 = q[:, :-1], q[:, 1:]       # (N,T-1,1)
    v0, v1 = v[:, :-1], v[:, 1:]

    # Need grads on q0, v0, v1 for autograd
    q0 = q0.requires_grad_(True)
    v0 = v0.requires_grad_(True)
    v1 = v1.requires_grad_(True)

    # -------- ∂L/∂q  and  ∂L/∂v  at t -------------------------------
    L0   = lnn(torch.cat([q0, v0], dim=-1)).sum()
    dLdq0, dLdv0 = torch.autograd.grad(L0, (q0, v0), create_graph=True)

    # -------- ∂L/∂v  at t+1  (needed for finite difference) ---------
    L1   = lnn(torch.cat([q1, v1], dim=-1)).sum()
    dLdv1 = torch.autograd.grad(L1, v1, create_graph=True)[0]

    # -------- residual  --------------------------------------------
    dLdv_dt = (dLdv1 - dLdv0) / dt                     # finite difference
    res     = (dLdv_dt - dLdq0).square().squeeze(-1)   # (N,T-1)
    curve_t = res.mean(0)                              # (T-1,)

    curve   = curve_t.detach().cpu().numpy()
    # simple linear slope (least-squares fit) for drift-rate
    t = np.arange(curve.shape[0], dtype=np.float32) * dt
    if curve.size == 0:
        slope = 0.0
    else:
        # avoid sklearn dependency here
        denom = (t - t.mean()).dot(t - t.mean()) + 1e-12
        slope = float(((curve - curve.mean()).dot(t - t.mean())) / denom)

    return (curve if return_curve else None), slope

# --- Per-frame image→latent regression metrics across ALL frames ----
def perframe_img_metrics(θ_true, ω_true, θ_img, ω_img) -> Dict[str, float]:
    """
    Aggregate regression metrics across all N*T predictions.
    Inputs: torch Tensors (N,T) on CPU.
    """
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error

    θt = θ_true.numpy().reshape(-1)
    ωt = ω_true.numpy().reshape(-1)
    θp = θ_img.numpy().reshape(-1)
    ωp = ω_img.numpy().reshape(-1)

    return dict(
        r2_theta_img  = float(r2_score(θt, θp)),
        mse_theta_img = float(mean_squared_error(θt, θp)),
        r2_omega_img  = float(r2_score(ωt, ωp)),
        mse_omega_img = float(mean_squared_error(ωt, ωp)),
    )

# --- Rollout vs ground-truth MSE over ALL frames --------------------
def rollout_mse(θ_true, ω_true, θ_roll, ω_roll) -> Dict[str, float]:
    errθ = torch.mean((θ_roll - θ_true)**2).item()
    errω = torch.mean((ω_roll - ω_true)**2).item()
    return dict(mse_theta_roll=errθ, mse_omega_roll=errω)

# --- One-step self-consistency:  roll from (θ_img[t],ω_img[t]) and
#     compare to (θ_img[t+1], ω_img[t+1]) -----------------------------
def one_step_consistency(
    θ_img: torch.Tensor, ω_img: torch.Tensor, *,
    dt: float, hnn=None, lnn=None, combine="hnn", integrator="verlet"
) -> Dict[str, float]:
    """
    Measures whether the learned latent dynamics agree with the encoder’s
    next-frame predictions. Lower is better.
    Works regardless of where θ_img/ω_img live (CPU/GPU).
    """
    # ---- choose the working device ----
    if hnn is not None:
        dev = next(hnn.parameters()).device
    elif lnn is not None:
        dev = next(lnn.parameters()).device
    else:
        dev = θ_img.device  # no physics nets: stay where we are

    # move copies to the physics device
    θ = θ_img.to(dev)
    ω = ω_img.to(dev)

    # local accel helper: ALWAYS build inputs on `dev`
    def _accel(q, v):
        with torch.set_grad_enabled(True):
            if combine == "hnn" and (hnn is not None):
                x = torch.stack([q, v], dim=1).to(dev).requires_grad_(True)
                td = hnn.time_derivative(x)
                return td[:, 1] if (td.ndim == 2 and td.size(-1) >= 2) else td.reshape(-1)
            if combine == "lnn" and (lnn is not None):
                return lnn_accel(lnn, q.to(dev), v.to(dev), dt=dt)
            if combine == "mean" and (hnn is not None) and (lnn is not None):
                x = torch.stack([q, v], dim=1).to(dev).requires_grad_(True)
                td = hnn.time_derivative(x)
                a_h = td[:, 1] if (td.ndim == 2 and td.size(-1) >= 2) else td.reshape(-1)
                a_l = lnn_accel(lnn, q.to(dev), v.to(dev), dt=dt)
                return 0.5 * (a_h + a_l)
            if isinstance(combine, tuple) and combine[0] == "blend" and (hnn is not None) and (lnn is not None):
                w = float(combine[1])
                x = torch.stack([q, v], dim=1).to(dev).requires_grad_(True)
                td = hnn.time_derivative(x)
                a_h = td[:, 1] if (td.ndim == 2 and td.size(-1) >= 2) else td.reshape(-1)
                a_l = lnn_accel(lnn, q.to(dev), v.to(dev), dt=dt)
                return w * a_h + (1.0 - w) * a_l
            # fallbacks
            if hnn is not None:
                x = torch.stack([q, v], dim=1).to(dev).requires_grad_(True)
                td = hnn.time_derivative(x)
                return td[:, 1] if (td.ndim == 2 and td.size(-1) >= 2) else td.reshape(-1)
            if lnn is not None:
                return lnn_accel(lnn, q.to(dev), v.to(dev), dt=dt)
            return torch.zeros_like(v, device=dev)

    N, T = θ.shape
    if T < 2:
        return dict(cons_theta_mse=0.0, cons_omega_mse=0.0)

    q = θ[:, :-1].contiguous()
    v = ω[:, :-1].contiguous()

    qf_list, vf_list = [], []
    for t in range(T - 1):
        q_t = q[:, t].detach()
        v_t = v[:, t].detach()
        a_t = _accel(q_t, v_t)

        if integrator == "euler":
            q_next = q_t + v_t * dt
            v_next = v_t + a_t * dt
        else:
            q_next = q_t + v_t * dt + 0.5 * a_t * (dt ** 2)
            v_half = v_t + 0.5 * a_t * dt
            a_next = _accel(q_next.detach(), v_half.detach())
            v_next = v_half + 0.5 * a_next * dt

        qf_list.append(q_next)
        vf_list.append(v_next)

    θ_one = torch.stack(qf_list, dim=1)            # (N, T-1) on dev
    ω_one = torch.stack(vf_list, dim=1)            # (N, T-1) on dev

    θ_target = θ[:, 1:]                            # (N, T-1) on dev
    ω_target = ω[:, 1:]

    return dict(
        cons_theta_mse = torch.mean((θ_one - θ_target)**2).item(),
        cons_omega_mse = torch.mean((ω_one - ω_target)**2).item(),
    )