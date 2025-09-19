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


# =====================================================================
# 4 · Euler–Lagrange residual
# =====================================================================
def el_residual_metric(
    lnn,
    theta: torch.Tensor,             # (N,T)
    omega: torch.Tensor,             # (N,T)
    *,
    dt: float
) -> float:
    """
    Mean squared Euler–Lagrange residual R<sub>EL</sub> over the first
    three frames *(T ≥ 3 required)*.

    Implementation details
    ----------------------
    •  `lnn.lagrangian_residual` expects (B,T,d) on **the same device**
       as the LNN’s parameters — tensors are moved automatically.  
    •  Only *θ* and *ω* are fed; momentum terms are not needed for a
       single-pendulum.
    """
    q = theta[:, :3, None]                          # (N,3,1)   θ
    v = omega[:, :3, None]                          # (N,3,1)   ω
    lnn_device = next(lnn.parameters()).device      # LNN’s own device
    return lnn.lagrangian_residual(q.to(lnn_device),
                                   v.to(lnn_device),
                                   dt).item()


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

# -------------------------------------------------------------------
# 7c · Euler–Lagrange residual curve + rate
# -------------------------------------------------------------------
def el_residual_curve(
    lnn,
    θ: torch.Tensor, ω: torch.Tensor,
    *,
    dt: float,
    return_curve: bool = False
) -> Tuple[np.ndarray | None, float]:
    """
    Per-time-step Euler–Lagrange residual ‖d/dt ∂L/∂v – ∂L/∂q‖².

    Parameters
    ----------
    lnn
        Trained :class:`~utils.models.LNN` (expects concat [q, v]).
    θ, ω
        Trajectories, each shape ``(N, T)`` **on CPU**.
    dt
        Time increment between frames (must match roll-out).
    return_curve
        • **True**  → return *(curve, slope)*  
        • **False** → return *(None,  slope)*  (saves memory)

    Returns
    -------
    curve : np.ndarray | None   (length **T-1**)
    slope : float               linear drift-rate  d(residual)/dt
    """
    # -------- move inputs to the same device as LNN -----------------
    dev = next(lnn.parameters()).device
    q = θ.to(dev)[:, :, None]          # (N,T,1)
    v = ω.to(dev)[:, :, None]

    # -------- split t   and  t+1  -----------------------------------
    q0, q1 = q[:, :-1], q[:, 1:]       # (N,T-1,1)
    v0, v1 = v[:, :-1], v[:, 1:]

    # Need grads on q0, v0, v1 for autograd
    q0, v0, v1 = q0.requires_grad_(True), v0.requires_grad_(True), v1.requires_grad_(True)

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

    curve   = curve_t.detach().cpu().numpy()           # ← detach fixes crash
    slope   = _linear_slope(curve, dt)

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