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