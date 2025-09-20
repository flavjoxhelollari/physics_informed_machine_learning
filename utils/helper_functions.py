"""
helper_functions.py
===================

Thin, **self-contained** helper layer that sits *between* your core
models (V-JEPA, HNN, LNN) and the surrounding experimentation notebooks.

Motivation
----------
The same snippets kept re-appearing in your notebooks:
“compute energy”, “roll out latent states”, “plot phase-space …”.
This module centralises those utilities **with exhaustive inline
comments** so that:

* Every research script can simply `import helper_functions as hf`.
* Newcomers can read one file and understand the maths + tensor shapes.
* Unit tests & linting tools have explicit type annotations to rely on.

Public API
~~~~~~~~~~
compute_energy()            – analytic simple-pendulum energy  
plot_energy_per_episode()   – per-episode energy curves  
plot_true_phase_space()     – scatter of (θ, ω) ground-truth pairs  
lnn_accel()                 – numerically stable α from a trained LNN  
rollout()                   – latent → phase-space trajectory (θ, ω, α)

All functions work on **CPU or CUDA** tensors; the selected `device` is
defined once at the top and can be overridden per call.
"""

from __future__ import annotations

import os                                       # file-system helpers
from typing import Optional, Tuple              # type hints

import numpy as np                              # maths / plotting
import matplotlib.pyplot as plt                 # lightweight viz

import torch                                    # tensors & autograd
import torch.nn.functional as F                # loss helpers
import tqdm as tqdm                          # nice progress bars

# ---------------------------------------------------------------------
# Global device used by default (override via keyword args if desired)
# ---------------------------------------------------------------------
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# 0 · Analytic energy function
# =====================================================================
def compute_energy(
    theta: np.ndarray | torch.Tensor,           # angle(s)  [rad]
    omega: np.ndarray | torch.Tensor,           # ang. vel. [rad s⁻¹]
    *,
    m: float = 1.0,                             # mass       [kg]
    l: float = 1.0,                             # length     [m]
    g: float = 9.81                             # gravity    [m s⁻²]
) -> np.ndarray:
    """
    Simple-pendulum energy

        E = ½ m l² ω²  +  m g l (1 − cos θ)

    Both NumPy arrays **and** torch tensors are accepted; output is
    returned as a NumPy array to play nicely with Matplotlib.

    Examples
    --------
    >>> E = compute_energy(theta_np, omega_np, l=0.5)      # NumPy
    >>> E = compute_energy(theta_torch, omega_torch).mean()
    """
    # Convert torch.Tensors → np.ndarray for unified downstream maths
    if torch.is_tensor(theta):
        theta = theta.detach().cpu().numpy()
    if torch.is_tensor(omega):
        omega = omega.detach().cpu().numpy()

    kinetic   = 0.5 * m * (l ** 2) * omega ** 2              # ½ m l² ω²
    potential = m * g * l * (1 - np.cos(theta))              # m g l (1-cosθ)
    return kinetic + potential                               # element-wise sum


# =====================================================================
# 1 · Episode-wise energy curves
# =====================================================================
def plot_energy_per_episode(
    dataset,
    *,
    episode_length: int,
    m: float       = 1.0,
    l: float       = 1.0,
    g: float       = 9.81,
    save_path: Optional[str] = None
) -> None:
    """
    Plot total mechanical energy for *each* episode in a **flat**
    `PendulumDataset`.

    Assumptions
    -----------
    * `dataset[i]` returns ``(img, label)`` where ``label == (theta, omega)``.
    * Samples are *ordered* episode-by-episode.  `episode_length` must
      match the generator arguments used for the data-set.
    """
    n_epi  = len(dataset) // episode_length        # total episodes
    energy = []                                    # list[np.ndarray]

    ptr = 0                                        # dataset index pointer
    for _ in range(n_epi):
        theta_seq, omega_seq = [], []
        for _ in range(episode_length):            # iterate one episode
            _, lbl = dataset[ptr]
            theta_seq.append(float(lbl[0]))        # θ_t  as Python float
            omega_seq.append(float(lbl[1]))        # ω_t
            ptr += 1                               # advance pointer
        energy.append(
            compute_energy(np.array(theta_seq), np.array(omega_seq),
                           m=m, l=l, g=g) )        # per-time-step energy

    # ---------- visualisation ----------------------------------
    plt.figure(figsize=(10, 4))
    for e in energy:
        plt.plot(e, alpha=.6)                      # one line per episode
    plt.xlabel("time-step"); plt.ylabel("energy [J]")
    plt.title("Pendulum energy per episode"); plt.grid(True)
    plt.tight_layout()
    if save_path:                                 # mkdir if needed
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=120)
    plt.show()


# =====================================================================
# 2 · Raw data phase-space scatter
# =====================================================================
def plot_true_phase_space(
    dataset,
    *,
    num_samples: int       = 500,
    save_path  : Optional[str] = None
) -> None:
    """
    Scatter of ground-truth *(θ, ω)* pairs for quick sanity checking.

    Only the **first** `num_samples` items are used to keep the plot
    lightweight in notebooks.
    """
    θ_list, ω_list = [], []
    for i in range(min(num_samples, len(dataset))):
        _, lbl = dataset[i]
        θ_list.append(float(lbl[0]))               # collect theta
        ω_list.append(float(lbl[1]))               # collect omega

    plt.figure(figsize=(5, 5))
    plt.scatter(θ_list, ω_list, s=10, alpha=.35)
    plt.xlabel("θ [rad]"); plt.ylabel("ω [rad s⁻¹]")
    plt.title("True phase-space samples"); plt.grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=120)
    plt.show()


# =====================================================================
# 3 · LNN-based acceleration (stable)
# =====================================================================
def lnn_accel(
    lnn,
    q : torch.Tensor,               # (B,)  angle     [rad]
    v : torch.Tensor,               # (B,)  ang. vel. [rad s⁻¹]
    *,
    dt       : float,
    eps      : float = 1e-6,
    clip_val : Optional[float] = 25.0
) -> torch.Tensor:
    """
    Compute angular acceleration **α** from a trained `LNN` by applying
    the Euler–Lagrange operator.

    α = ( d/dt ∂L/∂v − ∂L/∂q ) / ∂²L/∂v²

    Gradient bookkeeping
    --------------------
    * The incoming *q*,*v* need **no** grad – the function creates a
      differentiable clone internally.
    * Finite-difference d/dt term uses a single‐step Euler estimate.

    Returns
    -------
    torch.Tensor  – shape (B,) on the *same* device as input tensors.
    """
    qv = torch.stack([q, v], dim=1).requires_grad_(True)    # (B,2) w/ grad

    # Compute scalar L(q,v) over the batch
    L = lnn(qv).sum()                                       # forward pass

    # First derivatives  ∂L/∂q , ∂L/∂v
    dLdqv = torch.autograd.grad(L, qv, create_graph=True)[0]# (B,2)
    dLdq, dLdv = dLdqv[:, 0], dLdqv[:, 1]

    # Time derivative of ∂L/∂v  via forward Euler
    dLdv_dt = (dLdv - dLdv.detach()) / dt                   # (B,)

    # Second derivative  ∂²L/∂v²  (diagonal element)
    d2Ldv2 = torch.autograd.grad(dLdv.sum(), qv,
                                 create_graph=True)[0][:, 1]

    # Stabilise denominator to avoid exploding values
    denom  = d2Ldv2.abs().clamp_min(eps) * d2Ldv2.sign()

    α_pred = (dLdv_dt - dLdq) / denom                       # final formula
    if clip_val is not None:
        α_pred = α_pred.clamp(-clip_val, clip_val)          # optional clamp
    return α_pred


# ------------------------------------------------------------------
# 4 · Latent roll-out helper   ––  now supports `combine`
# ------------------------------------------------------------------
# @torch.no_grad()
# def rollout(
#     vjepa,
#     head,
#     *,
#     eval_loader,                   # deterministic grid
#     horizon   : int,
#     dt        : float,
#     hnn       = None,
#     lnn       = None,
#     combine   : str | tuple[str, float] | None = None
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Integration in latent (θ, ω) space with optional learned acceleration.

#     Parameters
#     ----------
#     hnn, lnn
#         Instances or *None* – may supply **one or both**.
#     combine
#         Behaviour when *both* nets are supplied:

#         * ``None``      → pick whichever net is available (HNN ≻ LNN)
#         * ``"hnn"``     → ignore LNN, use HNN
#         * ``"lnn"``     → ignore HNN, use LNN
#         * ``"mean"``    → α = ½(α_HNN + α_LNN)
#         * ``"sum"``     → α = α_HNN + α_LNN
#         * ``("blend", w)`` → weighted mix 0 ≤ w ≤ 1
#     """
#     if horizon < 2:
#         raise ValueError("horizon must be ≥ 2")

#     # --------------- sanity on combine spec -------------------------
#     if combine is None:
#         picker = "hnn" if hnn is not None else "lnn"          # auto-pick
#     elif combine in {"hnn", "lnn", "mean", "sum"}:
#         picker = combine
#     elif isinstance(combine, tuple) and combine[0] == "blend":
#         picker = ("blend", float(combine[1]))
#     else:
#         raise ValueError(f"rollout(): unknown combine spec {combine!r}")

#     # ensure nets in eval
#     vjepa.eval(); head.eval()
#     if hnn: hnn.eval()
#     if lnn: lnn.eval()

#     θ_buf, ω_buf, α_buf = [], [], []

#     for seq, _ in tqdm.tqdm(eval_loader, desc="rollout", leave=False):
#         imgs0 = seq[:, 0].to(device)

#         # latent → initial θ, ω
#         z0 = vjepa.context_encoder(
#                  vjepa.patch_embed(imgs0) + vjepa.pos_embed).mean(1)
#         θ, ω = head(z0).split(1, 1);  θ, ω = θ.squeeze(), ω.squeeze()

#         Θ, Ω, Α = [θ], [ω], []

#         for _ in range(horizon - 1):
#             q, v = Θ[-1].detach(), Ω[-1].detach()

#             # ---------------- choose / fuse acceleration ------------
#             if (picker == "hnn") and (hnn is not None):
#                 with torch.set_grad_enabled(True):
#                     α = hnn.time_derivative(torch.stack([q, v], 1)
#                                             .requires_grad_(True))[:, 1]

#             elif (picker == "lnn") and (lnn is not None):
#                 with torch.set_grad_enabled(True):
#                     α = lnn_accel(lnn, q, v, dt=dt)

#             elif (picker == "mean") and (hnn is not None) and (lnn is not None):
#                 with torch.set_grad_enabled(True):
#                     α_h = hnn.time_derivative(torch.stack([q, v], 1)
#                                               .requires_grad_(True))[:, 1]
#                     α_l = lnn_accel(lnn, q, v, dt=dt)
#                     α   = 0.5 * (α_h + α_l)

#             elif (picker == "sum") and (hnn is not None) and (lnn is not None):
#                 with torch.set_grad_enabled(True):
#                     α_h = hnn.time_derivative(torch.stack([q, v], 1)
#                                               .requires_grad_(True))[:, 1]
#                     α_l = lnn_accel(lnn, q, v, dt=dt)
#                     α   = α_h + α_l

#             elif isinstance(picker, tuple) and picker[0] == "blend" \
#                  and (hnn is not None) and (lnn is not None):
#                 w = float(picker[1])
#                 with torch.set_grad_enabled(True):
#                     α_h = hnn.time_derivative(torch.stack([q, v], 1)
#                                               .requires_grad_(True))[:, 1]
#                     α_l = lnn_accel(lnn, q, v, dt=dt)
#                     α   = w * α_h + (1.0 - w) * α_l

#             else:
#                 # fallback: if requested fuse not possible, choose available net
#                 if hnn is not None:
#                     with torch.set_grad_enabled(True):
#                         α = hnn.time_derivative(torch.stack([q, v], 1)
#                                                 .requires_grad_(True))[:, 1]
#                 elif lnn is not None:
#                     with torch.set_grad_enabled(True):
#                         α = lnn_accel(lnn, q, v, dt=dt)
#                 else:
#                     α = torch.zeros_like(v)

#             # ---------------- Euler update ---------------------------
#             Θ.append(q + v * dt)
#             Ω.append(v + α * dt)
#             Α.append(α)

#         Α = [torch.zeros_like(Α[0])] + Α          # prepend α₀ = 0
#         θ_buf.append(torch.stack(Θ, 1))
#         ω_buf.append(torch.stack(Ω, 1))
#         α_buf.append(torch.stack(Α, 1))

#     return torch.cat(θ_buf).cpu(), torch.cat(ω_buf).cpu(), torch.cat(α_buf).cpu()

# @torch.no_grad()
# def rollout(
#     vjepa,
#     head,
#     *,
#     eval_loader,                   # deterministic grid
#     horizon   : int,
#     dt        : float,
#     hnn       = None,
#     lnn       = None,
#     combine   : str | tuple[str, float] | None = None
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Integration in latent (θ, ω) space with optional learned acceleration.

#     Parameters
#     ----------
#     hnn, lnn
#         Instances or *None* – may supply **one or both**.
#     combine
#         Behaviour when *both* nets are supplied:

#         * ``None``      → pick whichever net is available (HNN ≻ LNN)
#         * ``"hnn"``     → ignore LNN, use HNN
#         * ``"lnn"``     → ignore HNN, use LNN
#         * ``"mean"``    → α = ½(α_HNN + α_LNN)
#         * ``"sum"``     → α = α_HNN + α_LNN
#         * ``("blend", w)`` → weighted mix 0 ≤ w ≤ 1
#     """
#     def _hnn_accel_from_td(hnn, q, v):
#         """
#         Returns acceleration from an HNN time_derivative() that might output:
#           - (B,2): [dq/dt, dv/dt]  -> take column 1
#           - (B,1) or (B,) : acceleration directly
#         q,v can be (B,), (B,1), or scalars; we normalize to (B,).
#         """
#         # normalize q,v -> (B,)
#         q = q.reshape(-1)
#         v = v.reshape(-1)
    
#         x = torch.stack([q, v], dim=1).requires_grad_(True)  # (B,2)
#         td = hnn.time_derivative(x)                          # (B,2) or (B,1)/(B,)
    
#         if td.ndim == 2 and td.size(-1) >= 2:
#             α = td[:, 1]             # dv/dt
#         else:
#             α = td.reshape(-1)       # accel-only
#         return α
    
#     if horizon < 2:
#         raise ValueError("horizon must be ≥ 2")

#     # --------------- sanity on combine spec -------------------------
#     if combine is None:
#         picker = "hnn" if hnn is not None else "lnn"          # auto-pick
#     elif combine in {"hnn", "lnn", "mean", "sum"}:
#         picker = combine
#     elif isinstance(combine, tuple) and combine[0] == "blend":
#         picker = ("blend", float(combine[1]))
#     else:
#         raise ValueError(f"rollout(): unknown combine spec {combine!r}")

#     # ensure nets in eval
#     vjepa.eval(); head.eval()
#     if hnn: hnn.eval()
#     if lnn: lnn.eval()

#     θ_buf, ω_buf, α_buf = [], [], []

#     for seq, _ in tqdm.tqdm(eval_loader, desc="rollout", leave=False):
#         imgs0 = seq[:, 0].to(device)

#         # latent → initial θ, ω
#         z0 = vjepa.context_encoder(
#                  vjepa.patch_embed(imgs0) + vjepa.pos_embed).mean(1)
#         θ, ω = head(z0).split(1, dim=1)   # (B,1), (B,1)
#         θ, ω = θ.squeeze(1), ω.squeeze(1) # (B,), (B,)

#         Θ, Ω, Α = [θ], [ω], []

#         for _ in range(horizon - 1):
#             q, v = Θ[-1].detach(), Ω[-1].detach()

#             # ---------------- choose / fuse acceleration ------------
#             if (picker == "hnn") and (hnn is not None):
#                 with torch.set_grad_enabled(True):
#                     α = _hnn_accel_from_td(hnn, q, v)

#             elif (picker == "lnn") and (lnn is not None):
#                 with torch.set_grad_enabled(True):
#                     α = lnn_accel(lnn, q, v, dt=dt)

#             elif (picker == "mean") and (hnn is not None) and (lnn is not None):
#                 with torch.set_grad_enabled(True):
#                     α_h = _hnn_accel_from_td(hnn, q, v)
#                     α_l = lnn_accel(lnn, q, v, dt=dt)
#                     α   = 0.5 * (α_h + α_l)

#             elif (picker == "sum") and (hnn is not None) and (lnn is not None):
#                 with torch.set_grad_enabled(True):
#                     α_h = _hnn_accel_from_td(hnn, q, v)
#                     α_l = lnn_accel(lnn, q, v, dt=dt)
#                     α   = α_h + α_l

#             elif isinstance(picker, tuple) and picker[0] == "blend" \
#                  and (hnn is not None) and (lnn is not None):
#                 w = float(picker[1])
#                 with torch.set_grad_enabled(True):
#                     α_h = _hnn_accel_from_td(hnn, q, v)
#                     α_l = lnn_accel(lnn, q, v, dt=dt)
#                     α   = w * α_h + (1.0 - w) * α_l

#             else:
#                 # fallback: if requested fuse not possible, choose available net
#                 if hnn is not None:
#                     with torch.set_grad_enabled(True):
#                         α = _hnn_accel_from_td(hnn, q, v)
#                 elif lnn is not None:
#                     with torch.set_grad_enabled(True):
#                         α = lnn_accel(lnn, q, v, dt=dt)
#                 else:
#                     α = torch.zeros_like(v)

#             # ---------------- Euler update ---------------------------
#             Θ.append(q + v * dt)
#             Ω.append(v + α * dt)
#             Α.append(α)

#         Α = [torch.zeros_like(Α[0])] + Α          # prepend α₀ = 0
#         θ_buf.append(torch.stack(Θ, 1))
#         ω_buf.append(torch.stack(Ω, 1))
#         α_buf.append(torch.stack(Α, 1))

#     return torch.cat(θ_buf).cpu(), torch.cat(ω_buf).cpu(), torch.cat(α_buf).cpu()

@torch.no_grad()
def rollout(
    vjepa,
    head,                                 # legacy combined head (384→2) or None
    *,
    eval_loader,
    horizon: int,
    dt: float,
    hnn=None,
    lnn=None,
    combine=None,
    # NEW: optional separate heads
    theta_head_1f: torch.nn.Module | None = None,   # 384→1
    omega_head_2f: torch.nn.Module | None = None,   # 768→1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    If theta_head_1f & omega_head_2f are provided, use:
      θ0 = theta_head_1f(z0) from frame-0
      ω0 = omega_head_2f([z0 ⊕ z1]) from frames (0,1)
    Otherwise fall back to legacy: head(z0) → (θ0, ω0).
    Integration (θ, ω) uses α from HNN/LNN exactly as before.
    """
    def _hnn_accel_from_td(hnn, q, v):
        q = q.reshape(-1); v = v.reshape(-1)
        x = torch.stack([q, v], dim=1).requires_grad_(True)
        td = hnn.time_derivative(x)
        return td[:, 1] if (td.ndim == 2 and td.size(-1) >= 2) else td.reshape(-1)

    if horizon < 2:
        raise ValueError("horizon must be ≥ 2")

    # choose accel source (unchanged)
    if combine is None:
        picker = "hnn" if hnn is not None else "lnn"
    elif combine in {"hnn", "lnn", "mean", "sum"}:
        picker = combine
    elif isinstance(combine, tuple) and combine[0] == "blend":
        picker = ("blend", float(combine[1]))
    else:
        raise ValueError(f"rollout(): unknown combine spec {combine!r}")

    vjepa.eval()
    if head is not None: head.eval()
    if theta_head_1f is not None: theta_head_1f.eval()
    if omega_head_2f is not None: omega_head_2f.eval()
    if hnn: hnn.eval()
    if lnn: lnn.eval()

    θ_buf, ω_buf, α_buf = [], [], []

    for seq, _ in tqdm.tqdm(eval_loader, desc="rollout", leave=False):
        imgs0 = seq[:, 0].to(device)
        z0 = vjepa.context_encoder(vjepa.patch_embed(imgs0) + vjepa.pos_embed).mean(1)  # (B,D)

        if (theta_head_1f is not None) and (omega_head_2f is not None) and (seq.size(1) >= 2):
            imgs1 = seq[:, 1].to(device)
            z1 = vjepa.context_encoder(vjepa.patch_embed(imgs1) + vjepa.pos_embed).mean(1)
            θ0 = theta_head_1f(z0).squeeze(1)                           # (B,)
            ω0 = omega_head_2f(torch.cat([z0, z1], dim=1)).squeeze(1)   # (B,)
        else:
            # legacy combined head
            assert head is not None, "rollout needs either (head) or (theta_head_1f & omega_head_2f)"
            θ0, ω0 = head(z0).split(1, dim=1)
            θ0, ω0 = θ0.squeeze(1), ω0.squeeze(1)

        Θ, Ω, Α = [θ0], [ω0], []

        for _ in range(horizon - 1):
            q, v = Θ[-1].detach(), Ω[-1].detach()
            if (picker == "hnn") and (hnn is not None):
                with torch.set_grad_enabled(True):
                    α = _hnn_accel_from_td(hnn, q, v)
            elif (picker == "lnn") and (lnn is not None):
                with torch.set_grad_enabled(True):
                    α = lnn_accel(lnn, q, v, dt=dt)
            elif (picker == "mean") and (hnn is not None) and (lnn is not None):
                with torch.set_grad_enabled(True):
                    α = 0.5 * (_hnn_accel_from_td(hnn, q, v) + lnn_accel(lnn, q, v, dt=dt))
            elif (picker == "sum") and (hnn is not None) and (lnn is not None):
                with torch.set_grad_enabled(True):
                    α = _hnn_accel_from_td(hnn, q, v) + lnn_accel(lnn, q, v, dt=dt)
            elif isinstance(picker, tuple) and picker[0] == "blend" and (hnn is not None) and (lnn is not None):
                w = float(picker[1])
                with torch.set_grad_enabled(True):
                    α = w * _hnn_accel_from_td(hnn, q, v) + (1.0 - w) * lnn_accel(lnn, q, v, dt=dt)
            else:
                if hnn is not None:
                    with torch.set_grad_enabled(True):
                        α = _hnn_accel_from_td(hnn, q, v)
                elif lnn is not None:
                    with torch.set_grad_enabled(True):
                        α = lnn_accel(lnn, q, v, dt=dt)
                else:
                    α = torch.zeros_like(v)

            Θ.append(q + v * dt)
            Ω.append(v + α * dt)
            Α.append(α)

        Α = [torch.zeros_like(Α[0])] + Α
        θ_buf.append(torch.stack(Θ, 1))
        ω_buf.append(torch.stack(Ω, 1))
        α_buf.append(torch.stack(Α, 1))

    return torch.cat(θ_buf).cpu(), torch.cat(ω_buf).cpu(), torch.cat(α_buf).cpu()