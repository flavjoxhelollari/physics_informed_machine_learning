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

@torch.no_grad()
def rollout(
    vjepa,
    head,
    *,
    eval_loader,                   # deterministic grid
    horizon   : int,
    dt        : float,
    hnn       = None,
    lnn       = None,
    combine   : str | tuple[str, float] | None = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Integration in latent (θ, ω) space with optional learned acceleration.

    Parameters
    ----------
    hnn, lnn
        Instances or *None* – may supply **one or both**.
    combine
        Behaviour when *both* nets are supplied:

        * ``None``      → pick whichever net is available (HNN ≻ LNN)
        * ``"hnn"``     → ignore LNN, use HNN
        * ``"lnn"``     → ignore HNN, use LNN
        * ``"mean"``    → α = ½(α_HNN + α_LNN)
        * ``"sum"``     → α = α_HNN + α_LNN
        * ``("blend", w)`` → weighted mix 0 ≤ w ≤ 1
    """
    def _hnn_accel_from_td(hnn, q, v):
        """
        Returns acceleration from an HNN time_derivative() that might output:
          - (B,2): [dq/dt, dv/dt]  -> take column 1
          - (B,1) or (B,) : acceleration directly
        q,v can be (B,), (B,1), or scalars; we normalize to (B,).
        """
        # normalize q,v -> (B,)
        q = q.reshape(-1)
        v = v.reshape(-1)
    
        x = torch.stack([q, v], dim=1).requires_grad_(True)  # (B,2)
        td = hnn.time_derivative(x)                          # (B,2) or (B,1)/(B,)
    
        if td.ndim == 2 and td.size(-1) >= 2:
            α = td[:, 1]             # dv/dt
        else:
            α = td.reshape(-1)       # accel-only
        return α
    
    if horizon < 2:
        raise ValueError("horizon must be ≥ 2")

    # --------------- sanity on combine spec -------------------------
    if combine is None:
        picker = "hnn" if hnn is not None else "lnn"          # auto-pick
    elif combine in {"hnn", "lnn", "mean", "sum"}:
        picker = combine
    elif isinstance(combine, tuple) and combine[0] == "blend":
        picker = ("blend", float(combine[1]))
    else:
        raise ValueError(f"rollout(): unknown combine spec {combine!r}")

    # ensure nets in eval
    vjepa.eval(); head.eval()
    if hnn: hnn.eval()
    if lnn: lnn.eval()

    θ_buf, ω_buf, α_buf = [], [], []

    for seq, _ in tqdm.tqdm(eval_loader, desc="rollout", leave=False):
        imgs0 = seq[:, 0].to(device)

        # latent → initial θ, ω
        z0 = vjepa.context_encoder(
                 vjepa.patch_embed(imgs0) + vjepa.pos_embed).mean(1)
        θ, ω = head(z0).split(1, dim=1)   # (B,1), (B,1)
        θ, ω = θ.squeeze(1), ω.squeeze(1) # (B,), (B,)

        Θ, Ω, Α = [θ], [ω], []

        for _ in range(horizon - 1):
            q, v = Θ[-1].detach(), Ω[-1].detach()

            # ---------------- choose / fuse acceleration ------------
            if (picker == "hnn") and (hnn is not None):
                with torch.set_grad_enabled(True):
                    α = _hnn_accel_from_td(hnn, q, v)

            elif (picker == "lnn") and (lnn is not None):
                with torch.set_grad_enabled(True):
                    α = lnn_accel(lnn, q, v, dt=dt)

            elif (picker == "mean") and (hnn is not None) and (lnn is not None):
                with torch.set_grad_enabled(True):
                    α_h = _hnn_accel_from_td(hnn, q, v)
                    α_l = lnn_accel(lnn, q, v, dt=dt)
                    α   = 0.5 * (α_h + α_l)

            elif (picker == "sum") and (hnn is not None) and (lnn is not None):
                with torch.set_grad_enabled(True):
                    α_h = _hnn_accel_from_td(hnn, q, v)
                    α_l = lnn_accel(lnn, q, v, dt=dt)
                    α   = α_h + α_l

            elif isinstance(picker, tuple) and picker[0] == "blend" \
                 and (hnn is not None) and (lnn is not None):
                w = float(picker[1])
                with torch.set_grad_enabled(True):
                    α_h = _hnn_accel_from_td(hnn, q, v)
                    α_l = lnn_accel(lnn, q, v, dt=dt)
                    α   = w * α_h + (1.0 - w) * α_l

            else:
                # fallback: if requested fuse not possible, choose available net
                if hnn is not None:
                    with torch.set_grad_enabled(True):
                        α = _hnn_accel_from_td(hnn, q, v)
                elif lnn is not None:
                    with torch.set_grad_enabled(True):
                        α = lnn_accel(lnn, q, v, dt=dt)
                else:
                    α = torch.zeros_like(v)

            # ---------------- Euler update ---------------------------
            Θ.append(q + v * dt)
            Ω.append(v + α * dt)
            Α.append(α)

        Α = [torch.zeros_like(Α[0])] + Α          # prepend α₀ = 0
        θ_buf.append(torch.stack(Θ, 1))
        ω_buf.append(torch.stack(Ω, 1))
        α_buf.append(torch.stack(Α, 1))

    return torch.cat(θ_buf).cpu(), torch.cat(ω_buf).cpu(), torch.cat(α_buf).cpu()

import torch, tqdm
from typing import Tuple, Optional

@torch.no_grad()
def rollout_sim_better(
    vjepa,
    head,
    *,
    eval_loader,                   # (B, T, C,H,W) sequences
    horizon   : int,
    dt        : float,
    hnn       = None,
    lnn       = None,
    combine   : str | tuple[str, float] | None = None,
    device    = None,
    integrator: str = "verlet",    # "verlet" | "euler"
    context_mode: Optional[str] = None,  # None | "t0" | "perframe"
    reanchor_every: Optional[int] = None,
    reanchor_beta: float = 0.1,
    normalize: Optional[callable] = None,  # fn(q,v) -> (q_n,v_n) if you used norm in training
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Improved 'simulation' rollout:
      - velocity-Verlet (symplectic) integrator
      - optional context z_t to HNN/LNN (t0 or perframe)
      - optional periodic re-anchoring toward per-frame encoder predictions

    Returns:
      θ_roll, ω_roll, α_roll   [all (N, horizon)]
    """
    if device is None:
        device = next(head.parameters()).device
    if horizon < 2:
        raise ValueError("horizon must be ≥ 2")

    # --- picker for HNN/LNN fusion ---
    if combine is None:
        picker = "hnn" if hnn is not None else "lnn"
    elif combine in {"hnn", "lnn", "mean", "sum"}:
        picker = combine
    elif isinstance(combine, tuple) and combine[0] == "blend":
        picker = ("blend", float(combine[1]))
    else:
        raise ValueError(f"unknown combine spec {combine!r}")

    def _hnn_accel(hnn, q, v, z_ctx=None):
        # Inputs are (B,)
        x = torch.stack([q.reshape(-1), v.reshape(-1)], dim=1).requires_grad_(True)
        # If your HNN can take context, extend inputs or set an internal buffer here.
        td = hnn.time_derivative(x)
        if td.ndim == 2 and td.size(-1) >= 2:
            return td[:, 1]
        return td.reshape(-1)

    def _lnn_accel(lnn, q, v, z_ctx=None):
        # If your LNN accepts context, extend signature there; otherwise ignore z_ctx.
        return lnn_accel(lnn, q, v, dt=dt)

    def _accel(q, v, z_ctx):
        # Optionally normalize to match physics-net training
        if normalize is not None:
            qn, vn = normalize(q, v)
        else:
            qn, vn = q, v
        if (picker == "hnn") and (hnn is not None):
            with torch.set_grad_enabled(True):
                return _hnn_accel(hnn, qn, vn, z_ctx)
        if (picker == "lnn") and (lnn is not None):
            with torch.set_grad_enabled(True):
                return _lnn_accel(lnn, qn, vn, z_ctx)
        if (picker == "mean") and (hnn is not None) and (lnn is not None):
            with torch.set_grad_enabled(True):
                return 0.5*(_hnn_accel(hnn, qn, vn, z_ctx) + _lnn_accel(lnn, qn, vn, z_ctx))
        if (picker == "sum") and (hnn is not None) and (lnn is not None):
            with torch.set_grad_enabled(True):
                return _hnn_accel(hnn, qn, vn, z_ctx) + _lnn_accel(lnn, qn, vn, z_ctx)
        if isinstance(picker, tuple) and picker[0] == "blend" and (hnn is not None) and (lnn is not None):
            w = float(picker[1])
            with torch.set_grad_enabled(True):
                return w*_hnn_accel(hnn, qn, vn, z_ctx) + (1.0-w)*_lnn_accel(lnn, qn, vn, z_ctx)
        # fallback
        if hnn is not None:
            with torch.set_grad_enabled(True):
                return _hnn_accel(hnn, qn, vn, z_ctx)
        if lnn is not None:
            with torch.set_grad_enabled(True):
                return _lnn_accel(lnn, qn, vn, z_ctx)
        return torch.zeros_like(v)

    # --- eval modes ---
    vjepa.eval(); head.eval()
    if hnn: hnn.eval()
    if lnn: lnn.eval()

    θ_buf, ω_buf, α_buf = [], [], []

    for seq, *_ in tqdm.tqdm(eval_loader, desc="rollout_sim_better", leave=False):
        seq = seq.to(device)                         # (B, T, C,H,W)
        B, T = seq.shape[0], seq.shape[1]
        T_eff = min(T, horizon)

        # Encode first frame and (optionally) all frames for context/re-anchoring
        imgs0 = seq[:, 0]
        z0 = vjepa.context_encoder(vjepa.patch_embed(imgs0) + vjepa.pos_embed).mean(1)  # (B,D)
        θ0, ω0 = head(z0).split(1, dim=1)
        q = θ0.squeeze(1)   # (B,)
        v = ω0.squeeze(1)   # (B,)

        if context_mode == "perframe":
            Z = []
            for t in range(T_eff):
                zt = vjepa.context_encoder(
                        vjepa.patch_embed(seq[:, t]) + vjepa.pos_embed).mean(1)
                Z.append(zt)
            Z = torch.stack(Z, dim=1)  # (B, T_eff, D)
        else:
            Z = None

        Θ, Ω, Α = [q], [v], []
        # initial accel
        z_ctx0 = z0 if context_mode == "t0" else (Z[:, 0] if (Z is not None) else None)
        a = _accel(q.detach(), v.detach(), z_ctx0)

        for t in range(horizon - 1):
            if integrator == "euler":
                # explicit Euler
                q_next = q + v*dt
                v_next = v + a*dt
                # next accel
                z_ctx = (z0 if context_mode == "t0"
                         else (Z[:, min(t+1, T_eff-1)] if (Z is not None) else None))
                a_next = _accel(q_next.detach(), v_next.detach(), z_ctx)
            else:
                # velocity-Verlet (symplectic)
                q_next = q + v*dt + 0.5*a*(dt**2)
                v_half = v + 0.5*a*dt
                z_ctx = (z0 if context_mode == "t0"
                         else (Z[:, min(t+1, T_eff-1)] if (Z is not None) else None))
                a_next = _accel(q_next.detach(), v_half.detach(), z_ctx)
                v_next = v_half + 0.5*a_next*dt

            Θ.append(q_next); Ω.append(v_next); Α.append(a)
            q, v, a = q_next, v_next, a_next

            # optional periodic re-anchoring toward per-frame encoder predictions
            if (reanchor_every is not None) and ((t+1) % reanchor_every == 0) and (t+1 < T_eff):
                with torch.no_grad():
                    z_corr = (Z[:, t+1] if (Z is not None) else
                              vjepa.context_encoder(vjepa.patch_embed(seq[:, t+1]) + vjepa.pos_embed).mean(1))
                    θ_corr, ω_corr = head(z_corr).split(1, dim=1)
                    θ_corr = θ_corr.squeeze(1); ω_corr = ω_corr.squeeze(1)
                    q = (1 - reanchor_beta)*q + reanchor_beta*θ_corr
                    v = (1 - reanchor_beta)*v + reanchor_beta*ω_corr
                # refresh accel after correction
                a = _accel(q.detach(), v.detach(), z_ctx)

        # pad α_0 for T alignment
        Α = [torch.zeros_like(Α[0])] + Α
        θ_buf.append(torch.stack(Θ, dim=1))
        ω_buf.append(torch.stack(Ω, dim=1))
        α_buf.append(torch.stack(Α, dim=1))

    return torch.cat(θ_buf).cpu(), torch.cat(ω_buf).cpu(), torch.cat(α_buf).cpu()

@torch.no_grad()
def rollout_align_with_goal(
    vjepa,
    head,
    *,
    eval_loader,                   # (B, T, C,H,W)
    horizon   : int,
    dt        : float,
    hnn       = None,
    lnn       = None,
    combine   : str | tuple[str, float] | None = None,
    device    = None,
    integrator: str = "verlet",
    context_mode: Optional[str] = None,    # None | "t0" | "perframe"
    normalize: Optional[callable] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      θ_img  : (N, T)   per-frame predictions from images
      ω_img  : (N, T)
      θ_roll : (N, T)   physics rollout from t=0 using (θ_img[:,0], ω_img[:,0])
      ω_roll : (N, T)
      α_roll : (N, T)
    """
    θ_img_buf, ω_img_buf = [], []
    θ_roll_buf, ω_roll_buf, α_roll_buf = [], [], []

    # Reuse the accel and integrator from rollout_sim_better
    def _hnn_accel(hnn, q, v, z_ctx=None):
        x = torch.stack([q.reshape(-1), v.reshape(-1)], dim=1).requires_grad_(True)
        td = hnn.time_derivative(x)
        return (td[:,1] if (td.ndim==2 and td.size(-1)>=2) else td.reshape(-1))

    def _lnn_accel(lnn, q, v, z_ctx=None):
        return lnn_accel(lnn, q, v, dt=dt)

    if combine is None:
        picker = "hnn" if hnn is not None else "lnn"
    elif combine in {"hnn", "lnn", "mean", "sum"}:
        picker = combine
    elif isinstance(combine, tuple) and combine[0] == "blend":
        picker = ("blend", float(combine[1]))
    else:
        raise ValueError(f"unknown combine spec {combine!r}")

    def _accel(q, v, z_ctx):
        if normalize is not None:
            q, v = normalize(q, v)
        if (picker == "hnn") and (hnn is not None):
            with torch.set_grad_enabled(True):
                return _hnn_accel(hnn, q, v, z_ctx)
        if (picker == "lnn") and (lnn is not None):
            with torch.set_grad_enabled(True):
                return _lnn_accel(lnn, q, v, z_ctx)
        if (picker == "mean") and (hnn is not None) and (lnn is not None):
            with torch.set_grad_enabled(True):
                return 0.5*(_hnn_accel(hnn, q, v, z_ctx) + _lnn_accel(lnn, q, v, z_ctx))
        if (picker == "sum") and (hnn is not None) and (lnn is not None):
            with torch.set_grad_enabled(True):
                return _hnn_accel(hnn, q, v, z_ctx) + _lnn_accel(lnn, q, v, z_ctx)
        if isinstance(picker, tuple) and picker[0] == "blend" and (hnn is not None) and (lnn is not None):
            w = float(picker[1])
            with torch.set_grad_enabled(True):
                return w*_hnn_accel(hnn, q, v, z_ctx) + (1.0-w)*_lnn_accel(lnn, q, v, z_ctx)
        if hnn is not None:
            with torch.set_grad_enabled(True):
                return _hnn_accel(hnn, q, v, z_ctx)
        if lnn is not None:
            with torch.set_grad_enabled(True):
                return _lnn_accel(lnn, q, v, z_ctx)
        return torch.zeros_like(v)

    # Modes
    vjepa.eval(); head.eval()
    if hnn: hnn.eval()
    if lnn: lnn.eval()
    if device is None:
        device = next(head.parameters()).device

    for seq, *_ in tqdm.tqdm(eval_loader, desc="rollout_align_with_goal", leave=False):
        seq = seq.to(device)
        B, T = seq.shape[0], min(seq.shape[1], horizon)

        # Per-frame image→latent predictions
        θ_list, ω_list, Z = [], [], []
        for t in range(T):
            z_t = vjepa.context_encoder(vjepa.patch_embed(seq[:, t]) + vjepa.pos_embed).mean(1)
            θ_t, ω_t = head(z_t).split(1, dim=1)
            θ_list.append(θ_t.squeeze(1))
            ω_list.append(ω_t.squeeze(1))
            Z.append(z_t)
        θ_img = torch.stack(θ_list, dim=1)     # (B,T)
        ω_img = torch.stack(ω_list, dim=1)     # (B,T)
        Z = torch.stack(Z, dim=1)              # (B,T,D)

        # Physics rollout from t=0
        q = θ_img[:, 0].clone()
        v = ω_img[:, 0].clone()
        Θ, Ω, Α = [q], [v], []
        # choose context
        if context_mode == "t0":
            z_ctx0 = Z[:, 0]
        elif context_mode == "perframe":
            z_ctx0 = Z[:, 0]
        else:
            z_ctx0 = None
        a = _accel(q.detach(), v.detach(), z_ctx0)

        for t in range(T-1):
            if integrator == "euler":
                q_next = q + v*dt
                v_next = v + a*dt
                z_ctx = (Z[:, t+1] if context_mode in ("t0","perframe") else None)
                a_next = _accel(q_next.detach(), v_next.detach(), (Z[:, 0] if context_mode=="t0" else z_ctx))
            else:
                q_next = q + v*dt + 0.5*a*(dt**2)
                v_half = v + 0.5*a*dt
                z_ctx = (Z[:, t+1] if context_mode in ("t0","perframe") else None)
                a_next = _accel(q_next.detach(), v_half.detach(), (Z[:, 0] if context_mode=="t0" else z_ctx))
                v_next = v_half + 0.5*a_next*dt

            Θ.append(q_next); Ω.append(v_next); Α.append(a)
            q, v, a = q_next, v_next, a_next

        Α = [torch.zeros_like(Α[0])] + Α
        θ_img_buf.append(θ_img.cpu())
        ω_img_buf.append(ω_img.cpu())
        θ_roll_buf.append(torch.stack(Θ, 1).cpu())
        ω_roll_buf.append(torch.stack(Ω, 1).cpu())
        α_roll_buf.append(torch.stack(Α, 1).cpu())

    θ_img_all = torch.cat(θ_img_buf, 0)
    ω_img_all = torch.cat(ω_img_buf, 0)
    θ_roll_all = torch.cat(θ_roll_buf, 0)
    ω_roll_all = torch.cat(ω_roll_buf, 0)
    α_roll_all = torch.cat(α_roll_buf, 0)
    return θ_img_all, ω_img_all, θ_roll_all, ω_roll_all, α_roll_all