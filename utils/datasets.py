"""
utils/pendulum_dataset.py
=========================

A **fully-featured, self-documenting** PyTorch ``Dataset`` for the classic
*Pendulum-v1* environment, designed for vision-based physics research.

Why would you use this instead of the raw `gymnasium.make("Pendulum-v1")`
loop?

* It **renders** every state into a *clean, white-background RGB image*
  (no mujoco viewer needed), so the exact pixel input is deterministic.
* Three research-oriented **knobs** are exposed:

  ┌───────────────────────────────────────────────────────────────────────┐
  │ ``sub_steps``     – densify dynamics: env is stepped `k` times for   │
  │                     every stored frame ⇒ temporal resolution ↑       │
  │ ``init_grid``     – supply a list of `(θ₀, ω₀)` pairs to *override*  │
  │                     random starts and obtain a deterministic grid    │
  │ ``random_action`` – toggle the usual random torque √ / conservative  │
  │                     swing-up ✗                                        │
  └───────────────────────────────────────────────────────────────────────┘

* **Windowed access**:  ``__getitem__`` returns a *sequence*
  of length `seq_len` *(T,C,H,W)* **and** its $(θ,ω)$ labels *(T,2)*,
  which downstream LNN / HNN losses require.

The defaults (``sub_steps=1``, ``init_grid=None``, ``random_action=True``)
reproduce your *original* dataset byte-for-byte, so existing notebooks
keep working.

---------------------------------------------------------------------------
Public API
---------------------------------------------------------------------------

.. autosummary::

   PendulumDataset
"""

from __future__ import annotations

import os
import math
from typing import List, Tuple, Optional

import gym             # ⟵ gymnasium ≥ 0.29 works too
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image, ImageDraw

# Torch tensor device used for *all* outputs (override via env var if   )
# ----------------------------------------------------------------------
device = torch.device(os.getenv("PND_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))


# ═══════════════════════════════════════════════════════════════════════
# PendulumDataset
# ═══════════════════════════════════════════════════════════════════════
class PendulumDataset(Dataset):
    """
    Rendered pendulum roll-outs, returned **as image *sequences***.

    Parameters
    ----------
    num_episodes : int, default=100
        Number of *random* episodes generated **if** `init_grid` is
        *not* supplied.
    episode_length : int, default=200
        Length (in stored frames) of **each** episode.
    img_size : int, default=64
        Square side length of the rendered RGB frames (pixels).
    seq_len : int, default=3
        Length **T** of each training sample window
        *(T,C,H,W) + (T,2)*.
    sub_steps : int, default=1
        If > 1 the Gym env is advanced ``sub_steps`` times per stored
        frame, increasing temporal resolution.
    init_grid : list[(float, float)] | None
        If supplied, **each** tuple *(θ₀, ω₀)* becomes *one* episode
        and `num_episodes` is ignored.  Useful for deterministic test
        grids.
    random_action : bool, default=True
        • True ⇒ env torque is sampled every step (original behaviour)  
        • False ⇒ action = 0 → conservative swing-up only.
    transform : callable | None
        Optional torchvision-style transform applied to the *tensor*
        sequence (after normalisation to 0-1).

    Notes
    -----
    *  ``seq_len`` must be ≥ 2 otherwise Euler-Lagrange residuals cannot
       be computed downstream.
    *  The underlying *Gym* environment is **re-used** across episodes
       to avoid slow `close()` / `make()` pairs.
    """

    # ----------------------------- constructor -------------------------
    def __init__(
        self,
        num_episodes : int = 100,
        episode_length: int = 200,
        img_size      : int = 64,
        seq_len       : int = 3,
        *,
        sub_steps     : int = 1,
        init_grid     : Optional[List[Tuple[float, float]]] = None,
        random_action : bool = True,
        transform     = None,
    ) -> None:
        super().__init__()

        # ------------ basic argument checks ---------------------------
        if seq_len < 2:
            raise ValueError("seq_len must be ≥ 2 for physics losses")
        if sub_steps < 1:
            raise ValueError("sub_steps must be ≥ 1")

        # ------------ store public attributes -------------------------
        self.img_size      = img_size
        self.seq_len       = seq_len
        self.sub_steps     = sub_steps
        self.init_grid     = init_grid
        self.random_action = random_action
        self.transform     = transform

        # ------------ internal buffers (filled by _generate) ----------
        self.frames  : List[np.ndarray]        = []   # flat list of H×W×3 uint8
        self.states  : List[Tuple[float,float]]= []   # flat list of (θ, ω)
        self.indices : List[int]               = []   # window start → frame idx

        # ------------ kick off synthetic roll-outs --------------------
        self._generate(num_episodes, episode_length)

    # ==================================================================
    #  Private helpers
    # ==================================================================
    def _render_pendulum(self, theta: float) -> np.ndarray:
        """
        Render a single RGB frame given `theta` (rad).

        The arm is drawn with antialiased lines; the blue bob radius
        scales with image size for readability in small resolutions.
        """
        L      = self.img_size * 0.4                  # physical arm length in px
        c      = self.img_size // 2                   # centre pixel
        end_x  = int(c + L * math.sin(theta))         # bob centre (x,y)
        end_y  = int(c + L * math.cos(theta))

        img  = Image.new("RGB", (self.img_size, self.img_size), "white")
        draw = ImageDraw.Draw(img)

        draw.line([(c, c), (end_x, end_y)], fill="black", width=3)
        draw.ellipse([(c-5, c-5), (c+5, c+5)], fill="red")         # pivot
        draw.ellipse([(end_x-8, end_y-8), (end_x+8, end_y+8)],
                     fill="blue")                                  # bob
        return np.asarray(img)                                     # (H,W,3) uint8

    # ------------------------------------------------------------------
    def _generate(self, n_episodes: int, epi_len: int) -> None:
        """
        Populate ``self.frames / self.states / self.indices`` in place.
        """
        print("Generating pendulum trajectories …")
        env = gym.make("Pendulum-v1")                               # mujoco-free env

        # choose seeds --------------------------------------------------
        episode_seeds = (
            self.init_grid
            if self.init_grid is not None
            else [None] * n_episodes
        )

        for seed in tqdm(episode_seeds):
            # ---------- initialise episode --------------------------
            if seed is None:
                obs, _ = env.reset()
            else:                                                    # deterministic
                theta0, omega0 = seed
                env.reset()
                env.unwrapped.state = np.array([theta0, omega0], dtype=np.float32)
                obs = np.array([np.cos(theta0), np.sin(theta0), omega0],
                               dtype=np.float32)

            ep_imgs, ep_states = [], []

            # ---------- main roll-out loop ---------------------------
            for _ in range(epi_len):

                # --- densify time: inner loop --------------------
                for _ in range(self.sub_steps):
                    action = (env.action_space.sample()
                              if self.random_action
                              else np.array([0.0], dtype=np.float32))
                    obs, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:                      # rare in v1
                        break

                # convert observation to (θ, ω)
                theta = float(np.arctan2(obs[1], obs[0]))
                omega = float(obs[2])

                ep_imgs  .append(self._render_pendulum(theta))
                ep_states.append((theta, omega))

            # ---------- window indexing ------------------------------
            for t0 in range(0, len(ep_imgs) - self.seq_len + 1):
                self.indices.append(len(self.frames) + t0)

            self.frames.extend(ep_imgs)
            self.states.extend(ep_states)

        env.close()
        print(f"Created {len(self.indices)} windows (seq_len={self.seq_len})")

    # ==================================================================
    #  PyTorch Dataset protocol
    # ==================================================================
    def __len__(self) -> int:                                   # number of windows
        return len(self.indices)

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        imgs : torch.FloatTensor
            Shape (T, 3, H, W), normalised to 0-1.
        states : torch.FloatTensor
            Shape (T, 2) with columns (θ, ω).
        """
        start = self.indices[idx]
        end   = start + self.seq_len

        # ----------- stack and normalise images -----------------------
        imgs = [
            torch.from_numpy(self.frames[i]).float()
                 .permute(2, 0, 1) / 255.0                      # (3,H,W)
            for i in range(start, end)
        ]
        imgs = torch.stack(imgs)                                # (T,3,H,W)

        # ----------- stack states ------------------------------------
        states = torch.tensor(self.states[start:end],
                              dtype=torch.float32)              # (T,2)

        # ----------- optional torchvision transform ------------------
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, states

import math
from typing import Dict, Optional, Union
import numpy as np
import torch
from torch.utils.data import DataLoader

@torch.no_grad()
def compute_theta_omega_stats(
    data: Union[DataLoader, torch.utils.data.Dataset],
    *,
    batch_size: int = 512,         # used if `data` is a Dataset
    use_all_timesteps: bool = True, # True = use all frames in the window; False = only t=0
    max_samples: Optional[int] = None, # cap total frames considered (for speed)
    device: Optional[torch.device] = None,
    theta_wrap: bool = True         # if True, also report circular mean/std for θ
) -> Dict[str, Dict[str, float]]:
    """
    Scans the dataset/loader and returns summary stats for θ and ω.

    Returns a dict with keys 'theta' and 'omega', each containing:
      - count, mean, std, min, max, p01, p50, p99
      - (for theta) circ_mean, circ_std if theta_wrap=True

    Assumptions:
      dataset __getitem__ returns (seq, states) with states shape (T, 2)
      where states[...,0] = theta, states[...,1] = omega.
    """
    # Get a DataLoader either way
    if isinstance(data, DataLoader):
        loader = data
    else:
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    theta_vals = []
    omega_vals = []
    seen = 0

    for batch in loader:
        # supports datasets that return (seq, states) or (seq, states, ...)
        seq, states = batch[0], batch[1]     # states: (B, T, 2)
        states = states if device is None else states.to(device)

        if use_all_timesteps:
            θ = states[..., 0].reshape(-1)   # (B*T,)
            ω = states[..., 1].reshape(-1)
        else:
            θ = states[:, 0, 0].reshape(-1)  # (B,)
            ω = states[:, 0, 1].reshape(-1)

        theta_vals.append(θ.cpu())
        omega_vals.append(ω.cpu())

        seen += θ.numel()
        if (max_samples is not None) and (seen >= max_samples):
            break

    if not theta_vals:
        raise ValueError("No data found to compute stats.")

    θ_all = torch.cat(theta_vals)
    ω_all = torch.cat(omega_vals)

    if max_samples is not None and θ_all.numel() > max_samples:
        θ_all = θ_all[:max_samples]
        ω_all = ω_all[:max_samples]

    def _tensor_stats(x: torch.Tensor) -> Dict[str, float]:
        x_np = x.numpy()
        return dict(
            count       = float(x_np.size),
            mean        = float(np.mean(x_np)),
            std         = float(np.std(x_np, ddof=0)),
            min         = float(np.min(x_np)),
            p01         = float(np.percentile(x_np, 1)),
            p50         = float(np.percentile(x_np, 50)),
            p99         = float(np.percentile(x_np, 99)),
            max         = float(np.max(x_np)),
        )

    def _circular_stats(x: torch.Tensor) -> Dict[str, float]:
        # x in radians; computes circular mean and std
        x_np = x.numpy()
        s, c = np.sin(x_np), np.cos(x_np)
        mean_ang = math.atan2(np.mean(s), np.mean(c))  # in [-pi, pi]
        R = np.hypot(np.mean(c), np.mean(s))           # mean resultant length
        # circular std: sqrt(-2 ln R)  (Fisher 1993)
        circ_std = float(np.sqrt(max(0.0, -2.0 * np.log(max(R, 1e-12)))))
        return dict(circ_mean=float(mean_ang), circ_std=circ_std, R=float(R))

    theta_stats = _tensor_stats(θ_all)
    omega_stats = _tensor_stats(ω_all)

    if theta_wrap:
        theta_stats.update(_circular_stats(θ_all))

    return {"theta": theta_stats, "omega": omega_stats}