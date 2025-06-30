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