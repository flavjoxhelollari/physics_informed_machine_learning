"""
loading_functions.py
────────────────────

High-level checkpoint I/O for the Pendulum-VJEPA research project.

This module provides robust utilities for loading model weights in two
formats without changing your import code:

1. Prefixed "combo" checkpoint:
   - A single file `model_<mode><suffix>.pt` containing keys
     `vjepa.*`, `theta_head.*`, `hnn.*`, `lnn.*`.

2. Flat pair of files:
   - `model_<mode><suffix>.pt` — VJEPA backbone only.
   - `theta_<mode><suffix>.pt` — linear θ→ω head only.

All functions return four *un-prefixed* state dictionaries:

    vjepa_sd, theta_sd, hnn_sd, lnn_sd

which can be fed directly into:

    vjepa.load_state_dict(vjepa_sd)
    head.load_state_dict(theta_sd)
    if hnn_sd: hnn.load_state_dict(hnn_sd)
    if lnn_sd: lnn.load_state_dict(lnn_sd)

Public API
----------
- split_ckpt(model_path, theta_path=None, *, map_location="cpu")
- load_components(mode, *, suffix="_dense", base_dir=".", map_location="cpu")
- load_net(mode, *, suffix="_dense", model_dir=".", map_location="cpu")
"""

# ─────────────────────────────────────────────────────────────────────────────
# loading_functions.py (patched)
# ─────────────────────────────────────────────────────────────────────────────
import os
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from utils.models import VJEPA, HNN, LNN  # your model classes

DEFAULT_DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Small helper: if all keys start with a prefix (e.g. "net."), strip it.
def _strip_prefix_keys(sd: Optional[Dict[str, torch.Tensor]], prefix: str) -> Optional[Dict[str, torch.Tensor]]:
    if sd is None or len(sd) == 0:
        return sd
    keys = list(sd.keys())
    if all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in sd.items()}
    return sd


def split_ckpt(
    model_path: str,
    theta_path: Optional[str] = None,
    *,
    map_location: Union[str, torch.device] = "cpu"
) -> Tuple[
    Dict[str, torch.Tensor],            # vjepa_sd
    Optional[Dict[str, torch.Tensor]],  # theta_sd
    Optional[Dict[str, torch.Tensor]],  # omega_sd
    Optional[Dict[str, torch.Tensor]],  # hnn_sd
    Optional[Dict[str, torch.Tensor]]   # lnn_sd
]:
    """
    Load a checkpoint file and split into un-prefixed state dicts:
    vjepa, theta_head, omega_head, HNN, LNN.

    Supports two layouts:
      A) Prefixed combo in a single file `model_<mode>.pt` with keys:
         "vjepa.*", "theta_head.*", "omega_head.*", "hnn.*", "lnn.*".
      B) Flat VJEPA-only file + separate theta file:
         `model_<mode>.pt` + `theta_<mode>.pt` (legacy; no omega file).

    Returns a 5-tuple: (vjepa_sd, theta_sd, omega_sd, hnn_sd, lnn_sd).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: '{model_path}'")

    ckpt: Dict[str, torch.Tensor] = torch.load(model_path, map_location=map_location)

    # Combo layout: detect presence of "vjepa." keys
    if any(k.startswith("vjepa.") for k in ckpt):
        def _strip(prefix: str) -> Dict[str, torch.Tensor]:
            return {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}

        vjepa_sd  = _strip("vjepa.")
        theta_sd  = _strip("theta_head.") or None
        omega_sd  = _strip("omega_head.") or None
        hnn_sd    = _strip("hnn.")        or None
        lnn_sd    = _strip("lnn.")        or None

        # Handle nested "net.*" saved modules (common with small wrappers)
        theta_sd  = _strip_prefix_keys(theta_sd, "net.")
        omega_sd  = _strip_prefix_keys(omega_sd, "net.")
        return vjepa_sd, theta_sd, omega_sd, hnn_sd, lnn_sd

    # Flat layout (legacy): model file is VJEPA; head file is theta only
    if theta_path is None:
        raise ValueError(
            f"Flat checkpoint detected at '{model_path}' but no `theta_path` provided."
        )
    if not os.path.exists(theta_path):
        raise FileNotFoundError(f"Head checkpoint not found: '{theta_path}'")

    vjepa_sd = ckpt
    theta_sd: Dict[str, torch.Tensor] = torch.load(theta_path, map_location=map_location)
    theta_sd = _strip_prefix_keys(theta_sd, "net.")
    omega_sd = None  # legacy layout has no separate omega file
    return vjepa_sd, theta_sd, omega_sd, None, None


def load_components(
    mode: str,
    *,
    suffix: str = "_dense",
    base_dir: str = ".",
    map_location: Union[str, torch.device] = "cpu"
) -> Tuple[
    Dict[str, torch.Tensor],
    Optional[Dict[str, torch.Tensor]],
    Optional[Dict[str, torch.Tensor]],
    Optional[Dict[str, torch.Tensor]],
    Optional[Dict[str, torch.Tensor]]
]:
    """
    Locate `model_<mode><suffix>.pt` (and optionally `theta_<mode><suffix>.pt`)
    and split into five un-prefixed state dicts:
      vjepa_sd, theta_sd, omega_sd, hnn_sd, lnn_sd
    """
    model_file = os.path.join(base_dir, f"model_{mode}{suffix}.pt")
    head_file  = os.path.join(base_dir, f"theta_{mode}{suffix}.pt")  # legacy single-head fallback

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model checkpoint not found: '{model_file}'")

    return split_ckpt(model_file, head_file, map_location=map_location)


# Simple head shells to match the two-head setup (keeps eval/train code tidy)
class ThetaHead1F(nn.Module):
    def __init__(self, D: int):
        super().__init__()
        self.net = nn.Linear(D, 1)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class OmegaHead2F(nn.Module):
    def __init__(self, D: int):
        super().__init__()
        self.net = nn.Linear(2*D, 1)
    def forward(self, z_t: torch.Tensor, z_tp1: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, z_tp1], dim=1))


def load_net(
    mode: str,
    *,
    suffix: str = "_dense",
    model_dir: str = ".",
    map_location: Union[str, torch.device] = DEFAULT_DEVICE,
    embed_dim: int = 384,
    h_hidden: int = 256,
    l_hidden: int = 256,
) -> Tuple[VJEPA, ThetaHead1F, OmegaHead2F, Optional[HNN], Optional[LNN]]:
    """
    Instantiate VJEPA + (theta_head, omega_head) + optional HNN/LNN and load weights.

    Returns
    -------
    vjepa, theta_head, omega_head, hnn, lnn
    """
    vjepa_sd, theta_sd, omega_sd, hnn_sd, lnn_sd = load_components(
        mode, suffix=suffix, base_dir=model_dir, map_location=map_location
    )

    # VJEPA
    vjepa = VJEPA(embed_dim=embed_dim, depth=6, num_heads=6).to(map_location)
    vjepa.load_state_dict(vjepa_sd, strict=True)

    # Heads
    theta_head = ThetaHead1F(embed_dim).to(map_location)
    if theta_sd is not None:
        theta_head.load_state_dict(theta_sd, strict=True)
    else:
        print(f"[{mode}] warning: no theta_head weights found; using random init")

    omega_head = OmegaHead2F(embed_dim).to(map_location)
    if omega_sd is not None:
        omega_head.load_state_dict(omega_sd, strict=True)
    else:
        print(f"[{mode}] warning: no omega_head weights found; using random init")

    # Physics nets (optional)
    hnn = None
    if hnn_sd is not None:
        hnn = HNN(hidden_dim=h_hidden).to(map_location)
        hnn.load_state_dict(hnn_sd, strict=True)

    lnn = None
    if lnn_sd is not None:
        lnn = LNN(input_dim=2, hidden_dim=l_hidden).to(map_location)
        lnn.load_state_dict(lnn_sd, strict=True)

    return vjepa, theta_head, omega_head, hnn, lnn