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

import os
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

# Type-only imports to avoid heavy module loading
from utils.models import VJEPA, HNN, LNN  # your model classes

# Default device selection; override via `map_location` if desired
DEFAULT_DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_ckpt(
    model_path: str,
    theta_path: Optional[str] = None,
    *,
    map_location: Union[str, torch.device] = "cpu"
) -> Tuple[
    Dict[str, torch.Tensor],         # vjepa_sd
    Optional[Dict[str, torch.Tensor]],  # theta_sd
    Optional[Dict[str, torch.Tensor]],  # hnn_sd
    Optional[Dict[str, torch.Tensor]]   # lnn_sd
]:
    """
    Load a checkpoint file and split its contents into four un-prefixed
    state dictionaries: vjepa, theta_head, HNN, and LNN.

    Supports two layouts:
      A) Prefixed combo: keys start with "vjepa.", "theta_head.", "hnn.", "lnn."
      B) Flat VJEPA-only: requires `theta_path` for the head weights.

    Parameters
    ----------
    model_path : str
        Path to the VJEPA checkpoint (combo or flat).
    theta_path : str, optional
        Path to the θ→ω head checkpoint (required for flat layout).
    map_location : str or torch.device
        Device spec passed to `torch.load`.

    Returns
    -------
    vjepa_sd  : Dict[str, torch.Tensor]
        State-dict for the VJEPA backbone.
    theta_sd  : Dict[str, torch.Tensor] or None
        State-dict for the linear head, or None if included in combo.
    hnn_sd    : Dict[str, torch.Tensor] or None
        State-dict for the HNN, or None if not present.
    lnn_sd    : Dict[str, torch.Tensor] or None
        State-dict for the LNN, or None if not present.

    Raises
    ------
    FileNotFoundError
        If `model_path` (or required `theta_path`) does not exist.
    ValueError
        If flat layout detected (no "vjepa." prefixes) but `theta_path` is None.
    """
    # Load the checkpoint into CPU or specified device
    ckpt: Dict[str, torch.Tensor] = torch.load(model_path, map_location=map_location)

    # Detect combo layout by presence of any "vjepa." key
    if any(key.startswith("vjepa.") for key in ckpt):
        # Helper to extract sub-dicts by prefix
        def _strip(prefix: str) -> Dict[str, torch.Tensor]:
            return {
                key[len(prefix):]: val
                for key, val in ckpt.items()
                if key.startswith(prefix)
            }

        vjepa_sd = _strip("vjepa.")
        theta_sd = _strip("theta_head.") or None
        hnn_sd   = _strip("hnn.")        or None
        lnn_sd   = _strip("lnn.")        or None
        return vjepa_sd, theta_sd, hnn_sd, lnn_sd

    # Otherwise assume flat layout: VJEPA only, head separate
    if theta_path is None:
        raise ValueError(
            f"Flat checkpoint detected at '{model_path}' "
            "but no `theta_path` provided."
        )
    if not os.path.exists(theta_path):
        raise FileNotFoundError(f"Head checkpoint not found: '{theta_path}'")

    vjepa_sd = ckpt
    theta_sd: Dict[str, torch.Tensor] = torch.load(theta_path, map_location=map_location)
    return vjepa_sd, theta_sd, None, None


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
    Optional[Dict[str, torch.Tensor]]
]:
    """
    Convenience wrapper: locate the checkpoint files for a given `mode`
    under `base_dir` and split them.

    Expects:
      - `model_<mode><suffix>.pt` (mandatory)
      - `theta_<mode><suffix>.pt` (if flat layout)

    Parameters
    ----------
    mode : str
        Experiment mode, e.g. "plain", "hnn", "lnn", "hnn+lnn".
    suffix : str
        Filename suffix, e.g. "_dense" or "".
    base_dir : str
        Directory containing the checkpoint files.
    map_location : str or torch.device
        Passed to `split_ckpt`.

    Returns
    -------
    Same as `split_ckpt`.
    """
    # Build expected paths
    model_file = os.path.join(base_dir, f"model_{mode}{suffix}.pt")
    head_file  = os.path.join(base_dir, f"theta_{mode}{suffix}.pt")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model checkpoint not found: '{model_file}'")

    # Delegate to split_ckpt
    return split_ckpt(model_file, head_file, map_location=map_location)


def load_net(
    mode: str,
    *,
    suffix: str = "_dense",
    model_dir: str = ".",
    map_location: Union[str, torch.device] = DEFAULT_DEVICE
) -> Tuple[VJEPA, nn.Linear, Optional[HNN], Optional[LNN]]:
    """
    Instantiate and load the full set of networks for a given experiment mode.

    Parameters
    ----------
    mode : str
        One of "plain", "hnn", "lnn", "hnn+lnn".
    suffix : str
        Checkpoint filename suffix.
    model_dir : str
        Directory containing the checkpoint files.
    map_location : str or torch.device
        Device spec for model loading and instantiation.

    Returns
    -------
    vjepa : VJEPA
        The V-JEPA backbone with loaded weights.
    head : nn.Linear
        Linear layer mapping latent → (θ̂, ω̂).
    hnn : HNN or None
        Hamiltonian NN if present in checkpoint.
    lnn : LNN or None
        Lagrangian NN if present in checkpoint.
    """
    # 1) Retrieve raw state dicts
    vjepa_sd, theta_sd, hnn_sd, lnn_sd = load_components(
        mode, suffix=suffix, base_dir=model_dir, map_location=map_location
    )

    # 2) Instantiate and load VJEPA backbone
    vjepa = VJEPA(embed_dim=384, depth=6, num_heads=6).to(map_location)
    vjepa.load_state_dict(vjepa_sd, strict=True)

    # 3) Instantiate and load linear θ→ω head
    head = nn.Linear(384, 2).to(map_location)
    head.load_state_dict(theta_sd, strict=True)  # type: ignore[arg-type]

    # 4) Instantiate optional physics nets
    hnn = None
    if hnn_sd is not None:
        hnn = HNN(hidden_dim=256).to(map_location)
        hnn.load_state_dict(hnn_sd, strict=True)

    lnn = None
    if lnn_sd is not None:
        lnn = LNN(input_dim=2, hidden_dim=256).to(map_location)
        lnn.load_state_dict(lnn_sd, strict=True)

    return vjepa, head, hnn, lnn