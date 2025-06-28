import torch
import os
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# 1 · Universal splitter
# ─────────────────────────────────────────────────────────────
def split_ckpt(model_path, theta_path=None, *, map_location="cpu"):
    """
    Return four state-dicts:
        vjepa_sd   (dict)
        theta_sd   (dict  or  None)
        hnn_sd     (dict  or  None)
        lnn_sd     (dict  or  None)
    Works for *prefixed* as well as *flat* checkpoints.
    """

    ckpt = torch.load(model_path, map_location=map_location)

    # ---------- Case A: prefixed combined dict -----------------
    if any(k.startswith("vjepa.") for k in ckpt):
        strip = lambda p, d: {k[len(p):]: v for k, v in d.items()
                              if k.startswith(p)}
        vjepa_sd = strip("vjepa.",       ckpt)
        theta_sd = strip("theta_head.",  ckpt) or None
        hnn_sd   = strip("hnn.",         ckpt) or None
        lnn_sd   = strip("lnn.",         ckpt) or None
        print("Found prefixed checkpoint:", model_path)
        return vjepa_sd, theta_sd, hnn_sd, lnn_sd

    # ---------- Case B: flat dict ------------------------------
    print("Loading flat checkpoint:", model_path)
    vjepa_sd, hnn_sd, lnn_sd = ckpt, None, None

    if theta_path is None:
        raise ValueError("theta_path required for flat checkpoints")

    if not os.path.exists(theta_path):
        raise FileNotFoundError(theta_path)

    theta_sd = torch.load(theta_path, map_location=map_location)
    return vjepa_sd, theta_sd, hnn_sd, lnn_sd


# ─────────────────────────────────────────────────────────────
# 2 · Convenience loader
# ─────────────────────────────────────────────────────────────
def load_components(mode, *, suffix="_dense", base_dir=".", map_location="cpu"):
    """
    Wrapper that calls `split_ckpt` with the correct paths.
    """
    model_path = os.path.join(base_dir, f"model_{mode}{suffix}.pt")
    theta_path = os.path.join(base_dir, f"theta_{mode}{suffix}.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    return split_ckpt(model_path, theta_path, map_location=map_location)

# -----------------------------------------------------------
# Revised load_net – now directory-aware
# -----------------------------------------------------------
def load_net(mode: str,
             *,                       # keyword-only args
             suffix    = "_dense",
             model_dir = ".",         # ← new
             map_location = "cpu"):

    # 1) grab four state-dicts via the robust helper
    v_sd, t_sd, hnn_sd, lnn_sd = load_components(
        mode,
        suffix       = suffix,
        base_dir     = model_dir,
        map_location = map_location
    )

    # 2) instantiate modules
    vjepa = VJEPA(embed_dim=384, depth=6, num_heads=6).to(device)
    head  = torch.nn.Linear(384, 2).to(device)
    vjepa.load_state_dict(v_sd, strict=True)
    head.load_state_dict(t_sd, strict=True)

    hnn = lnn = None
    if hnn_sd:
        hnn = HNN(hidden_dim=256).to(device)
        hnn.load_state_dict(hnn_sd, strict=True)
    if lnn_sd:
        lnn = LNN(input_dim=2, hidden_dim=256).to(device)
        lnn.load_state_dict(lnn_sd, strict=True)

    return vjepa, head, hnn, lnn