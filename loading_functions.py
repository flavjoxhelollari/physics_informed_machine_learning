import torch
import os

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