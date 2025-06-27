import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------
# Splitter that works for flat  **and**  prefixed checkpoints
# ----------------------------------------------------------------
def split_ckpt(path, map_location="cpu"):
    """
    Returns  vjepa_sd , theta_sd , hnn_sd , lnn_sd  (some may be None)
    """
    ckpt = torch.load(path, map_location=map_location)

    # prefixed ⇒ strip prefixes ----------------------------------
    if any(k.startswith("vjepa.") for k in ckpt):
        vjepa_sd = {k[6:]  : v for k,v in ckpt.items() if k.startswith("vjepa.")}
        theta_sd = {k[11:]: v for k,v in ckpt.items() if k.startswith("theta_head.")}
        hnn_sd   = {k[4:]  : v for k,v in ckpt.items() if k.startswith("hnn.")} or None
        lnn_sd   = {k[4:]  : v for k,v in ckpt.items() if k.startswith("lnn.")} or None
        return vjepa_sd, theta_sd, hnn_sd, lnn_sd

    # flat ⇒ everything belongs to V-JEPA ------------------------
    return ckpt, None, None, None


def load_components(mode, suffix):
    """load V-JEPA, theta-head and (optionally) HNN / LNN states"""
    vjepa_sd, theta_sd, hnn_sd, lnn_sd = split_ckpt(f"model_{mode}{suffix}.pt")

    # theta head saved separately in the current pipeline
    if theta_sd is None:
        theta_sd = torch.load(f"theta_{mode}{suffix}.pt", map_location="cpu")

    return vjepa_sd, theta_sd, hnn_sd, lnn_sd