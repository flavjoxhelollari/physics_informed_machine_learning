import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------
#  Metric Helpers for Physics-Informed Machine Learning
# ----------------------------------------------------------------
def neighbour_divergence(θ, ω, ε=0.1, step=-1):
    phase0 = torch.stack([θ[:, 0], ω[:, 0]], 1)
    dist   = torch.cdist(phase0, phase0, p=2)
    mask   = (dist > 0) & (dist < ε)
    i, j   = torch.nonzero(mask, as_tuple=True)
    diff   = torch.stack([θ[i, step] - θ[j, step],
                          ω[i, step] - ω[j, step]], 1)
    return diff.norm(dim=1).mean().item()


def energy_drift(θ, ω, m=1, g=9.8, l=1):
    E0 = 0.5 * m * (l ** 2) * ω[:, 0] ** 2 + m * g * l * (1 - torch.cos(θ[:, 0]))
    Et = 0.5 * m * (l ** 2) * ω[:, -1] ** 2 + m * g * l * (1 - torch.cos(θ[:, -1]))
    return (Et - E0).abs().mean().item()


def accel_mse(θ, alpha_pred, m, g, l):
    alpha_true = -g / l * torch.sin(θ[:, :-1])        # skip last step
    return F.mse_loss(alpha_pred[:, :-1], alpha_true).item()


def el_residual_metric(lnn, θ, ω, dt):
    q = θ[:, :3, None]       # (N,3,1)
    v = ω[:, :3, None]
    return lnn.lagrangian_residual(q.to(device),
                                   v.to(device), dt).item()


def latent_R2(vjepa, head, n_samples=500, eval_loader=None):
    """
    Linear-regression R² for predicting θ from the *latent* vector.
    """
    vjepa.eval(); head.eval()
    Z, θ = [], []
    collected = 0
    for seq, st in eval_loader:
        imgs0 = seq[:, 0].to(device)
        z = vjepa.context_encoder(
                vjepa.patch_embed(imgs0) + vjepa.pos_embed).mean(1)
        Z.append(z.detach().cpu())                      # ← detach()
        θ.extend(st[:, 0, 0].tolist())
        collected += imgs0.size(0)
        if collected >= n_samples:
            break
    Z = torch.cat(Z, 0).numpy()[:n_samples]
    θ = np.array(θ)[:n_samples]
    return r2_score(θ, LinearRegression().fit(Z, θ).predict(Z))
