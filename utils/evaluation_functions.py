"""
evaluation_functions.py
───────────────────────
Inference-side utilities for the Pendulum-VJEPA study.

This module is intentionally **stand-alone**: the only external
dependencies are `models.py`, `loading_functions.py`,
`evaluation_metrics.py` and `helper_functions.py`.

Public API
----------
EvalConfig            – central container for *all* evaluation knobs  
evaluate_mode         – run a single checkpoint, compute metrics, dump JSON  
evaluate_all_modes    – loop `evaluate_mode` over a list of modes  
collect_head_scatter  – gather (θ_true, ω_true, θ̂, ω̂) arrays  
head_regression_metrics – tiny helper for (R², MSE) of the latent head  
normalise_keys        – legacy →   loss_*  key renamer  
plot_loss             – quick multi-run loss visualiser
"""

from __future__ import annotations  # ← forward-ref typing (Py < 3.11)

import os, json                                   # file I/O helpers
from dataclasses import dataclass, asdict         # lightweight settings
from typing      import Dict, Iterable, List      # static typing

import numpy as np                                # numeric arrays
import torch                                      # tensors & autograd
import matplotlib.pyplot as plt                   # plotting
from torch.utils.data import DataLoader           # batching
from tqdm       import tqdm                  # nice progress-bars
from sklearn.metrics import r2_score, mean_squared_error  # regression
from sklearn.linear_model import LinearRegression          # linear-fit

# --------------------------------------------------------------------
# Project-local imports (paths must be on PYTHONPATH when importing)
# --------------------------------------------------------------------
from utils.models              import *                # NN classes
from utils.loading_functions   import load_components                # ckpt splitter
from utils.evaluation_metrics  import *
from utils.helper_functions    import *                        # latent rollout

# CUDA if available, otherwise CPU
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====================================================================
# 1 · Dataclass holding every evaluation hyper-parameter
# ====================================================================
@dataclass
class EvalConfig:
    """All knobs for a single evaluation run (defaults = your current)."""
    suffix           : str   = "_dense"      # checkpoint suffix
    model_dir        : str   = "./models"    # where model_*.pt live
    out_dir          : str   = "./metrics"   # where metrics_*.json are written

    # roll-out parameters (used by `rollout`)
    horizon          : int   = 60            # time-steps (3 s if dt=0.05)
    dt               : float = 0.05          # simulation Δt

    # physical constants for energy / accel ground-truth
    m                : float = 1.0
    g                : float = 9.81
    l                : float = 1.0

    # latent-head scatter settings
    scatter_samples  : int   = 500
    scatter_batch    : int   = 64

    #batch-size
    batch_size       : int   = 64

    # --- NEW: neighbour divergence options ---
    # mode: "pairwise" (original O(N^2)), "grid" (adjacent ICs), "window" (Δθ/Δω thresholds), "knn"
    ndiv_mode: str = "grid"          # good default for gridded eval
    ndiv_stencil: str = "8"          # used when ndiv_mode="grid" ("4" or "8")
    ndiv_k: int = 8                  # used when ndiv_mode="knn"
    ndiv_dtheta: float | None = None # used when ndiv_mode="window"
    ndiv_domega: float | None = None # used when ndiv_mode="window"
    ndiv_eps: float = 0.1

    # NEW: how to reduce across time for the scalar Δ_div
    # "step" → use ndiv_step index; "mean" → average over all T
    ndiv_reduce: Literal["step","mean"] = "step"
    ndiv_step: int = -1  # used only when ndiv_reduce == "step"

    # NEW: target normalization stats (same used in training)
    theta_mu         : float = 0.0
    theta_std        : float = 1.0
    omega_mu         : float = 0.0
    omega_std        : float = 1.0


# ====================================================================
# 2 · Head scatter collector
# ====================================================================
def collect_head_scatter(
    model: VJEPA,
    theta_head: torch.nn.Module,
    omega_head: torch.nn.Module,
    dataset,
    *,
    n_samples: int,
    batch:     int,
    dt: float,
    theta_mu: float, theta_std: float,
    omega_mu: float, omega_std: float,
) -> tuple[np.ndarray, ...]:
    """
    Collect (θ_true, ω_true) and predictions at the earliest available times:
      θ at t=0 (single-frame), ω at t=1 using (z1, z0).
    Returns arrays of shape (n_samples,).
    """
    model.eval(); theta_head.eval()
    if hasattr(omega_head, "eval"): omega_head.eval()

    loader = DataLoader(dataset, batch_size=batch, shuffle=True)
    θ_true, ω_true, θ_pred, ω_pred = [], [], [], []
    collected = 0

    for seq, states in loader:
        seq = seq.to(device)                         # (B,T,C,H,W)
        B, T, C, H, W = seq.shape
        imgs = seq.reshape(B*T, C, H, W)
        with torch.no_grad():
            z_flat = model.context_encoder(model.patch_embed(imgs) + model.pos_embed).mean(1)
            z = z_flat.reshape(B, T, -1)            # (B,T,D)
            # θ at t=0
            θ_pn = theta_head(z[:,0,:])             # (B,1) normalized
            θ_hat = (θ_pn * theta_std + theta_mu).squeeze(-1)  # (B,)
            # ω at t=1 using (z1, z0) if available
            if T >= 2:
                ω_pn = omega_head(z[:,1,:], z[:,0,:], dt)      # (B,1) normalized
                ω_hat = (ω_pn * omega_std + omega_mu).squeeze(-1)
            else:
                ω_hat = torch.zeros(B, device=seq.device)

        θ_true.extend(states[:, 0, 0].cpu().numpy())
        ω_true.extend(states[:, 1, 1].cpu().numpy() if T>=2 else np.zeros(B))
        θ_pred.extend(θ_hat.detach().cpu().numpy())
        ω_pred.extend(ω_hat.detach().cpu().numpy())

        collected += B
        if collected >= n_samples:
            break

    return (np.array(θ_true)[:n_samples],
            np.array(ω_true)[:n_samples],
            np.array(θ_pred)[:n_samples],
            np.array(ω_pred)[:n_samples])


# ====================================================================
# 3 · Tiny regression helper (latent-head)
# ====================================================================
def head_regression_metrics(
    θ_t: np.ndarray, ω_t: np.ndarray,
    θ_p: np.ndarray, ω_p: np.ndarray
) -> Dict[str, float]:
    """Return R² and MSE for θ and ω."""
    return dict(
        r2_theta  = float(r2_score(θ_t, θ_p)),
        mse_theta = float(mean_squared_error(θ_t, θ_p)),
        r2_omega  = float(r2_score(ω_t, ω_p)),
        mse_omega = float(mean_squared_error(ω_t, ω_p)),
    )


# ====================================================================
# 4 · Evaluate ONE mode  ––  **now aware of `combine`**
# ====================================================================
def evaluate_mode(
    mode       : str,
    dataset,
    cfg        : EvalConfig,
    *,
    combine           : str | tuple[str, float] | None = None,
    plot_head_scatter : bool = False,
    save_curves       : bool = False,
) -> Dict[str, float]:

    # 1) -------- load state-dicts -----------------------------------
    v_sd, t_sd, hnn_sd, lnn_sd = load_components(
        mode, suffix=cfg.suffix, base_dir=cfg.model_dir)

    # 2) -------- rebuild modules & weights --------------------------
    model = VJEPA(embed_dim=384, depth=6, num_heads=6).to(device)
    model.load_state_dict(v_sd, strict=True)
    
    theta_head = ThetaHead(384).to(device)
    omega_head = OmegaHead(384).to(device)
    
    def _looks_like_linear2(sd: dict) -> bool:
        return ("weight" in sd) and ("bias" in sd) and sd["weight"].dim() == 2 and sd["weight"].size(0) == 2
    
    def _strip_prefix(sd: dict, prefix: str) -> dict:
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    
    loaded_variant = None
    try:
        # Case A: new two-head format with prefixes
        theta_sd = _strip_prefix(t_sd, "theta_head.")
        omega_sd = _strip_prefix(t_sd, "omega_head.")
        if theta_sd and omega_sd:
            theta_head.load_state_dict(theta_sd, strict=True)
            omega_head.load_state_dict(omega_sd, strict=True)
            loaded_variant = "two-head (namespaced)"
        else:
            raise KeyError("namespaced heads not both present")
    except Exception:
        # Case B: legacy single Linear(384->2)
        if _looks_like_linear2(t_sd):
            legacy = torch.nn.Linear(384, 2).to(device)
            legacy.load_state_dict(t_sd, strict=True)
    
            class _LegacyWrap(torch.nn.Module):
                def __init__(self, lin): super().__init__(); self.lin = lin
                def theta(self, z):
                    y = self.lin(z); return y[..., :1]
                def omega(self, z):
                    y = self.lin(z); return y[..., 1:]
            L = _LegacyWrap(legacy)
    
            class ThetaFromLegacy(torch.nn.Module):
                def __init__(self, L): super().__init__(); self.L = L
                def forward(self, z):
                    if z.dim()==3: B,T,D = z.shape; return self.L.theta(z.reshape(B*T,D)).reshape(B,T,1)
                    return self.L.theta(z).unsqueeze(-1)
            class OmegaFromLegacy(torch.nn.Module):
                def __init__(self, L): super().__init__(); self.L = L
                def forward(self, z_t, z_tm1, dt):
                    # no two-frame info in legacy; use z_t only
                    if z_t.dim()==3: B,Tp,D = z_t.shape; return self.L.omega(z_t.reshape(B*Tp,D)).reshape(B,Tp,1)
                    return self.L.omega(z_t).unsqueeze(-1)
    
            theta_head = ThetaFromLegacy(L).to(device)
            omega_head = OmegaFromLegacy(L).to(device)
            loaded_variant = "legacy linear(2) split"
        else:
            # Case C: single MLP head state-dict (e.g., ThetaHead only)
            has_mlp_keys = any(k.startswith("net.") for k in t_sd.keys())
            if has_mlp_keys:
                try:
                    theta_head.load_state_dict(t_sd, strict=True)
                    # omega_head stays randomly init (no weights present)
                    loaded_variant = "single MLP head (theta only)"
                except Exception as err:
                    raise RuntimeError(f"Unrecognized head state-dict format (single MLP) – {err}") from err
            else:
                # Final fallback: maybe t_sd held a namespaced *module* dict under an unexpected root
                # Try to find any subprefix automatically.
                prefixes = set(k.split(".", 1)[0] for k in t_sd.keys() if "." in k)
                matched = False
                for pref in prefixes:
                    sub = _strip_prefix(t_sd, pref + ".")
                    try:
                        theta_head.load_state_dict(sub, strict=True)
                        loaded_variant = f"heuristic: loaded theta_head from '{pref}.'"
                        matched = True
                        break
                    except Exception:
                        continue
                if not matched:
                    raise RuntimeError(
                        "Could not parse head checkpoint. Supported: "
                        "[theta_head.*, omega_head.*] or Linear(384->2) or single MLP (net.*). "
                        f"Keys seen: {list(t_sd.keys())[:6]} ..."
                    )
    
    # physics heads (optional)
    hnn = lnn = None
    if hnn_sd:
        hnn = HNN(hidden_dim=256).to(device); hnn.load_state_dict(hnn_sd, strict=True)
    if lnn_sd:
        lnn = LNN(input_dim=2, hidden_dim=256).to(device); lnn.load_state_dict(lnn_sd, strict=True)

    # 3) -------- deterministic eval loader --------------------------
    eval_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    # 4) -------- per-frame preds and rollout from t=0 ---------------
    θ_img, ω_img, θ_roll, ω_roll, α_roll = rollout_align_with_goal(
        vjepa=model,
        theta_head=theta_head,
        omega_head=omega_head,
        eval_loader=eval_loader,
        horizon=cfg.horizon,
        dt=cfg.dt,
        hnn=hnn, lnn=lnn, combine=combine,
        integrator="verlet",         # better long-term stability
        context_mode="t0",
        theta_mu=cfg.theta_mu, theta_std=cfg.theta_std,
        omega_mu=cfg.omega_mu, omega_std=cfg.omega_std,
    )

    # 5) -------- gather ground-truth aligned with loader order ------
    θ_true_list, ω_true_list = [], []
    for _, states in eval_loader:
        θ_true_list.append(states[:, :θ_img.shape[1], 0])
        ω_true_list.append(states[:, :ω_img.shape[1], 1])
    θ_true = torch.cat(θ_true_list, 0).cpu()
    ω_true = torch.cat(ω_true_list, 0).cpu()

    # 6) -------- base physics metrics (use rollout) -----------------
    metrics: Dict[str, float] = {
        "Δ_div"  : neighbour_divergence(θ_roll, ω_roll),
        "E_drift": energy_drift(θ_roll, ω_roll, m=cfg.m, g=cfg.g, l=cfg.l),
    }
    if lnn is not None:
        metrics["EL_res"] = el_residual_metric(lnn, θ_roll, ω_roll, dt=cfg.dt)

    Δ_curve, Δ_rate = neighbour_divergence_curve(θ_roll, ω_roll, return_curve=True)
    E_curve, E_rate = energy_drift_curve(θ_roll, ω_roll, m=cfg.m, g=cfg.g, l=cfg.l, return_curve=True)
    metrics.update(Δ_rate=Δ_rate, E_rate=E_rate)

    if lnn is not None:
        EL_curve, EL_rate = el_residual_curve(lnn, θ_roll, ω_roll, dt=cfg.dt, return_curve=True)
        metrics["EL_rate"] = EL_rate

    if save_curves:
        metrics["Δ_curve"]  = Δ_curve.tolist()
        metrics["E_curve"]  = E_curve.tolist()
        if lnn is not None:
            metrics["EL_curve"] = EL_curve.tolist()

    # 7) -------- per-frame image→latent metrics (central claim) ----
    metrics.update(perframe_img_metrics(θ_true, ω_true, θ_img, ω_img))

    # 8) -------- rollout-vs-truth MSE (for completeness) -----------
    metrics.update(rollout_mse(θ_true, ω_true, θ_roll, ω_roll))

    # 9) -------- one-step self-consistency --------------------------
    metrics.update(one_step_consistency(
        θ_img, ω_img, dt=cfg.dt, hnn=hnn, lnn=lnn, combine=(combine or "hnn"),
        integrator="verlet"
    ))

    # 10) ------- acceleration MSE vs analytic (uses rollout α) -----
    metrics["accel_mse_true"] = accel_mse(θ_true, α_roll, g=cfg.g, l=cfg.l)

    # 11) ------- head-scatter diagnostic (frame-0 / two-head aware) -
    try:
        θ_t, ω_t, θ_p, ω_p = collect_head_scatter(
            model, theta_head, omega_head, dataset,
            n_samples=cfg.scatter_samples,
            batch    =cfg.scatter_batch,
            dt       =cfg.dt,
            theta_mu =cfg.theta_mu, theta_std=cfg.theta_std,
            omega_mu =cfg.omega_mu, omega_std=cfg.omega_std
        )
    except TypeError:
        # Back-compat: fall back to legacy collector that expects a single head
        single_head = torch.nn.Linear(384, 2).to(device)
        # try to map from t_sd into a plain Linear (best-effort)
        try:
            single_sd = {k[len("theta_head."):]: v for k, v in t_sd.items()}
            single_head.load_state_dict(single_sd, strict=False)
        except Exception:
            try:
                single_head.load_state_dict(t_sd, strict=False)
            except Exception:
                pass
        θ_t, ω_t, θ_p, ω_p = collect_head_scatter(
            model, single_head, dataset,
            n_samples=cfg.scatter_samples,
            batch    =cfg.scatter_batch
        )
    metrics.update(head_regression_metrics(θ_t, ω_t, θ_p, ω_p))

    # 12) ------- optional scatter plot ------------------------------
    if plot_head_scatter:
        plt.figure(figsize=(10, 5))
        # θ
        plt.subplot(1, 2, 1)
        plt.scatter(θ_t, θ_p, s=8, alpha=.6)
        plt.xlabel("true θ"); plt.ylabel("pred θ")
        plt.grid(True); plt.xlim(-4, 4); plt.ylim(-4, 4)
        plt.gca().set_aspect("equal", adjustable="box")
        # ω
        plt.subplot(1, 2, 2)
        plt.scatter(ω_t, ω_p, s=8, alpha=.6)
        plt.xlabel("true ω"); plt.ylabel("pred ω")
        plt.grid(True); plt.xlim(-10, 10); plt.ylim(-10, 10)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.suptitle(f"Head predictions vs truth – {mode}")
        plt.tight_layout(); plt.show()

    # 13) ------- write JSON ----------------------------------------
    os.makedirs(cfg.out_dir, exist_ok=True)
    out_path = os.path.join(cfg.out_dir, f"metrics_{mode}{cfg.suffix}.json")
    with open(out_path, "w") as fp:
        json.dump(metrics, fp, indent=2)

    print(f"{mode:8} →", {k: round(v, 4) if isinstance(v, float) else '…' for k, v in metrics.items()},
          f"(saved → {out_path})")
    return metrics
    
# def evaluate_mode(
#     mode       : str,
#     dataset,
#     cfg        : EvalConfig,
#     *,
#     combine           : str | tuple[str, float] | None = None,
#     plot_head_scatter : bool = False,
#     save_curves       : bool = False,          # ← NEW FLAG
# ) -> Dict[str, float]:
#     """
#     Inference + physics metrics for one checkpoint.

#     Parameters
#     ----------
#     …
#     save_curves
#         If **True**, also dump the full per-step curves
#         ``Δ_curve``, ``E_curve`` and (when LNN present) ``EL_curve`` to
#         the output JSON.  Beware: curves are length ≈ `cfg.horizon` and
#         stored as full-precision floats, so the file grows linearly with
#         horizon.

#     Returns
#     -------
#     Dict[str, float]
#         Keys:
#             Δ_div, E_drift, EL_res (if LNN),
#             Δ_rate, E_rate, EL_rate (if LNN),
#             (optionally the three *_curve lists)
#             + latent-head regression stats.
#     """
#     # 1) -------- load state-dicts -----------------------------------
#     v_sd, t_sd, hnn_sd, lnn_sd = load_components(
#         mode, suffix=cfg.suffix, base_dir=cfg.model_dir)

#     # 2) -------- rebuild modules & weights --------------------------
#     model = VJEPA(embed_dim=384, depth=6, num_heads=6).to(device)
#     head  = torch.nn.Linear(384, 2).to(device)
#     model.load_state_dict(v_sd, strict=True)
#     head .load_state_dict(t_sd, strict=True)

#     hnn = lnn = None
#     if hnn_sd:
#         hnn = HNN(hidden_dim=256).to(device); hnn.load_state_dict(hnn_sd, strict=True)
#     if lnn_sd:
#         lnn = LNN(input_dim=2, hidden_dim=256).to(device); lnn.load_state_dict(lnn_sd, strict=True)

#     # 3) -------- fixed evaluation DataLoader ------------------------
#     eval_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

#     # 4) -------- latent rollout -------------------------------------
#     θ, ω, _ = rollout(
#         model, head,
#         hnn         = hnn,
#         lnn         = lnn,
#         combine     = combine,
#         horizon     = cfg.horizon,
#         dt          = cfg.dt,
#         eval_loader = eval_loader
#     )

#     # 5) -------- base physics metrics -------------------------------
#     metrics: Dict[str, float] = {
#         "Δ_div"  : neighbour_divergence(θ, ω),
#         "E_drift": energy_drift(θ, ω, m=cfg.m, g=cfg.g, l=cfg.l),
#     }
#     if lnn is not None:
#         metrics["EL_res"] = el_residual_metric(lnn, θ, ω, dt=cfg.dt)

#     # 6) -------- per-step curves + rates ----------------------------
#     Δ_curve, Δ_rate = neighbour_divergence_curve(θ, ω)
#     E_curve, E_rate = energy_drift_curve(θ, ω, m=cfg.m, g=cfg.g, l=cfg.l)
#     metrics.update(Δ_rate=Δ_rate, E_rate=E_rate)

#     if lnn is not None:
#         EL_curve, EL_rate = el_residual_curve(lnn, θ, ω, dt=cfg.dt)
#         metrics["EL_rate"] = EL_rate

#     if save_curves:
#         # keep JSON (back-compat)
#         metrics["Δ_curve"] = Δ_curve.tolist()
#         metrics["E_curve"] = E_curve.tolist()
#         if lnn is not None:
#             metrics["EL_curve"] = EL_curve.tolist()
    
#         # add compact sidecar
#         curves_path = os.path.join(cfg.out_dir, f"curves_{mode}{cfg.suffix}.npz")
#         np.savez_compressed(
#             curves_path,
#             delta_curve=np.asarray(Δ_curve, dtype=np.float32),
#             energy_curve=np.asarray(E_curve, dtype=np.float32),
#             **({"el_curve": np.asarray(EL_curve, dtype=np.float32)} if lnn is not None else {})
#         )
#         metrics["curves_npz"] = curves_path  # tiny pointer in JSON

#     # 7) -------- latent-head diagnostics ---------------------------
#     θ_t, ω_t, θ_p, ω_p = collect_head_scatter(
#         model, head, dataset,
#         n_samples=cfg.scatter_samples,
#         batch    =cfg.scatter_batch)
#     metrics.update(head_regression_metrics(θ_t, ω_t, θ_p, ω_p))

#     # 8) -------- optional scatter plot -----------------------------
#     if plot_head_scatter:
#         plt.figure(figsize=(10, 5))
#         # θ
#         plt.subplot(1, 2, 1)
#         plt.scatter(θ_t, θ_p, s=8, alpha=.6)
#         plt.xlabel("true θ"); plt.ylabel("pred θ")
#         plt.grid(True); plt.xlim(-4, 4); plt.ylim(-4, 4)
#         plt.gca().set_aspect("equal", adjustable="box")
#         # ω
#         plt.subplot(1, 2, 2)
#         plt.scatter(ω_t, ω_p, s=8, alpha=.6)
#         plt.xlabel("true ω"); plt.ylabel("pred ω")
#         plt.grid(True); plt.xlim(-10, 10); plt.ylim(-10, 10)
#         plt.gca().set_aspect("equal", adjustable="box")
#         plt.suptitle(f"Head predictions vs truth – {mode}")
#         plt.tight_layout(); plt.show()

#     # 9) -------- write JSON ----------------------------------------
#     os.makedirs(cfg.out_dir, exist_ok=True)
#     out_path = os.path.join(cfg.out_dir, f"metrics_{mode}{cfg.suffix}.json")
#     with open(out_path, "w") as fp:
#         json.dump(metrics, fp, indent=2)

#     print(f"{mode:8} →", {k: round(v, 4) if isinstance(v, float) else '…' for k, v in metrics.items()},
#           f"(saved → {out_path})")
#     return metrics
# ====================================================================
# 5 · Evaluate *many* modes (auto-detects whether fusion is possible)
# ====================================================================
def evaluate_all_modes(
    modes        : Iterable[str],
    dataset,
    cfg          : EvalConfig,
    *,
    combine      : str | tuple[str, float] | None = "mean",
    plot_scatter : bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Run :func:`evaluate_mode` for every string in *modes*.

    The helper **checks each checkpoint first** and forwards the *combine*
    argument **only when *both* physics nets are present** (otherwise the
    call falls back to ``combine=None`` so that the run is *not* skipped).

    Parameters
    ----------
    modes
        Iterable with any subset of ``{"plain","hnn","lnn","hnn+lnn"}``.
    dataset
        Evaluation dataset (e.g. deterministic Pendulum grid).
    cfg
        Shared :class:`EvalConfig` (horizon, dt, paths…).
    combine
        Fusion rule when *both* HNN **and** LNN exist in the checkpoint
        (ignored otherwise).  Supported values:

        ================  ============================================
        ``None``          do **not** fuse – use HNN alone  
        ``"mean"``        α = ½ (α_HNN + α_LNN)  ← **default**  
        ``"sum"``         α = α_HNN + α_LNN  
        ``("blend", w)``  α = w·α_HNN + (1-w)·α_LNN  (0 ≤ w ≤ 1)  
        ================  ============================================

    plot_scatter
        Forwarded to :func:`evaluate_mode(plot_head_scatter=…)`.

    Returns
    -------
    dict
        ``{mode: metric-dict}`` for every *successfully* evaluated mode.
        The same object is also written as JSON to
        ``cfg.out_dir / metrics_all<suffix>.json``.
    """
    all_metrics: Dict[str, Dict[str, float]] = {}

    for mode in modes:
        try:
            # --- lightweight peek: are both physics nets stored? ----
            _, _, hnn_sd, lnn_sd = load_components(
                mode,
                suffix       = cfg.suffix,
                base_dir     = cfg.model_dir,
                map_location = "cpu")               # tiny, no GPU alloc

            # decide on the fusion rule to pass downstream
            comb_arg = combine if (hnn_sd and lnn_sd) else None

            # ------------------ full evaluation --------------------
            all_metrics[mode] = evaluate_mode(
                mode,
                dataset,
                cfg,
                combine           = comb_arg,
                plot_head_scatter = plot_scatter
            )
        except FileNotFoundError as err:
            print(f"[{mode}] skipped – checkpoint not found: {err}")
        except ValueError as err:
            # propagate unexpected errors but keep loop alive
            print(f"[{mode}] skipped – {err}")

    # ------------------ persist combined dict -----------------------
    os.makedirs(cfg.out_dir, exist_ok=True)
    out_file = os.path.join(cfg.out_dir, f"metrics_all{cfg.suffix}.json")
    with open(out_file, "w") as fh:
        json.dump(all_metrics, fh, indent=2)

    print(f"\n✓ combined metrics saved → {out_file}")
    return all_metrics

# ====================================================================
# 6 · Legacy plotting helpers
# ====================================================================
_REMAP = {"total":"loss_total","jepa":"loss_jepa",
          "hnn":"loss_hnn","lnn":"loss_lnn","sup":"loss_sup"}

metrics_to_plot: list[tuple[str, str]] = [
    ("loss_total", "total objective"),
    ("loss_hnn",   "HNN residual"),
    ("loss_lnn",   "LNN residual"),
    ("loss_jepa",  "JEPA reconstruction"),
    ("loss_sup",   "θ supervision"),
]

def normalise_keys(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Map legacy keys (total/jepa/…) → new loss_* names."""
    return { _REMAP.get(k, k): v for k,v in arrays.items() }

def plot_loss(
    component: str,
    logs: Dict[str, Dict[str, np.ndarray]],
    *,
    ylabel: str | None = None
) -> None:
    """
    Overlay loss curves from multiple runs.

    Parameters
    ----------
    component : e.g. 'loss_total', 'loss_hnn', …
    logs      : mode → arrays (already `normalise_keys`-processed)
    """
    plt.figure(figsize=(7,4))
    for mode, rec in logs.items():
        if component in rec:
            plt.plot(rec[component], label=mode)
    plt.xlabel("epoch")
    plt.ylabel(ylabel or component)
    plt.title(component)
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
    
def plot_selected_losses(
    logs: Dict[str, Dict[str, np.ndarray]],
    *,
    items: list[tuple[str, str]] = metrics_to_plot
) -> None:
    """
    Iterate over `items` and draw a loss curve **only if** at least one
    run in `logs` contains the given key.

    Parameters
    ----------
    logs
        Dict mapping *mode* → dict of numpy arrays
        (already passed through `normalise_keys` if needed).
    items
        Sequence of ``(loss_key, y-label)`` tuples.  Defaults to the
        canonical list above.
    """
    for key, label in items:
        if any(key in rec for rec in logs.values()):  # at least one run has it?
            plot_loss(key, logs, ylabel=label)        # use the existing helper
        else:
            print(f"[skip] no log contains '{key}'")