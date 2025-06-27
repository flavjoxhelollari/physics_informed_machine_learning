import torch, numpy as np, matplotlib.pyplot as plt, json, os, re
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────
# 0 · helper: run head → (θ̂, ω̂) and collect points
# ────────────────────────────────────────────────────────────────
def _scatter_head(model, head, dataset, num_samples=500, batch_size=64):
    model.eval(); head.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    θ_true, ω_true, θ_pred, ω_pred = [], [], [], []
    collected = 0

    for seq, states in loader:
        imgs = seq[:, 0].to(device)                       # (B,C,H,W)
        with torch.no_grad():
            z = model.patch_embed(imgs) + model.pos_embed
            z = model.context_encoder(z).mean(1)          # (B,D)
            θ̂, ω̂ = head(z).split(1, 1)

        θ_true.extend(states[:, 0, 0].cpu().numpy())
        ω_true.extend(states[:, 0, 1].cpu().numpy())
        θ_pred.extend(θ̂.squeeze().cpu().numpy())
        ω_pred.extend(ω̂.squeeze().cpu().numpy())

        collected += imgs.size(0)
        if collected >= num_samples:
            break

    # trim exactly
    θ_t = np.array(θ_true)[:num_samples]
    ω_t = np.array(ω_true)[:num_samples]
    θ_p = np.array(θ_pred)[:num_samples]
    ω_p = np.array(ω_pred)[:num_samples]

    return θ_t, ω_t, θ_p, ω_p


# ────────────────────────────────────────────────────────────────
# 1 · analyse & plot for all modes
# ────────────────────────────────────────────────────────────────
def analyse_modes(modes,
                  dataset,
                  model_ctor,
                  head_ctor,
                  samples=500,
                  out_json="head_metrics_dense.json"):
    metrics = {}

    for m in modes:
        mdl_file  = f"model_{m}_dense.pt"
        head_file = f"theta_{m}_dense.pt"
        if not (os.path.exists(mdl_file) and os.path.exists(head_file)):
            print(f"[{m}] checkpoints not found → skipping"); continue

        # --- load nets -----------------------------------------------------
        model = model_ctor().to(device)
        model.load_state_dict(torch.load(mdl_file, map_location=device))
        head  = head_ctor().to(device)
        head.load_state_dict(torch.load(head_file, map_location=device))

        # --- collect points ------------------------------------------------
        θ_t, ω_t, θ_p, ω_p = _scatter_head(model, head, dataset,
                                           num_samples=samples)

        # --- metrics -------------------------------------------------------
        r2θ  = r2_score(θ_t, θ_p);   mseθ = mean_squared_error(θ_t, θ_p)
        r2ω  = r2_score(ω_t, ω_p);   mseω = mean_squared_error(ω_t, ω_p)
        metrics[m] = dict(r2_theta=r2θ, mse_theta=mseθ,
                          r2_omega=r2ω, mse_omega=mseω)
        print(f"\n{m.upper()}:  θ R²={r2θ:.3f}  ω R²={r2ω:.3f}")

        # --- scatter plots -------------------------------------------------
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1); plt.scatter(θ_t, θ_p, s=8, alpha=.6)
        plt.xlabel("true θ"); plt.ylabel("pred θ"); plt.grid()
        plt.subplot(1,2,2); plt.scatter(ω_t, ω_p, s=8, alpha=.6)
        plt.xlabel("true ω"); plt.ylabel("pred ω"); plt.grid()
        plt.suptitle(f"Head predictions vs truth  –  {m}")
        plt.tight_layout(); plt.show()

    with open(out_json, "w") as f: json.dump(metrics, f, indent=2)
    print("\nSaved metrics →", out_json)
    
    
# ------------------------------------------------------------
# phase-space comparison for multiple trained models
# ------------------------------------------------------------
def plot_phase_space_models(modes,
                            dataset,
                            model_ctor,
                            head_ctor,
                            num_samples=500):
    """
    For every mode in `modes`:
      • load model_<mode>.pt  and  theta_<mode>.pt
      • scatter true (θ, ω)  vs. predicted (θ̂, ω̂) for up to num_samples
    """
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    results = {}          # mode → dict of arrays

    for mode in modes:
        mdl_file  = f"model_{mode}_dense.pt"
        head_file = f"theta_{mode}_dense.pt"
        if not (os.path.exists(mdl_file) and os.path.exists(head_file)):
            print(f"[{mode}] checkpoints missing — skipping")
            continue

        # -------- load nets ----------
        model = model_ctor().to(device)
        model.load_state_dict(torch.load(mdl_file, map_location=device))
        model.eval()

        theta_head = head_ctor().to(device)
        theta_head.load_state_dict(torch.load(head_file, map_location=device))
        theta_head.eval()

        θ_true, ω_true, θ_pred, ω_pred = [], [], [], []
        collected = 0

        for seq, states in loader:
            imgs  = seq[:, 0].to(device)          # (B,C,H,W) frame 0
            with torch.no_grad():
                z = model.patch_embed(imgs) + model.pos_embed
                z = model.context_encoder(z).mean(1)
                θ̂, ω̂ = theta_head(z).split(1,1)

            θ_true.extend(states[:,0,0].cpu().numpy())
            ω_true.extend(states[:,0,1].cpu().numpy())
            θ_pred.extend(θ̂.squeeze().cpu().numpy())
            ω_pred.extend(ω̂.squeeze().cpu().numpy())

            collected += imgs.size(0)
            if collected >= num_samples:
                break

        # trim to num_samples exactly
        θ_true = np.array(θ_true)[:num_samples]
        ω_true = np.array(ω_true)[:num_samples]
        θ_pred = np.array(θ_pred)[:num_samples]
        ω_pred = np.array(ω_pred)[:num_samples]

        results[mode] = dict(theta_true=θ_true,
                             omega_true=ω_true,
                             theta_pred=θ_pred,
                             omega_pred=ω_pred)

        # -------- plotting ----------
        plt.figure(figsize=(5,4))
        plt.scatter(θ_true, ω_true, s=10, alpha=.35, label="true")
        plt.scatter(θ_pred, ω_pred, s=10, alpha=.35, label="pred")
        plt.xlabel("θ  (rad)")
        plt.ylabel("ω  (rad/s)")
        plt.title(f"Phase-space – {mode}")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    return results


# ------------------------------------------------------------------
# your existing regression helper, slightly tweaked to *return* metrics
# ------------------------------------------------------------------
def latent_phase_regression(model, dataset, batch_size=64, num_samples=500):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    latents, theta_list, omega_list = [], [], []
    collected = 0

    for seq, states in loader:
        imgs  = seq[:, 0].to(next(model.parameters()).device)
        theta = states[:, 0, 0].cpu().numpy()
        omega = states[:, 0, 1].cpu().numpy()

        with torch.no_grad():
            z = model.patch_embed(imgs) + model.pos_embed
            z = model.context_encoder(z).mean(1).cpu()

        latents.append(z)
        theta_list.extend(theta)
        omega_list.extend(omega)

        collected += imgs.size(0)
        if collected >= num_samples:
            break

    Z = torch.cat(latents, 0).numpy()[:num_samples]
    θ = np.array(theta_list)[:num_samples].reshape(-1,1)
    ω = np.array(omega_list)[:num_samples].reshape(-1,1)

    reg_θ = LinearRegression().fit(Z, θ)
    reg_ω = LinearRegression().fit(Z, ω)

    θ̂, ω̂ = reg_θ.predict(Z), reg_ω.predict(Z)
    return dict(
        r2_theta = float(r2_score(θ, θ̂)),
        mse_theta= float(mean_squared_error(θ, θ̂)),
        r2_omega = float(r2_score(ω, ω̂)),
        mse_omega= float(mean_squared_error(ω, ω̂)),
    )

# ------------------------------------------------------------------
# load-and-analyse loop
# ------------------------------------------------------------------
def analyse_saved_models(modes, dataset,
                         model_ctor, head_ctor,
                         result_json="metrics_all_dense.json"):
    out = {}
    for m in modes:
        print(f"\n>>> {m.upper()} <<<")

        model = model_ctor().to(device)
        head  = head_ctor().to(device)

        mdl_ckpt  = f"model_{m}_dense.pt"
        head_ckpt = f"theta_{m}_dense.pt"
        if not (os.path.exists(mdl_ckpt) and os.path.exists(head_ckpt)):
            print("  checkpoints missing, skipping.")
            continue
        model.load_state_dict(torch.load(mdl_ckpt,  map_location=device))
        head.load_state_dict (torch.load(head_ckpt, map_location=device))

        # graft head (only for forward pass clarity)
        model.theta_head = head

        # run regression
        metrics = latent_phase_regression(model, dataset)
        out[m] = metrics
        print(metrics)

    # save metrics to one json for later
    with open(result_json, "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved metrics →", result_json)
    return out

# ------------------------------------------------------------------
# quick comparison plot  (example: R² for θ̂)
# ------------------------------------------------------------------
def plot_metric(all_metrics, key, ylabel):
    plt.figure(figsize=(6,4))
    modes = sorted(all_metrics.keys())
    vals  = [all_metrics[m][key] for m in modes]
    plt.bar(modes, vals)
    plt.ylabel(ylabel); plt.title(key.replace("_"," "))
    plt.grid(axis="y"); plt.tight_layout(); plt.show()
    
    
def _normalise_keys(ndict):
    """map total→loss_total, jepa→loss_jepa, … if needed"""
    mapping = {
        "total":"loss_total", "jepa":"loss_jepa",
        "hnn":"loss_hnn",     "lnn":"loss_lnn",
        "sup":"loss_sup"
    }
    out = {}
    for k,v in ndict.items():
        out[ mapping.get(k,k) ] = v
    return out

# ------------------------------------------------------------
# 2)  Generic multi-line plot
# ------------------------------------------------------------
def plot_loss(comp, ylabel=None, logs= {}):
    plt.figure(figsize=(7,4))
    for mode,rec in logs.items():
        if comp in rec:
            plt.plot(rec[comp], label=mode)
    plt.xlabel("epoch")
    plt.ylabel(ylabel or comp)
    plt.title(f"{comp} across experiments")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()