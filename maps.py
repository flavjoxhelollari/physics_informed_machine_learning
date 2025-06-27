lambda_map = {
        "plain":    dict(λ_h=0.0,   λ_l=0.0),
        "hnn":      dict(λ_h=1e-3,  λ_l=0.0),
        "lnn":      dict(λ_h=0.0,   λ_l=1e-3),
        "hnn+lnn":  dict(λ_h=5e-4,  λ_l=1e-3),
    }

metrics_to_plot = [
    ("loss_total", "total objective"),
    ("loss_hnn",   "HNN residual"),
    ("loss_lnn",   "LNN residual"),
    ("loss_jepa",  "JEPA recon."),
    ("loss_sup",   "θ supervision"),
]