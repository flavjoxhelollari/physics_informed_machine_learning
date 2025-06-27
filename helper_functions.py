import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_energy(theta, omega, m=1.0, l=1.0, g=9.81):
    """
    Compute the total mechanical energy of the pendulum
    E = 0.5 * m * l^2 * omega^2 + m * g * l * (1 - cos(theta))
    """
    kinetic = 0.5 * m * (l**2) * omega**2
    potential = m * g * l * (1 - np.cos(theta))
    return kinetic + potential

# Here we plot the energy of the pendulum system for each episode in the dataset.
# Dataset has each item : input and label. label contains theta omega. The
# episode length is te nr of steps in each episode

def plot_energy_per_episode(dataset, episode_length=100):
    thetas = []
    omegas = []
    energies = []
    num_episodes = len(dataset) // episode_length

    # go through all episodes and each time step of the episode
    for ep in range(num_episodes):
        ep_thetas = []
        ep_omegas = []
        for i in range(episode_length):
            idx = ep * episode_length + i
            _, label = dataset[idx]
            ep_thetas.append(label[0].item())
            ep_omegas.append(label[1].item())
        theta_np = np.array(ep_thetas)
        omega_np = np.array(ep_omegas)
        # calculate energy
        energy = compute_energy(theta_np, omega_np)
        energies.append(energy)

    # Plot each episode separately
    plt.figure(figsize=(12, 4))
    for ep_energy in energies:
        plt.plot(ep_energy, alpha=0.7)
    plt.xlabel("Time Step (within episode)")
    plt.ylabel("Energy (J)")
    plt.title("Pendulum Energy per Episode")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("energy_per_episode.png")
    plt.show()

# plot the true phase space trajectory, i.e. theta vs theta dot
# dataset laid out like above

def plot_true_phase_space(dataset, num_samples=500):
    theta_vals = []
    omega_vals = []
    # loop over the nr of samples
    for i in range(min(num_samples, len(dataset))):
        _, label = dataset[i]
        theta_vals.append(label[0].item())
        omega_vals.append(label[1].item())

    plt.figure(figsize=(6, 5))
    plt.plot(theta_vals, omega_vals, '.', alpha=0.3)
    plt.xlabel("Theta (rad)")
    plt.ylabel("Theta dot (rad/s)")
    plt.title("True Phase Space Trajectory (θ vs θ̇)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("true_phase_space.png")
    plt.show()
    
    
# -------------------------------------------------------------
# 4.Stable LNN acceleration helper (single grad w.r.t. qv)
# -------------------------------------------------------------
def lnn_accel(lnn, q, v, dt=None, eps=1e-6, a_clip=25.0):
    """
    Stable acceleration from a learned Lagrangian NN
      a = ( d/dt ∂L/∂v  –  ∂L/∂q )  /  ∂²L/∂v²
    q, v : (B,)   – 1-D tensors of θ and ω
    returns a : (B,)  – predicted ω̇
    """
    # ---------------------------------------------
    # 1) build (q,v) with grad tracking
    # ---------------------------------------------
    qv = torch.stack([q, v], dim=1).requires_grad_(True)   # (B,2)

    # ---------------------------------------------
    # 2) L(q,v)  and first derivatives
    # ---------------------------------------------
    L = lnn(qv).sum()
    dLdqv = torch.autograd.grad(L, qv, create_graph=True)[0]  # (B,2)
    dLdq, dLdv = dLdqv[:, 0], dLdqv[:, 1]                     # (B,)

    # time derivative  d/dt ∂L/∂v  ~  (current − stop-grad) / dt
    dLdv_dt = (dLdv - dLdv.detach()) / dt                    # (B,)

    # ---------------------------------------------
    # 3) second derivative  ∂²L/∂v²   (diagonal entry)
    # ---------------------------------------------
    d2Ldv2 = torch.autograd.grad(dLdv.sum(), qv,
                                 create_graph=True)[0][:, 1] # (B,)

    # clamp denominator to avoid division blow-up
    denom = d2Ldv2.abs().clamp_min(eps) * d2Ldv2.sign()      # preserve sign

    # ---------------------------------------------
    # 4) acceleration  and optional clipping
    # ---------------------------------------------
    a_pred = (dLdv_dt - dLdq) / denom                        # (B,)
    if a_clip is not None:
        a_pred = a_pred.clamp(-a_clip, a_clip)

    return a_pred

# ----------------------------------------------------------------
# 5 · rollout latent phase trajectory
# ----------------------------------------------------------------
@torch.no_grad()                         
def rollout(vjepa, head, *, hnn=None, lnn=None, horizon=None, dt=None, eval_loader=None):
    """
    Returns three tensors θ, ω, α  of shape (N,T).

    α = learned angular acceleration (zeros if neither HNN nor LNN is used)
    """
    vjepa.eval(); head.eval()
    if hnn: hnn.eval()
    if lnn: lnn.eval()

    θ_list, ω_list, α_list = [], [], []
    for seq, _ in tqdm(eval_loader, desc="rollout", leave=False):
        imgs0 = seq[:, 0].to(device)

        # -- initial latent → (θ̂₀, ω̂₀) ---------------------------
        z0        = vjepa.context_encoder(
                        vjepa.patch_embed(imgs0) + vjepa.pos_embed).mean(1)
        θ0, ω0    = head(z0).split(1, 1)           # (B,1) each
        θs, ωs    = [θ0.squeeze()], [ω0.squeeze()]
        αs        = []                             # will pad later

        # -- roll out ---------------------------------------------
        for _ in range(horizon - 1):
            q, v = θs[-1].detach(), ωs[-1].detach()

            # enable grads ONLY for the physics net we need
            if hnn is not None:
                with torch.set_grad_enabled(True):
                    qp   = torch.stack([q, v], 1).requires_grad_(True)
                    a    = hnn.time_derivative(qp)[:, 1]          # dp/dt
            elif lnn is not None:
                with torch.set_grad_enabled(True):
                    a    = lnn_accel(lnn, q, v, dt)               # from LNN
            else:
                a = torch.zeros_like(v)                           # plain V-JEPA

            q_next = q + v * dt
            v_next = v + a * dt

            θs.append(q_next); ωs.append(v_next); αs.append(a)

        # pad α with a zero for t = 0
        αs = [torch.zeros_like(αs[0])] + αs

        θ_list.append(torch.stack(θs, 1))
        ω_list.append(torch.stack(ωs, 1))
        α_list.append(torch.stack(αs, 1))

    θ = torch.cat(θ_list, 0).cpu()
    ω = torch.cat(ω_list, 0).cpu()
    α = torch.cat(α_list, 0).cpu()
    return θ, ω, α        # each (N, T)