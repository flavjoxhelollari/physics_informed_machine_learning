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
    
    
# ------------------------------------------------------------------
# Stable LNN acceleration helper
# ------------------------------------------------------------------
def lnn_accel(lnn, q, v, *, dt, eps=1e-6, a_clip=25.0):
    """
    a = ( d/dt ∂L/∂v  –  ∂L/∂q ) / ∂²L/∂v²
    q, v : (B,) – tensors (no grads required on entry)
    """
    qv = torch.stack([q, v], 1).requires_grad_(True)  # (B,2)

    L        = lnn(qv).sum()
    dLdqv    = torch.autograd.grad(L, qv, create_graph=True)[0]
    dLdq, dLdv = dLdqv[:, 0], dLdqv[:, 1]

    dLdv_dt  = (dLdv - dLdv.detach()) / dt
    d2Ldv2   = torch.autograd.grad(dLdv.sum(), qv,
                                   create_graph=True)[0][:, 1]

    denom    = d2Ldv2.abs().clamp_min(eps) * d2Ldv2.sign()
    a_pred   = (dLdv_dt - dLdq) / denom
    return a_pred.clamp(-a_clip, a_clip)


# ------------------------------------------------------------------
# Roll-out latent phase trajectory
# ------------------------------------------------------------------

@torch.no_grad()
def rollout(vjepa, head, *,                    # pos-only before *
            hnn=None, lnn=None,
            horizon, dt,                       # ← REQUIRED
            eval_loader):

    if horizon is None or dt is None:
        raise ValueError("rollout: ‘horizon’ and ‘dt’ must be supplied")

    vjepa.eval(); head.eval()
    if hnn: hnn.eval()
    if lnn: lnn.eval()

    θ_buf, ω_buf, α_buf = [], [], []

    for seq, _ in tqdm.tqdm(eval_loader, desc="rollout", leave=False):
        imgs0 = seq[:, 0].to(device)
        z0    = vjepa.context_encoder(
                    vjepa.patch_embed(imgs0) + vjepa.pos_embed).mean(1)
        θ, ω  = head(z0).split(1, 1)
        θ, ω  = θ.squeeze(), ω.squeeze()

        Θ, Ω, Α = [θ], [ω], []

        for _ in range(horizon - 1):
            q, v = Θ[-1].detach(), Ω[-1].detach()

            if hnn is not None:
                with torch.set_grad_enabled(True):
                    qp = torch.stack([q, v], 1).requires_grad_(True)
                    α  = hnn.time_derivative(qp)[:, 1]
            elif lnn is not None:
                with torch.set_grad_enabled(True):
                    α  = lnn_accel(lnn, q, v, dt=dt)
            else:
                α = torch.zeros_like(v)

            Θ.append(q + v * dt)
            Ω.append(v + α * dt)
            Α.append(α)

        Α = [torch.zeros_like(Α[0])] + Α
        θ_buf.append(torch.stack(Θ, 1))
        ω_buf.append(torch.stack(Ω, 1))
        α_buf.append(torch.stack(Α, 1))

    return (torch.cat(θ_buf).cpu(),
            torch.cat(ω_buf).cpu(),
            torch.cat(α_buf).cpu())
