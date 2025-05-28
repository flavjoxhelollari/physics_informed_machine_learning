# Physics-Informed ML: Pendulum Dynamics with V-JEPA, HNN, and LNN

This project demonstrates how physics-informed machine learning models—V-JEPA (video-based), Hamiltonian Neural Networks (HNN), and Lagrangian Neural Networks (LNN)—can learn and analyze the dynamics of a physical pendulum. The code and analyses are designed to help you explore how these models encode physical laws, conserve energy, and relate latent representations to true physical variables.

---

## Features

- **Pendulum Data Generation**: Synthesize pendulum trajectories (angle, angular velocity) and corresponding rendered images.
- **Model Training**: Train V-JEPA, HNN, and LNN models on pendulum data.
- **Physics Analysis**:
  - Compute and plot total mechanical energy (kinetic + potential) to assess conservation.
  - Visualize true phase space trajectories.
  - Analyze the relationship between learned latent representations and true physical variables (for V-JEPA).
- **Visualization**: Generate plots for energy over time, phase space, and latent variable correlations.

---
## Analysis Workflow

1. **Generate or load pendulum data** (images and physical states).
2. **Train the chosen model** (V-JEPA, HNN, or LNN).
3. **Evaluate model performance**:
 - Plot energy conservation.
 - Visualize phase space.
 - For V-JEPA: Analyze how latent features relate to θ and θ̇.
4. **Interpret results**:  
 - Check if the model conserves energy (HNN/LNN).
 - Assess if latent representations encode physical variables (V-JEPA).

---

## Requirements

- Python 3.7+
- torch, numpy, matplotlib, moviepy, celluloid, gym, jax, jaxlib, torchvision, scikit-learn, PIL

---

## References

- [V-JEPA: Intuitive physics from self-supervised video](https://github.com/facebookresearch/jepa-intuitive-physics)
- [Hamiltonian Neural Networks](https://github.com/greydanus/hamiltonian-nn)
- [Lagrangian Neural Networks](https://github.com/MilesCranmer/lagrangian_nns)

---


**Note:** For detailed usage, see the source code and notebooks in each repository.
