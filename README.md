# Physics informed ML
## Exploring how ML models *understand* physics. 



References : 
**********************************************************************************

1. Intuitive physics understanding emerges from self-supervised pretraining
on natural videos Code: https://github.com/facebookresearch/jepa-intuitive-physics

The paper "Intuitive physics understanding emerges from self-supervised pretraining on natural videos" explores how models can develop an understanding of intuitive physics by learning from video data. The key aspects are:

Model Architecture (V-JEPA): The Video Joint Embedding Predictive Architecture (V-JEPA) learns representations of video frames by predicting masked regions. This enables the model to capture physical properties like object permanence and continuity.

Violation-of-Expectation Framework: The model is tested on its ability to detect physically implausible scenarios (e.g., objects disappearing). It measures "surprise" based on prediction errors, indicating its grasp of intuitive physics concepts.

Findings: The V-JEPA model demonstrated strong generalization capabilities, suggesting that predictive coding in a learned representation space is sufficient for acquiring intuitive physics knowledge, even without explicit task-specific training.

This approach highlights how self-supervised learning can mimic human-like reasoning about the physical world.

**********************************************************************************

2. Hamiltonian Neural Networks, Sam Greydanus, Misko Dzamba, Jason Yosinski,
NeurIPS 2019 Code: github.com/greydanus/hamiltonian-nn

HNNs incorporate principles from Hamiltonian mechanics, focusing on conservation laws such as energy:

Core Idea: HNNs learn the Hamiltonian function of a system, which encodes its total energy. By doing so, they enforce physical priors like energy conservation directly into the learning process.

Applications: HNNs have been applied to tasks such as simulating mass-spring systems and celestial mechanics. They excel in scenarios where conservation laws are critical, producing reversible and stable predictions over time.

Advantages: Compared to traditional neural networks, HNNs generalize better and train faster on physics-related problems due to their inductive biases.

However, HNNs require canonical coordinates (e.g., position and momentum), which limits their applicability in some scenarios.


**********************************************************************************


3. Lagrangian Neural Networks, Miles Cranmer, Sam Greydanus, Stephan Hoyer,
Peter Battaglia, David Spergel, Shirley Ho, ICML Deep Differential Equations Workshop,
2020
Code: https://github.com/MilesCranmer/lagrangian_nns

LNNs extend the idea of physics-informed ML by parameterizing Lagrangians, which describe a system's dynamics using generalized coordinates:

Key Features:

Unlike HNNs, LNNs do not require canonical coordinates, making them suitable for systems where these are difficult to compute (e.g., double pendulum).

They enforce energy conservation while allowing for more flexible representations of physical systems.

Applications:

Modeling complex dynamics such as relativistic particles and wave equations.

Learning dynamics directly from data without predefined equations.

Comparison with HNNs: While both methods conserve energy, LNNs are more versatile in handling arbitrary coordinates and non-holonomic systems.

**********************************************************************************


NOTE : ML models like V-JEPA, HNNs, and LNNs demonstrate distinct approaches to understanding physics. V-JEPA focuses on intuitive physics through video-based learning, while HNNs and LNNs embed physical laws into their architectures for more accurate and interpretable simulations of dynamical systems. These advancements suggest that integrating domain knowledge into ML models can significantly enhance their ability to model and reason about the physical world.



**********************************************************************************

## How to run the code from the 3 papers above? 

git clone https://github.com/facebookresearch/jepa.git
cd jepa
pip install torch torchvision moviepy celluloid
/absolute_file_path.[mp4, webvid, etc.] $integer_class_label
python app/main.py --config-file configs/pretrain/your_config.yaml


git clone https://github.com/greydanus/hamiltonian-nn.git
cd hamiltonian-nn
pip install torch numpy scipy matplotlib celluloid gym
cd experiment/
python experiment.py
cd pixels/
python pendulum.py



git clone https://github.com/MilesCranmer/lagrangian_nns.git
cd lagrangian_nns
pip install jax jaxlib numpy moviepy celluloid


Initial Report - Phase 1
________________________

Hamiltonian and Lagrangian Neural Networks for modeling pendulum dynamics



