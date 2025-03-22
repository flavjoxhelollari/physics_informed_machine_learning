# physics_informed_machine_learning
Exploring how ML models *understand* physics. 



References : 
1. Intuitive physics understanding emerges from self-supervised pretraining
on natural videos Code: https://github.com/facebookresearch/jepa-intuitive-physics

2. Hamiltonian Neural Networks, Sam Greydanus, Misko Dzamba, Jason Yosinski,
NeurIPS 2019 Code: github.com/greydanus/hamiltonian-nn

3. Lagrangian Neural Networks, Miles Cranmer, Sam Greydanus, Stephan Hoyer,
Peter Battaglia, David Spergel, Shirley Ho, ICML Deep Differential Equations Workshop,
2020
Code: https://github.com/MilesCranmer/lagrangian_nns


How to run the code from the 3 papers above? 

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



