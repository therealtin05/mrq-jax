# MRQ-JAX

JAX implementation of **MR.Q: Towards General-Purpose Model-Free Reinforcement Learning**.

Paper: https://arxiv.org/abs/2501.16142

## Status

Implemented:

* Continuous control
* Episodic environments (haven't benchmark yet)
* Core MR.Q algorithm
* WandB logging


Not implemented:

* Image observations
* Discrete actions
* LAP (Loss Adjusted Prioritization)

## Results

![DMC Comparison](media/dmc_comparision.png)

## Installation

```bash
conda create -n mrq-jax python=3.10
conda activate mrq-jax

pip install -r requirements.txt
pip install -e .
```

## Training

Example:

```bash
python3 main.py env.env_name=humanoid-run env.backend=dmc
python3 main.py env.env_name=HalfCheetah-v4 env.backend=gymnasium mrq.episodic=true
```

## Acknowledgements

This implementation is inspired by:

* https://github.com/adaptive-intelligent-robotics/QDax
* https://github.com/ShaneFlandermeyer/tdmpc2-jax

Some code structure and implementation details follow ideas from these projects.
