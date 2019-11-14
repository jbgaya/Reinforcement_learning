# Policy gradients for deep reinforcement learning

This folder aims to present some policy gradients implementations on gym environments.

## Available algorithms

Go to `agents` folder to find the implementations.

- Actor critic (A2C) from [Konda et al.](http://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)
	- Online version
	- Batch version

- Proximal policy optimization (PPO) from [Schulman et al.](https://arxiv.org/pdf/1707.06347.pdf)
	- Clipped version
	- Kulbach-Leibler version

## Available environments

- [`CartPole-v1`](https://gym.openai.com/envs/CartPole-v1/)
- [`LunarLander-v2`](https://gym.openai.com/envs/LunarLander-v2/)
