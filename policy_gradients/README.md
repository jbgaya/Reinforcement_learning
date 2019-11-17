# Policy gradients for deep reinforcement learning

In this folder you can find some policy gradients implementations on gym environments.

## A simple interface to test policy gradients



![image](https://i.ibb.co/pQXfVnd/Capture-d-cran-de-2019-11-17-18-12-55.png)

**Installation** :
- make sure you already installed [gym](https://github.com/openai/gym), [Pytorch](https://pytorch.org/get-started/locally/),
and [PyInquirer](https://github.com/CITGuru/PyInquirer) packages.
- download this folder
- Run `main.py`. An interface will show up and guide you to make your experiments.

## Available algorithms

Go to `agents` folder to find the code.

- **Actor critic** (A2C) from [Konda et al.](http://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf) :
	- Online version : the policy is updated at each action the agent is taking (quite unstable)
	- Batch version : the policy is updated on a whole trajectory, making the algorithm more stable

- **Proximal policy optimization** (PPO) from [Schulman et al.](https://arxiv.org/pdf/1707.06347.pdf) :
	- Clipped version : as described in the paper, the loss is clipped to reduce high variance from rare trajectories.
	- Kulbach-Leibler version : a KL cost is added to th original loss to prevent the algorithm from taking too big and non-optimal steps

## Available environments

For each environment, a specific feature function is designed to extract important information from the state :

- [`CartPole-v1`](https://gym.openai.com/envs/CartPole-v1/) : identity function (no transformation)
- [`LunarLander-v2`](https://gym.openai.com/envs/LunarLander-v2/) : identity function (no transformation)
