import matplotlib
import math
import numpy as np
matplotlib.use("TkAgg")
import gym
import gridworld
from torch.utils.tensorboard import SummaryWriter

import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn
from torch import optim
import copy

from random import sample
from collections import deque
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device = ",device)

class NN(nn.Module):
    def __init__(self, inSize, outSize, layers=[]):
        super(NN, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
            self.layers.append(nn.Linear(inSize, outSize))

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.leaky_relu(x)
            x = self.layers[i](x)
        return x

class Memory():
    def __init__(self, N=500000):
        self.data = deque(maxlen=N)

    def sample(self,n):
        return sample(self.data,n)

    def store(self,last_obs,a,r,obs,done):
        self.data.append([last_obs,a,r,obs,done])

class DQN_agent():
    def __init__(self,d_in,d_out,epsilon=1.0,gamma=0.9999,layers=[200],C=2000,n=100,tau=0.001):
        self.Q = NN(d_in,d_out,layers).to(device,torch.double)
        self.Q_hat = NN(d_in,d_out,layers).to(device,torch.double)
        self.loss = nn.SmoothL1Loss()
        self.Q_hat.load_state_dict(self.Q.state_dict())

        self.memory = Memory()

        self.last_obs = torch.zeros(0)
        self.last_a = None

        self.epsilon = epsilon
        self.gamma = gamma
        self.C = C
        self.c = 0
        self.batch = n
        self.tau = tau

        self.i = 0
        self.writer = SummaryWriter("runs/"+env.spec.id+"/DQN - smoothed target network tau="+str(self.tau))

        self.opt = torch.optim.Adam(self.Q.parameters(),lr=1e-3)
        self.opt.zero_grad()

    def phi(self,obs):
        return torch.Tensor(obs).to(device,torch.double)

    def update(self):
        self.i += 1
        data = self.memory.sample(self.batch)
        X = torch.cat([self.Q.forward( last_obs )[act].reshape(1) for last_obs,act,_,_,_ in data]).T.to(device,torch.double)
        with torch.no_grad():
            Y = torch.cat([r if(done) else r + self.gamma * torch.max(self.Q_hat.forward(obs)) for _,_,r,obs,done in data]).to(device,torch.double)
        loss_func = self.loss( X , Y )
        self.writer.add_scalar('Q_loss',loss_func.item(),self.i)
        loss_func.backward()
        self.opt.step()
        self.opt.zero_grad()

        self.c += 1
        #if(self.c>=self.C):
        #    self.Q_hat.load_state_dict(self.Q.state_dict())
        #    self.c = 0

        #3- smooth update of Q
        for p_target,p in zip(self.Q_hat.parameters(),self.Q.parameters()):
            p_target.data.copy_( self.tau * p.data + (1-self.tau) * p_target.data )

    def act(self,obs,r,done):
        obs = self.phi(obs)
        r = torch.Tensor([r]).to(device,torch.double)

        with torch.no_grad():
            action = np.random.randint(2) if (np.random.rand() < self.epsilon) else int(torch.argmax(self.Q.forward(obs)))

        if len(self.last_obs) != 0:
            self.memory.store(self.last_obs,self.last_a,r,obs,done)
        if len(self.memory.data)>=self.batch:
            self.update()

        self.last_obs = obs
        self.last_a = action
        self.epsilon *= 0.9995
        return action

    def act_notrain(self,obs,r,done):
        obs = self.phi(obs)
        with torch.no_grad():
            action = int(torch.argmax(self.Q.forward(obs)))
        return action


if __name__ == "__main__":
    #Initializing environment, agent and variables
    env = gym.make('CartPole-v1')
    #env = gym.make('LunarLander-v2')
    d_in = env.observation_space.shape[0]
    d_out = env.action_space.n
    agent = agent = DQN_agent(d_in,d_out)
    env.seed(0)
    reward = 0
    done = False
    rsum = 0
    episode_count = 10000
    rsum_count = []
    test_episode_count = 100
    #Training phase
    print("Starting training phase on",episode_count,"episodes :")
    for i in range(1,episode_count+1):
        obs = env.reset()
        j = 0
        rsum = 0

        while True:
            action = agent.act(obs,reward,done)
            obs, reward, done, _ = env.step(action)
            rsum += reward
            j += 1
            #if env.verbose:
                #env.render()
            if done:
                rsum_count.append(rsum)
                agent.writer.add_scalar('train rewards',rsum,i)
                print("Episode : " + str(i) + " rsum=" + str(round(rsum,2)) + ", " + str(j) + " actions")
                break


    #Testing phase
    print("Starting testing phase on ",test_episode_count," episodes :")
    for i in range(1,test_episode_count+1):
        obs = env.reset()
        j = 0
        rsum = 0

        while True:
            action = agent.act_notrain(obs,reward,done)
            obs, reward, done, _ = env.step(action)
            rsum += reward
            j += 1
            if done:
                agent.writer.add_scalar('test rewards',rsum,i)
                print("Episode : " + str(i) + " rsum=" + str(round(rsum,2)) + ", " + str(j) + " actions")
                break
    env.close()
