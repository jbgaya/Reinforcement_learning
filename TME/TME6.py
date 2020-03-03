from random import sample
import math
import numpy as np
import torch
from torch.nn.functional import kl_div
from torch.distributions.multinomial import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import gym
device = torch.device('cpu')
print("device = ",device)

class NN(nn.Module):
    def __init__(self, inSize, outSize, layers=[],softmax=False):
        super(NN, self).__init__()
        self.softmax = softmax
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.tanh(x)
            x = self.layers[i](x)
        return torch.softmax(x,0) if self.softmax else x

def phi(obs,device):
    return torch.Tensor(obs).to(device,torch.double)


class Clip_agent():

    def __init__(self,d_in,d_out,layers_V=[200],layers_f=[200],K=100,epochs=50,minibatchs=20,
    alpha=0.,epsilon=0.3,gamma=0.99,entropy=0.01,device=device):
        self.device = device

        #NN to approximate V (baseline)
        self.V = NN(d_in,1,layers_V).to(self.device,torch.double)
        self.loss_V = nn.SmoothL1Loss()
        self.opt_V = torch.optim.Adam(self.V.parameters(),lr=1e-3)
        self.opt_V.zero_grad()

        #NN to approximate f (policy)
        self.f = NN(d_in,d_out,layers_f,softmax=True).to(self.device,torch.double)
        self.lastf = NN(d_in,d_out,layers_f,softmax=True).to(self.device,torch.double)
        self.opt_f = torch.optim.Adam(self.f.parameters(),lr=1e-3)
        self.opt_f.zero_grad()

        self.episode = []
        self.batch = []
        self.K = K
        self.k = 0
        self.minibatchs = minibatchs
        self.epochs = epochs
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.entropy = entropy

        self.lastobs = None
        self.lastaction = None
        self.start = True

        self.i = 0
        self.j = 0
        self.writer = SummaryWriter("runs/"+env.spec.id+"/Clipped PPO - epsilon="+str(self.epsilon)+" - alpha="+str(self.alpha))

    def update(self):

        batch = []
        for episode in self.batch:
            R = torch.zeros(1,dtype=torch.double)
            for i,(lastobs,action,obs,r) in enumerate(reversed(episode)):
                R = r + self.gamma * R
                batch.append((lastobs,action,obs,r,R))

        #1- fit V (baseline function)
        for epoch in range(self.epochs):
            self.i += 1
            sample_batch = sample(batch,self.minibatchs)
            Vloss = torch.zeros(1,requires_grad=True,dtype=torch.double)
            for lastobs,action,obs,_,R in sample_batch:
                Vloss = Vloss + self.loss_V( self.V.forward(lastobs) , R )
            Vloss = Vloss / self.minibatchs
            self.writer.add_scalar('critic loss',Vloss.item(),self.i)
            Vloss.backward()
            self.opt_V.step()
            self.opt_V.zero_grad()

        #2- fit f (policy function)
        self.lastf.load_state_dict(self.f.state_dict())
        for epoch in range(self.epochs):
            self.j += 1
            #total_loss = 0
            sample_batch = sample(batch,self.minibatchs)
            A = torch.zeros(1,dtype=torch.double)
            floss = torch.zeros(1,requires_grad=True,dtype=torch.double)
            for lastobs,action,obs,r,_ in sample_batch:
                with torch.no_grad():
                    #delta = r + self.gamma * self.V.forward(obs) - self.V.forward(lastobs)
                    #A = self.gamma * self.alpha * A + delta
                    A = r + self.gamma * self.V.forward(obs) - self.V.forward(lastobs)
                    lastpi = self.lastf.forward(lastobs)
                newpi =  self.f.forward(lastobs)
                entr = - (newpi * newpi.log()).sum()
                #print("--------------")
                #print(newpi[action]/lastpi[action])
                #print(torch.clamp((newpi[action]/lastpi[action]),1-self.epsilon,1+self.epsilon))
                floss = floss - min( (newpi[action]/lastpi[action]) * A,
                    torch.clamp((newpi[action]/lastpi[action]),1-self.epsilon,1+self.epsilon) * A ) + self.entropy * entr
                #print(floss.item())
            #total_loss += -round(floss.item(),2)
            self.writer.add_scalar('actor loss',floss.item(),self.j)
            floss = floss / self.minibatchs
            floss.backward()
            self.opt_f.step()
            self.opt_f.zero_grad()
            #print("epoch ",epoch,":",total_loss)

    def act(self,obs,r,done):
        obs = phi(obs,self.device)
        with torch.no_grad():
            pi = Categorical(probs=self.f.forward(obs))
        action =pi.sample()

        if(self.start):
            self.start = False
        else:
            self.episode.append((self.lastobs,self.lastaction,obs,r))
        self.lastobs = obs
        self.lastaction = action

        if done:
            self.batch.append(self.episode)
            self.k += len(self.episode)
            self.episode = []
            self.lastaction = None
            self.start = True

            if self.k >= self.K:
                self.update()
                self.batch = []
                self.k = 0
        return int(action)

    def act_notrain(self,obs,r,done):
        obs = phi(obs,self.device)
        with torch.no_grad():
            pi = Categorical(probs=self.f.forward(obs))
        action =pi.sample()

        return int(action)

class KL_agent():

    def __init__(self,d_in,d_out,layers_V=[200],layers_f=[200],minibatchs=100,
    K=100,epochs=20,alpha=1.,beta=0.3,delta=0.005,gamma=0.99,entropy=0.,device=torch.device("cpu")):
        self.device = device

        #NN to approximate V_hat (baseline) and f (distribution function)
        self.V = NN(d_in,1,layers_V).to(device,torch.double)
        self.loss_V = nn.SmoothL1Loss()
        self.opt_V = torch.optim.Adam(self.V.parameters(),lr=1e-3)
        self.opt_V.zero_grad()

        self.f = NN(d_in,d_out,layers_f,softmax=True).to(device,torch.double)
        self.lastf = NN(d_in,d_out,layers_f,softmax=True).to(device,torch.double)
        self.opt_f = torch.optim.Adam(self.f.parameters(),lr=1e-3)
        self.opt_f.zero_grad()

        self.episode = []
        self.batch = []
        self.k = 0
        self.K = K
        self.epochs = epochs
        self.minibatchs = minibatchs
        self.gamma = gamma
        self.beta = beta
        self.delta = delta
        self.alpha = alpha
        self.entropy = entropy

        self.lastobs = None
        self.lastaction = None
        self.start = True

        self.i = 0
        self.j = 0
        self.writer = SummaryWriter("runs/"+env.spec.id+"/KL PPO - alpha="+str(alpha))

    def update_kl(self,kl):

        #print("1.5 * delta: ",1.5 * self.delta)
        #print("kl:\t",kl.item())
        #print("delta / 1.5: ",self.delta / 1.5)
        #print("beta:\t",self.beta)
        if kl > 1.5 * self.delta:
            self.beta *= 2
        elif kl < self.delta / 1.5 :
            self.beta /= 2

    def update(self):

        batch = []
        for episode in self.batch:
            R = torch.zeros(1,dtype=torch.double)
            for i,(lastobs,action,obs,r) in enumerate(reversed(episode)):
                R = r + self.gamma * R
                batch.append((lastobs,action,obs,r,R))

        #1- fit V (baseline function)
        for epoch in range(self.epochs):
            self.i += 1
            sample_batch = sample(batch,self.minibatchs)
            Vloss = torch.zeros(1,requires_grad=True,dtype=torch.double)
            for lastobs,action,obs,_,R in sample_batch:
                Vloss = Vloss + self.loss_V( self.V.forward(lastobs) , R )
            Vloss = Vloss / self.minibatchs
            self.writer.add_scalar('critic loss',Vloss.item(),self.i)
            Vloss.backward()
            self.opt_V.step()
            self.opt_V.zero_grad()

        #2- fit f (policy function)
        self.lastf.load_state_dict(self.f.state_dict())
        for epoch in range(self.epochs):
            self.j += 1
            sample_batch = sample(batch,self.minibatchs)
            A = torch.zeros(1,dtype=torch.double)
            floss = torch.zeros(1,requires_grad=True,dtype=torch.double)
            for lastobs,action,obs,r in reversed(episode):
                with torch.no_grad():
                    delta = r + self.gamma * self.V.forward(obs) - self.V.forward(lastobs)
                    A = self.gamma * self.alpha * A + delta
                    lastpi = self.lastf.forward(lastobs)
                newpi =  self.f.forward(lastobs)
                entr = - (newpi * newpi.log()).sum()
                kl = kl_div(newpi.log(),lastpi,reduction="sum")
                floss = floss - (newpi[action] / lastpi[action]) * A + self.beta * kl + self.entropy * entr
                #print(floss.item())
            floss = floss / self.minibatchs
            self.writer.add_scalar('actor loss',floss.item(),self.j)
            floss.backward()
            self.opt_f.step()
            self.opt_f.zero_grad()
        with torch.no_grad():
            self.update_kl(kl_div(self.f.forward(lastobs).log(),lastpi,reduction="sum"))

    def act(self,obs,r,done):
        obs = phi(obs,self.device)
        with torch.no_grad():
            pi = Categorical(probs=self.f.forward(obs))
        action =pi.sample()

        if(self.start):
            self.start = False
        else:
            self.episode.append((self.lastobs,self.lastaction,obs,r))
        self.lastobs = obs
        self.lastaction = action

        if done:
            self.batch.append(self.episode)
            self.k += len(self.episode)
            self.episode = []
            self.lastaction = None
            self.start = True

            if self.k >= self.K:
                self.update()
                self.batch = []
                self.k = 0
        return int(action)

    def act_notrain(self,obs,r,done):
        obs = phi(obs,self.device)
        with torch.no_grad():
            pi = Categorical(probs=self.f.forward(obs))
        action =pi.sample()

        return int(action)

if __name__ == "__main__":
    #Initializing environment, agent and variables
    env = gym.make('CartPole-v1')
    #env = gym.make('LunarLander-v2')
    d_in = env.observation_space.shape[0]
    d_out = env.action_space.n
    #agent = KL_agent(d_in,d_out)
    agent = Clip_agent(d_in,d_out)
    env.seed(0)
    reward = 0
    done = False
    rsum = 0
    episode_count = 3000
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
                if (i % 100 == 0):
                    print("Episode : " + str(i) + " rsum=" + str(round(rsum,2)) + ", " + str(j) + " actions")
                break
        if sum(rsum_count[-50:]) == 500*50:
            print("Agent trained ! ")
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
