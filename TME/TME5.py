import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device = ",device)

import gym

class NN(nn.Module):
    def __init__(self, inSize, outSize, layers,softmax=False):
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
            x = torch.relu(x)
            x = self.layers[i](x)
        return torch.softmax(x,0) if self.softmax else x

def phi(obs,device):
    return torch.Tensor(obs).to(device,torch.double)

class Batch_agent():
    def __init__(self,d_in,d_out,layers_V=[300,50],layers_f=[300,50],alpha=1.,gamma=0.99,device=device):


        self.device = device
        #NN to approximate V_hat (baseline) and f (distribution function)
        self.V = NN(d_in,1,layers_V).to(self.device,torch.double)
        self.loss_V = nn.SmoothL1Loss()
        self.opt_V = Adam(self.V.parameters(),lr=1e-3)
        self.opt_V.zero_grad()
        self.f = NN(d_in,d_out,layers_f,softmax=True).to(self.device,torch.double)
        self.opt_f = Adam(self.f.parameters(),lr=1e-3)
        self.opt_f.zero_grad()

        self.episode = []
        self.gamma = gamma
        self.alpha = alpha

        self.lastobs = None
        self.lastaction = None
        self.start = True
        self.i = 0

        self.writer = SummaryWriter("runs/"+env.spec.id+"/Batch A2C - alpha="+str(alpha))

    def update(self):
        self.i += 1
        self.opt_V.zero_grad()
        self.opt_f.zero_grad()


        #1- fit V (baseline function)
        R = torch.zeros(1,dtype=torch.double,device=self.device)
        Vloss = torch.zeros(1,requires_grad=True,dtype=torch.double,device=self.device)
        for lastobs,action,obs,r in reversed(self.episode):
            R = r + self.gamma * R
            Vloss = Vloss + self.loss_V( self.V.forward(lastobs) , R)
        Vloss = Vloss #/ len(self.episode)
        self.writer.add_scalar('critic loss',Vloss.item(),i)
        Vloss.backward()
        self.opt_V.step()


        #2- fit f (policy function)
        A = torch.zeros(1,dtype=torch.double,device=self.device)
        floss = torch.zeros(1,requires_grad=True,dtype=torch.double,device=self.device)
        for lastobs,action,obs,r in reversed(self.episode):
            with torch.no_grad():
                delta = r + self.gamma * self.V.forward(obs) - self.V.forward(lastobs)
                A = self.gamma * self.alpha * A + delta
            floss = floss - torch.log(self.f.forward(lastobs)[action]) * A
        floss = floss #/ len(self.episode)
        self.writer.add_scalar('actor loss',floss.item(),i)
        floss.backward()
        self.opt_f.step()

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
            self.update()
            self.episode = []
            self.lastaction = None
            self.start = True

        return int(action)

    def act_notrain(self,obs,r,done):
        obs = phi(obs,self.device)
        with torch.no_grad():
            pi = Categorical(probs=self.f.forward(obs))
        action =pi.sample()
        return int(action)



class Online_agent():

    def __init__(self,d_in,d_out,layers_V=[100],layers_f=[100],alpha=1.,gamma=0.9999,device=torch.device("cpu")):
        self.device = device

        #NN to approximate V_hat (baseline) and f (distribution function)
        self.V = NN(d_in,1,layers_V).to(self.device,torch.double)
        self.loss_V = nn.SmoothL1Loss()
        self.opt_V = torch.optim.Adam(self.V.parameters(),lr=1e-7)
        self.opt_V.zero_grad()

        self.f = NN(d_in,d_out,layers_f,softmax=True).to(self.device,torch.double)
        self.opt_f = torch.optim.Adam(self.f.parameters(),lr=1e-7)
        self.opt_f.zero_grad()

        self.gamma = gamma
        self.alpha = alpha

        self.lastobs = None
        self.lastaction = None
        self.start = True

        self.i = 0
        self.writer = SummaryWriter("runs/"+env.spec.id+"/Online A2C - alpha="+str(alpha))

    def update(self,lastobs,action,obs,r):
        self.i += 1

        self.opt_f.zero_grad()
        self.opt_V.zero_grad()

        #1- fit V (baseline function)
        R = torch.zeros(1,dtype=torch.double,device=self.device)
        Vloss = torch.zeros(1,requires_grad=True,dtype=torch.double,device=self.device)
        with torch.no_grad():
            R = r + self.V.forward(obs)
        Vloss = self.loss_V( self.V.forward(lastobs) , R )
        self.writer.add_scalar('critic loss',Vloss.item(),self.i)
        Vloss.backward()
        self.opt_V.step()

        #2- fit f (policy function)
        A = torch.zeros(1,dtype=torch.double,device=self.device)
        floss = torch.zeros(1,requires_grad=True,dtype=torch.double,device=self.device)
        with torch.no_grad():
            delta = r + self.gamma * self.V.forward(obs) - self.V.forward(lastobs)
            A = self.gamma * self.alpha * A + delta
        floss = - torch.log(self.f.forward(lastobs)[action]) * A
        self.writer.add_scalar('actor loss',floss.item(),self.i)
        floss.backward()
        self.opt_f.step()


    def act(self,obs,r,done):
        obs = phi(obs,self.device)
        with torch.no_grad():
            pi = Categorical(probs=self.f.forward(obs))
        action =pi.sample()

        if(self.start):
            self.start = False
        else:
            self.update(self.lastobs,self.lastaction,obs,r)

        if done:
            self.lastaction = None
            self.start = True
        else:
            self.lastobs = obs
            self.lastaction = action

        return int(action)

    def act_notrain(self,obs,r,done):
        obs = phi(obs,self.device)
        with torch.no_grad():
            pi = Categorical(probs=self.f.forward(obs))
        action =pi.sample()
        return int(action)

if __name__ == "__main__":
    #Initializing environment, agent and variables
    #env = gym.make('CartPole-v1')
    env = gym.make('LunarLander-v2')
    d_in = env.observation_space.shape[0]
    d_out = env.action_space.n
    agent = Batch_agent(d_in,d_out)
    #agent = Online_agent(d_in,d_out)
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
                if (i % 100 == 0):
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
                agent.add_scalar('test rewards',rsum,i)
                print("Episode : " + str(i) + " rsum=" + str(round(rsum,2)) + ", " + str(j) + " actions")
                break
    env.close()
