
from agents.helper import *

class Clip_agent():

    def __init__(self,d_in,d_out,layers_V=[200],layers_f=[200],K=10,epochs=10,
    alpha=0.96,beta=0.5,epsilon=0.1,gamma=0.9999,entropy=0.01,device=torch.device("cpu")):
        self.device = device

        #NN to approximate V (baseline)
        self.V = NN(d_in,1,layers_V).to(self.device,torch.double)
        self.loss_V = nn.SmoothL1Loss()
        self.opt_V = torch.optim.Adam(self.V.parameters(),lr=0.001)
        self.opt_V.zero_grad()

        #NN to approximate f (policy)
        self.f = NN(d_in,d_out,layers_f,softmax=True).to(self.device,torch.double)
        self.lastf = NN(4,2,layers_f,softmax=True).to(self.device,torch.double)
        self.opt_f = torch.optim.Adam(self.f.parameters(),lr=0.001)
        self.opt_f.zero_grad()

        self.episode = []
        self.batch = []
        self.K = K
        self.epochs = epochs
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        self.epsilon = epsilon
        self.entropy = entropy

        self.lastobs = None
        self.lastaction = None
        self.start = True

    def update(self):

        #1- fit V (baseline function)
        for episode in self.batch:
            R = torch.zeros(1,dtype=torch.double)
            Vloss = torch.zeros(1,requires_grad=True,dtype=torch.double)
            for lastobs,action,obs,r in reversed(episode):
                R = r + self.gamma * R
                Vloss = Vloss + self.loss_V( self.V.forward(lastobs) , R )
            Vloss.backward()
            self.opt_V.step()
            self.opt_V.zero_grad()

        #2- fit f (policy function)
        for epoch in range(self.epochs):
            self.lastf.load_state_dict(self.f.state_dict())
            for episode in self.batch:
                A = torch.zeros(1,dtype=torch.double)
                floss = torch.zeros(1,requires_grad=True,dtype=torch.double)
                for lastobs,action,obs,r in reversed(episode):
                    with torch.no_grad():
                        delta = r + self.gamma * self.V.forward(obs) - self.V.forward(lastobs)
                        A = self.gamma * self.alpha * A + delta
                        lastpi = self.lastf.forward(lastobs)
                    newpi =  self.f.forward(lastobs)
                    entr = - (newpi * newpi.log()).sum()
                    #print("--------------")
                    #print(newpi[action]/lastpi[action])
                    #print(torch.clamp((newpi[action]/lastpi[action]),1-self.epsilon,1+self.epsilon))
                    floss = floss - min( (newpi[action]/lastpi[action]) * A,
                        torch.clamp((newpi[action]/lastpi[action]),1-self.epsilon,1+self.epsilon) * A) + self.entropy * entr
                #print(floss.item())
                floss.backward()
                self.opt_f.step()
                self.opt_f.zero_grad()

    def act(self,obs,r,done):
        obs = phi(obs,self.device)
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
            self.episode = []
            self.lastaction = None
            self.start = True

            if len(self.batch) == self.K:
                #print("updating PPO agent policy...")
                self.update()
                self.batch = []

        return int(action)

    def act_notrain(self,obs,r,done):
        obs = phi(obs,self.device)
        with torch.no_grad():
            pi = Categorical(probs=self.f.forward(obs))
        action =pi.sample()

        return int(action)

class KL_agent():

    def __init__(self,d_in,d_out,gamma=0.9999,layers_V=[200],layers_f=[200],
    K=10,epochs=5,alpha=0.96,beta=0.5,delta=0.1,entropy=0.01,device=torch.device("cpu")):
        self.device = device

        #NN to approximate V_hat (baseline) and f (distribution function)
        self.V = NN(d_in,1,layers_V).to(device,torch.double)
        self.loss_V = nn.SmoothL1Loss()
        self.opt_V = torch.optim.Adam(self.V.parameters(),lr=0.001)
        self.opt_V.zero_grad()

        self.f = NN(d_in,d_out,layers_f,softmax=True).to(device,torch.double)
        self.lastf = NN(d_in,d_out,layers_f,softmax=True).to(device,torch.double)
        self.opt_f = torch.optim.Adam(self.f.parameters(),lr=0.001)
        self.opt_f.zero_grad()

        self.episode = []
        self.batch = []
        self.K = K
        self.epochs = epochs
        self.gamma = gamma
        self.beta = beta
        self.delta = delta
        self.alpha = alpha
        self.entropy = entropy

        self.lastobs = None
        self.lastaction = None
        self.start = True

    def phi(self,obs):
        return torch.Tensor(obs).to(device,torch.double)

    def update_kl(self,kl):

        print("1.5 * delta: ",1.5 * self.delta)
        print("kl:\t",kl.item())
        print("delta / 1.5: ",self.delta / 1.5)
        print("beta:\t",self.beta)
        if kl > 1.5 * self.delta:
            self.beta *= 2
        if kl < self.delta / 1.5 :
            self.beta /= 2

    def update(self):

        #fit V (baseline function)
        for episode in self.batch:
            R = torch.zeros(1,dtype=torch.double)
            Vloss = torch.zeros(1,requires_grad=True,dtype=torch.double)
            for lastobs,action,obs,r in reversed(episode):
                R = r + self.gamma * R
                Vloss = Vloss + self.loss_V( self.V.forward(lastobs) , R )
            Vloss.backward()
            self.opt_V.step()
            self.opt_V.zero_grad()

        #2- fit f (policy function)
        for epoch in range(self.epochs):
            self.lastf.load_state_dict(self.f.state_dict())
            for episode in self.batch:
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
                floss.backward()
                self.opt_f.step()
                self.opt_f.zero_grad()
        with torch.no_grad():
            self.update_kl(kl)

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
            self.episode = []
            self.lastaction = None
            self.start = True

            if len(self.batch) == self.K:
                #print("updating PPO agent policy...")
                self.update()
                self.batch = []

        return int(action)

    def act_notrain(self,obs,r,done):
        obs = phi(obs,self.device)
        with torch.no_grad():
            pi = Categorical(probs=self.f.forward(obs))
        action =pi.sample()

        return int(action)
