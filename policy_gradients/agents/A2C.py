from agents.helper import *

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
            x = torch.tanh(x)
            x = self.layers[i](x)
        return torch.softmax(x,0) if self.softmax else x

def phi(obs,device):
    return torch.Tensor(obs).to(device,torch.double)

class Batch_agent():
    def __init__(self,d_in,d_out,layers_V=[200],layers_f=[200],alpha=0.92,gamma=0.99,device=torch.device("cpu")):


        self.device = device
        #NN to approximate V_hat (baseline) and f (distribution function)
        self.V = NN(d_in,1,layers_V).to(self.device,torch.double)
        self.loss_V = nn.SmoothL1Loss()
        self.opt_V = torch.optim.Adam(self.V.parameters(),lr=0.001)
        self.opt_V.zero_grad()
        self.f = NN(d_in,d_out,layers_f,softmax=True).to(self.device,torch.double)
        self.opt_f = torch.optim.Adam(self.f.parameters(),lr=0.001)
        self.opt_f.zero_grad()

        self.episode = []
        self.gamma = gamma
        self.alpha = alpha

        self.lastobs = None
        self.lastaction = None
        self.start = True

    def update(self):
        self.opt_V.zero_grad()
        self.opt_f.zero_grad()


        #1- fit V (baseline function)
        R = torch.zeros(1,dtype=torch.double)
        Vloss = torch.zeros(1,requires_grad=True,dtype=torch.double)
        for lastobs,action,obs,r in reversed(self.episode):
            R = r + self.gamma * R
            Vloss = Vloss + self.loss_V( self.V.forward(lastobs) , R )
        Vloss.backward()
        self.opt_V.step()


        #2- fit f (policy function)
        A = torch.zeros(1,dtype=torch.double)
        floss = torch.zeros(1,requires_grad=True,dtype=torch.double)
        for lastobs,action,obs,r in reversed(self.episode):
            with torch.no_grad():
                delta = r + self.gamma * self.V.forward(obs) - self.V.forward(lastobs)
                A = self.gamma * self.alpha * A + delta
            floss = floss - torch.log(self.f.forward(lastobs)[action]) * A
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

    def __init__(self,d_in,d_out,layers_V=[200],layers_f=[200],alpha=0.96,gamma=0.99,device=torch.device("cpu")):
        self.device = device

        #NN to approximate V_hat (baseline) and f (distribution function)
        self.V = NN(d_in,1,layers_V).to(self.device,torch.double)
        self.loss_V = nn.SmoothL1Loss()
        self.opt_V = torch.optim.Adam(self.V.parameters(),lr=0.001)
        self.opt_V.zero_grad()

        self.f = NN(d_in,d_out,layers_f,softmax=True).to(self.device,torch.double)
        self.opt_f = torch.optim.Adam(self.f.parameters(),lr=0.001)
        self.opt_f.zero_grad()

        self.gamma = gamma
        self.alpha = alpha

        self.lastobs = None
        self.lastaction = None
        self.start = True

    def update(self,lastobs,action,obs,r):
        self.opt_f.zero_grad()
        self.opt_V.zero_grad()

        #1- fit V (baseline function)
        with torch.no_grad():
            R = r + self.V.forward(obs)
        Vloss = self.loss_V( self.V.forward(lastobs) , R )
        Vloss.backward()
        self.opt_V.step()

        #2- fit f (policy function)
        with torch.no_grad():
            delta = r + self.gamma * self.V.forward(obs) - self.V.forward(lastobs)
            A = self.gamma * self.alpha * A + delta
        floss = - torch.log(self.f.forward(lastobs)[action]) * A
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
