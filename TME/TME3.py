import matplotlib
matplotlib.use("TkAgg")
import gridworld
import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import wrappers, logger
import copy
import warnings
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings(action='once')
warnings.filterwarnings('ignore')

#Agent QLearning
class Q_learning_agent():

    def __init__(self,alpha=0.01,epsilon=0.9,gamma=0.9,epsilon_decay=0.999):
        self.Q = {}
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lastobs = None
        self.lasta = None

    def update(self,obs,lastobs,lasta,r,done):
        if done:
            self.Q[lastobs][lasta] += self.alpha * (r - self.Q[lastobs][lasta])
        else:
            self.Q[lastobs][lasta] += self.alpha * (r + self.gamma * max([v for _,v in self.Q[obs].items()]) - self.Q[lastobs][lasta])


    def act(self,obs,r,done):
        self.Q[obs] = self.Q.get(obs,{x:y for x,y in zip(range(4),np.random.rand(4))})

        if self.lastobs != None:
            #print(states[self.lastobs]," ",states[obs])
            self.update(obs,self.lastobs,self.lasta,r,done)

        self.lastobs = obs
        action = np.random.randint(4) if (np.random.rand() < self.epsilon) else max(self.Q[obs],key=self.Q[obs].get)
        self.lasta = action
        self.epsilon *= self.epsilon_decay
        return action

# Execution avec un Agent Q-learning
alpha = 0.01
epsilon = 1.
gamma = 0.99
epsilon_decay = 0.9994
agent = Q_learning_agent(alpha,epsilon,gamma,epsilon_decay)

# Faire un fichier de log sur plusieurs scenarios
n = input("Enter the map number you want to choose (between 0 and 10) : ")
env = gym.make('gridworld-v0')
env.setPlan("gridworldPlans/plan"+str(n)+".txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
env.seed()  # Initialiser le pseudo aleatoire
episode_count = 3000
reward = 0
done = False
rsum = 0
FPS = 0.0001
writer = SummaryWriter("runs/gridworld/plan"+str(n)+"/Qlearning:alpha="+str(alpha)+",gamma="+str(gamma))
print("Starting training agent...")
for i in range(episode_count):
    if(i == episode_count-1):
        agent.epsilon = 0
    obs = env.reset()
    env.verbose = (i % 1000 == 0 and i > 0)  # afficher 1 episode sur 100
    if env.verbose:
        env.render(FPS)
    j = 0
    rsum = 0
    while True:
        action = agent.act(env.state2str(obs), reward, done)
        obs, reward, done, _ = env.step(action)
        rsum += reward
        j += 1
        if env.verbose:
            env.render(FPS)
        if done:
            writer.add_scalar('train rewards',rsum,i)
            print("Episode : " + str(i) + " rsum=" + str(round(rsum,2)) + ", " + str(j) + " actions")

            break

print("done")
env.close()
