import gridworld
import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import wrappers, logger
import time

def compute_value(state,action,P,V_prev,gamma):
    return sum([ prob * (r + gamma * V_prev[new_state]) for prob,new_state,r,end in P[state][action]])

def evaluate_pi(state,P,V,gamma):
    ev = {a:sum([ prob * (r + gamma * V[new_state]) for prob,new_state,r,end in P[state][a]]) for a in [0,1,2,3]}
    return max(ev,key=ev.get)

class Policy_agent():

    def __init__(self,states,P,gamma=0.9):
        #Policy
        self.pi = dict(zip(P.keys(),np.random.randint(4, size=len(P))))
        self.lastpi = dict(zip(P.keys(),np.random.randint(4, size=len(P))))

        #Hyperparameters
        self.e = 1e-10
        self.gamma = gamma

    def act(self,state):
        return self.pi[state]

    def fit(self,states,P):
        self.i = 0

        while self.lastpi != self.pi:
            self.lastpi = self.pi.copy()

            #Value
            self.V =  {v:np.random.rand() if v in P else 0 for v in states}
            self.lastV = {v:np.random.rand() if v in P else 0 for v in states}

            while max([abs(self.lastV[v] - self.V[v]) for v in self.lastV]) > self.e:
                self.i += 1
                self.lastV = self.V.copy()
                self.V = {v:compute_value(v,self.pi[v],P,self.lastV,self.gamma) if v in self.pi else 0 for v in self.V}

            self.pi = {s:evaluate_pi(s,P,self.V,self.gamma) for s in self.pi}

    def show_map(self):

        actions = {0:"↓",1:"↑ ",2:"←",3:"→"}

        #Computing value and policy arrays for vizualization
        mapval = env.str2state(next(iter(self.V.keys()))).astype(float)
        for v in self.V:
            mapval[np.where(env.str2state(v)==2)] = round(self.V[v],2)

        mappi = np.array([[" " for  _ in range(mapval.shape[0])] for _ in range(mapval.shape[1])])
        for v in self.pi:
            mappi[np.where(env.str2state(v)==2)] = actions[self.pi[v]]


        #PLOT
        fig, ax = plt.subplots(1,2)
        fig.set_size_inches(20.5, 20.5)
        #Plotting MDP values
        im = ax[0].imshow(mapval,cmap="gist_gray")
        for i in range(mapval.shape[0]):
            for j in range(mapval.shape[1]):
                if mapval[i,j] not in [0,1]:
                    text = ax[0].text(j, i, mapval[i, j],
                               ha="center", va="center", color="r",size="x-small")
        ax[0].set_title("MDP Values")
        #Plotting MDP vpolicy
        im2 = ax[1].imshow(np.zeros(mappi.shape),cmap="gist_yarg")
        for i in range(mappi.shape[0]):
            for j in range(mappi.shape[1]):
                text = ax[1].text(j, i, mappi[i, j],
                           ha="center", va="center", color="r",size="x-large")
        ax[1].set_title("MDP Policy")
        plt.show()

def compute_value_2(state,action,P,V_prev,gamma):
    v = float("-inf")
    for action in [0,1,2,3]:
        v = max(v,sum([ prob * (r + gamma * V_prev[new_state]) for prob,new_state,r,end in P[state][action]]))
    return v

class Value_agent():

    def __init__(self,states,P,gamma=0.9):
        #Policy
        self.pi = dict(zip(P.keys(),np.random.randint(4, size=len(P))))

        #Hyperparameters
        self.e = 1e-10
        self.gamma = gamma

    def act(self,state):
        return self.pi[state]

    def fit(self,states,P):

        #Value
        self.V =  {v:np.random.rand() if v in P else 0 for v in states}
        self.lastV = {v:np.random.rand() if v in P else 0 for v in states}

        while sum([abs(self.lastV[v] - self.V[v]) for v in self.lastV]) > self.e:
            self.lastV = self.V.copy()
            self.V = {v:compute_value_2(v,self.pi[v],P,self.lastV,self.gamma) if v in self.pi else 0 for v in self.V}

        self.pi = {s:evaluate_pi(s,P,self.V,self.gamma) for s in self.pi}

    def show_map(self):

        actions = {0:"↓",1:"↑ ",2:"←",3:"→"}

        #Computing value and policy arrays for vizualization
        mapval = env.str2state(next(iter(self.V.keys()))).astype(float)
        for v in self.V:
            mapval[np.where(env.str2state(v)==2)] = round(self.V[v],2)

        mappi = np.array([[" " for  _ in range(mapval.shape[0])] for _ in range(mapval.shape[1])])
        for v in self.pi:
            mappi[np.where(env.str2state(v)==2)] = actions[self.pi[v]]


        #PLOT
        fig, ax = plt.subplots(1,2)
        fig.set_size_inches(20.5, 20.5)
        #Plotting MDP values
        im = ax[0].imshow(mapval,cmap="gist_gray")
        for i in range(mapval.shape[0]):
            for j in range(mapval.shape[1]):
                if mapval[i,j] not in [0,1]:
                    text = ax[0].text(j, i, mapval[i, j],
                               ha="center", va="center", color="r",size="x-small")
        ax[0].set_title("MDP Values")
        #Plotting MDP vpolicy
        im2 = ax[1].imshow(np.zeros(mappi.shape),cmap="gist_yarg")
        for i in range(mappi.shape[0]):
            for j in range(mappi.shape[1]):
                text = ax[1].text(j, i, mappi[i, j],
                           ha="center", va="center", color="r",size="x-large")
        ax[1].set_title("MDP Policy")
        plt.show()

if __name__ == "__main__":
    plan = str(input("Enter the id of the map you want to try (int 0 to 9) :"))
    discount = float(input("Enter the discount rate you want to try (float 0 to 1, 0.95 suggested) :"))
    algo = input("Enter the algorithm you want to try : 0 for Policy iteration, 1 for Value iteration :")

    #Initializing environment
    env = gym.make('gridworld-v0')
    rewards = {0:-0.001,3:1,4:1,5:-1,6:-1}
    env.setPlan("gridworldPlans/plan"+plan+".txt",rewards)
    env.verbose = True

    #Initializing MDP and agent
    states,P = env.getMDP()
    agent = Value_agent(states,P,gamma=discount) if algo else Policy_agent(states,P,gamma=discount)

    #Fitting policy
    print("fitting agent ...")
    start_time = time.clock()
    agent.fit(states,P)
    end_time = time.clock()
    print("Time elapsed during the fitting : {:.1e}".format(end_time - start_time)," sec.")

    #Displaying map if no intermediate rewards
    if 4 not in env.str2state(list(P.keys())[0]):
        print("close the tab to start the test phase")
        agent.show_map()

    #Testing agent policy
    env.seed()
    episode_count = 200
    reward = 0
    done = False
    rsum = 0

    print("testing the agent on ",episode_count," episodes :")
    for i in range(1,episode_count+1):
        obs = env.reset()
        j = 0
        rsum = 0
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        while True:
            action = agent.act(env.state2str(obs))
            obs, reward, done, _ = env.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render()
                time.sleep(0.2)
            if done:
                print("Episode : ",i," rsum=",round(rsum,2),", ",j," actions")
                break

    print("done")
    env.close()
