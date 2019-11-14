import gym
import gridworld
from gym import wrappers, logger
import agents
from agents import A2C,PPO

#
episode_count = 500
test_episode_count = 10


#Initializing environment and variables
env = gym.make("CartPole-v1")
env.seed(0)
reward = 0
done = False
rsum = 0
d_in = env.observation_space.shape[0]
d_out = env.action_space.n

#Initializing agent
#agent = A2C.Batch_agent(d_in,d_out)
agent = PPO.Clip_agent(d_in,d_out)

#Training phase
print("Starting training phase on ",episode_count," episodes :")
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
            if (i % 10 == 0):
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
            print("Episode : " + str(i) + " rsum=" + str(round(rsum,2)) + ", " + str(j) + " actions")
            break
env.close()
