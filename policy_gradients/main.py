import agents
from agents import A2C,PPO,interface

#Initializing environment, agent and variables
env,agent,episode_count = interface.agent_interface()
env.seed(0)
reward = 0
done = False
rsum = 0

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
