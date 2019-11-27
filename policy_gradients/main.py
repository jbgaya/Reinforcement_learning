import click
from agents import interface

@click.command()
@click.argument('environment')
@click.argument('algorithm')
@click.option(
    '--episodes', '-n',
    help='nb of episodes',
    type=click.INT
)
@click.option(
    '--minibatchs', '-K',
    help='minibatch size',
    type=click.INT
)
@click.option(
    '--epochs', '-ep',
    help='nb of epochs',
    type=click.INT
)
@click.option(
    '--alpha', '-a',
    help='generalized advantage estimation rate',
    type=click.FLOAT
)
@click.option(
    '--gamma', '-g',
    help='discount rate',
    type=click.FLOAT
)
@click.option(
    '--entropy', '-ent',
    help='entropy rate [0.01]',
    type=click.FLOAT
)
@click.option(
    '--beta', '-b',
    help='KL rate (for PPO_KL) [0.5]',
    type=click.FLOAT
)
@click.option(
    '--delta', '-d',
    help='KL treshold (for PPO_KL) [0.1]',
    type=click.FLOAT
)
@click.option(
    '--epsilon', '-e',
    help='clipping value (for PPO_Clipped) [0.2]',
    type=click.FLOAT
)

def main(environment,algorithm,episodes,minibatchs,epochs,
alpha,gamma,entropy,beta,delta,epsilon):
    """
    A little RL tool that helps to experiment quickly some policy gradient
    algorithms by adding hyperparameters in option.
    Default arguments are displayed in [ ].
    """
    #Initializing environment, agent and variables
    env,agent,episode_count = interface.agent_interface(environment,algorithm,episodes,minibatchs,epochs,
    alpha,gamma,entropy,beta,delta,epsilon)
    env.seed(0)
    reward = 0
    done = False
    rsum = 0
    test_episode_count = 10

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

if __name__ == "__main__":
    main()
