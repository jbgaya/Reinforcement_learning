import click
import gym
from gym import wrappers, logger
from agents.A2C import *
from agents.PPO import *

def agent_interface(environment,algorithm,episodes,minibatchs,epochs,
alpha,gamma,entropy,beta,delta,epsilon):
    click.clear()
    env_list = ["CartPole-v1", "LunarLander-v2"]
    algorithm_list = ["A2C_Online","A2C_Batch","PPO_Clipped","PPO_KL"]
    print("------------------------------")
    if(environment not in env_list):
        environment = click.prompt(
            "- environment ",
            default="CartPole-v1",
            type=click.Choice(env_list, case_sensitive=False)
            )
    else:
        print(f"- environment : {environment}")
    if(algorithm not in algorithm_list):
        algorithm = click.prompt(
            "- algorithm ",
            default="PPO_Clipped",
            type=click.Choice(algorithm_list, case_sensitive=False)
            )
    else:
        print(f"- algorithm : {algorithm}")

    if episodes == None:
        episodes = click.prompt(
            "- nb of episodes ",
            default=10000,
            type=click.INT
            )
    else:
        print(f"- nb of episodes : {episodes}")
    if alpha == None:
        alpha = click.prompt(
            "- alpha ",
            default=0.97,
            type=click.FLOAT
            )
    else:
        print(f"- alpha : {alpha}")
    if gamma == None:
        gamma = click.prompt(
            "- gamma ",
            default=0.9999,
            type=click.FLOAT
            )
    else:
        print(f"- gamma : {gamma}")

    env = gym.make(environment)
    d_in = env.observation_space.shape[0]
    d_out = env.action_space.n

    if algorithm == "A2C_Online":
        agent = Online_agent(d_in,d_out,layers_V=[200],layers_f=[200],
        alpha=alpha,gamma=gamma,device=torch.device("cpu"))

    elif algorithm == "A2C_Batch":
        agent = Batch_agent(d_in,d_out,layers_V=[200],layers_f=[200],
        alpha=alpha,gamma=gamma,device=torch.device("cpu"))

    elif algorithm == "PPO_Clipped":
        if minibatchs == None:
            minibatchs = click.prompt(
                "- minimum samples before update: ",
                default=100,
                type=click.INT
                )
        else:
            print(f"- minibatch size : {minibatchs}")
        if epochs == None:
            epochs = click.prompt(
                "- nb of epochs ",
                default=50,
                type=click.INT
                )
        else:
            print(f"- nb of epochs : {epochs}")
        if epsilon == None:
            epsilon = click.prompt(
                "- epsilon ",
                default=0.2,
                type=click.FLOAT
                )
        else:
            print(f"- epsilon: {epsilon}")
        if entropy == None:
            entropy = click.prompt(
                "- entropy ",
                default=0.01,
                type=click.FLOAT
                )
        else:
            print(f"- entropy: {entropy}")
        agent = Clip_agent(d_in,d_out,layers_V=[200],layers_f=[200],
        K=minibatchs,epochs=epochs,alpha=alpha,gamma=gamma,
        epsilon=epsilon,entropy=entropy,device=torch.device("cpu"))

    elif algorithm == "PPO_KL":
        if minibatchs == None:
            minibatchs = click.prompt(
                "- minibatch size ",
                default=10,
                type=click.INT
                )
        else:
            print(f"- minibatch size : {minibatchs}")
        if epochs == None:
            epochs = click.prompt(
                "- nb of epochs ",
                default=50,
                type=click.INT
                )
        else:
            print(f"- nb of epochs : {epochs}")
        if beta == None:
            epochs = click.prompt(
                "- beta ",
                default=0.5,
                type=click.FLOAT
                )
        else:
            print(f"- beta: {beta}")
        if delta == None:
            epochs = click.prompt(
                "- delta ",
                default=1.,
                type=click.FLOAT
                )
        else:
            print(f"- delta: {delta}")
        if entropy == None:
            epochs = click.prompt(
                "- entropy ",
                default=0.01,
                type=click.FLOAT
                )
        else:
            print(f"- entropy: {entropy}")
        agent = KL_agent(d_in,d_out,layers_V=[200],layers_f=[200],
        K=minibatchs,epochs=epochs,alpha=alpha,beta=beta,delta=delta,
        gamma=gamma,entropy=entropy,device=torch.device("cpu"))

    print("------------------------------")
    return env,agent,episodes
