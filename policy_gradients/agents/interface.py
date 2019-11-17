from __future__ import print_function, unicode_literals
from PyInquirer import prompt, print_json
from PyInquirer import style_from_dict, Token, prompt, Separator,Validator,ValidationError
from agents.A2C import *
from agents.PPO import *
import gym
from gym import wrappers, logger

class floatValidator(Validator):
    def validate(self, document):
        try:
            float(document.text)
        except ValueError:
            raise ValidationError(
                message='/!\ must be a positive float',
                cursor_position=len(document.text))  # Move cursor to end
        if float(document.text)<0 :
            raise ValidationError(
                message='/!\ must be a positive float',
                cursor_position=len(document.text))  # Move cursor to end

class intValidator(Validator):
    def validate(self, document):
        try:
            int(document.text)
        except ValueError:
            raise ValidationError(
                message='/!\ must be a strictly positive int',
                cursor_position=len(document.text))  # Move cursor to end
        if int(document.text)<=0 :
            raise ValidationError(
                message='/!\ must be a strictly positive int',
                cursor_position=len(document.text))  # Move cursor to end

questions = [
    {
        'type': 'list',
        'name': 'environment',
        'message': 'Choose the environment :',
        'choices': [
            {
                'name': "CartPole-v1"
            },
            {
                'name': "LunarLander-v2"
            }
            ]
    },
    {
        'type': 'list',
        'name': 'agent',
        'message': 'Choose the policy gradient algorithm :',
        'choices': [
            {
                'name': "A2C Online version"
            },
            {
                'name': 'A2C Batch version'
            },
            {
                'name': 'PPO Clipped version'
            },
            {
                'name': 'PPO Kullback Leibler version'
            }
            ]
    }]

questions2 = {
"A2C Online version":[
    {
        'type': 'input',
        'name': 'episode_count',
        'message': 'nb of episodes=',
        'validate': intValidator,
        'filter': lambda val: int(val)
    },
    {
        'type': 'input',
        'name': 'K',
        'message': 'minibatchs=',
        'validate': intValidator,
        'filter': lambda val: int(val)
    },
    {
        'type': 'input',
        'name': 'epochs',
        'message': 'epochs=',
        'validate': intValidator,
        'filter': lambda val: int(val)
    },
    {
        'type': 'input',
        'name': 'alpha',
        'message': 'alpha=',
        'validate': floatValidator,
        'filter': lambda val: float(val)
    },
    {
        'type': 'input',
        'name': 'gamma',
        'message': 'gamma=',
        'validate': floatValidator,
        'filter': lambda val: float(val)
    }
],
"A2C Batch version":[
    {
        'type': 'input',
        'name': 'episode_count',
        'message': 'nb of episodes=',
        'validate': intValidator,
        'filter': lambda val: int(val)
    },
    {
        'type': 'input',
        'name': 'K',
        'message': 'minibatchs=',
        'validate': intValidator,
        'filter': lambda val: int(val)
    },
    {
        'type': 'input',
        'name': 'epochs',
        'message': 'epochs=',
        'validate': intValidator,
        'filter': lambda val: int(val)
    },
    {
        'type': 'input',
        'name': 'alpha',
        'message': 'alpha=',
        'validate': floatValidator,
        'filter': lambda val: float(val)
    },
    {
        'type': 'input',
        'name': 'gamma',
        'message': 'gamma=',
        'validate': floatValidator,
        'filter': lambda val: float(val)
    }
],
"PPO Clipped version":[
    {
        'type': 'input',
        'name': 'episode_count',
        'message': 'nb of episodes=',
        'validate': intValidator,
        'filter': lambda val: int(val)
    },
    {
        'type': 'input',
        'name': 'K',
        'message': 'minibatchs=',
        'validate': intValidator,
        'filter': lambda val: int(val)
    },
    {
        'type': 'input',
        'name': 'epochs',
        'message': 'epochs=',
        'validate': intValidator,
        'filter': lambda val: int(val)
    },
    {
        'type': 'input',
        'name': 'alpha',
        'message': 'alpha=',
        'validate': floatValidator,
        'filter': lambda val: float(val)
    },
    {
        'type': 'input',
        'name': 'gamma',
        'message': 'gamma=',
        'validate': floatValidator,
        'filter': lambda val: float(val)
    },
    {
        'type': 'input',
        'name': 'epsilon',
        'message': 'epsilon=',
        'validate': floatValidator,
        'filter': lambda val: float(val)
    },
    {
        'type': 'input',
        'name': 'entropy',
        'message': 'entropy=',
        'validate': floatValidator,
        'filter': lambda val: float(val)
    }
],
"PPO Kullback Leibler version":[
    {
        'type': 'input',
        'name': 'episode_count',
        'message': 'nb of episodes=',
        'validate': intValidator,
        'filter': lambda val: int(val)
    },
    {
        'type': 'input',
        'name': 'K',
        'message': 'minibatchs=',
        'validate': intValidator,
        'filter': lambda val: int(val)
    },
    {
        'type': 'input',
        'name': 'epochs',
        'message': 'epochs=',
        'validate': intValidator,
        'filter': lambda val: int(val)
    },
    {
        'type': 'input',
        'name': 'alpha',
        'message': 'alpha=',
        'validate': floatValidator,
        'filter': lambda val: float(val)
    },
    {
        'type': 'input',
        'name': 'beta',
        'message': 'beta=',
        'validate': floatValidator,
        'filter': lambda val: float(val)
    },
    {
        'type': 'input',
        'name': 'delta',
        'message': 'delta=',
        'validate': floatValidator,
        'filter': lambda val: float(val)
    },
    {
        'type': 'input',
        'name': 'gamma',
        'message': 'gamma=',
        'validate': floatValidator,
        'filter': lambda val: float(val)
    },
    {
        'type': 'input',
        'name': 'entropy',
        'message': 'entropy=',
        'validate': floatValidator,
        'filter': lambda val: float(val)
    }
]
}

def agent_interface():
    answers = prompt(questions)
    params = prompt(questions2[answers["agent"]])

    episode_count = params["episode_count"]
    env = gym.make(answers["environment"])
    d_in = env.observation_space.shape[0]
    d_out = env.action_space.n

    if answers["agent"] == "A2C Online version":
        agent = Online_agent(d_in,d_out,layers_V=[200],layers_f=[200],
        alpha=params["alpha"],gamma=params["gamma"],device=torch.device("cpu"))

    elif answers["agent"] == "A2C Batch version":
        agent = Batch_agent(d_in,d_out,layers_V=[200],layers_f=[200],
        alpha=params["alpha"],gamma=params["gamma"],device=torch.device("cpu"))

    elif answers["agent"] == "PPO Clipped version":
        agent = Clip_agent(d_in,d_out,layers_V=[200],layers_f=[200],
        K=params["K"],epochs=params["epochs"],alpha=params["alpha"],epsilon=params["epsilon"],gamma=params["gamma"],entropy=params["entropy"],device=torch.device("cpu"))

    elif answers["agent"] == "PPO Kullback Leibler version":
        agent = KL_agent(d_in,d_out,layers_V=[200],layers_f=[200],
        K=params["K"],epochs=params["epochs"],alpha=params["alpha"],beta=params["beta"],delta=params["delta"],gamma=params["gamma"],entropy=params["entropy"],device=torch.device("cpu"))

    return env,agent,episode_count
