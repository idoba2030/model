import numpy as np


def check_reward(params, choice):
    reward = np.random.choice(
        a=range(0, 2),
        p=[1 - params["reward_prob"][choice], params["reward_prob"][choice]],
    )  # every stimuli has a fixed prob for reward
    return reward

