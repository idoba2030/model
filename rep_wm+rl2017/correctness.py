import numpy as np


def check_correct(params, stimulus, choice):
    if (
        params["correct_answer"][stimulus, choice] == 1
    ):  # it means there is 1 in this column - so you are right - now reward(RL) is 1/2 and outcome(WM) is 1
        reward = np.random.choice(
            a=range(1, 3),
            p=[1 - params["prob_2"][stimulus], params["prob_2"][stimulus],],
        )  # every stimuli has a fixed 2_prob
        outcome = 1  # this is for WM
    else:
        outcome = 0
        reward = 0
    return reward, outcome

