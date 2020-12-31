import pandas as pd


def append_data(simul, trial, stimulus, choice, outcome, reward, df):

    df_to_append = [simul, trial, stimulus, choice, outcome, reward]
    df_length = len(df)
    df.loc[df_length] = df_to_append

    return df


import pandas as pd
import numpy as np


def calculate(df, params):
    total_p_correct = np.zeros((params["simulations"], params["trials"]))
    for sim in range(params["simulations"]):
        df_simul = df[df.iloc[:, 0] == sim]
        encounters_stimuli1 = df_simul.stimuli.cumsum()
        count_correct_stimuli1 = df_simul.outcomes.cumsum()
        p_correct_stimuli1 = count_correct_stimuli1 / encounters_stimuli1
        total_p_correct[sim, : len(p_correct_stimuli1)] = p_correct_stimuli1
        total_p_correct[
            sim, len(p_correct_stimuli1) : params["trials"]
        ] = total_p_correct[sim, len(p_correct_stimuli1) - 1]
    return np.mean(total_p_correct, axis=0)


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


def forget(params, params1, Q_WM, stimulus, choice):
    # phi is a decay parameter
    Q_WM[stimulus, choice] = Q_WM[stimulus, choice] + params["phi"] * (
        params1["uniform_policy"][choice] - Q_WM[stimulus, choice]
    )  # QWM +phi(Q0-QWM)


def initialize(params, params1):
    import numpy as np

    Q_RL = (
        np.zeros((params1["number_of_stimuli"], params["key_presses"]))
        + 1 / params["key_presses"]
    )
    policy_RL = (
        np.zeros((params1["number_of_stimuli"], params["key_presses"]))
        + 1 / params["key_presses"]
    )
    policy_WM = (
        np.zeros((params1["number_of_stimuli"], params["key_presses"]))
        + 1 / params["key_presses"]
    )
    # (number of stimuli, number of key presses) adding initial uniform Qval according to key presses
    Q_WM = (
        np.zeros((params1["number_of_stimuli"], params["key_presses"]))
        + 1 / params["key_presses"]
    )  # (number of stimuli, number of key presses)

    uniform_policy = np.array(
        [
            1 / params["key_presses"],
            1 / params["key_presses"],
            1 / params["key_presses"],
        ]
    )

    return Q_WM, Q_RL, policy_WM, policy_RL, uniform_policy


import numpy as np


def make_choice(
    params, params1, Q_RL, Q_WM, policy_RL, policy_WM, stimulus, block, capacity
):
    policy_RL[stimulus] = np.exp(params["beta"] * Q_RL[stimulus]) / sum(
        np.exp(params["beta"] * Q_RL[stimulus])
    )  # agent calculates prob to press each key for this given stimuli based on RL
    policy_WM[stimulus] = np.exp(params["beta"] * Q_WM[stimulus]) / sum(
        np.exp(params["beta"] * Q_WM[stimulus])
    )  # agent calculates prob to press each key for this given stimuli based on WM (rho=confidence in WM, k=capacity)
    p_WM = params["rho"][0] * min(
        capacity / block, 1
    )  # agent calculates how much should can he rely on WM (his initial confidence on WM times his ability in this set-size)
    net_policy = (
        p_WM * policy_WM[stimulus] + (1 - p_WM) * policy_RL[stimulus]
    )  # integrating the two softmax policies into one according to p_WM.

    # adding slips of undirectednoise
    # net_policy_with_noise=noise*uniform_policy+(1-noise)*net_policy

    choice = np.random.choice(a=range(3), p=net_policy)
    return choice


def pick_stimuli(params1):
    import numpy as np

    stimulus = np.random.choice(
        a=range(params1["number_of_stimuli"])
    )  # computer shows a stimuli of possible set_size
    return stimulus


def update_Q_RL(params, reward, stimulus, Q_RL, choice):
    PE_RL = (
        reward - Q_RL[stimulus, choice]
    )  # update the PE for Qval of the key press for this stimuli
    Q_RL[stimulus, choice] = Q_RL[stimulus, choice] + params["alpha_RL"] * PE_RL


def update_Q_WM(params, outcome, stimulus, Q_WM, choice):
    PE_WM = (
        outcome - Q_WM[stimulus, choice]
    )  # update the PE for Qval of the key press for this stimuli
    Q_WM[stimulus, choice] = Q_WM[stimulus, choice] + params["alpha_WM"] * PE_WM
