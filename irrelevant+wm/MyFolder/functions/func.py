def append_data(simul, capacity, load, trial, fractal, key_press, reward, df):
    import pandas as pd

    df_to_append = [
        simul,
        capacity,
        load,
        trial,
        fractal,
        key_press[0][0],
        reward,
    ]
    df_length = len(df)
    df.loc[df_length] = df_to_append
    df.subject = df.subject.astype(int)
    df.capacity = df.capacity.astype(int)
    df.trial = df.trial.astype(int)
    df.chosen_fractal = df.chosen_fractal.astype(int)
    df.key_press = df.key_press.astype(int)
    df.reward = df.reward.astype(int)
    df = df.round(2)
    return df


import pandas as pd
import numpy as np


def reward_previous(df, params):
    reward_previous = pd.Series([None])
    reward_previous = reward_previous.append(df.reward, ignore_index=True)
    df["reward_previous"] = reward_previous
    return df


def key_previous(df, params):
    key_previous = pd.Series([None])
    key_previous = key_previous.append(df.key_press, ignore_index=True)
    df["key_previous"] = key_previous
    return df


def stay(df, params):
    stay_key = df.key_press == df.key_previous
    df["stay_key"] = stay_key
    df.stay_key = df.stay_key.astype(float)
    return df


def calculate(df, params):
    loads = np.sort(df.load.unique())
    stay_differences = {}
    for load in loads:
        stay_rewarded = np.mean(
            df.stay_key[(df.reward_previous == 1) & (df.load == load)]
        )
        stay_unrewarded = np.mean(
            df.stay_key[(df.reward_previous == 0) & (df.load == load)]
        )
        stay_diff = stay_rewarded - stay_unrewarded
        stay_differences["load" + str(load)] = stay_diff
        print(stay_rewarded)
        print(stay_unrewarded)
    return stay_differences, loads


import numpy as np


def check_reward(params, choice):
    reward = np.random.choice(
        a=range(0, 2),
        p=[1 - params["reward_prob"][choice], params["reward_prob"][choice]],
    )  # every stimuli has a fixed prob for reward
    return reward


import numpy as np


def make_choice(
    params, capacity, Q_RL_frac, Q_RL_key, Q_RL_net, policy_RL, stimuli, set_size,
):
    # get a load factor 1,2,3 according to the set_size and the capacity
    WM_load_factor = capacity
    # updating the Qnet according to the Q_key and Q_frac
    Q_RL_net[stimuli[0], 0] = Q_RL_key[0] * WM_load_factor + Q_RL_frac[stimuli[0]]
    Q_RL_net[stimuli[1], 1] = Q_RL_key[1] * WM_load_factor + Q_RL_frac[stimuli[1]]
    # updating is for the key (0,1) and for the frac (stimuli[0],stimuli[1]) accordingly
    Q_to_policy_RL = np.array([Q_RL_net[stimuli[0], 0], Q_RL_net[stimuli[1], 1]])
    # agent calculates softmax prob to press each key for this given stimuli
    policy_RL = np.exp(params["beta"] * Q_to_policy_RL) / np.sum(
        np.exp(params["beta"] * Q_to_policy_RL)
    )
    # choose a fractal from the two options
    chosen_fractal = np.random.choice(a=stimuli, p=policy_RL)
    # find the location of the chosen fractal (0/1)
    key_press = np.where(stimuli == chosen_fractal)
    return key_press, chosen_fractal, WM_load_factor


def pick_stimuli(set_size):
    import numpy as np

    stimuli = np.random.choice(
        range(set_size), 2, replace=False
    )  # computer shows a stimuli of possible set_size
    return stimuli


import numpy as np


def update_Q_RL_frac(params, Q_RL_frac, reward, chosen_fractal):
    PE_RL_frac = (
        reward - Q_RL_frac[chosen_fractal]
    )  # update the PE for Qval of the key press for this stimuli
    Q_RL_frac[chosen_fractal] = (
        Q_RL_frac[chosen_fractal] + params["alpha_RL"] * PE_RL_frac
    )


def update_Q_RL_key(params, Q_RL_key, reward, key_press):
    PE_RL_key = (
        reward - Q_RL_key[key_press]
    )  # update the PE for Qval of the key press for this stimuli
    Q_RL_key[key_press] = Q_RL_key[key_press] + params["alpha_RL"] * PE_RL_key


def initialize(params, set_size):
    import numpy as np

    # key array (2 on 1)
    Q_RL_key = np.zeros((params["key_presses"], 1)) + 0.5
    # fractal array (2/4/6 on 1)
    Q_RL_frac = np.zeros((set_size, 1)) + 0.5
    # combined arrays (2/4/6 on 2)
    Q_RL_net = np.zeros((set_size, params["key_presses"])) + 0.5
    policy_RL = np.zeros((set_size, params["key_presses"])) + 0.5
    return (
        Q_RL_frac,
        Q_RL_key,
        Q_RL_net,
        policy_RL,
    )

