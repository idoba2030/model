# import
import numpy as np
import numpy.matlib as npmat
import random as rand
import matplotlib.pyplot as plt
import pdb
import pandas as pd
from initialize import initialize
from forget import forget_frac, forget_key, forget_combined
from correctness import check_reward
from pick_stimuli import pick_stimuli
from make_choice import make_choice
from update_Q_RL import update_Q_RL_frac, update_Q_RL_key, update_Q_RL_combined
from update_Q_WM import update_Q_WM_frac, update_Q_WM_key, update_Q_WM_combined
from append_data import append_data
from calculate import calculate
import matplotlib.pyplot as plt

###
# This is the main script of the model.
# The script simulates 3 different blocks which represent set-sizes varying from 2 to 6 (params[Ncards]).
# In every block we have 100 trials (params[trials]).
# The trials are replicated in 100 simulations (params[simulations])
# The model's decision making consists of a RL and WM part.
# Both the RL and the WM module learn about the key,the fractal and their combination.
# In each trial the computer draws two stimuli from the possible stimuli(Ncards) it has in this block (pick_stimuli())
# Then the agent makes a choice according to its learned or initialized (initialize()) Q_RL and Q_WM values.
# The agent calculates if the choice was correct (make_choice() in correctness file).
# The agent updates its Q_RL and Q_WM values accordingly (update_Q_WM,update_Q_R).
# Moreover, the agent forgets its learned Q_WM values in (forget()).
###
# define parameters
params = {
    "trials": 5,
    "beta": 8,  # inverse temp
    "phi": 0.1,  # decay param
    "epsilon": 0.05,  # error param
    "k": [2, 4, 6],  # capacity
    "rho": [0.8, 0.9],  # confidence in WM
    "reward": [0, 1],
    "reward_prob": [0.2, 0.5, 0.8, 0.2, 0.5, 0.8],
    "simulations": 1,
    "alpha_WM": 1,
    "alpha_RL": 0.1,
    "key_presses": 2,
    "Ncards": [2, 4, 6],
    "w_RLkey": [0.2, 0.3, 0.5],
    "w_RLcombined": [0.2, 0.3, 0.5],
    "w_WMkey": [0.2, 0.3, 0.5],
    "w_WMcombined": [0.2, 0.3, 0.5],
}
params1 = {
    "number_of_stimuli": params["Ncards"][-1],
    "number_of_blocks": len(params["Ncards"]),
}

results = {}
# initialize arrays
# calculate data for block
# calculate data for simulation

for block in params["Ncards"]:
    df = pd.DataFrame(
        columns=["simul", "trial", "chosen_fractal", "key_press", "rewards"]
    )
    uniform_policy = npmat.repmat(
        1 / params["key_presses"], block, params["key_presses"]
    )
    for simul in range(params["simulations"]):
        (
            Q_RL_frac,
            Q_RL_key,
            Q_RL_combined,
            Q_RL_net,
            policy_RL,
            Q_WM_frac,
            Q_WM_key,
            Q_WM_combined,
            Q_WM_net,
            policy_WM,
            uniform_policy,
        ) = initialize(params, params1, block)
        for t in range(params["trials"]):
            # calculate data for trial

            # pick a stimuli
            stimuli = pick_stimuli(block)
            # make a choice
            key_press, chosen_fractal = make_choice(
                params,
                params1,
                Q_RL_frac,
                Q_RL_key,
                Q_RL_combined,
                Q_RL_net,
                policy_RL,
                Q_WM_frac,
                Q_WM_key,
                Q_WM_combined,
                Q_WM_net,
                policy_WM,
                stimuli,
                block,
            )
            # check the outcome and the reward
            reward = check_reward(params, chosen_fractal)
            # update Qval_RL
            update_Q_RL_frac(params, Q_RL_frac, reward, chosen_fractal)
            update_Q_RL_key(params, Q_RL_key, reward, key_press)
            update_Q_RL_combined(
                params, Q_RL_combined, reward, key_press, chosen_fractal
            )
            # update Qval_WM
            update_Q_WM_frac(params, Q_WM_frac, reward, chosen_fractal)
            update_Q_WM_key(params, Q_WM_key, reward, key_press)
            update_Q_WM_combined(
                params, Q_WM_combined, reward, key_press, chosen_fractal
            )
            # forgetting
            forget_frac(params, Q_WM_frac, chosen_fractal, key_press, uniform_policy)
            forget_key(params, Q_WM_key, chosen_fractal, key_press, uniform_policy)
            forget_combined(
                params, Q_WM_combined, chosen_fractal, key_press, uniform_policy
            )
            # enter data into lists
            # if stimulus == 1:
            # append_data(simul, t, stimulus, choice,reward, df)
    # mean_of_block = calculate(df, params)
    # results[str(block)] = mean_of_block

    # plot the data
# plt.plot(results["2"][:17], label="ns2")
# plt.plot(results["3"][:17], label="ns3")
# plt.plot(results["4"][:17], label="ns4")
# plt.plot(results["5"][:17], label="ns5")
# plt.plot(results["6"][:17], label="ns6")

