# import
import numpy as np
import numpy.matlib as npmat
import random as rand
import matplotlib.pyplot as plt
import pdb
import pandas as pd
from MyFolder.functions.func import *
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
    "trials": 200,
    "beta": 8,  # inverse temp
    "k": np.arange(0, 1, 0.2),  # capacity
    "reward": [0, 1],
    "reward_prob": [0.6, 0.4, 0.6, 0.4, 0.6, 0.4],
    "simulations": 5,
    "alpha_RL": 0.4,
    "key_presses": 2,
    "Ncards": [2],
    "prior_w_key": 1,
}
# add set_size to df
df = pd.DataFrame(
    columns=[
        "subject",
        "capacity",
        "load",
        "trial",
        "chosen_fractal",
        "key_press",
        "reward",
    ]
)
# new subject
for simul in range(params["simulations"]):
    # assigning random capacity and w_key for this subject
    capacity = params["k"][simul % 30]
    # new block
    for set_size in params["Ncards"]:
        (Q_RL_frac, Q_RL_key, Q_RL_net, policy_RL) = initialize(params, set_size)
        # new trial
        for t in range(params["trials"]):
            # pick a stimuli
            stimuli = pick_stimuli(set_size)
            # make a choice
            key_press, chosen_fractal, load = make_choice(
                params,
                capacity,
                Q_RL_frac,
                Q_RL_key,
                Q_RL_net,
                policy_RL,
                stimuli,
                set_size,
            )
            # check the outcome and the reward
            reward = check_reward(params, chosen_fractal)
            # update Qval_RL
            update_Q_RL_frac(params, Q_RL_frac, reward, chosen_fractal)
            update_Q_RL_key(params, Q_RL_key, reward, key_press)

            # enter data into lists
            # if stimulus == 1:
            append_data(
                simul, capacity, load, t, chosen_fractal, key_press, reward, df,
            )
            print(simul, set_size, t)
reward_previous(df, params)
key_previous(df, params)
stay(df, params)
stay_differences, loads = calculate(df, params)
# results[str(block)] = mean_of_block
print(stay_differences)
# plot the data
plt.plot(loads, list(stay_differences.values()), label=stay_differences.keys())
plt.ylabel("Stay_key|rewarded-Stay_key|unrewarded")
plt.xlabel("Working Memory Load")
plt.show()
# plt.plot(results["3"][:17], label="ns3")
# plt.plot(results["4"][:17], label="ns4")
# plt.plot(results["5"][:17], label="ns5")
# plt.plot(results["6"][:17], label="ns6")
