# import
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import pdb
import pandas as pd
from initialize import initialize
from forget import forget
from correctness import check_correct
from pick_stimuli import pick_stimuli
from make_choice import make_choice
from update_Q_RL import update_Q_RL
from update_Q_WM import update_Q_WM
from append_data import append_data
from calculate import calculate
import matplotlib.pyplot as plt

# define parameters
params = {
    "trials": 100,
    "alpha": 0.1,
    "beta": 8,
    "phi": 0.1,
    "epsilon": 0.05,
    "k": np.arange(2, 7),
    "rho": [0.8, 0.9],
    "reward": [0, 1, 2],
    "ns": np.arange(2, 7),
    "prob_2": [0.2, 0.5, 0.8, 0.2, 0.5, 0.8],
    "simulations": 100,
    "alpha_WM": 1,
    "alpha_RL": 0.1,
    "key_presses": 3,
    "correct_answer": np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ),  # connect every stimulus with its correct key-press action
}
params1 = {
    "number_of_stimuli": params["ns"][-1],
    "number_of_blocks": len(params["ns"]),
    "uniform_policy": np.array(
        [
            1 / params["key_presses"],
            1 / params["key_presses"],
            1 / params["key_presses"],
        ]
    ),
}

results = {}
# initialize arrays
# calculate data for block
# calculate data for simulation

for block in params["ns"]:
    df = pd.DataFrame(
        columns=["simul", "trial", "stimuli", "choices", "outcomes", "rewards"]
    )
    for simul in range(params["simulations"]):
        Q_WM, Q_RL, policy_WM, policy_RL, uniform_policy = initialize(params, params1)
        for t in range(params["trials"]):
            # calculate data for trial

            # pick a stimuli
            stimulus = pick_stimuli(params1)
            # make a choice
            choice = make_choice(
                params, params1, Q_RL, Q_WM, policy_RL, policy_WM, stimulus, block
            )
            # check the outcome and the reward
            reward, outcome = check_correct(params, stimulus, choice)
            # update Qval_RL
            update_Q_RL(params, reward, stimulus, Q_RL, choice)
            # update Qval_WM
            update_Q_WM(params, outcome, stimulus, Q_WM, choice)
            # forgetting
            forget(params, params1, Q_WM, stimulus, choice)
            # enter data into lists
            if stimulus == 1:
                # into_lists(
                # stimulus, stimuli, choice, choices, reward, rewards, outcome, outcomes
                # )
                append_data(simul, t, stimulus, choice, outcome, reward, df)
    mean_of_block = calculate(df, params)
    results[str(block)] = mean_of_block
# plot the data
plt.plot(results["2"][:17], label="ns2")
plt.plot(results["3"][:17], label="ns3")
plt.plot(results["4"][:17], label="ns4")
plt.plot(results["5"][:17], label="ns5")
plt.plot(results["6"][:17], label="ns6")
plt.legend()
plt.show()
