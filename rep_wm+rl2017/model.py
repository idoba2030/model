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

###
# This is the main script of the model.
# The script simulates 5 different blocks which represent set-sizes varying from 2 to 6 (params[ns]).
# In every block we have 100 trials (params[trials]).
# The trials are replicated in 100 simulations (params[simulations])
# The model's decision making consists of a RL and WM part.
# In each trial the computer draws a stimulus from the possible stimuli it has in this block (pick_stimuli())
# Then the agent makes a choice according to its learned or initialized (initialize()) Q_RL and Q_WM values.
# The agent calculates if the choice was correct (make_choice() in correctness file).
# The agent updates its Q_RL and Q_WM values accordingly (update_Q_WM,update_Q_R).
# Moreover, the agent forgets its learned Q_WM values in (forget()).
# The model gathers the results regarding stimulus=1 which appears in every set_size and has a 0.5 probability for double reward.
# The model registers the data about stimulus1 from every trial and simulation in (append_data()).
# The model calculates the percentchage of correct decisions about stimulus 1 in every block in (calculate()).
# Finally, the model enters the calculated data into a dictionary and plots it.
###
# define parameters
params = {
    "trials": 100,
    "alpha": 0.1,
    "beta": 8,
    "phi": 0.1,
    "epsilon": 0.05,
    "k": [2, 3, 6],
    "rho": [0.8, 0.9],
    "reward": [0, 1, 2],
    "ns": np.arange(2, 7),
    "prob_2": [0.2, 0.5, 0.8, 0.2, 0.5, 0.8],
    "simulations": 300,
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
for capacity in params["k"]:
    for block in params["ns"]:
        df = pd.DataFrame(
            columns=["simul", "trial", "stimuli", "choices", "outcomes", "rewards"]
        )
        for simul in range(params["simulations"]):
            Q_WM, Q_RL, policy_WM, policy_RL, uniform_policy = initialize(
                params, params1
            )
            for t in range(params["trials"]):
                # calculate data for trial

                # pick a stimuli
                stimulus = pick_stimuli(params1)
                # make a choice
                choice = make_choice(
                    params,
                    params1,
                    Q_RL,
                    Q_WM,
                    policy_RL,
                    policy_WM,
                    stimulus,
                    block,
                    capacity,
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
    fig, ax1, ax2, ax3 = plt.subplots(3, 1)
    ax1.plot(results["2"][:17], label="ns2")
    ax1.plot(results["3"][:17], label="ns3")
    ax1.plot(results["4"][:17], label="ns4")
    ax1.plot(results["5"][:17], label="ns5")
    ax1.plot(results["6"][:17], label="ns6")
    ax1.legend()
    ax1.show()
    ax2.plot(results["2"][:17], label="ns2")
    ax2.plot(results["3"][:17], label="ns3")
    ax2.plot(results["4"][:17], label="ns4")
    ax2.plot(results["5"][:17], label="ns5")
    ax2.plot(results["6"][:17], label="ns6")
    ax2.legend()
    ax2.show()
    ax3.plot(results["2"][:17], label="ns2")
    ax3.plot(results["3"][:17], label="ns3")
    ax3.plot(results["4"][:17], label="ns4")
    ax3.plot(results["5"][:17], label="ns5")
    ax3.plot(results["6"][:17], label="ns6")
    ax3.legend()
    ax3.show()
