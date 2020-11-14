### Read
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import pdb
import pandas as pd

# @title Define Paramaters
# Define paramaters: trials=number of simulations alpha = learning rate, beta=softmax inverse temprature (exploit/explore),
# phi=decay of WM, epsilon= slips of undirected noise, k=WM capacity, rho=confidence in WM
# reward=possible outcomes, ns= number of stimulus in set, prob_2= the probability to get reward of 2, if correct.
params = {
    "trials": 100,
    "alpha": 0.1,
    "beta": 8,
    "phi": 0.1,
    "epsilon": 0.05,
    "k": [2, 3],
    "rho": [0.8, 0.9],
    "reward": [0, 1, 2],
    "ns": np.arange(2, 3),
    "prob_2": [0.2, 0.5, 0.8, 0.2, 0.5, 0.8],
    "simulations": 3,
    "alpha_WM": 1,
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

p_correct = np.zeros((ns[-1], trials, len(ns), simulations))
encounters = np.zeros((ns[-1], len(ns), simulations)).astype(
    int
)  # (number of stimuli, number of trials,number of blocks,number of simulations)
count_correct = np.zeros((ns[-1], len(ns), simulations))
p_correct_total = np.zeros((ns[-1], trials, len(ns), simulations))

# helper functions

for simul in range(simulations):
    # initiating
    # every stimulus has a fixed prob for double reward
    correct_answer = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )  # connect every stimulus with its correct key-press action

    # WM learning is very fast

    # how many key_presses are there? (A1_A2_A3)
    uniform_policy = np.array([1 / key_presses, 1 / key_presses, 1 / key_presses])

    # for loop for every set size (2,3,4,5,6)
    for set_size in ns:
        choices = np.zeros((trials)).astype(
            int
        )  # arrays as integers so that indexing would be possible
        rewards = np.zeros((trials)).astype(int)
        stimuli = np.zeros((trials)).astype(int)
        outcomes = np.zeros((trials)).astype(int)
        Q_RL = (
            np.zeros((set_size, key_presses)) + 1 / key_presses
        )  # (number of stimuli, number of key presses) adding initial uniform Qval according to key presses
        Q_WM = (
            np.zeros((set_size, key_presses)) + 1 / key_presses
        )  # (number of stimuli, number of key presses)

        # for loop for every trial
        for t in range(trials):
            stimuli[t] = np.random.choice(
                a=range(Q_RL.shape[0])
            )  # computer shows a stimuli of possible set_size
            policy_RL = np.exp(beta * Q_RL[stimuli[t]]) / sum(
                np.exp(beta * Q_RL[stimuli[t]])
            )  # agent calculates prob to press each key for this given stimuli based on RL
            policy_WM = np.exp(beta * Q_WM[stimuli[t]]) / sum(
                np.exp(beta * Q_WM[stimuli[t]])
            )  # agent calculates prob to press each key for this given stimuli based on WM
            p_WM = confidence_WM[0] * min(
                capacity[0] / set_size, 1
            )  # agent calculates how much should can he rely on WM (his initial confidence on WM times his ability in this set-size)
            net_policy = (
                p_WM * policy_WM + (1 - p_WM) * policy_RL
            )  # integrating the two softmax policies into one according to p_WM.
            # net_policy_with_noise=noise*uniform_policy+(1-noise)*net_policy #adding slips of undirectednoise
            choices[t] = np.random.choice(
                a=range(3), p=net_policy
            )  # agent picks a keypress A1/2/3 - this is constant in every condition - according to his integrated policy
            # we update the number of encounters with this specific stimuli for this set_size(block) in this simul
            encounters[
                stimuli[t], set_size - 2, simul
            ] += 1  # count the encounters with each stimuli in each trial for every set size
            # now we should check if the choice the agent made is like in the correct_answer array
            if (
                correct_answer[stimuli[t], choices[t]] == 1
            ):  # it means there is 1 in this column - so you are right - now reward(RL) is 1/2 and outcome(WM) is 1
                rewards[t] = np.random.choice(
                    a=range(1, 3),
                    p=[
                        1 - prob_for_double_reward[stimuli[t]],
                        prob_for_double_reward[stimuli[t]],
                    ],
                )  # every stimuli has a fixed 2_prob
                outcomes[t] = 1  # this is for WM
                count_correct[
                    stimuli[t], set_size - 2, simul
                ] += 1  # we count correct answers to check the p_correct
            else:
                rewards[t] = 0
                outcomes[t] = 0
            encounters_current_stimuli = encounters[
                stimuli[t], set_size - 2, simul
            ]  # what encounter is it for this stimuli
            # the current p_correct for this stimuli is its count_correct divided by the encounters until now.
            p_correct[
                stimuli[t], encounters_current_stimuli - 1, set_size - 2, simul
            ] = (
                count_correct[stimuli[t], set_size - 2, simul]
                / encounters[stimuli[t], set_size - 2, simul]
            )
            # p_correct[:,0:t+1,set_size-2,simul]=np.cumsum(count_correct[:,0:t+1,set_size-2],axis=1)/np.cumsum(encounters[:,0:t+1,set_size-2],axis=1) #check the cumulative p_correct for every stimuli, according to encounters the agent had with it
            # p_correct=np.nan_to_num(p_correct,nan=1/3)
            PE_RL = (
                rewards[t] - Q_RL[stimuli[t], choices[t]]
            )  # update the PE for Qval of the key press for this stimuli
            PE_WM = (
                outcomes[t] - Q_WM[stimuli[t], choices[t]]
            )  # update the PE for Qval of the key press for this stimuli
            Q_RL[stimuli[t], choices[t]] = (
                Q_RL[stimuli[t], choices[t]] + alpha_RL * PE_RL
            )
            Q_WM[stimuli[t], choices[t]] = (
                Q_WM[stimuli[t], choices[t]] + alpha_WM * PE_WM
            )
            # forgetting
            Q_WM[stimuli[t], choices[t]] = Q_WM[
                stimuli[t], choices[t]
            ] + decay_param * (
                Q_WM[stimuli[t], choices[t]] * 0
                + 1 / key_presses
                - Q_WM[stimuli[t], choices[t]]
            )  # QWM +phi(Q0-QWM),
    p_correct_total = p_correct_total + p_correct
p_correct_avg = np.sum(p_correct_total, axis=3) / simulations

# @title Capacity = 2
plt.figure(figsize=(20, 10))
plt.plot(
    np.mean(np.cumsum(encounters[0:2, :, 0], axis=1), axis=0),
    np.mean(p_correct_avg[0:2, :, 0], axis=0),
    label="ns2",
)
plt.plot(
    np.mean(np.cumsum(encounters[0:3, :, 1], axis=1), axis=0),
    np.mean(p_correct_avg[0:3, :, 1], axis=0),
    label="ns3",
)
plt.plot(
    np.mean(np.cumsum(encounters[0:4, :, 2], axis=1), axis=0),
    np.mean(p_correct_avg[0:4, :, 2], axis=0),
    label="ns4",
)
plt.plot(
    np.mean(np.cumsum(encounters[0:5, :, 3], axis=1), axis=0),
    np.mean(p_correct_avg[0:5, :, 3], axis=0),
    label="ns5",
)
plt.plot(
    np.mean(np.cumsum(encounters[0:6, :, 4], axis=1), axis=0),
    np.mean(p_correct_avg[0:6, :, 4], axis=0),
    label="ns6",
)
plt.legend()
plt.ylabel("p_correct")
plt.xlabel("encounters")
plt.axis([0, 16, 0, 1])
plt.show()

# @title Capacity = 3
plt.figure(figsize=(20, 10))
plt.plot(
    np.mean(np.cumsum(encounters, axis=1)[0:1, :, 0], axis=0),
    np.mean(p_correct_avg[0:2, :, 0], axis=0),
    label="ns2",
)
plt.plot(
    np.mean(np.cumsum(encounters, axis=1)[0:2, :, 1], axis=0),
    np.mean(p_correct_avg[0:3, :, 1], axis=0),
    label="ns3",
)
plt.plot(
    np.mean(np.cumsum(encounters, axis=1)[0:3, :, 2], axis=0),
    np.mean(p_correct_avg[0:4, :, 2], axis=0),
    label="ns4",
)
plt.plot(
    np.mean(np.cumsum(encounters, axis=1)[0:4, :, 3], axis=0),
    np.mean(p_correct_avg[0:5, :, 3], axis=0),
    label="ns5",
)
plt.plot(
    np.mean(np.cumsum(encounters, axis=1)[0:5, :, 4], axis=0),
    np.mean(p_correct_avg[0:6, :, 4], axis=0),
    label="ns6",
)
plt.legend()
plt.ylabel("p_correct")
plt.xlabel("encounters")
plt.axis([0, 16, 0, 1])
plt.show()
