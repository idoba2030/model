def initialize(params, params1):
    import numpy as np

    choices = []  # arrays as integers so that indexing would be possible
    rewards = []
    stimuli = []
    outcomes = []
    Q_RL = (
        np.zeros((params1["number_of_stimuli"], params["key_presses"]))
        + 1 / params["key_presses"]
    )  # (number of stimuli, number of key presses) adding initial uniform Qval according to key presses
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

    return choices, rewards, stimuli, outcomes, Q_WM, Q_RL, uniform_policy

