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

