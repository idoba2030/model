def initialize(params, params1, block):
    import numpy as np

    # key arrays (2 on 1)
    Q_WM_key = np.zeros((params["key_presses"], 1)) + 0.5
    Q_RL_key = np.zeros((params["key_presses"], 1)) + 0.5
    # fractal arrays (2/4/6 on 1)
    Q_WM_frac = np.zeros((block, 1)) + 0.5
    Q_RL_frac = np.zeros((block, 1)) + 0.5
    # combined arrays (2/4/6 on 2)
    Q_WM_combined = np.zeros((block, params["key_presses"])) + 0.5
    uniform_policy = np.zeros((block, params["key_presses"])) + 0.5
    policy_WM = np.zeros((block, params["key_presses"])) + 0.5
    Q_WM_net = np.zeros((block, params["key_presses"])) + 0.5
    Q_RL_net = np.zeros((block, params["key_presses"])) + 0.5
    Q_RL_combined = np.zeros((block, params["key_presses"])) + 0.5
    policy_RL = np.zeros((block, params["key_presses"])) + 0.5
    return (
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
    )

