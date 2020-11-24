import numpy as np


def make_choice(
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
):
    Q_RL_net[stimuli[0], 0] = (
        Q_RL_key[0] * params["w_RLkey"][0]
        + Q_RL_frac[stimuli[0]] * (1 - params["w_RLkey"][0] - params["w_RLcombined"][1])
        + Q_RL_combined[stimuli[0], 0] * params["w_RLcombined"][1]
    )
    Q_RL_net[stimuli[1], 1] = (
        Q_RL_key[1] * params["w_RLkey"][0]
        + Q_RL_frac[stimuli[1]] * (1 - params["w_RLkey"][0] - params["w_RLcombined"][1])
        + Q_RL_combined[stimuli[1], 1] * params["w_RLcombined"][1]
    )
    Q_WM_net[stimuli[0], 0] = (
        Q_WM_key[0] * params["w_WMkey"][0]
        + Q_WM_frac[stimuli[0]] * (1 - params["w_WMkey"][0] - params["w_WMcombined"][1])
        + Q_WM_combined[stimuli[0], 0] * params["w_WMcombined"][1]
    )
    Q_WM_net[stimuli[1], 1] = (
        Q_WM_key[1] * params["w_WMkey"][0]
        + Q_WM_frac[stimuli[1]] * (1 - params["w_WMkey"][0] - params["w_WMcombined"][1])
        + Q_WM_combined[stimuli[1], 1] * params["w_WMcombined"][1]
    )
    Q_to_policy_RL = np.array([Q_RL_net[stimuli[0], 0], Q_RL_net[stimuli[1], 1]])
    Q_to_policy_WM = np.array([Q_WM_net[stimuli[0], 0], Q_WM_net[stimuli[1], 1]])
    policy_RL = np.exp(params["beta"] * Q_to_policy_RL) / sum(
        np.exp(params["beta"] * Q_to_policy_RL)
    )
    # agent calculates prob to press each key for this given stimuli based on RL
    policy_WM = np.exp(params["beta"] * Q_to_policy_WM) / sum(
        np.exp(params["beta"] * Q_to_policy_WM)
    )  # agent calculates prob to press each key for this given stimuli based on WM (rho=confidence in WM, k=capacity)
    p_WM = params["rho"][0] * min(
        params["k"][0] / block, 1
    )  # agent calculates how much should can he rely on WM (his initial confidence on WM times his ability in this set-size)
    net_policy = (
        p_WM * policy_WM + (1 - p_WM) * policy_RL
    )  # integrating the two softmax policies into one according to p_WM.

    # adding slips of undirectednoise
    # net_policy_with_noise=noise*uniform_policy+(1-noise)*net_policy

    # choose a fractal from the two options
    chosen_fractal = np.random.choice(a=stimuli, p=net_policy)
    # find the location of the chosen fractal (0/1)
    key_press = np.where(stimuli == chosen_fractal)
    return key_press, chosen_fractal

