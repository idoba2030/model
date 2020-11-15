import numpy as np


def make_choice(params, params1, Q_RL, Q_WM, policy_RL, policy_WM, stimulus, block):
    policy_RL[stimulus] = np.exp(params["beta"] * Q_RL[stimulus]) / sum(
        np.exp(params["beta"] * Q_RL[stimulus])
    )  # agent calculates prob to press each key for this given stimuli based on RL
    policy_WM[stimulus] = np.exp(params["beta"] * Q_WM[stimulus]) / sum(
        np.exp(params["beta"] * Q_WM[stimulus])
    )  # agent calculates prob to press each key for this given stimuli based on WM (rho=confidence in WM, k=capacity)
    p_WM = params["rho"][0] * min(
        params["k"][4] / block, 1
    )  # agent calculates how much should can he rely on WM (his initial confidence on WM times his ability in this set-size)
    net_policy = (
        p_WM * policy_WM[stimulus] + (1 - p_WM) * policy_RL[stimulus]
    )  # integrating the two softmax policies into one according to p_WM.

    # adding slips of undirectednoise
    # net_policy_with_noise=noise*uniform_policy+(1-noise)*net_policy

    choice = np.random.choice(a=range(3), p=net_policy)
    return choice

