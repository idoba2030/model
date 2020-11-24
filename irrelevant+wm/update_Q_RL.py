def update_Q_RL_frac(params, Q_RL_frac, reward, chosen_fractal):
    PE_RL_frac = (
        reward - Q_RL_frac[chosen_fractal]
    )  # update the PE for Qval of the key press for this stimuli
    Q_RL_frac[chosen_fractal] = (
        Q_RL_frac[chosen_fractal] + params["alpha_RL"] * PE_RL_frac
    )


def update_Q_RL_key(params, Q_RL_key, reward, key_press):
    PE_RL_key = (
        reward - Q_RL_key[key_press]
    )  # update the PE for Qval of the key press for this stimuli
    Q_RL_key[key_press] = Q_RL_key[key_press] + params["alpha_RL"] * PE_RL_key


def update_Q_RL_combined(params, Q_RL_combined, reward, key_press, chosen_fractal):
    PE_RL_combined = (
        reward - Q_RL_combined[chosen_fractal, key_press]
    )  # update the PE for Qval of the key press for this stimuli
    Q_RL_combined[chosen_fractal, key_press] = (
        Q_RL_combined[chosen_fractal, key_press] + params["alpha_RL"] * PE_RL_combined
    )
