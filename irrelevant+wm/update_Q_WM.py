def update_Q_WM_frac(params, Q_WM_frac, reward, chosen_fractal):
    PE_WM_frac = (
        reward - Q_WM_frac[chosen_fractal]
    )  # update the PE for Qval of the key press for this stimuli
    Q_WM_frac[chosen_fractal] = (
        Q_WM_frac[chosen_fractal] + params["alpha_WM"] * PE_WM_frac
    )


def update_Q_WM_key(params, Q_WM_key, reward, key_press):
    PE_WM_key = (
        reward - Q_WM_key[key_press]
    )  # update the PE for Qval of the key press for this stimuli
    Q_WM_key[key_press] = Q_WM_key[key_press] + params["alpha_WM"] * PE_WM_key


def update_Q_WM_combined(params, Q_WM_combined, reward, key_press, chosen_fractal):
    PE_WM_combined = (
        reward - Q_WM_combined[chosen_fractal, key_press]
    )  # update the PE for Qval of the key press for this stimuli
    Q_WM_combined[chosen_fractal, key_press] = (
        Q_WM_combined[chosen_fractal, key_press] + params["alpha_WM"] * PE_WM_combined
    )
