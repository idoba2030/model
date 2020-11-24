def forget_frac(params, Q_WM_frac, chosen_fractal, key_press, uniform_policy):
    # phi is a decay parameter
    Q_WM_frac[chosen_fractal] = Q_WM_frac[chosen_fractal] + params["phi"] * (
        uniform_policy[chosen_fractal, key_press] - Q_WM_frac[chosen_fractal]
    )  # QWM +phi(Q0-QWM)


def forget_key(params, Q_WM_key, chosen_fractal, key_press, uniform_policy):
    # phi is a decay parameter
    Q_WM_key[key_press] = Q_WM_key[key_press] + params["phi"] * (
        uniform_policy[chosen_fractal, key_press] - Q_WM_key[key_press]
    )  # QWM +phi(Q0-QWM)


def forget_combined(params, Q_WM_combined, chosen_fractal, key_press, uniform_policy):
    # phi is a decay parameter
    Q_WM_combined[chosen_fractal, key_press] = Q_WM_combined[
        chosen_fractal, key_press
    ] + params["phi"] * (
        uniform_policy[chosen_fractal, key_press]
        - Q_WM_combined[chosen_fractal, key_press]
    )  # QWM +phi(Q0-QWM)
