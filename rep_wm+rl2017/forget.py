def forget(params, params1, Q_WM, stimulus, choice):
    # phi is a decay parameter
    Q_WM[stimulus, choice] = Q_WM[stimulus, choice] + params["phi"] * (
        params1["uniform_policy"][choice] - Q_WM[stimulus, choice]
    )  # QWM +phi(Q0-QWM)
