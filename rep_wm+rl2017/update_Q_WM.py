def update_Q_WM(params, outcome, stimulus, Q_WM, choice):
    PE_WM = (
        outcome - Q_WM[stimulus, choice]
    )  # update the PE for Qval of the key press for this stimuli
    Q_WM[stimulus, choice] = Q_WM[stimulus, choice] + params["alpha_WM"] * PE_WM

