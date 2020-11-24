def update_Q_RL(params, reward, stimulus, Q_RL, choice):
    PE_RL = (
        reward - Q_RL[stimulus, choice]
    )  # update the PE for Qval of the key press for this stimuli
    Q_RL[stimulus, choice] = Q_RL[stimulus, choice] + params["alpha_RL"] * PE_RL
