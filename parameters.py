def define_params(params):
    ###
    # this function takes a dictionary of parameters and returns the values necessary from it
    ###
    set_sizes = params["ns"]  # set sizes
    number_of_blocks = len(set_sizes)  # number of blocks
    trials = params["trials"]  # number of trials to run in each block
    simulations = params["simulations"]  # the number of simulations [subjects]
    prob_for_double_reward = params["prob_2"]  # the possibilities for double rewards
    decay_param = params["phi"]  # paramater of counting on WM
    key_presses = params["key_presses"]  # number of options the subject has
    alpha_RL = params["alpha"]  # RL learning is 0.1
    alpha_WM = params["alpha_WM"]  # WM learning is 1
    capacity = params["k"]  # WM capacity can be 2/3
    confidence_WM = params["rho"]  # how much to rely on WM? 0.8/0.9
    noise = params["epsilon"]  # undirected noise parameter
    beta = params["beta"]  # inverse temperature

    return (
        set_sizes,
        number_of_blocks,
        trials,
        simulations,
        prob_for_double_reward,
        decay_param,
        key_presses,
        alpha_RL,
        alpha_WM,
        capacity,
        confidence_WM,
        noise,
        beta,
    )

