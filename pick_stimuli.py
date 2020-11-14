def pick_stimuli(params1):
    import numpy as np

    stimulus = np.random.choice(
        a=range(params1["number_of_stimuli"])
    )  # computer shows a stimuli of possible set_size
    return stimulus
