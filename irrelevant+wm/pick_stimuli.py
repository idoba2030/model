def pick_stimuli(block):
    import numpy as np

    stimuli = np.random.choice(
        range(block), 2, replace=False
    )  # computer shows a stimuli of possible set_size
    return stimuli
