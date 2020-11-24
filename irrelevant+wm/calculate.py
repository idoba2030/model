import pandas as pd
import numpy as np


def calculate(df, params):
    total_p_correct = np.zeros((params["simulations"], params["trials"]))
    for sim in range(params["simulations"]):
        df_simul = df[df.iloc[:, 0] == sim]
        encounters_stimuli1 = df_simul.stimuli.cumsum()
        count_correct_stimuli1 = df_simul.outcomes.cumsum()
        p_correct_stimuli1 = count_correct_stimuli1 / encounters_stimuli1
        total_p_correct[sim, : len(p_correct_stimuli1)] = p_correct_stimuli1
        total_p_correct[
            sim, len(p_correct_stimuli1) : params["trials"]
        ] = total_p_correct[sim, len(p_correct_stimuli1) - 1]
    return np.mean(total_p_correct, axis=0)
