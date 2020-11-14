import pandas as pd
import numpy as np


def calculate(df, df1, params):
    for sim in range(params["simulations"]):
        df_simul = df[df.iloc[:, 0] == sim]
        encounters_stimuli1 = df_simul.stimuli.cumsum()
        count_correct_stimuli1 = df_simul.outcomes.cumsum()
        p_correct_stimuli1 = count_correct_stimuli1 / encounters_stimuli1
        df_to_append = [
            encounters_stimuli1.tolist(),
            count_correct_stimuli1.tolist(),
            p_correct_stimuli1.tolist(),
        ]
        df1_length = len(df1)
        df1.loc[df1_length] = df_to_append
