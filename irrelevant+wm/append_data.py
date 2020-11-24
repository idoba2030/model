def append_data(simul, trial, stimulus, choice, outcome, reward, df):
    import pandas as pd

    df_to_append = [simul, trial, stimulus, choice, outcome, reward]
    df_length = len(df)
    df.loc[df_length] = df_to_append

    return df

