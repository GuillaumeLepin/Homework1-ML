import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path, encoding="latin-1", sep=";")
    return df

def remove_extreme_stations(df):
    return df[~df["station name"].isin(["Mont Ventoux", "Pic du Midi"])]