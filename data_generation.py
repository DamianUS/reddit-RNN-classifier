import pandas as pd
import numpy as np

np.random.seed(42)

def get_reddit_labeled_weekends():
    original_df = pd.read_csv('data/reddit_2015_labeled_weekdays.csv', header=None)
    interactions = original_df.iloc[:, 0].to_numpy()
    labels = original_df.iloc[:, 1].to_numpy()
    #split interactions and labels in sequences of 24 hours
    splitted_interactions = np.split(interactions, 365)
    splitted_labels = np.split(labels, 365)
    return splitted_interactions, splitted_labels

def get_reddit_labeled_starting_hours(n_samples, sequence_length):
    np.random.seed(42)
    original_df = pd.read_csv('data/reddit_2015_labeled_hours.csv', header=None)
    idx = np.arange(original_df.shape[0]-sequence_length)
    np.random.shuffle(idx)
    interactions = original_df.iloc[:, 0].to_numpy()
    labels = original_df.iloc[:, 1].to_numpy()
    #split interactions and labels in sequences of 24 hours
    splitted_interactions = [interactions[index:index+sequence_length] for index in idx[:n_samples]]
    splitted_labels = [labels[index] for index in idx[:n_samples]]
    return splitted_interactions, splitted_labels