import pandas as pd
from datetime import datetime


def is_weekend(date_string):
    return int(datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S').weekday() in [5, 6])

original_df = pd.read_csv('reddit_2015.csv')

original_df['is_weekend'] = original_df.apply(lambda row: is_weekend(row[0]), axis=1)
original_df.drop(['timestamp'], inplace=True, axis=1)
print("weekend rows", len(original_df[original_df['is_weekend'] == 1]))
print("weekday rows", len(original_df[original_df['is_weekend'] == 0]))
original_df.to_csv('reddit_2015_labeled_weekdays.csv', header=False, index=False)
