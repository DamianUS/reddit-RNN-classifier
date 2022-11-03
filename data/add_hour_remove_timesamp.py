import pandas as pd
from datetime import datetime


def is_weekend(date_string):
    return int(datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S').weekday() in [5, 6])


original_df = pd.read_csv('reddit_2015.csv')

original_df['hour'] = original_df.apply(lambda row: datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S').hour, axis=1)
original_df.drop(['timestamp'], inplace=True, axis=1)
for hour in range(24):
    print(f'Number of rows at hour {hour}', len(original_df[original_df['hour'] == hour]))
original_df.to_csv('reddit_2015_labeled_hours.csv', header=False, index=False)
