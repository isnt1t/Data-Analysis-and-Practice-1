import warnings
warnings.filterwarnings("ignore")

import pandas as pd


'''
All ratings are contained in the file "ratings.dat" and are in the following format:

UserID::MovieID::Rating::Timestamp

UserIDs range between 1 and 6040
MovieIDs range between 1 and 3952
Ratings are made on a 5-star scale (whole-star ratings only)
Timestamp is represented in seconds since the epoch as returned by time(2)
Each user has at least 20 ratings
'''

# load data
MovieLens_df = pd.read_table('ratings.dat', sep='::', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

# drop unnecessary column
MovieLens_df.drop(['Timestamp'], axis=1, inplace=True)

# create user-item rating matirx
MovieLens_pivot = MovieLens_df.pivot_table(index='UserID', columns='MovieID', values='Rating')

# replace NaN to 0 & convert float to int
MovieLens_pivot.fillna(0, inplace=True)
MovieLens_pivot = MovieLens_pivot.astype('int64')

# check the matrix
print(MovieLens_pivot)
