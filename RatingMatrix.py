import numpy as np
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
MovieLens_df = pd.read_table('ratings.dat', sep='::', engine='python', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

# drop unnecessary column
MovieLens_df.drop(['Timestamp'], axis=1, inplace=True)

# create user-item rating matrix
MovieLens_pivot = MovieLens_df.pivot_table(index='UserID', columns='MovieID', values='Rating')

# rating이 많은 사용자 1,000명
row_index = MovieLens_pivot.isnull().sum(axis = 1).sort_values() # NaN이 적은 순으로 행 뽑아내기
MovieLens_pivot = pd.DataFrame(MovieLens_pivot, index=row_index.index)
MovieLens_pivot = MovieLens_pivot.iloc[:1000, :]

# rating이 많은 영화 1,000개
col_index = MovieLens_pivot.isnull().sum(axis = 0).sort_values() # NaN이 적은 순으로 열 뽑아내기
MovieLens_pivot = pd.DataFrame(MovieLens_pivot, columns=col_index.index)
MovieLens_pivot = MovieLens_pivot.iloc[:, :1000]

MovieLens_pivot.fillna(0, inplace=True)
MovieLens_pivot = MovieLens_pivot.astype('int64')
MovieLens_pivot = np.array(MovieLens_pivot) # 함수 인자로 넣기 위해 numpy array 형태로 변경

# check the matrix
if __name__ == "__main__":
    print(MovieLens_pivot)
