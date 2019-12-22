from __future__ import (absolute_import, division, print_function, unicode_literals)

import os

import numpy as np
import pandas as pd

from surprise import Dataset
from surprise import Reader
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import SVD


# 0. Data Load - Movie lens 1M data

# -> 데이터가 없으면 아래 과정 수행 (Y해서 데이터 로드)
# data = Dataset.load_builtin('ml-1m')
# df = pd.DataFrame(data.raw_ratings, columns=["user", "item", "rate", "id"])

# -> 데이터가 있으면 아래 과정 수행
file_path = os.path.expanduser('~/.surprise_data/ml-1m/ml-1m/ratings.dat')
reader = Reader(line_format = 'user item rating timestamp', sep = '::')
data = Dataset.load_from_file(file_path, reader = reader)
df = pd.DataFrame(data.raw_ratings, columns = ['uid', 'iid', 'rate', 'timestamp'])


trainset = data.build_full_trainset()
sim_options = {'name' : 'cosine', 'user_based' : True}


class PredictedRating:
    def __init__(self, data, trainset):
        self.data = data
        self.trainset = trainset

        self.user = list(set(data.uid))
        self.item = list(set(data.iid))
        self.user_i = list(map(int, self.user))
        self.item_i = list(map(int, self.item))

        self.algorithm = None

    def setAlgorithm(self, algorithm):
        self.algorithm = algorithm

    def PredictedRating(self):
        algo = self.algorithm
        algo.fit(self.trainset)
        pred_rating = np.zeros((max(self.user_i) + 1, max(self.item_i) + 1))
        for u in self.user:
            iids = self.data[self.data.uid == u]
            for i in range(1, len(iids) + 1):
                iid = iids[i - 1:i].iid.values[0]
                r_ui = iids[i - 1:i].rate.values[0]
                pred = algo.predict(u, iid, r_ui, verbose=False)
                pred_rating[int(u)][int(iid)] = pred.est

        return pred_rating



PredictedRating = PredictedRating(data, trainset)
# 1. Basic CF algorithm
PredictedRating.setAlgorithm(KNNBasic(k = 40, min_k = 1, simoptions = sim_options))
PR_BasicCF = PredictedRating.PredictedRating()

# 2. CF algorithm with mean
PredictedRating.setAlgorithm(KNNWithMeans(k = 40, min_k = 1, simoptions = sim_options))
PR_CFwithMean =PredictedRating.PredictedRating()

# 3. CF algorithm with z-score
PredictedRating.setAlgorithm(KNNWithZScore(k = 40, min_k = 1, simoptions = sim_options))
PR_CFwithZ = PredictedRating.PredictedRating()

# 4. SVD
PredictedRating.setAlgorithm(SVD(n_factors = 100, n_epochs = 20, biased = False, lr_all = 0.005, reg_all = 0))
PR_SVD = PredictedRating.PredictedRating()

# 5. PMF
PredictedRating.setAlgorithm(SVD(n_factors = 100, n_epochs = 20, biased = False, lr_all = 0.005, reg_all = 0.02))
PR_PMF = PredictedRating.PredictedRating()

# 6. PMF with biased
PredictedRating.setAlgorithm(SVD(n_factors = 100, n_epochs = 20, biased = True, lr_all = 0.005, reg_all = 0.02))
PMFwithBias = PredictedRating.PredictedRating()


if __name__ == "__main__":
    print(PR_BasicCF)
    print(PR_CFwithMean)
    print(PR_CFwithZ)
    print(PR_SVD)
    print(PR_PMF)
    print(PMFwithBias)