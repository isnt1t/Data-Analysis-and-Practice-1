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

# -> 데이터가 없으면 행아래 과정 수행 (Y해서 데이터 로드)
# data = Dataset.load_builtin('ml-1m')
# df = pd.DataFrame(data.raw_ratings, columns=["user", "item", "rate", "id"])

# -> 데이터가 있으면 아래 과정 수행
file_path = os.path.expanduser('~/.surprise_data/ml-1m/ml-1m/ratings.dat')
reader = Reader(line_format = 'user item rating timestamp', sep = '::')
data = Dataset.load_from_file(file_path, reader = reader)
df = pd.DataFrame(data.raw_ratings, columns = ['uid', 'iid', 'rate', 'timestamp'])

user = list(set(df.uid))
item = list(set(df.iid))
user_i = list(map(int, user))
item_i = list(map(int, item))

trainset = data.build_full_trainset()
sim_options = {'name' : 'cosine', 'user_based' : True}


# 1. Basic CF algorithm
algo = KNNBasic(k = 40, min_k = 1, simoptions = sim_options)
algo.fit(trainset)
pred_rating_basic = np.zeros((max(user_i) + 1, max(item_i) + 1))
for u in user:
    iids = df[df.uid == u]
    for i in range(1, len(iids) + 1):
        iid = iids[i - 1:i].iid.values[0]
        r_ui = iids[i - 1:i].rate.values[0]
        pred = algo.predict(u, iid, r_ui, verbose = False)
        pred_rating_basic[int(u)][int(iid)] = pred.est


# 2. CF algorithm with mean
algo = KNNWithMeans(k = 40, min_k = 1, simoptions = sim_options)
algo.fit(trainset)
pred_rating_mean = np.zeros((max(user_i) + 1, max(item_i) + 1))
for u in user:
    iids = df[df.uid == u]
    for i in range(1, len(iids) + 1):
        iid = iids[i - 1:i].iid.values[0]
        r_ui = iids[i - 1:i].rate.values[0]
        pred = algo.predict(u, iid, r_ui, verbose = False)
        pred_rating_mean[int(u)][int(iid)] = pred.est


# 3. CF algorithm with z-score
algo = KNNWithZScore(k = 40, min_k = 1, simoptions = sim_options)
algo.fit(trainset)
pred_rating_z = np.zeros((max(user_i) + 1, max(item_i) + 1))
for u in user:
    iids = df[df.uid == u]
    for i in range(1, len(iids) + 1):
        iid = iids[i - 1:i].iid.values[0]
        r_ui = iids[i - 1:i].rate.values[0]
        pred = algo.predict(u, iid, r_ui, verbose = False)
        pred_rating_z[int(u)][int(iid)] = pred.est


# 4. SVD
algo = SVD(n_factors = 100, n_epochs = 20, biased = False, lr_all = 0.005, reg_all = 0)
algo.fit(trainset)
pred_rating_svd = np.zeros((max(user_i) + 1, max(item_i) + 1))
for u in user:
    iids = df[df.uid == u]
    for i in range(1, len(iids) + 1):
        iid = iids[i - 1:i].iid.values[0]
        r_ui = iids[i - 1:i].rate.values[0]
        pred = algo.predict(u, iid, r_ui, verbose = False)
        pred_rating_svd[int(u)][int(iid)] = pred.est


# 5. PMF
algo = SVD(n_factors = 100, n_epochs = 20, biased = False, lr_all = 0.005, reg_all = 0.02)
algo.fit(trainset)
pred_rating_pmf = np.zeros((max(user_i) + 1, max(item_i) + 1))
for u in user:
    iids = df[df.uid == u]
    for i in range(1, len(iids) + 1):
        iid = iids[i - 1:i].iid.values[0]
        r_ui = iids[i - 1:i].rate.values[0]
        pred = algo.predict(u, iid, r_ui, verbose = False)
        pred_rating_pmf[int(u)][int(iid)] = pred.est


# 6. PMF with biased
algo = SVD(n_factors = 100, n_epochs = 20, biased = True, lr_all = 0.005, reg_all = 0.02)
algo.fit(trainset)
pred_rating_pmf_bi = np.zeros((max(user_i) + 1, max(item_i) + 1))
for u in user:
    iids = df[df.uid == u]
    for i in range(1, len(iids) + 1):
        iid = iids[i - 1:i].iid.values[0]
        r_ui = iids[i - 1:i].rate.values[0]
        pred = algo.predict(u, iid, r_ui, verbose = False)
        pred_rating_pmf_bi[int(u)][int(iid)] = pred.est


if __name__ == "__main__":
    print(pred_rating_basic)
    print(pred_rating_mean)
    print(pred_rating_z)
    print(pred_rating_svd)
    print(pred_rating_pmf)
    print(pred_rating_pmf_bi)