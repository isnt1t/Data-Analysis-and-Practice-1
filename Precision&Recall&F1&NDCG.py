from __future__ import (absolute_import, division, print_function, unicode_literals)
from collections import defaultdict

import numpy as np
import pandas as pd

from surprise import Dataset
from surprise import KNNWithMeans
from surprise import SVD
from surprise.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')


# 0. Data Load - Movie lens 1M data
data = Dataset.load_builtin('ml-1m')
kf = KFold(n_splits = 5)
sim_options = {'name' : 'cosine', 'user_based' : True}


# 1. Precision & Recall & F1-measure
# https://github.com/NicolasHug/Surprise/blob/master/examples/precision_recall_at_k.py
def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls

def get_P_R_F(data, algo):
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k = 5, threshold = 4)

        P = sum(prec for prec in precisions.values()) / len(precisions)
        R = sum(rec for rec in recalls.values()) / len(recalls)
        F1 = 2 * P * R / (P + R)

        print('precision : ', P)
        print('recall : ', R)
        print('F1 : '  , F1)

# 1-1. CF with mean
CFwM_algo = KNNWithMeans(k = 40, min_k = 1, simoptions = sim_options)

# 1-2. SVD
SVD_algo = SVD(n_factors = 100, n_epochs = 20, biased = False, lr_all = 0.005, reg_all = 0)

# 1-3. PMF
PMF_algo = SVD(n_factors = 100, n_epochs = 20, biased = False, lr_all = 0.005, reg_all = 0.02)

# 1-4. PMF with bias
PMFwB_algo = SVD(n_factors = 100, n_epochs = 20, biased = True, lr_all = 0.005, reg_all = 0.02)


# 2. NDCG
df = pd.DataFrame(data.raw_ratings, columns = ['uid', 'iid', 'rate', 'timestamp'])
del df['timestamp']
user = list(set(df.uid))

def addRating(uid, pred_data):
    add = df[df.uid == uid]
    add['pred'] = 0.0
    for iid, i in zip(add.iid, add.index):
        add.set_value(i, 'pred', pred_data.iloc[int(uid) - 1][int(iid) + 1])

    return add

def DCG(data):
    pred_sort = list(data.sort_values(by=['pred'], axis=0, ascending=False).rate)
    dcg = pred_sort[0]
    for i in range(1, len(pred_sort)):
        dcg += pred_sort[i] / np.log2(i + 1)

    return dcg

def IDCG(data):
    rate_sort = list(data.sort_values(by=['rate'], axis=0, ascending=False).rate)
    idcg = rate_sort[0]
    for i in range(1, len(rate_sort)):
        idcg += rate_sort[i] / np.log2(i + 1)

    return idcg

def NDCG(uid, pred_data):
    table = addRating(uid, pred_data)
    dcg = DCG(table)
    idcg = IDCG(table)

    return dcg / idcg

def get_NDCG(user, pred_data):
    NDCG_ = {}
    for u in user:
        value = NDCG(u, pred_data)
        NDCG_[u] = value
    return NDCG_;

# Assignment#4에서 구한 예측 레이팅 불러오기
pred_rating_mean = pd.read_csv('pred_rating_mean.csv')
pred_rating_svd = pd.read_csv('pred_rating_svd.csv')
pred_rating_pmf = pd.read_csv('pred_rating_pmf.csv')
pred_rating_pmf_bi = pd.read_csv('pred_rating_pmf_bi.csv')

# 2-1. CF with mean
NDCG_CFwM = get_NDCG(user, pred_rating_mean)

# 2-2. SVD
NDCG_SVD = get_NDCG(user, pred_rating_svd)

# 2-3. PMF
NDCG_PMF = get_NDCG(user, pred_rating_pmf)

# 2-4. PMF WITH bias
NDCG_PMFwM = get_NDCG(user, pred_rating_pmf_bi)


if __name__ == "__main__":
    print(get_P_R_F(data, CFwM_algo))
    print(get_P_R_F(data, SVD_algo))
    print(get_P_R_F(data, PMF_algo))
    print(get_P_R_F(data, PMFwB_algo))

    print(NDCG_CFwM)
    print(NDCG_SVD)
    print(NDCG_PMF)
    print(NDCG_PMFwM)