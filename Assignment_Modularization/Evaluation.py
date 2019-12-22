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

import PredictedRating


# 0. Data Load - Movie lens 1M data
data = Dataset.load_builtin('ml-1m')
kf = KFold(n_splits = 5)
sim_options = {'name' : 'cosine', 'user_based' : True}


# 1. Precision & Recall & F1-measure
class Precision_Recall_F1:
    def __init__(self, data, algo):
        self.data = data
        self.algo = algo

    def precision_recall_at_k(self, predictions, k=10, threshold=3.5):
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

    def P_R_F(self): # 최종적인 결과
        for trainset, testset in kf.split(self.data):
            self.algo.fit(trainset)
            predictions = self.algo.test(testset)
            precisions, recalls = self.precision_recall_at_k(predictions, k = 5, threshold = 4)

            P = sum(prec for prec in precisions.values()) / len(precisions)
            R = sum(rec for rec in recalls.values()) / len(recalls)
            F1 = 2 * P * R / (P + R)

            print('precision : ', P)
            print('recall : ', R)
            print('F1 : '  , F1)

# 1-1. CF with mean
CFwM_algo = KNNWithMeans(k = 40, min_k = 1, simoptions = sim_options)
PRF_CFwM = Precision_Recall_F1(data, CFwM_algo).P_R_F()

# 1-2. SVD
SVD_algo = SVD(n_factors = 100, n_epochs = 20, biased = False, lr_all = 0.005, reg_all = 0)
PRF_SVD = Precision_Recall_F1(data, SVD_algo).P_R_F()

# 1-3. PMF
PMF_algo = SVD(n_factors = 100, n_epochs = 20, biased = False, lr_all = 0.005, reg_all = 0.02)
PMF_SVD = Precision_Recall_F1(data, PMF_algo).P_R_F()

# 1-4. PMF with bias
PMFwB_algo = SVD(n_factors = 100, n_epochs = 20, biased = True, lr_all = 0.005, reg_all = 0.02)
PMFwB_SVD = Precision_Recall_F1(data, PMFwB_algo).P_R_F()


# 2. NDCG
df = pd.DataFrame(data.raw_ratings, columns = ['uid', 'iid', 'rate', 'timestamp'])
del df['timestamp']
user = list(set(df.uid))

class NDCG:
    def __init__(self, user, true_data, pred_data):
        self.user = user
        self.true_data = true_data
        self.pred_data = pred_data

    def addRating(self, uid):
        add = self.true_data[self.true_data.uid == uid] # 해당 사용자 행만 추출
        add['pred'] = 0.0 # 데이터프레임에 예측 레이팅 열 추가
        for iid, i in zip(add.iid, add.index):
            add.set_value(i, 'pred', self.pred_data.iloc[int(uid) - 1][int(iid) + 1])

        return add

    def DCG(self, data):
        pred_sort = list(data.sort_values(by=['pred'], axis=0, ascending=False).rate)
        # 예측 레이팅을 기준으로 내림차순으로 정렬한 실제 레이팅 리스트
        dcg = pred_sort[0] # 예측 레이팅이 가장 높은 아이템의 실제 레이팅 저장
        for i in range(1, len(pred_sort)): # 리스트 돌며
            dcg += pred_sort[i] / np.log2(i + 1) # 순서대로 가중치를 줄여가며 더해주기

        return dcg

    def IDCG(self, data):
        rate_sort = list(data.sort_values(by=['rate'], axis=0, ascending=False).rate)
        # 실제 레이팅을 기준으로 내림차순으로 정렬한 실제 레이팅 리스트
        idcg = rate_sort[0] # 실제 레이팅이 가장 높은 아이템의 실제 레이팅 저장
        for i in range(1, len(rate_sort)): # 리스트 돌며
            idcg += rate_sort[i] / np.log2(i + 1) # 순서대로 가중치를 줄여가며 더해주기

        return idcg

    def NDCG_oneUser(self, uid):  # 한 사용자의 NDCG
        table = self.addRating(uid)  # 예측 레이팅 열 추가한 데이터프레임
        dcg = self.DCG(table)
        idcg = self.IDCG(table)

        return dcg / idcg

    def NDCG_byUser(self): # 모든 사용자의(사용자 ID별) NDCG
        NDCG = {} # 딕셔너리 생성
        for u in self.user: # 각 유저를 돌며
            value = self.NDCG_oneUser(u) # NDCG 값 구하고
            NDCG[u] = value # key로는 user_id, value로는 NDCG 값 저장
        return NDCG;

    def NDCG(self): # 최종적인 결과
        return sum(self.NDCG_byUser().values()) / len(self.NDCG_byUser())

# Assignment#4에서 구한 예측 레이팅 불러오기
pred_rating_cf_mean = PredictedRating.PR_CFwithMean
pred_rating_svd = PredictedRating.PR_SVD
pred_rating_pmf = PredictedRating.PR_PMF
pred_rating_pmf_bias = PredictedRating.PMFwithBias


# 2-1. CF with mean
NDCG_CFwM = NDCG(user, df, pred_rating_cf_mean).NDCG()

# 2-2. SVD
NDCG_SVD = NDCG(user, df, pred_rating_svd).NDCG()

# 2-3. PMF
NDCG_PMF = NDCG(user, df, pred_rating_pmf).NDCG()

# 2-4. PMF WITH bias
NDCG_PMFwB = NDCG(user, df, pred_rating_pmf_bias).NDCG()


if __name__ == "__main__":
    print(PRF_CFwM)
    print(PRF_SVD)
    print(PMF_SVD)
    print(PMFwB_SVD)

    print(NDCG_CFwM)
    print(NDCG_SVD)
    print(NDCG_PMF)
    print(NDCG_PMFwB)