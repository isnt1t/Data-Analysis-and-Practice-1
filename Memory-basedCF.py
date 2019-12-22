import numpy as np
import pandas as pd

from UserSimilarity import MovieLens_pivot # rating이 많은 사용자 1,000명, 아이템 1,000개로 구성된 1000x1000 numpy array
from UserSimilarity import UserSimilarity

UserSimilarity = UserSimilarity(MovieLens_pivot)
COS = UserSimilarity.COS()
PCC = UserSimilarity.PCC()


class MemorybasedCF:
    def __init__(self, data, sim, k):
        self.data = data
        self.sim = sim
        self. k = k


    def CFwithBaseline(self):
        # initializing (1000, 1000) numpy array with zeros
        predicted_rating = np.zeros(self.data.shape)

        # calculating means
        mean = np.nanmean(np.where(self.data != 0, self.data, np.nan))  # the mean of all ratings
        mean_u = np.nanmean(np.where(self.data != 0, self.data, np.nan), axis=1)  # the mean of all users
        mean_i = np.nanmean(np.where(self.data != 0, self.data, np.nan), axis=0)  # the mean of all items

        # base user, item
        b_u = mean_u - mean # users' baseline
        b_i = mean_i - mean # items' baseline

        # selecting similarity function
        if self.sim == 'COS':
            sim = COS
        elif self.sim == 'PCC':
            sim = PCC

        # selecting top k users by sorting similarity array
        k_users = np.argsort(-sim)
        k_users = np.delete(k_users, np.s_[self.k:], 1)  # delete colomn under k

        # number of users with axis = 0 condition
        num_users = np.size(self.data, axis=0)
        num_items = np.size(self.data, axis=1)

        # calculating predicted ratings
        for u in range(0, num_users):
            list_sim = sim[u, k_users[u]]  # selecting top k similarity
            for i in range(0, num_items):
                list_rating = self.data[k_users[u], i].astype('float64')  # k users' ratings on item i

                b_ui = mean + b_u[u] + b_i[i]  # scalar
                # b_u[u]: user u의 baseline
                b_vi = mean + b_u[k_users[u]] + b_i[i]  # list
                # b_i[i]: item i의 baseline

                # calculation
                mom = np.sum(list_sim)  # 분모
                son = np.sum(list_sim * (list_rating - b_vi))  # 분자
                predicted_rating[u, i] = b_ui + son / mom

        return predicted_rating

# User-based & COS & k = 2
User_COS_k2 = pd.DataFrame(MemorybasedCF(MovieLens_pivot, 'COS', 2).CFwithBaseline())

# User-based & PCC & k = 2
User_PCC_k2 = pd.DataFrame(MemorybasedCF(MovieLens_pivot, 'PCC', 2).CFwithBaseline())

# Item-based & COS & k = 2
Item_COS_k2 = pd.DataFrame(MemorybasedCF(MovieLens_pivot.T, 'COS', 2).CFwithBaseline())

# Item-based & PCC & k = 2
Item_PCC_k2 = pd.DataFrame(MemorybasedCF(MovieLens_pivot.T, 'PCC', 2).CFwithBaseline())


if __name__ == "__main__":
    print(User_COS_k2)
    print(User_PCC_k2)
    print(Item_COS_k2)
    print(Item_PCC_k2)