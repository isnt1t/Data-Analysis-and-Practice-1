import numpy as np
import pandas as pd

from RatingMatrix import MovieLens_pivot # rating이 많은 사용자 1,000명, 아이템 1,000개로 구성된 1000x1000 numpy array
from UserSimilarity import COS
from UserSimilarity import PCC

def basic_baseline(data, sim, k):
    # initializing (1000, 1000) numpy array with zeros
    predicted_rating = np.zeros(data.shape)

    # calculating means
    mean = np.nanmean(np.where(data != 0, data, np.nan))  # the mean of all ratings
    mean_u = np.nanmean(np.where(data != 0, data, np.nan), axis=1)  # the mean of all users
    mean_i = np.nanmean(np.where(data != 0, data, np.nan), axis=0)  # the mean of all items

    # base user, item
    b_u = mean_u - mean # users' baseline
    b_i = mean_i - mean # items' baseline

    # selecting similarity function
    if sim == 'COS':
        sim = COS(data)
    elif sim == 'PCC':
        sim = PCC(data)

    # selecting top k users by sorting similarity array
    k_users = np.argsort(-sim)
    k_users = np.delete(k_users, np.s_[k:], 1)  # delete colomn under k

    # number of users with axis = 0 condition
    num_users = np.size(data, axis=0)
    num_items = np.size(data, axis=1)

    # calculating predicted ratings
    for u in range(0, num_users):
        list_sim = sim[u, k_users[u]]  # selecting top k similarity
        for i in range(0, num_items):
            list_rating = data[k_users[u], i].astype('float64')  # k users' ratings on item i

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
User_COS_k2 = pd.DataFrame(basic_baseline(MovieLens_pivot, 'COS', 2))

# User-based & PCC & k = 2
User_PCC_k2 = pd.DataFrame(basic_baseline(MovieLens_pivot, 'PCC', 2))

# Item-based & COS & k = 2
Item_COS_k2 = pd.DataFrame(basic_baseline(MovieLens_pivot.T, 'COS', 2))

# Item-based & PCC & k = 2
Item_PCC_k2 = pd.DataFrame(basic_baseline(MovieLens_pivot.T, 'PCC', 2))


if __name__ == "__main__":
    print(User_COS_k2)
    print(User_PCC_k2)
    print(Item_COS_k2)
    print(Item_PCC_k2)