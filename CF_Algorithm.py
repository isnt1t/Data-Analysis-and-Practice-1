import numpy as np
from RatingMatrix import MovieLens_pivot
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
    b_u = mean_u - mean
    b_i = mean_i - mean

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
            # b_u[u] : user u의 baseline
            b_vi = mean + b_u[k_users[u]] + b_i[i]  # list
            # b_i[i] : item i의 baseline


            # calculation
            mom = np.sum(list_sim)  # 분모
            son = np.sum(list_sim * (list_rating - b_vi))  # 분자
            predicted_rating[u, i] = b_ui + son / mom

    return predicted_rating


'''
Optional(Extra point)
1) Item-based
기존의 basic_baseline 함수의 parameter 중 data 대신 data.T를 넣어준다.
'''


if __name__ == "__main__":
    print("** User-based & COS & k = 2 **\n", basic_baseline(MovieLens_pivot, 'COS', 2))
    print("** User-based & PCC & k = 2 **\n", basic_baseline(MovieLens_pivot, 'PCC', 2))

    print("** Item-based & COS & k = 2 **\n", basic_baseline(MovieLens_pivot.T, 'COS', 2))
    print("** Item-based & PCC & k = 2 **\n", basic_baseline(MovieLens_pivot.T, 'PCC', 2))