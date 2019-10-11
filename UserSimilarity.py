import numpy as np
from RatingMatrix import MovieLens_pivot


def COS(data):
    num_users = np.size(data, axis=0)
    simCOS = np.zeros((num_users,num_users))  # 0으로 초기화 된 행렬 생성
    for u in range(0, num_users):  # 각 유저별로 for문 반복
        arridx_u = np.where(data[u, ] == 0)
        for v in range(u + 1, num_users):
            arridx_v = np.where(data[v, ] == 0)
            arridx = np.unique(np.concatenate((arridx_u, arridx_v), axis=None))

            U = np.delete(data[u, ], arridx)
            V = np.delete(data[v, ], arridx)

            if np.linalg.norm(U) == 0 or np.linalg.norm(V) == 0:
                simCOS[u, v] = 0
            else:
                simCOS[u, v] = np.dot(U, V) / (np.linalg.norm(U) * np.linalg.norm(V))

            simCOS[v, u] = simCOS[u, v]

    return simCOS


def PCC(data):
    num_users = np.size(data, axis=0)
    simPCC = np.zeros((num_users,num_users))  # 0으로 초기화 된 행렬 생성
    mean = np.nanmean(np.where(data != 0, data, np.nan), axis=1)
    for u in range(0, num_users):
        arridx_u = np.where(data[u, ] == 0)
        for v in range(u + 1, num_users):
            arridx_v = np.where(data[v, ] == 0)
            arridx = np.unique(np.concatenate((arridx_u, arridx_v), axis=None))

            U = np.delete(data[u, ], arridx) - mean[u]
            V = np.delete(data[v, ], arridx) - mean[v]

            if np.linalg.norm(U) == 0 or np.linalg.norm(V) == 0:
                simPCC[u, v] = 0
            else:
                simPCC[u, v] = np.dot(U, V) / (np.linalg.norm(U) * np.linalg.norm(V))

            simPCC[v, u] = simPCC[u, v]

    return simPCC


if __name__ == "__main__":
    print("COS:", COS(MovieLens_pivot))
    print("PCC:", PCC(MovieLens_pivot))
