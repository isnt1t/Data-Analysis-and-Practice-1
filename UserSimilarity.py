import numpy as np
import pandas as pd

from RatingMatrix import MovieLens_pivot_NaN

# rating이 많은 사용자 1,000명 구하기
row_index = MovieLens_pivot_NaN.isnull().sum(axis = 1).sort_values() # NaN이 적은 순으로 행 뽑아내기
MovieLens_pivot_NaN = pd.DataFrame(MovieLens_pivot_NaN, index=row_index.index)
MovieLens_pivot_NaN = MovieLens_pivot_NaN.iloc[:1000,:]

# rating이 많은 영화 1,000개 구하기
col_index = MovieLens_pivot_NaN.isnull().sum(axis = 0).sort_values() # NaN이 적은 순으로 열 뽑아내기
MovieLens_pivot_NaN = pd.DataFrame(MovieLens_pivot_NaN, columns=col_index.index)
MovieLens_pivot_NaN = MovieLens_pivot_NaN.iloc[:,:1000]

user_index = MovieLens_pivot_NaN.index.values # User ID 기억하기
user_index = pd.Series(user_index) # Index rename 인자로 넣기 위해 형태 변경

MovieLens_pivot = MovieLens_pivot_NaN.fillna(0)
MovieLens_pivot = np.array(MovieLens_pivot) # 함수 인자로 넣기 위해 numpy array 형태로 변경


class UserSimilarity:
    def __init__(self, data):
        self.data = data

    # 1. Cosine Simiarity
    def COS(self):
        size = np.size(self.data, axis=0)
        simCOS = np.zeros(shape=(size, size))  # 0으로 초기화 된 행렬 생성

        for i in range(0, size):  # 각 유저별로 for문 반복
            for j in range(i, size):
                normI = np.linalg.norm(self.data[i,])  # i벡터의 크기 계산
                normJ = np.linalg.norm(self.data[j,])  # j벡터의 크기 계산
                inputData = np.dot(self.data[i,], self.data[j,]) / (normI * normJ)  # Cosine similarity 공식
                simCOS[i, j] = inputData  # 행렬에 계산 값 대입하기
                simCOS[j, i] = inputData  # 대각선 값 대입하기

        return simCOS  # 최종 행렬값 반환


    # 2. Pearson Correlation Coefficient
    def cal_PCC(self, i, j):  # 두 벡터의 PCC값을 계산해 주는 함수
        i = np.array(i)  # numpy array로 형변환
        j = np.array(j)  # numpy array로 형변환

        # 벡터의 0인 원소를 null값으로 바꾸고 null값을 제외한 평균값
        mean_i = np.nanmean(np.where(i != 0, i, np.nan))
        mean_j = np.nanmean(np.where(j != 0, j, np.nan))

        # 벡터의 원소가 0인 부분의 인덱스를 저장한 리스트 생성
        zero_i = np.where(i == 0)
        zero_j = np.where(j == 0)

        # zero_i와 zero_j 벡터를 하나로 합쳐준다.
        zeros = np.concatenate((zero_i, zero_j), axis=None)

        # 각 벡터의 원소가 0인 인덱스를 삭제
        del_i = np.delete(i, zeros)
        del_j = np.delete(j, zeros)

        # 각 벡터의 원소에서 평균값을 빼준다.
        del_i = del_i - mean_i
        del_j = del_j - mean_j

        return np.dot(del_i, del_j) / (np.linalg.norm(del_i) * np.linalg.norm(del_j))
        # PCC 공식을 통한 similarity 값 반환

    def PCC(self):
        size = np.size(self.data, axis=0)
        simPCC = np.zeros(shape=(size, size)) # 0으로 초기화 된 행렬 생성

        for i in range(0, size):
            for j in range(i, size):
                inputData = self.cal_PCC(self.data[i,], self.data[j,]) # cal_PCC함수를 이용
                simPCC[i, j] = inputData
                simPCC[j, i] = inputData

        return simPCC



    # 3. JMSD
    def MSD(self, data):
        size = np.size(data, axis=0)
        simMSD = np.zeros(shape=(size, size)) # 0으로 초기화 된 행렬 생성

        for i in range(0, size):
            for j in range(i, size):
                # 벡터의 원소가 0인 부분을 null값으로 변경
                nozero_i = np.where(data[i,] == 0, np.nan, data[i,])
                nozero_j = np.where(data[j,] == 0, np.nan, data[j,])

                # i벡터에서 j벡터의 각 원소를 빼고 제곱한 결과
                SquaredSum = np.square(nozero_i - nozero_j)

                # 제곱한 결과가 null값인 것을 제외
                SquaredSum = SquaredSum[~np.isnan(SquaredSum)]

                # MSD 공식에 따른 similarity 값
                AllItems = np.size(SquaredSum, axis=0)
                tmp = np.sum(SquaredSum) / AllItems

                # 행렬에 구한 similarity 값 대입
                simMSD[i, j] = tmp
                simMSD[j, i] = simMSD[i, j]

        return simMSD

    def JAC(self):
        size = np.size(self.data, axis=0)
        simJAC = np.zeros(shape=(size, size))  # 0으로 초기화 된 행렬 생성

        for i in range(0, size):
            for j in range(i, size):
                # 벡터의 원소가 0이상을 만족하는 인덱스만 1, 0이라면 0
                I = np.array(self.data[i,] > 0, dtype=np.int)
                J = np.array(self.data[j,] > 0, dtype=np.int)

                # I, J벡터의 합
                SumIJ = I + J

                # 합한 원소가 1보다 크면 교집합
                Inter = np.sum(np.array(SumIJ > 1, dtype=np.int))
                # 0보다 큰 모든 원소의 합은 합집합
                Union = np.sum(np.array(SumIJ > 0, dtype=np.int))

                # JAC 공식을 따른 similarity
                tmp = Inter / Union

                # 결과 값 simJAC행렬에 대입
                simJAC[i, j] = tmp
                simJAC[j, i] = simJAC[i, j]

        return simJAC

    def JMSD(self, max=5):
        return self.JAC() * (1 - (self.MSD(self.data/max))) # JMSD 공식에 따른 similarity, max는 최대 평가 값



COS_result = UserSimilarity(MovieLens_pivot).COS()
COS_result = pd.DataFrame(COS_result)  # 결과 행렬을 보기 쉽게 데이터프레임으로 변환
COS_result.rename(index=user_index, columns=user_index, inplace=True)  # User ID mapping

PCC_result = UserSimilarity(MovieLens_pivot).PCC()
PCC_result = pd.DataFrame(PCC_result)  # 결과 행렬을 보기 쉽게 데이터프레임으로 변환
PCC_result.rename(index=user_index, columns=user_index, inplace=True)  # User ID mapping

JMSD_result = UserSimilarity(MovieLens_pivot).JMSD()
JMSD_result = pd.DataFrame(JMSD_result)  # 결과 행렬을 보기 쉽게 데이터프레임으로 변환
JMSD_result.rename(index=user_index, columns=user_index, inplace=True)  # User ID mapping


if __name__ == "__main__":
    print(COS_result)
    print(PCC_result)
    print(JMSD_result)