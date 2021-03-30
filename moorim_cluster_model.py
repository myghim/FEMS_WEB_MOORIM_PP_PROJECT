# 1. 목적 :
# 2. 목표 :
# 3. 특징 :
# 4. 대상 :
# 5. 기대효과 :
# --------------------------------------------------------------------------------------------------------

# 환경 : Anaconda Python 3.7
# 식별자 : 스네이크 표기법
# 라이브러리
# numpy :
# pandas :
# scikit-learn :
# time :
# random :

# --------------------------------------------------------------------------------------------------------

# 변수
# train_data : 학습용 데이터
# test_data : 테스트 데이터

# 라이브러리
import numpy as np
import pandas as pd
import sklearn as sk
import time
import random

class Cluster:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    # K-MEANS
    def model(self):
        train_data = self.train_data
        test_data = self.test_data

        # K-Means 모델 생성
        k_means = sk.cluster.KMeans() # 하이퍼파라미터 조절
        result = k_means.predict(train_data)

        # 하이퍼파라미터 최적화
        # exhaustive grid search

        # randomized parameter optimization

        # searching for optimal parameters with successive halving


        return result

    # KNN
    def model_2(self):
        result = self.train_data
        result_2 = self.test_data
        return result

    # LDA
    def model_3(self):
        result = self.train_data
        result_2 = self.test_data
        return result

    # Agglomerative Clustering
    def model_4(self):
        result = self.train_data
        result_2 = self.test_data
        return result

call = Cluster(3,3)
print(call.model())
print(call.model_2())
print(call.model_3())
print(call.model_4())
