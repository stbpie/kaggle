# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 00:12:05 2023

@author: NADA
"""

# 출처 : https://www.kaggle.com/code/jhotor/catbooooostclassifier
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
'''데이터 로드'''
train = pd.read_csv('train.zip')
test = pd.read_csv('test.zip')
train.sample(10)

'''train, test split'''
# 테스트 크기를 25%로 설정하여 CatBoostClassifier 모델을 사용함.
X_train, X_test, y_train, y_test = train_test_split(train.drop(columns=['author', 'id']),
                                                    train['author'], test_size=0.25, stratify=train['author'])
# stratify=train['author']는 분할을 할 때 작가(author) 클래스의 분포를 유지하도록 지시하는 것
# 학습 및 테스트 세트 각각에 작가(author) 클래스의 비율이 동일하게 유지되어 더 나은 일반화 성능 얻을 수 있다.

'''catboost 모델 생성'''
model = CatBoostClassifier(text_features=['text'], random_state=1234, auto_class_weights='Balanced', loss_function='MultiClass',
                           task_type='GPU', devices="0:1")
# text_features: 모델에서 텍스트로 처리해야하는 열(column)을 지정. 여기에서는 'text' 열을 텍스트로 처리하도록 지정
# auto_class_weights: 모델 학습 시 자동으로 클래스 가중치를 계산하는 방법을 지정.
#                     'Balanced' 값을 지정하면 클래스 불균형이 있는 경우 자동으로 균형을 맞춤.
# loss_function: 모델의 손실 함수를 지정. 여기에서는 다중 클래스 분류 문제를 해결하기 위해 'MultiClass'를 사용.
# task_type: 모델이 실행되는 환경을 지정. 'GPU' 값을 지정하면 GPU를 사용하여 모델을 더 빠르게 실행.
# devices: 이 매개 변수는 모델이 실행되는 GPU 장치를 지정. 여기에서는 '0:1'을 지정하여 첫 번째 GPU 장치를 사용.

'''모델 학습'''
model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=25, verbose=100)
# eval_set: 검증 세트의 입력 및 출력을 나타냄. 학습 중에 모델이 얼마나 잘 수행되고 있는지 평가하는 데 사용.
# early_stopping_rounds: 조기 정지를 수행할 때 사용. 모델 학습 중 검증 손실이 개선되지 않는 경우
#                         정해진 숫자(여기에서는 25) 만큼의 에폭이 지나면 학습을 조기 종료함.



'''prediction'''
print(model.predict_proba(X_test)[0:5])
print(y_test[0:5])
# model.predict_proba(X_test)는 모델이 각 클래스(저자)에 속할 확률을 예측한 2차원 배열을 반환

'''정확도 탐색'''
predictions = model.predict(X_test)
# predictions 변수에는 각 샘플(문서)의 예측 결과가 2차원 배열의 형태로 저장된다.
predictions = [item for sublist in predictions for item in sublist]
# predictions는 각 샘플(문서)에 대한 예측 결과가 포함된 1차원 배열이 된다.

print(classification_report(y_test, predictions))
# y_test와 predictions를 인수로 받아, 정밀도(precision), 재현율(recall), f1-score, 지원 데이터 개수 등의 분류 지표를 출력
# 각 클래스(저자)에 대한 분류 성능을 쉽게 파악할 수 있다.
print(confusion_matrix(y_test, predictions))
# y_test와 predictions 받아, 실제 클래스와 예측 클래스의 조합에 따라 각 클래스에 대한 정확한 예측 개수를 표시하는 혼동 행렬을 출력
# 이를 통해 어떤 클래스가 잘 예측되었고, 어떤 클래스가 혼란스러웠는지를 파악할 수 있다.





