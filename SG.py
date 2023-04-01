# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 01:47:30 2023

@author: NADA
"""
# 출처 : https://www.kaggle.com/code/wjd2165/notebookd5a16077f9/edit

'''1번째 스터디'''
import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.shape)
train.head()
train.info()
train.describe()

print(test.shape)
test.head()
test.info()
test.describe()
'''1번째 단계 - bar''''
train.groupby('Pclass').mean() # groupby는 집단화다. Pclass별로 값이 나오는 것이다.
train.groupby('Pclass').mean()['Survived'] # survived만 추출
train.groupby('Pclass').mean()['Survived'].plot(kind='bar') # 좌석등급이 1등급인 승객들의 생존률은 약 60% 2등급은 약 45%
train.groupby('Pclass').mean()['Survived'].plot(kind='bar', rot=0) #이렇게하면 숫자 누워있는거 해결

'''2번째 - 히스토그램'''
train['Age'].plot(kind='hist') #Age에 대해 히스토그램 그렸다.
train['Age'].plot(kind='hist', bins=30) # 그래프가 더 촘촘해진다.
train['Age'].plot(kind='hist', bins=30, grid=True) # 보조선이 생긴다.

'''3번째 - 산점도 DF.plot(x,y,kind = scatter)'''
train.plot(x = 'Age', y = 'Fare', kind = 'scatter')

'''4번쨰 - 결측값 확인하기 isna()'''
train.isna() # 결측값 여부를 찾아준다.
train.isna().sum() # 컬럼별로 결측치 여부를 알려준다. 0인 곳이 결측치가 없는 것이다.

'''5번째 - 결측치 채우기 fillna()'''
train['Age'].median() # Age의 중앙값 확인
train['Age'] = train['Age'].fillna(28) # 중앙값으로 결측값 채워줌
train.isna().sum() # 결측치가 없음을 확인할 수 있음.

train['Embarked'].value_counts() # Embarked의 속성 별 개수 확인
train['Embarked'] = train['Embarked'].fillna('S') # Embarked의 결측값들을 최빈값으로 채워줌

'''6번째 - 시리즈 내의 값 변환하기 map()'''
train['Sex'] # Sex는 문자로 이루어짐
train['Sex'] = train['Sex'].map({'male':0, 'female':1}) # 남자는 0, 여자는 1로 바꿈

train.head() # 바뀐것들을 보자



'''3번째 스터디'''
'''라이브러리 불러오기'''
import warnings # warnings모듈 가져옴
warnings.filterwarnings('ignore') # 경고메시지를 무시하도록 설정
import numpy as np
import pandas as pd
# Data Visualisation
import matplotlib.pyplot as plt 
import seaborn as sns

'''파일 DF형태로 읽어오기'''
advertising = pd.DataFrame(pd.read_csv("advertising.csv"))
advertising.head()
advertising.shape
advertising.info()
advertising.describe()
advertising.isnull().sum()

'''데이터 전처리'''
 # 결측값이 없는것을 확인
advertising.isnull().sum()*100/advertising.shape[0]

 # 이상치 확인
fig, axs = plt.subplots(3, figsize = (5,5)) # Matplotlib의 subplots 함수를 사용하여 하나의 figure에 3개의 subplot을 생성하고
                                            # fig 변수에 figure 객체를, axs 변수에 subplot 객체의 배열을 할당
plt1 = sns.boxplot(advertising['TV'], ax = axs[0])
plt2 = sns.boxplot(advertising['Newspaper'], ax = axs[1])
plt3 = sns.boxplot(advertising['Radio'], ax = axs[2])
plt.tight_layout()
        # 특별한 이상치가 없음을 확인하였다.

'''일변량 분석'''
sns.boxplot(advertising['Sales']) # sales에 대한 박스플롯
plt.show()

# sales와 다른 변수들과의 관계를 pairplot으로 시각화해보자
sns.pairplot(advertising, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()
            # TV와 Sales의 그래프가 선형의 모습을 보인다.

# 상관계수 알아보자
sns.heatmap(advertising.corr(), cmap="YlGnBu", annot = True)
plt.show()
         # 상관계수에서 확인할 수 있듯 TV와 Sales의 상관관계가 높다.
     # 이제 TV를 특징변수로 사용하여 간단한 선형 회귀 분석을 해보자.


'''회귀분석'''
X = advertising['TV'] # tv를 특징변수, Sales를 반응변수로 설정
y = advertising['Sales']

'''Train - Test Split'''
# 이제 변수를 train, test set으로 분할한다.
# sklearn.model_selection 라이브러리에서 train_test_split을 가져와 수행한다.
# 일반적으로 데이터의 70%는 train, 30%는 test에 보관하는 것이 좋다.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)

X_train.head()
y_train.head()

'''선형 모델(Linear Model) 만들기'''
import statsmodels.api as sm

# 절편을 가져오기 위해 상수 추가
X_train_sm = sm.add_constant(X_train)

# X_train 데이터 세트에 상수를 추가한 후에는 아래와 같이 통계 모형의 
# OLS(일반 최소 제곱) 속성을 사용하여 회귀선을 적합시킬 수 있다

# OLS를 사용하여 회귀선 적합
lr = sm.OLS(y_train, X_train_sm).fit()

# 매개변수(예 적합된 회귀선의 절편 및 기울기) 확인
lr.params

# summary를 출력하면 적합된 회귀선의 다른 모든 매개 변수가 나열된다.
print(lr.summary())

'''summary의 결과 해석'''
# TV의 회귀계수가 0.054, 매우 낮은 P값 : 계수가 통계적으로 유의미하다.
                                     # 따라서 연관성은 우연이 하니다.
# 결정계수(R제곱)가 0.816 : 매출 변동의 81.6%가 TV에 의해 설명됨을 의미한다.

# F통계량의 P값이 매우 낮음 : 모형 적합치가 통계적으로 유의미함.
                            #설명된 분산이 우연이 아니라는 뜻.



'''시각화 해보기'''
# 위의 결과로부터 얻은 선형 회귀 방정식은 다음과 같다.
                               # 𝑆𝑎𝑙𝑒𝑠=6.948+0.054×𝑇𝑉
plt.scatter(X_train, y_train)
plt.plot(X_train, 6.948 + 0.054*X_train, 'r') # 회귀선을 그림
plt.show()


'''모델 평가'''
# 오차항이 정규 분포를 따르는지 확인해야 한다.(선형 회귀 분석의 가정 중 하나)
# 오차항의 히스토그램을 그려서 어떻게 보이는지 알아보자
y_train_pred = lr.predict(X_train_sm)
res = (y_train - y_train_pred)

fig = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 
plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
plt.show()
       # 오차항이 정규분포의 형태를 보이고 있다. good!


# 잔차항에 대한 scatter plot 그려보자
plt.scatter(X_train,res)
plt.show()
     # 잔차 항의 정규성을 통해 계수에 대한 추론이 가능합니다.
     # 그러나 X로 증가하는 잔차의 분산은 이 모형이 설명할 수 없는 상당한 변동이 있음을 나타냅니다.

'''Test Set에 대한 예측'''
이제 test data에 대해 몇 가지 예측을 수행할 차례입니다.
이를 위해 먼저 X_train에 대해 수행한 것처럼 X_test 데이터에 상수를 추가한 다음 
적합 회귀선의 예측 속성을 사용하여 X_test에 해당하는 y 값을 예측할 수 있습니다.


# X_test에 상수 추가하기
X_test_sm = sm.add_constant(X_test)
# X_test_sm에 해당하는 y 값 예측
y_pred = lr.predict(X_test_sm)

y_pred.head()

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# 평균오차제곱(RMSE)
np.sqrt(mean_squared_error(y_test, y_pred))

# 결정계수
r_squared = r2_score(y_test, y_pred)
r_squared

plt.scatter(X_test, y_test)
plt.plot(X_test, 6.948 + 0.054 * X_test, 'r')
plt.show()











