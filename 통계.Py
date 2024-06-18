import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.api as sm

부동산가격= pd.read_csv("주택매매가격변동률_20240618123927.csv", header=None)
결혼수 = pd.read_csv("결혼수_1.csv", header=None)

X = 부동산가격.iloc[1]
Y = 결혼수.iloc[4]


# 상관계수

#상관계수 = X.corr(Y)
#print(상관계수)

X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()

print(model.summary())

intercept, slope = model.params
print(f"Intercept (절편): {intercept}")
print(f"Slope (기울기): {slope}")
