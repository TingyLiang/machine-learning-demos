import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

# 使用线性回归拟合直线
# create data
train_x = np.arange(-100, 100, 1).reshape(200, 1)
train_y = 2 * train_x + 2
# print(train_y.reshape(200, 1))
# data = np.hstack((train_x, train_y))
plot.figure()
plot.xlabel("x")
plot.ylabel("y")
# plot.plot(train_x, train_y, "")
# plot.show()

# create model
model = LinearRegression()

model.fit(train_x, train_y)

print(model.score(train_x, train_y))
print(model.predict([[200]]))
joblib.dump(model, "line.m")

# 使用线性回归拟合曲线,应该不适用
# x = np.arange(-100, 100, 1).reshape(200, 1)
# y = x ** 2 + 4 * x + 1
# model1 = LinearRegression()
# model1.fit(x, y)
# plot.plot(x, y)
# plot.show()
# print(model1.score(x, y))
# print(model1.predict([[101]]))
