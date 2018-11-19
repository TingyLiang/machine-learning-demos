import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression

# 使用线性回归拟合直线
# create data
train_x = np.arange(-100, 100, 1).reshape(200, 1)
train_y = 2 * train_x + 2
# print(train_y.reshape(200, 1))
# data = np.hstack((train_x, train_y))
plot.figure()
plot.xlabel("x")
plot.ylabel("y")
plot.plot(train_x, train_y, "")
plot.show()

# create model
model = LinearRegression()

model.fit(train_x, train_y)

print(model.score(train_x, train_y))
print(model.predict([[200]]))
