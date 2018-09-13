import numpy as np
from sklearn import datasets
from math import sqrt
from sklearn.preprocessing import StandardScaler

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target

    X = X[y < 50]
    y = y[y < 50]

    return X, y

# 分割数据集
def data_split(X, y, seed=None):
    if seed:
        np.random.seed(seed)
    # 随机打乱数据集的下标顺序
    shuffled_indexes = np.random.permutation(len(X))
    train_ratio = 0.8
    train_size = int(len(X) * train_ratio)
    # 将数据集2-8分，前80%是train集，后20%是test集
    train_indexes = shuffled_indexes[:train_size]
    test_indexes = shuffled_indexes[train_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, y_train, X_test, y_test

# 最小二乘法
def least_square(data, y):

    X = np.hstack([np.ones((len(data), 1)), data])
    weight = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    return weight

def error_rate(weight, X, y):
    # X(nx3) weight(3x1)
    return X.T.dot(X.dot(weight) - y) / len(y)

def gradAscent(data, y):  # 梯度下降求最优参数
    alpha = 0.01  # 步长
    maxCycles = 1000  # 设置迭代的次数
    X = np.hstack([np.ones((len(data), 1)), data])
    weights = np.ones(X.shape[1])  # 设置初始的参数，并都赋默认值为1。注意这里权重以矩阵形式表示三个参数。3x1

    for k in range(maxCycles):
        error = error_rate(weights, X, y)
        weights = weights - alpha * error  # 迭代更新权重

    return weights

def random_grad(data, y):
    maxCycles = 1000  # 设置迭代的次数
    X = np.hstack([np.ones((len(data), 1)), data])
    weights = np.ones(X.shape[1])  # 设置初始的参数，并都赋默认值为1。注意这里权重以矩阵形式表示三个参数。3x1

    for j in range(maxCycles):
        dataIndex = list(range(len(data)))
        for i in range(len(data)):
            alpha = 2 / (1.0 + j + i) + 0.0001
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            vectSum = np.sum(X[randIndex] * weights)
            errors = y[randIndex] - vectSum
            weights = weights + alpha * errors * X[randIndex]
            del(dataIndex[randIndex])

    return weights


# 预测
def predict(weight, X_predict):
    X = np.hstack([np.ones((len(X_predict), 1)), X_predict])

    return X.dot(weight)

# MSE
def mean_squared_error(y, y_predict):

    return np.sum((y - y_predict) ** 2) / len(y)

# RMSE
def root_mean_squared_error(y, y_predict):

    return sqrt(mean_squared_error(y, y_predict))

# MAE
def mean_absolute_error(y, y_predict):

    return np.sum(np.absolute(y - y_predict)) / len(y)

# R Squared
def r_square_score(y, y_predict):

    return 1 - mean_squared_error(y, y_predict) / np.var(y)

def score(weight, X_predict, y):

    y_predict = predict(weight, X_predict)

    return r_square_score(y, y_predict)


if __name__ == "__main__":

    X, y = load_data()
    X_train, y_train, X_test, y_test = data_split(X, y, seed=666)

    # # 最小二乘法
    # weight = least_square(X_train, y_train)
    # print(score(weight, X_test, y_test))

    # 梯度下降法
    standardScaler = StandardScaler()
    standardScaler.fit(X_train)
    # 数据归一化
    X_train_standard = standardScaler.transform(X_train)
    weight = random_grad(X_train_standard, y_train)
    X_test_standard = standardScaler.transform(X_test)
    print(score(weight, X_test_standard, y_test))
