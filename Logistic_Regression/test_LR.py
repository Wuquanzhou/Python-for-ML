import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# filename = 'E:\\ML_data\\LR_data.txt'  # 文件目录
# def loadDataSet():     # 读取数据（这里只有两个特征）
#     dataMat = []
#     labelMat = []
#     fr = open(filename)
#     for line in fr.readlines():
#         lineArr = line.strip().split()
#         # print(line.strip().split())
#         dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])   # 前面的1，表示方程的常量。比如两个特征X1,X2，共需要三个参数，W1+W2*X1+W3*X2
#         labelMat.append(int(lineArr[-1]))
#     fr.close()
#     return dataMat, labelMat

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

def sigmoid(x):  # sigmoid函数
    return 1.0 / (1 + np.exp(-x))

#计算当前损失函数值
def error_function(weight, X, y):

    y_predict = sigmoid(X.dot(weight))

    return -np.sum(y * np.log(y_predict) + (1-y) * np.log(1-y_predict)) / len(y)

def error_rate(weight, X, y):
    # X(nx3) weight(3x1)
    return X.T.dot(sigmoid(X.dot(weight)) - y) / len(y)

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
            alpha = 4 / (1.0 + j + i) + 0.0001
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            vectSum = np.sum(X[randIndex] * weights)
            grad = sigmoid(vectSum)
            errors = y[randIndex] - grad
            weights = weights + alpha * errors * X[randIndex]
            del(dataIndex[randIndex])

    return weights

# 预测
def predict_y(predict_data, weights):

    X = np.hstack([np.ones((len(predict_data), 1)), predict_data])
    pro_y = sigmoid(X.dot(weights))
    return np.array(pro_y >= 0.5, dtype='int')

# 评分
def score(y, y_predict):

    return np.sum(y == y_predict) / len(y)

def load_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X = X[y < 2, :2]
    y = y[y < 2]
    return X, y

def plotBestFit(weights):  # 决策面
    X, y = load_data()
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue")
    x1 = np.linspace(4, 8, 1000)
    plt.plot(x1, (-weights[0]-weights[1]*x1)/weights[2])
    plt.show()

# 分类决策面变化趋势
def plot_weights(data, y):
    weightlist = []
    alpha = 0.01  # 步长
    maxCycles = 1000  # 设置迭代的次数
    X = np.hstack([np.ones((len(data), 1)), data])
    weights = np.ones(X.shape[1])  # 设置初始的参数，并都赋默认值为1。注意这里权重以矩阵形式表示三个参数。3x1

    for k in range(maxCycles):
        error = error_rate(weights, X, y)
        weights = weights - alpha * error  # 迭代更新权重
        weightlist.append(weights)
    X, y = load_data()
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue")
    X = np.linspace(4, 8, 100)
    for i in range(1, len(weightlist)):
        if i % 200 == 0:
            weights = weightlist[i]
            Y = (-weights[0] - weights[1] * X) / weights[2]
            plt.plot(X, Y)
            plt.annotate("i:"+str(i), xy=(X[99], Y[99]))
    plt.show()

# 决策面截距变化
def plot_b(data, y):
    weightlist = []
    alpha = 0.01  # 步长
    maxCycles = 1000  # 设置迭代的次数
    X = np.hstack([np.ones((len(data), 1)), data])
    weights = np.ones(X.shape[1])  # 设置初始的参数，并都赋默认值为1。注意这里权重以矩阵形式表示三个参数。3x1

    for k in range(maxCycles):
        error = error_rate(weights, X, y)
        weights = weights - alpha * error  # 迭代更新权重
        weightlist.append(weights)

    # # 截距或斜率变化趋势
    # fig = plt.figure()
    # axes1 = plt.subplot(211)
    # axes2 = plt.subplot(212)
    # weightmat = np.zeros((maxCycles, X.shape[1]))
    # i = 0
    # for weight in weightlist:
    #     weightmat[i, :] = weight.T
    #     i += 1
    # x1 = np.linspace(0, maxCycles, maxCycles)
    #
    # # 截距：-weights[0] / weights[2]   斜率：-weights[1] / weights[2]
    # axes1.plot(x1[200:], -weightmat[200:, 0] / weightmat[200:, 2], c='b', linewidth=1, linestyle="-")
    # axes1.set_ylabel('intercept')
    # axes2.plot(x1[200:], -weightmat[200:, 1] / weightmat[200:, 2], c='r', linewidth=1, linestyle="-")
    # axes2.set_ylabel('slope')

    # 权重向量收敛评估
    fig = plt.figure()
    axes1 = plt.subplot(311)
    axes2 = plt.subplot(312)
    axes3 = plt.subplot(313)
    weightmat = np.zeros((maxCycles, X.shape[1]))
    i = 0
    for weight in weightlist:
        weightmat[i, :] = weight.T
        i += 1
    x1 = np.linspace(0, maxCycles, maxCycles)
    axes1.plot(x1, weightmat[:, 0], c='b', linewidth=1, linestyle="-")
    axes1.set_ylabel('weight[0]')
    axes2.plot(x1, weightmat[:, 1], c='r', linewidth=1, linestyle="-")
    axes2.set_ylabel('weight[1]')
    axes3.plot(x1, weightmat[:, 2], c='g', linewidth=1, linestyle="-")
    axes3.set_ylabel('weight[2]')
    plt.show()

def random_gra_b(data, y):
    weightlist = []
    maxCycles = 1000  # 设置迭代的次数
    X = np.hstack([np.ones((len(data), 1)), data])
    weights = np.ones(X.shape[1])  # 设置初始的参数，并都赋默认值为1。注意这里权重以矩阵形式表示三个参数。3x1

    for j in range(maxCycles):
        dataIndex = list(range(len(data)))
        for i in range(len(data)):
            alpha = 4 / (1.0 + j + i) + 0.0001
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            vectSum = np.sum(X[randIndex] * weights)
            grad = sigmoid(vectSum)
            errors = y[randIndex] - grad
            weights = weights + alpha * errors * X[randIndex]
            del (dataIndex[randIndex])
        weightlist.append(weights)

    # # 截距或斜率变化趋势
    # fig = plt.figure()
    # axes1 = plt.subplot(211)
    # axes2 = plt.subplot(212)
    # weightmat = np.zeros((maxCycles, X.shape[1]))
    # i = 0
    # for weight in weightlist:
    #     weightmat[i, :] = weight.T
    #     i += 1
    # x1 = np.linspace(0, maxCycles, maxCycles)
    #
    # # 截距：-weights[0] / weights[2]   斜率：-weights[1] / weights[2]
    # axes1.plot(x1, -weightmat[:, 0] / weightmat[:, 2], c='b', linewidth=1, linestyle="-")
    # axes1.set_ylabel('intercept')
    # axes2.plot(x1, -weightmat[:, 1] / weightmat[:, 2], c='r', linewidth=1, linestyle="-")
    # axes2.set_ylabel('slope')

    # 截距和斜率变化趋势
    fig = plt.figure()
    axes1 = plt.subplot(311)
    axes2 = plt.subplot(312)
    axes3 = plt.subplot(313)
    weightmat = np.zeros((maxCycles, X.shape[1]))
    i = 0
    for weight in weightlist:
        weightmat[i, :] = weight.T
        i += 1
    x1 = np.linspace(0, maxCycles, maxCycles)
    axes1.plot(x1, weightmat[:, 0], c='b', linewidth=1, linestyle="-")
    axes1.set_ylabel('weight[0]')
    axes2.plot(x1, weightmat[:, 1], c='r', linewidth=1, linestyle="-")
    axes2.set_ylabel('weight[1]')
    axes3.plot(x1, weightmat[:, 2], c='g', linewidth=1, linestyle="-")
    axes3.set_ylabel('weight[2]')
    plt.show()


if  __name__ ==  '__main__':
    X, y = load_data()
    X_train, y_train, X_test, y_test = data_split(X, y, seed=666)

    # 预测求准确率
    # 梯度下降
    # weights = gradAscent(X_train, y_train)
    # 随机梯度下降
    weights = random_grad(X_train, y_train)
    plotBestFit(weights)
    y_predict = predict_y(X_test, weights)
    print("test score:", score(y_test, y_predict))

    # test 决策面变化趋势
    # plot_weights(X_train, y_train)

    # test 决策面截距，斜率变化趋势
    # random_gra_b(X_train, y_train)