# coding:UTF-8
import numpy as np
import matplotlib.pyplot as plt
inputfile = 'E:\\ML_data\\SR_data.txt'

def load_data():
    '''导入训练数据
    input:  inputfile(string)训练样本的位置
    output: feature_data(mat)特征
            label_data(mat)标签
            k(int)类别的个数
    '''
    f = open(inputfile)  # 打开文件
    feature_data = []
    label_data = []
    for line in f.readlines():
        feature_tmp = []
        feature_tmp.append(1)  # 偏置项
        lines = line.strip().split("\t")
        for i in range(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        label_data.append(int(lines[-1]))

        feature_data.append(feature_tmp)
    f.close()  # 关闭文件
    return feature_data, label_data, len(set(label_data))


def cost(err, label_data):
    '''计算损失函数值
    input:  err(mat):exp的值
            label_data(mat):标签的值
    output: sum_cost / m(float):损失函数的值
    '''
    m = np.shape(err)[0]
    sum_cost = 0.0
    for i in range(m):
        if err[i, label_data[i, 0]] / np.sum(err[i, :]) > 0:
            sum_cost -= np.log(err[i, label_data[i, 0]] / np.sum(err[i, :]))
        else:
            sum_cost -= 0
    return sum_cost / m


def gradientAscent(feature_data, label_data, k, maxCycle, alpha):
    '''利用梯度下降法训练Softmax模型
    input:  feature_data(mat):特征
            label_data(mat):标签
            k(int):类别的个数
            maxCycle(int):最大的迭代次数
            alpha(float):学习率
    output: weights(mat)：权重
    '''
    dataMatrix = np.mat(feature_data)  # 将读取的数据转换为矩阵,dataMatrix : nx3
    classLabels = np.mat(label_data).transpose()  # 将读取的数据转换为矩阵,classLabels : nx1
    m, n = np.shape(dataMatrix)
    weights = np.mat(np.ones((n, k)))  # 权重的初始化（矩阵）
    print(np.shape(dataMatrix))
    print(np.shape(weights))
    i = 0
    while i <= maxCycle:
        err = np.exp(dataMatrix * weights)
        if i % 500 == 0:
            print("\t-----iter: ", i, ", cost: ", cost(err, classLabels))

        rowsum = -err.sum(axis=1)
        rowsum = rowsum.repeat(k, axis=1)
        err = err / rowsum
        for x in range(m):
            err[x, classLabels[x, 0]] += 1
        weights = weights + (alpha / m) * dataMatrix.T * err
        i += 1
    return weights


def save_model(file_name, weights):
    '''保存最终的模型
    input:  file_name(string):保存的文件名
            weights(mat):softmax模型
    '''
    f_w = open(file_name, "w")
    m, n = np.shape(weights)
    for i in range(m):
        w_tmp = []
        for j in range(n):
            w_tmp.append(str(weights[i, j]))
        f_w.write("\t".join(w_tmp) + "\n")
    f_w.close()

def plotBestFit(weights):
    feature_data, label_data, k = load_data()
    dataArr = np.array(feature_data)
    n = np.shape(dataArr)[0]
    xcord0 = []; ycord0 = []
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    xcord3 = []; ycord3 = []
    for i in range(n):
        if int(label_data[i]) == 0:
            xcord0.append(dataArr[i, 1])
            ycord0.append(dataArr[i, 2])
        if int(label_data[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        if int(label_data[i]) == 2:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
        if int(label_data[i]) == 3:
            xcord3.append(dataArr[i, 1])
            ycord3.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord0, ycord0, s=30, c='red', marker='*')
    ax.scatter(xcord1, ycord1, s=30, c='green', marker='+')
    ax.scatter(xcord2, ycord2, s=30, c='blue', marker='s')
    ax.scatter(xcord3, ycord3, s=30, c='black')
    x = np.arange(-3.0, 3.0, 0.1)
    y_01 = (-weights[0, 0] - weights[1, 0] * x) / weights[2, 0]      # 红点
    y_12 = (-weights[0, 1] - weights[1, 1] * x) / weights[2, 1]      # 绿点
    y_23 = (-weights[0, 2] - weights[1, 2] * x) / weights[2, 2]      # 蓝点
    y_34 = (-weights[0, 3] - weights[1, 3] * x) / weights[2, 3]      # 黑点
    ax.plot(x, y_01, x, y_12, x, y_23)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == "__main__":
    # 1、导入训练数据
    print
    "---------- 1.load data ------------"
    feature, label, k = load_data()
    # 2、训练Softmax模型
    print
    "---------- 2.training ------------"
    weights = gradientAscent(feature, label, k, 10000, 0.4).getA()
    plotBestFit(weights)
    # 3、保存最终的模型
    print
    "---------- 3.save model ------------"
    save_model("weights", weights)