import numpy as np
from collections import Counter

#简单数据集
def createDataSet():
    raw_data_X = [[3.393533211, 2.331273381],
                  [3.110073483, 1.781539638],
                  [1.343808831, 3.368360954],
                  [3.582294042, 4.679179110],
                  [2.280362439, 2.866990263],
                  [7.423436942, 4.696522875],
                  [5.745051997, 3.533989803],
                  [9.172168622, 2.511101045],
                  [7.792783481, 3.424088941],
                  [7.939820817, 0.791637231]
                  ]
    raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    X_train = np.array(raw_data_X)
    y_train = np.array(raw_data_y)
    return X_train, y_train

# KNN分类器
def knn_Classifier(X_in, X_train, y_train, k):

    # 使用欧氏距离公式求距离
    distances = [(np.sum((x_train - X_in)**2))**0.5
                 for x_train in X_train]

    #取最近的k个距离，argsort函数拿到前k个值的下标，对应到y_train中，将对应的标签存到topK_y中
    nearest = np.argsort(distances)
    topK_y = [y_train[neighbor] for neighbor in nearest[:k]]

    # 选出前k个标签中票数最多的一个标签作为最终的预测值
    votes = Counter(topK_y)
    predict_y = votes.most_common(1)[0][0]

    return predict_y

# 预测X_in的标签
def knn_predict(X_in, X_train, y_train, k):
    y_predict = [knn_Classifier(x_in, X_train, y_train, k)
                 for x_in in X_in]
    return np.array(y_predict)

# 分割数据集
def data_split(X, y):

    # 随机打乱数据集的下标顺序
    shuffled_indexes = np.random.permutation(len(X))
    test_ratio = 0.8
    test_size = int(len(X) * test_ratio)
    # 将数据集2-8分，前20%是test集，后80%是训练集
    train_indexes = shuffled_indexes[:test_size]
    test_indexes = shuffled_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, y_train, X_test, y_test

# 计算准确率
def accuracy_score(X, y):

    X_train, y_train, X_test, y_test = data_split(X, y)
    best_score = 0.0
    best_k = 0
    for k in range(1, 10):
        y_predict = knn_predict(X_test, X_train, y_train, k)
        score = sum(y_predict == y_test) / len(y_test)
        if score > best_score:
            best_score = score
            best_k = k
    # print("y_predict : ", y_predict)
    # print("y_test :    ", y_test)
    return best_score, best_k