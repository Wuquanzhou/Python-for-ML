import numpy as np
from collections import Counter

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
