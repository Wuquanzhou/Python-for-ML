import KNN
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    # 训练数据，标签
    X_train, y_train = KNN.createDataSet()
    # 预测数据
    x = np.array([8.093607318, 3.365731514])

    # 画出散点图
    plt.scatter(x[0], x[1], color='g')
    plt.annotate('?', (x[0], x[1]), (x[0] + 0.1, x[1]))
    for i in range(len(y_train)):
        if(y_train[i] == 0):
            # 画散点图
            plt.scatter(X_train[i, 0], X_train[i, 1], color='r')
            # 给每个点做标签
            plt.annotate(y_train[i], (X_train[i, 0], X_train[i, 1]), (X_train[i, 0] - 0.1, X_train[i, 1] + 0.1))
        else:
            plt.scatter(X_train[i, 0], X_train[i, 1], color='b')
            plt.annotate(y_train[i], (X_train[i, 0], X_train[i, 1]), (X_train[i, 0] - 0.1, X_train[i, 1] + 0.1))
    plt.show()

    predict_y = KNN.knn_Classifier(x, X_train, y_train, 6)
    print(predict_y)

