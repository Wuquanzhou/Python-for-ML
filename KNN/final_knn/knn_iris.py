import KNN
import numpy as np
from sklearn import datasets

if __name__ == "__main__":
    # 手写数字数据集
    # digits = datasets.load_digits()
    # # print(digits.keys())
    # X = digits.data  # 数据
    # y = digits.target  # 标签
    # print(KNN.accuracy_score(X, y))

    # 导入鸢尾花数据集
    iris = datasets.load_iris()
    # print(iris.keys())
    X = iris.data  # 数据
    y = iris.target  # 标签
    best_score, best_k = KNN.accuracy_score(X, y)
    print("best score is: ", best_score)
    print("best k is: ", best_k)