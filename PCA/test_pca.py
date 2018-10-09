import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

# 导入mnist手写数据集
def load_mnist_data():
    mnist = fetch_mldata('MNIST original', data_home='E:\MNIST_data')
    X, y = mnist['data'], mnist['target']
    X_train = np.array(X[:60000], dtype=float)
    y_train = np.array(y[:60000], dtype=float)
    X_test = np.array(X[60000:], dtype=float)
    y_test = np.array(y[60000:], dtype=float)
    return X_train, y_train, X_test, y_test

# 导入sklearn中的手写数据集
def load_sklearn_data():
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

    return X_train, y_train, X_test, y_test

# demean
def demean(X):
    return X - np.mean(X, axis=0)

# pca优化函数的梯度
def d_var(w, X):
    return X.T.dot(X.dot(w)) * 2. / len(X)

# 将w变为单位向量
def direction(w):
    return w / np.linalg.norm(w)

# 梯度上升法优化目标函数
def grad_component(X, initial_w):
    w = direction(initial_w)
    alpha = 0.01  # 步长
    maxCycles = 1000  # 设置迭代的次数

    for k in range(maxCycles):
        gradient = d_var(w, X)
        w = w + alpha * gradient
        w = direction(w)

    return w

# 求原始数据的前k个梯度，即求Wk
def pca_w(k, X):

    X_pca = X.copy()
    X_pca = demean(X_pca)
    w_k = np.empty((k, X.shape[1]))

    for i in range(k):
        initial_w = np.random.random(X_pca.shape[1])
        w = grad_component(X_pca, initial_w)
        w_k[i, :] = w
        X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w

    return w_k

# 求降维后的矩阵Xk
def transform(X, w_k):
    return X.dot(w_k.T)

def pca_cov(k, X):

    X_pca = X.copy()
    X_pca = demean(X_pca)

    covMat = np.cov(X_pca, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(k+1):-1]
    redEigVects = eigVects[:, eigValInd]

    return redEigVects

if __name__ == "__main__":

    # sklean中手写数据集使用pca降维
    X_train, y_train, X_test, y_test = load_sklearn_data()
    w_k = pca_cov(17, X_train)
    X_train_pca = X_train.dot(w_k)
    X_test_pca = X_test.dot(w_k)
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train_pca, y_train)
    print("基于最大可分性使用PCA准确率：", knn_clf.score(X_test_pca, y_test))

    # 网格搜索sklean中手写数据集性能达到最优时所降到的维数
    # best_score = 0.0
    # best_k = 0
    # for k in range(1, X_train.shape[1]):
    #     w_k = pca_w(k, X_train)
    #     X_train_pca = transform(X_train, w_k)
    #     X_test_pca = transform(X_test, w_k)
    #     knn_clf.fit(X_train_pca, y_train)
    #     if knn_clf.score(X_test_pca, y_test) > best_score:
    #         best_score = knn_clf.score(X_test_pca, y_test)
    #         best_k = k
    # print("best_score:", best_score)
    # print("best_k:", best_k)



    # X_train, y_train, X_test, y_test = load_mnist_data()
    # knn_clf = KNeighborsClassifier()
    # 未使用PCA
    # start = time.clock()
    # knn_clf.fit(X_train, y_train)
    # print("未使用PCA：", knn_clf.score(X_test, y_test))
    # end = time.clock()
    # print('Running time: %s mins' % float((end - start) / 60))

    # 绘图：横坐标是维数，纵坐标为累积方差百分比sum(Percentage of variance[:i+1])
    # pca = PCA(n_components=X_train.shape[1])
    # pca.fit(X_train)
    # plt.plot([i for i in range(X_train.shape[1])],
    #          [np.sum(pca.explained_variance_ratio_[:i+1]) for i in range(X_train.shape[1])])
    # plt.show()

    # 使用PCA
    # start = time.clock()
    # pca = PCA(0.9)
    # pca.fit(X_train)
    # X_train_pca = pca.transform(X_train)
    # X_test_pca = pca.transform(X_test)
    # # print(X_train_pca.shape)
    # knn_clf.fit(X_train_pca, y_train)
    # print("使用PCA：", knn_clf.score(X_test_pca, y_test))
    # end = time.clock()
    # print('Running time: %s mins' % float((end - start) / 60))