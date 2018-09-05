import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from collections import Counter

# 计算信息熵
def entropy(y):
    counter = Counter(y)   #生成一个字典（key：value），统计数据集中不同标签的个数
    entropy = 0.0
    for num in counter.values():
        p = num / len(y)
        entropy += -p * np.log(p)
    return entropy

# 计算基尼指数
def gini(y):
    counter = Counter(y)  # 生成一个字典（key：value），统计数据集中不同标签的个数
    gini_p = 0.0
    for num in counter.values():
        p = num / len(y)
        gini_p += p ** 2
    gini = 1 - gini_p

    return gini


# 绘制决策边界
def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0, x1, zz, cmap=custom_cmap)

# 划分数据集
def split_data(data, label, d, value):

    data_l = []
    data_r = []
    label_l = []
    label_r = []
    for x in range(data.shape[0]):
        if data[x, d] >= value:
            data_l.append(data[x])
            label_l.append(label[x])
        if data[x, d] < value:
            data_r.append(data[x])
            label_r.append(label[x])

    return data_l, data_r, label_l, label_r

# 构建决策树(二分法)
def dec_tree(data, label):
    best_entropy = float('inf')
    best_d, best_v = -1, -1
    for d in range(data.shape[1]):
        sorted_index = np.argsort(data[:, d])
        for i in range(1, len(data)):
            if data[sorted_index[i], d] != data[sorted_index[i - 1], d]:
                v = (data[sorted_index[i], d] + data[sorted_index[i - 1], d]) / 2
                data_l, data_r, label_l, label_r = split_data(data, label, d, v)
                p_l, p_r = len(data_l) / len(X), len(data_r) / len(X)
                e = p_l * entropy(label_l) + p_r * entropy(label_r)
                if e < best_entropy:
                    best_entropy, best_d, best_v = e, d, v

    return best_entropy, best_d, best_v

# 构建决策树
def build_tree(data, label):
    best_entropy = float('inf')
    best_d, best_v = -1, -1
    for d in range(data.shape[1]):
        feature_vlues = {}
        for sample in data:
            feature_vlues[sample[d]] = 1
        # print(len(feature_vlues))
        for value in feature_vlues.keys():
            data_l, data_r, label_l, label_r = split_data(data, label, d, value)
            p_l, p_r = len(data_l) / len(X), len(data_r) / len(X)
            e = p_l * entropy(label_l) + p_r * entropy(label_r)
            if e < best_entropy:
                best_entropy, best_d, best_v = e, d, value
    return best_entropy, best_d, best_v

if __name__ == "__main__":

    iris = datasets.load_iris()
    X = iris.data[:, 2:]
    y = iris.target

    best_entropy, best_d, best_v = build_tree(X, y)
    print("best_entropy =", best_entropy)
    print("best_d =", best_d)
    print("best_v =", best_v)

    X1_l, X1_r, y1_l, y1_r = split_data(X, y, best_d, best_v)
    best_entropy2, best_d2, best_v2 = build_tree(np.array(X1_l), y1_l)
    print("best_entropy =", best_entropy2)
    print("best_d =", best_d2)
    print("best_v =", best_v2)


    # t = np.linspace(0, 3)
    # plt.plot(best_v + 0*t, t, c='g', label="x=2.45")
    # t1 = np.linspace(best_v, 7.5)
    # plt.plot(t1, best_v2 + 0*t1, c='b', label="y=1.75")
    # plt.legend()
    # plt.scatter(X[y == 0, 0], X[y == 0, 1])
    # plt.scatter(X[y == 1, 0], X[y == 1, 1])
    # plt.scatter(X[y == 2, 0], X[y == 2, 1])
    # plt.show()


