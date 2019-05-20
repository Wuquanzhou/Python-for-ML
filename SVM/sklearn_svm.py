import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def load_data():

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X = X[y < 2, :2]
    y = y[y < 2]

    return X, y

def test_svm():

    X, y = load_data()
    # 数据归一化
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaler = scaler.transform(X)

    # 训练svm模型
    svc = LinearSVC(C=1000)
    svc.fit(X_scaler, y)

    w = svc.coef_[0]                 # w = (w1,w2)
    b = svc.intercept_[0]            # b

    x = np.linspace(-3, 3, 100)

    # w0 * x0 + w1 * x1 + b = 0
    # => x1 = - w0 / w1 * x0 - b / w1
    plt.plot(x, -w[0] / w[1] * x - b / w[1], c='r')
    # w0 * x0 + w1 * x1 + b = 1
    # => x1 = - w0 / w1 * x0 - b / w1 + 1 / w[1]
    plt.plot(x, -w[0] / w[1] * x - b / w[1] + 1 / w[1], c='black')
    # w0 * x0 + w1 * x1 + b = -1
    # => x1 = - w0 / w1 * x0 - b / w1 - 1 / w[1]
    plt.plot(x, -w[0] / w[1] * x - b / w[1] - 1 / w[1], c='black', label='C=1000')

    plt.scatter(X_scaler[y == 0, 0], X_scaler[y == 0, 1])
    plt.scatter(X_scaler[y == 1, 0], X_scaler[y == 1, 1])

    plt.axis([-3, 3, -3, 3])
    plt.legend()
    plt.show()


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

def rbf_svm():

    X, y = datasets.make_moons(noise=0.15, random_state=123)

    svc_pipe = Pipeline([
        ("std_scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", gamma=100))
    ])
    svc_pipe.fit(X, y)

    plot_decision_boundary(svc_pipe, axis=[-1.5, 2.5, -1.0, 1.5])
    plt.scatter(X[y == 0, 0], X[y == 0, 1])
    plt.scatter(X[y == 1, 0], X[y == 1, 1])
    plt.show()


if __name__ == "__main__":
    # test_svm()
    rbf_svm()