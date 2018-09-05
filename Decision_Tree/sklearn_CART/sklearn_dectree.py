import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

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

def iris_dectree():
    iris = datasets.load_iris()
    X = iris.data[:, 2:]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

    dec_tree = DecisionTreeClassifier(max_depth=2)
    dec_tree.fit(X_train, y_train)

    print("test score:", dec_tree.score(X_test, y_test))

    plot_decision_boundary(dec_tree, axis=[0.5, 7.5, 0, 3])
    plt.scatter(X[y == 0, 0], X[y == 0, 1])
    plt.scatter(X[y == 1, 0], X[y == 1, 1])
    plt.scatter(X[y == 2, 0], X[y == 2, 1])
    plt.show()

def moons_dectree():

    X, y = datasets.make_moons(noise=0.2, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

    dec_tree = DecisionTreeClassifier(max_depth=2)
    dec_tree.fit(X_train, y_train)

    print("test score:", dec_tree.score(X_test, y_test))

    plot_decision_boundary(dec_tree, axis=[-1.5, 2.5, -1.0, 1.5])
    plt.scatter(X[y == 0, 0], X[y == 0, 1])
    plt.scatter(X[y == 1, 0], X[y == 1, 1])
    plt.scatter(X[y == 2, 0], X[y == 2, 1])
    plt.show()

def boston_dectree():

    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # dec_tree = DecisionTreeRegressor()
    # dec_tree.fit(X_train, y_train)
    #
    # print("train score:", dec_tree.score(X_train, y_train))
    # print("test score:", dec_tree.score(X_test, y_test))

    best_h = -1
    best_split = -1
    best_leaf = -1
    best_nodes = -1
    best_score = 0.0

    for a in range(2, 10):
        for b in range(2, 10):
            for c in range(2, 10):
                    dec_tree = DecisionTreeRegressor(max_depth=a, min_samples_split=b, min_samples_leaf=c)
                    dec_tree.fit(X_train, y_train)
                    if dec_tree.score(X_test, y_test) > best_score:
                        best_score = dec_tree.score(X_test, y_test)
                        best_h = a
                        best_split = b
                        best_leaf = c


    print("best score:", best_score)
    print(best_h)
    print(best_split)
    print(best_leaf)


if __name__ == "__main__":

    # iris_dectree()
    # moons_dectree()
    boston_dectree()