import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression

def test_data():
    x = np.random.uniform(-3, 3, size=100)
    X = x.reshape(-1, 1)
    y = x ** 2 + 2 * x + 3 +np.random.normal(0, 1, 100)
    return x, X, y

def test_LRdata():
    np.random.seed(666)
    X = np.random.normal(0, 1, size=(200, 2))
    y = np.array((X[:, 0] ** 2 + X[:, 1] ** 2) < 1.5, dtype='int')

    return X, y

def linear_reg(X, y):
    lin = LinearRegression()
    lin.fit(X, y)
    return lin.predict(X)

def plot_lin_reg(x, X, y):
    y_predict = linear_reg(X, y)
    plt.plot(x, y_predict, c='r')
    plt.show()

    return X, y

def plot_poly_reg():
    x, X, y = test_data()
    plt.scatter(x, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
    poly_reg = Pipeline([
        ("poly", PolynomialFeatures(degree=2)),
        ("std_scaler", StandardScaler()),
        ("lin_reg", LinearRegression())
    ])
    poly_reg.fit(X_train, y_train)

    y_test_pre = poly_reg.predict(X_test)
    y_predict = poly_reg.predict(X)
    print(mean_squared_error(y_test, y_test_pre))
    plt.plot(np.sort(x), y_predict[np.argsort(x)], c='r')
    plt.show()

def test_poly():
    x, X, y = test_data()
    poly = PolynomialFeatures(degree=2)
    poly.fit(X)
    X2 = poly.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=666)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    print("coef:", lin_reg.coef_)
    print("intercept", lin_reg.intercept_)
    print("score:", lin_reg.score(X_test, y_test))

def test_linear():
    x, X, y = test_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    print("coef:", lin.coef_)
    print("intercept", lin.intercept_)
    print("score:", lin.score(X_test, y_test))

def test_LR():
    X, y = test_LRdata()
    plt.scatter(X[y == 0, 0], X[y == 0, 1])
    plt.scatter(X[y == 1, 0], X[y == 1, 1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    x1 = np.linspace(-3, 3, 1000)
    plt.plot(x1, (-log_reg.intercept_[0] - log_reg.coef_[0][0] * x1) / log_reg.coef_[0][1], c='r')
    plt.axis([-3, 3, -5, 5])
    plt.show()
    print("score:", log_reg.score(X_test, y_test))


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

def test_polyLR():
    X, y = test_LRdata()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
    poly_lr = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])
    poly_lr.fit(X_train, y_train)
    plot_decision_boundary(poly_lr, axis=[-4, 4, -4, 4])
    plt.scatter(X[y == 0, 0], X[y == 0, 1])
    plt.scatter(X[y == 1, 0], X[y == 1, 1])
    plt.show()
    print("score:", poly_lr.score(X_test, y_test))


if __name__ == "__main__":
    # test_poly()
    # test_linear()
    # plot_poly_reg()
    # test_LR()
    test_polyLR()