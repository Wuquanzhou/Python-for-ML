import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# 回归测试用例
def test_data():
    x = np.random.uniform(-3, 3, size=100)
    X = x.reshape(-1, 1)
    y = x ** 2 + 2 * x + 3 +np.random.normal(0, 1, 100)
    return x, X, y
# 分类测试用例
def test_LRdata():
    np.random.seed(666)
    X = np.random.normal(0, 1, size=(200, 2))
    y = np.array((X[:, 0] ** 2 + X[:, 1] ** 2) < 1.5, dtype='int')

    return X, y
# 线性回归预测
def linear_reg(X, y):
    lin = LinearRegression()
    lin.fit(X, y)
    return lin.predict(X)
# 绘制线性回归走势线
def plot_lin_reg(x, X, y):
    y_predict = linear_reg(X, y)
    plt.plot(x, y_predict, c='r')
    plt.show()

    return X, y

# 将多项式特征加入线性回归中,使用Pipeline
def plot_poly_reg():
    x, X, y = test_data()
    plt.scatter(x, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
    poly_reg = Pipeline([
        ("poly", PolynomialFeatures(degree=40)),
        ("std_scaler", StandardScaler()),
        ("lin_reg", LinearRegression())
    ])
    poly_reg.fit(X_train, y_train)

    y_test_pre = poly_reg.predict(X_test)
    y_predict = poly_reg.predict(X)
    print(mean_squared_error(y_test, y_test_pre))
    plt.plot(np.sort(x), y_predict[np.argsort(x)], c='r')
    plt.axis([-3, 3, 0, 17])
    plt.show()

# 测试岭回归
def test_ridge():
    x, X, y = test_data()
    plt.scatter(x, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
    poly_reg = Pipeline([
        ("poly", PolynomialFeatures(degree=40)),
        ("std_scaler", StandardScaler()),
        ("ridge_reg", Ridge(alpha=1000))
    ])
    poly_reg.fit(X_train, y_train)

    y_test_pre = poly_reg.predict(X_test)
    y_predict = poly_reg.predict(X)
    print(mean_squared_error(y_test, y_test_pre))
    plt.plot(np.sort(x), y_predict[np.argsort(x)], c='r', label='λ=1000')
    plt.legend()
    plt.axis([-3, 3, 0, 17])
    plt.show()

# 测试Lasso回归
def test_lasso():
    x, X, y = test_data()
    plt.scatter(x, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
    poly_reg = Pipeline([
        ("poly", PolynomialFeatures(degree=40)),
        ("std_scaler", StandardScaler()),
        ("ridge_reg", Lasso(alpha=10))
    ])
    poly_reg.fit(X_train, y_train)

    y_test_pre = poly_reg.predict(X_test)
    y_predict = poly_reg.predict(X)
    print(mean_squared_error(y_test, y_test_pre))
    plt.plot(np.sort(x), y_predict[np.argsort(x)], c='r', label='λ=10')
    plt.legend()
    plt.axis([-3, 3, 0, 17])
    plt.show()

# 将多项式特征加入线性回归中
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

# 测试sklearn中的LinearRegression()
def test_linear():
    x, X, y = test_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    print("coef:", lin.coef_)
    print("intercept", lin.intercept_)
    print("score:", lin.score(X_test, y_test))

# 测试sklearn中的LogisticRegression()
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

# 测试加入多项式特征的逻辑回归
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
    # test_ridge()
    test_lasso()
    # test_LR()
    # test_polyLR()