from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def sk_iris():

    # 导入鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 使用sklearn封装的分割方法将我们的数据集分成train set和test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # 使用KNeighborsClassifier，传入参数k，fit
    sk_knn = KNeighborsClassifier(n_neighbors=4)
    sk_knn.fit(X_train, y_train)
    score = sk_knn.score(X_test, y_test)

    return score

def sk_digits():

    #导入手写数字数据集
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    sk_knn = KNeighborsClassifier(n_neighbors=4)
    sk_knn.fit(X_train, y_train)
    score = sk_knn.score(X_test, y_test)

    return score
#调参
def adjust_par():
    '''
    这里我先构建自己的网格搜索:
    1.不考虑weights，即uniform，这时我需要考虑参数n_neighbors（1，10）；
    2.考虑weights，即distance，这时我们不仅要考虑n_neighbors（1，10），
    加上了距离权重，就要考虑使用什么距离，即闵可夫斯基距离中p等于多少时分类器最优（1，10）
    '''
    par = [
        {
            'weights': ['uniform'],
            'n_neighbors': [i for i in range(1, 10)]
        },
        {
            'weights': ['distance'],
            'n_neighbors': [i for i in range(1, 10)],
            'p': [i for i in range(1, 6)]
        }
    ]
    # 导入鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    sk_knn = KNeighborsClassifier()
    # 上面的网格搜索要进行54次（9+9*5），为了速度，我将计算机的全部核都用来跑代码，所以我将n_jobs设为-1
    grid_search = GridSearchCV(sk_knn, par, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_score_

if __name__ == "__main__":

    # print("iris准确率：", sk_iris())
    # print("digits准确率：", sk_digits())
    best_estimator, best_score = adjust_par()
    # 得到最高准确率时使用的参数
    print(best_estimator)
    # 打印准确率
    print(best_score)