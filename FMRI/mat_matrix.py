import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

def Dynamic_FC(filename):
    data = scio.loadmat(filename)
    DF_matrirx = data['DZStruct'][0][0]  # DZStruct

    # print(len(DF_matrirx))
    Last_Matrirx = np.empty((196, 4005))
    for j in range(len(DF_matrirx)):
        Alist = DF_matrirx[j][0, 1:].tolist()
        for i in range(1, len(DF_matrirx[j]) - 1):
            print(np.shape(DF_matrirx[0][i, i+1:].tolist()))
            Alist.extend(DF_matrirx[j][i, i + 1:].tolist())
        Last_Matrirx[j, :] = Alist

    Last_Matrirx_A = np.empty((1, 4005))
    for i in range(4005):
        Last_Matrirx_A[0, i] = np.mean(Last_Matrirx[:, i])

    return Last_Matrirx_A

if __name__ == "__main__":

    data1 = Dynamic_FC("F://TestData/01_PD/GretnaDFCMatrixZ/z170704_RBD_SQL_1.mat")
    data2 = Dynamic_FC("F://TestData/01_PD/GretnaDFCMatrixZ/z170710_RBD_SZF_1.mat")
    data3 = Dynamic_FC("F://TestData/01_PD/GretnaDFCMatrixZ/z170718_RBD_WYR_1.mat")
    data4 = Dynamic_FC("F://TestData/01_PD/GretnaDFCMatrixZ/z170814_RBD_YMF_1.mat")
    data5 = Dynamic_FC("F://TestData/01_PD/GretnaDFCMatrixZ/z170823_RBD_ZYP_1.mat")
    data6 = Dynamic_FC("F://TestData/01_PD/GretnaDFCMatrixZ/z171023_RBD_QYH_1.mat")
    data7 = Dynamic_FC("F://TestData/01_PD/GretnaDFCMatrixZ/z180411_PD_CC_CQL_1.mat")
    data8 = Dynamic_FC("F://TestData/01_PD/GretnaDFCMatrixZ/z180528_PD_CC_CWD_1.mat")
    data9 = Dynamic_FC("F://TestData/01_PD/GretnaDFCMatrixZ/z180528_PD_CC_ZWM_1.mat")
    data10 = Dynamic_FC("F://TestData/01_PD/GretnaDFCMatrixZ/z180613_PD_CC_LMM_1.mat")
    data11 = Dynamic_FC("F://TestData/01_PD/GretnaDFCMatrixZ/z180711_PD_CC_ZYH_1.mat")
    data12 = Dynamic_FC("F://TestData/01_PD/GretnaDFCMatrixZ/z180711_RBD_LEX_1.mat")
    data13 = Dynamic_FC("F://TestData/01_PD/GretnaDFCMatrixZ/z180828_PD_CC_QFY_1.mat")

    future_matrix = np.vstack([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13])
    print(np.shape(future_matrix))

    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0])
    # X_train, X_test, y_train, y_test = train_test_split(future_matrix, y)
    shuffled_indexes = np.random.permutation(len(y))
    train_ratio = 0.8
    train_size = int(len(y) * train_ratio)
    # 将数据集2-8分，前80%是train集，后20%是test集
    train_indexes = shuffled_indexes[:train_size]
    test_indexes = shuffled_indexes[train_size:]

    X_train = future_matrix[train_indexes]
    y_train = y[train_indexes]

    X_test = future_matrix[test_indexes]
    y_test = y[test_indexes]
    print(y_train)
    print(y_test)

    # 使用PCA
    pca = PCA(0.9)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    clf = SVC(kernel="poly")
    # knn_clf = KNeighborsClassifier()
    clf.fit(X_train_pca, y_train)
    print(clf.predict(X_test_pca))
    print("使用PCA：", clf.score(X_test_pca, y_test))



