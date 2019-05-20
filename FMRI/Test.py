import numpy as np
import scipy.io as scio
from scipy.stats import ks_2samp
from xlwt import *

def Dynamic_FC(filename):
    data = scio.loadmat(filename)
    DF_matrirx = data['FCM'][0][0][2]  # FCM
    # print(DF_matrirx)
    # print(np.shape(DF_matrirx[0][0].todense()))
    # print(DF_matrirx[0][0].todense()[:, 0])
    Last_Matrirx = np.empty((10, 89))
    for j in range(len(DF_matrirx)):
        matrirx = np.array(DF_matrirx[j][0].todense())
        # print(np.shape(matrirx))
        Alist = matrirx[33, :].tolist()
        del Alist[33]
        Last_Matrirx[j, :] = Alist

    Last_Matrirx_A = np.empty((1, 89))
    for i in range(89):
        Last_Matrirx_A[0, i] = np.round(np.mean(Last_Matrirx[:, i]), 2)


    return Last_Matrirx_A
def Test_data():
    data1 = Dynamic_FC("F:\Data\PD\ROI_wise_NiftiLabel_sw\FCM\PD_CC_CQL\TV_PD_CC_CQL_FCM.mat")
    data2 = Dynamic_FC("F:\Data\PD\ROI_wise_NiftiLabel_sw\FCM\PD_CC_CWD\TV_PD_CC_CWD_FCM.mat")
    data3 = Dynamic_FC("F:\Data\PD\ROI_wise_NiftiLabel_sw\FCM\PD_CC_LMM\TV_PD_CC_LMM_FCM.mat")
    data4 = Dynamic_FC("F:\Data\PD\ROI_wise_NiftiLabel_sw\FCM\PD_CC_QFY\TV_PD_CC_QFY_FCM.mat")
    data5 = Dynamic_FC("F:\Data\PD\ROI_wise_NiftiLabel_sw\FCM\PD_CC_ZWM\TV_PD_CC_ZWM_FCM.mat")
    data6 = Dynamic_FC("F:\Data\PD\ROI_wise_NiftiLabel_sw\FCM\PD_CC_ZYH\TV_PD_CC_ZYH_FCM.mat")
    data7 = Dynamic_FC("F:\Data\PD\ROI_wise_NiftiLabel_sw\FCM\RBD_LEX\TV_RBD_LEX_FCM.mat")
    data8 = Dynamic_FC("F:\Data\PD\ROI_wise_NiftiLabel_sw\FCM\RBD_QYH\TV_RBD_QYH_FCM.mat")
    data9 = Dynamic_FC("F:\Data\PD\ROI_wise_NiftiLabel_sw\FCM\RBD_SQL\TV_RBD_SQL_FCM.mat")
    data10 = Dynamic_FC("F:\Data\PD\ROI_wise_NiftiLabel_sw\FCM\RBD_SZF\TV_RBD_SZF_FCM.mat")
    data11 = Dynamic_FC("F:\Data\PD\ROI_wise_NiftiLabel_sw\FCM\RBD_WYR\TV_RBD_WYR_FCM.mat")
    data12 = Dynamic_FC("F:\Data\PD\ROI_wise_NiftiLabel_sw\FCM\RBD_YMF\TV_RBD_YMF_FCM.mat")
    data13 = Dynamic_FC("F:\Data\PD\ROI_wise_NiftiLabel_sw\FCM\RBD_ZYP\TV_RBD_ZYP_FCM.mat")
    future_matrix = np.vstack(
        [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13])

    data2_1 = Dynamic_FC("F:\Data\PDCCPS\ROI_wise_NiftiLabel_sw\FCM\PD_CC_CML\TV_PD_CC_CML_FCM.mat")
    data2_2 = Dynamic_FC("F:\Data\PDCCPS\ROI_wise_NiftiLabel_sw\FCM\PD_CC_CZ\TV_PD_CC_CZ_FCM.mat")
    data2_3 = Dynamic_FC("F:\Data\PDCCPS\ROI_wise_NiftiLabel_sw\FCM\PD_CC_GSX\TV_PD_CC_GSX_FCM.mat")
    data2_4 = Dynamic_FC("F:\Data\PDCCPS\ROI_wise_NiftiLabel_sw\FCM\PD_CC_WHG\TV_PD_CC_WHG_FCM.mat")
    data2_5 = Dynamic_FC("F:\Data\PDCCPS\ROI_wise_NiftiLabel_sw\FCM\PD_CC_ZJX\TV_PD_CC_ZJX_FCM.mat")
    data2_6 = Dynamic_FC("F:\Data\PDCCPS\ROI_wise_NiftiLabel_sw\FCM\RBD_MJH\TV_RBD_MJH_FCM.mat")
    data2_7 = Dynamic_FC("F:\Data\PDCCPS\ROI_wise_NiftiLabel_sw\FCM\RBD_WHQ\TV_RBD_WHQ_FCM.mat")
    future_matrix2 = np.vstack([data2_1, data2_2, data2_3, data2_4, data2_5, data2_6, data2_7])

    return future_matrix, future_matrix2

def write_Excel(future_matrix, future_matrix2):

    file = Workbook(encoding='utf-8')

    table = file.add_sheet('Cingulum_Mid_R')
    table.write(0, 0, u'PD')
    table.write(0, 1, u'PDCCPS')
    table.write(0, 2, u'H')
    table.write(0, 3, u'statistic')
    table.write(0, 4, u'pvalue')
    for i in range(1, 90):
        # print(future_matrix[:, i-1].tolist())

        table.write(i, 0, str(future_matrix[:, i - 1]))
        table.write(i, 1, str(future_matrix2[:, i - 1]))
        if ks_2samp(future_matrix[:, i - 1], future_matrix2[:, i - 1])[1] < 0.05:
            table.write(i, 2, u'H=1')
        else:
            table.write(i, 2, u'H=0')
        table.write(i, 3, ks_2samp(future_matrix[:, i - 1], future_matrix2[:, i - 1])[0])
        table.write(i, 4, ks_2samp(future_matrix[:, i - 1], future_matrix2[:, i - 1])[1])
    file.save('C://Users/Administrator/Desktop/ROI/Cingulum_Mid_R.xlsx')

if __name__ == "__main__":

    future_matrix, future_matrix2 = Test_data()
    write_Excel(future_matrix, future_matrix2)




