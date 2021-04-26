#!/usr/bin/env python
# -*- coding:utf-8 -*-

import time
from sklearn import metrics
import numpy as np
import pandas as pd
import scipy.io as sio
# from sklearn.externals import joblib
from joblib import *
import csv
import numpy as np
import os, glob
import scipy.io as sio
import xlwt
import xlrd

#import xlutils

from xlutils.copy import copy

# def storFile(data, fileName):
#     with open(fileName, 'w', newline='') as f:
#         mywrite = csv.writer(f)
#         for d in data:
#             mywrite.writerow([d])

import time
from sklearn import metrics
import numpy as np
import pandas as pd
import scipy.io as sio
# from sklearn.externals import joblib
from joblib import *
import csv
import numpy as np
import os, glob
import scipy.io as sio
import xlwt
import xlrd
import xlsxwriter

#import xlutils

from xlutils.copy import copy

# def storFile(data, fileName):
#     with open(fileName, 'w', newline='') as f:
#         mywrite = csv.writer(f)
#         for d in data:
#             mywrite.writerow([d])

# def storFile(data, a):
#     # f = xlwt.Workbook()  # 创建工作薄
#     f = xlrd.open_workbook('MY01_1DCAa.xls', formatting_info=True);
#     # sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
#     newWb = copy(f)
#     newWs = newWb.get_sheet(0)
#     j = 0
#     for g in data:
#         newWs.write(j, a, g)  # 循环写入 竖着写
#         j = j + 1
#     newWb.save('MY01_1DCAa.xls')  # 保存

# def xxwrite(heads, datas):
#     workbook = xlsxwriter.Workbook('MY01_1DCAa.xlsx')  # 新建excel表
#
#     worksheet = workbook.add_worksheet('sheet1')  # 新建sheet（sheet的名称为"sheet1"）
#
#     # headings = # 设置表头
#
#     # data = [
#     #     ['2017-9-1', '2017-9-2', '2017-9-3', '2017-9-4', '2017-9-5', '2017-9-6'],
#     #     [10, 40, 50, 20, 10, 50],
#     #     [30, 60, 70, 50, 40, 30],
#     # ]  # 自己造的数据
#     data.append(datas)
#     for cccc in range(len(data)):
#         worksheet.write_column(0, cccc, data[0])
#
#
#     # worksheet.write_row('A1', heads)
#
#     worksheet.write_column(, data[0])
#     worksheet.write_column('B2', data[1])
#     worksheet.write_column('C2', data[2])  # 将数据插入到表格中
#
#     workbook.close()

# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegressionCV
    model = LogisticRegressionCV(penalty='l2', multi_class='auto', cv=3)
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=200, oob_score='True', max_features='auto')
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=450, max_features=9,learning_rate=0.05, subsample=0.7, min_samples_leaf=60,
                                       min_samples_split=370, max_depth=12, random_state=2)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(gamma='auto', kernel='rbf', probability=True, C=4)
    model.fit(train_x, train_y)
    return model


# AdaBoost Classifier
def AdaBoost_Classifier(train_x, train_y):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=9, min_samples_split=14, min_samples_leaf=7,splitter='random', max_features=0.8),
                         algorithm="SAMME",
                         n_estimators=475, learning_rate=0.3, random_state=2)
    model.fit(train_x, train_y)
    return model

# XGBoost Classifier
def XGBoost_Classifier(train_x, train_y):
    from xgboost.sklearn import XGBClassifier
    model = XGBClassifier(max_depth=5, learning_rate=0.1, cv=3, n_estimators=200, silent=True, subsample=0.6)
    model.fit(train_x, train_y)
    return model
def CatBoost_Classifier(train_x,train_y):
    import catboost
    model = catboost.CatBoostClassifier(task_type='GPU',devices=[0,2,3])
    model.fit(train_x,train_y)
    return model
# SVM Classifier using cross validation
#def svm_cross_validation(train_x, train_y):
    #from sklearn.model_selection import GridSearchCV
    #from sklearn.svm import SVC
    #model = SVC(kernel='rbf', probability=True)
    #param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    #grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    #grid_search.fit(train_x, train_y)
    #best_parameters = grid_search.best_estimator_.get_params()
    #for para, val in best_parameters.items():
        #print(para, val)
    #model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    #model.fit(train_x, train_y)
    #return model




if __name__ == '__main__':
    # # for gh in range(32):_
    workbook1 = xlsxwriter.Workbook('./tweenter1SR/tweenter_ori.xlsx')
    workbook2 = xlsxwriter.Workbook('./tweenter1SR/tweenter_ori_avg.xlsx')
    worksheet1 = workbook1.add_worksheet('sheet1')
    worksheet2 = workbook2.add_worksheet('sheet1')
    data = []
    data1 =[]
    fileList = os.listdir('./tweenter1SR/')
    fileList.sort()
    # fileList.pop(0)
    print(fileList)
    print(len(fileList))
    final_list = []
    for dataname in fileList:
        if os.path.splitext(dataname)[1] == '.csv':  # 目录下包含.json的文件
            print(dataname)
            final_list.append(dataname)
    print(final_list)
    print(len(final_list))
    for gh in range(5):
        DCA_fmd_VcVa_c_R1_labels = pd.read_csv("/DATA/shihaowei/new/pycharmgive/label/twSR/tweenterSRlabel.csv", header=None)
        DCA_fmd_VcVa_c_R1_features1 = pd.read_csv("./tweenter1SR/dca/" + final_list[gh], header=None)
        # DCA_fmd_VcVa_c_R1_features2 = pd.read_csv("../fabric/mat/R/" + final_list[gh], header=None)
        # DCA_fmd_VcVa_c_R1_features3 = pd.read_csv("../fabric/mat/R/" + final_list[gh], header=None)
        # DCA_fmd_VcVa_c_R1_features4 = pd.read_csv("../fabric/mat/R/" + final_list[gh], header=None)
        train_index1 = pd.read_csv("/DATA/shihaowei/new/pycharmgive/label/twSR/tweenterSRtrain.csv", header=None)
        test_index1 = pd.read_csv("/DATA/shihaowei/new/pycharmgive/label/twSR/tweenterSRtest.csv", header=None)
        # train_index2 = pd.read_csv("../fabric/label/f2_7525_train.csv", header=None)
        # test_index2 = pd.read_csv("../fabric/label/f2_7525_test.csv", header=None)
        # train_index3 = pd.read_csv("../fabric/label/f3_7525_train.csv", header=None)
        # test_index3 = pd.read_csv("../fabric/label/f3_7525_test.csv", header=None)
        # train_index4 = pd.read_csv("../fabric/label/f4_7525_train.csv", header=None)
        # test_index4 = pd.read_csv("../fabric/label/f4_7525_test.csv", header=None)
        # Fx = [DCA_fmd_VcVa_c_R1_features1, DCA_fmd_VcVa_c_R1_features2, DCA_fmd_VcVa_c_R1_features3,
        #       DCA_fmd_VcVa_c_R1_features4]
        train_all = [train_index1]
        test_all = [test_index1]
        bbbb = []
        for gx in range(len(test_all)):
            train_x = DCA_fmd_VcVa_c_R1_features1[train_all[gx][0]].values
            test_x = DCA_fmd_VcVa_c_R1_features1[test_all[gx][0]].values
            train_y = DCA_fmd_VcVa_c_R1_labels[train_all[gx][0]].values.ravel()
            test_y = DCA_fmd_VcVa_c_R1_labels[test_all[gx][0]].values.ravel()

            # train_x, test_x, train_y, test_y = train_test_split(DCA_fmd_VcVa_c_R1_features, DCA_fmd_VcVa_c_R1_labels, test_size=0.3)

            test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT', 'AdaBoost', 'XGBoost','Catboost'] #
            num = ['1','2', '3', '4', '5', '6', '7', '8', '9','10'] #,
            classifiers = {
                           'Catboost': CatBoost_Classifier,
                            'NB': naive_bayes_classifier,
                            'KNN': knn_classifier,
                            'LR': logistic_regression_classifier,
                            'RF': random_forest_classifier,
                            'DT': decision_tree_classifier,
                            'SVM': svm_classifier,
                            'GBDT': gradient_boosting_classifier,
                            'AdaBoost': AdaBoost_Classifier,
                            'XGBoost': XGBoost_Classifier,

            }#

            print('reading training and testing data...')
            # train_x, train_y, test_x, test_y = read_data(data_file)
            num_train, num_feat = train_x.shape
            num_test, num_feat = test_x.shape
            is_binary_class = (len(np.unique(train_y)) == 2)
            print('******************** the feature of ' + final_list[gh] + '_' + str(gx + 1) + '*********************')
            print('#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat))
            aaa = []
            bbb = []
            aaa.append(final_list[gh])
            for classifier, i in zip(test_classifiers, num):
                print('******************* %s ********************' % classifier)
                train_time = time.clock()
                model = classifiers[classifier](train_x, train_y)
                print('training took %fs!' % (time.clock() - train_time))
                trd = time.clock() - train_time
                test_time = time.clock()
                predict = model.predict(test_x)
                print('testing took %fs!' % (time.clock() - test_time))
                ted = time.clock() - test_time
                # probility = model.predict_proba(test_x)
                # dic = {final_list[gh] + '_%s_%s_score' % (classifier, str(gx + 1)): probility}
                # sio.savemat('../fabric/mat/RSCORE/' + final_list[gh] + '_' + classifier + '_' + str(gx + 1) + '_score' + '.mat',
                #             dic)
                # dic = {'predict': predict}
                # sio.savemat('../fabric/mat/RSCORE/' + 'predict_' + final_list[gh] + '_' + classifier + '_' + str(
                #     gx + 1) + '_score' + '.mat', dic)
                # if model_save_file != None:
                #     model_save[classifier] = model
                # if is_binary_c lass:
                #     precision = metrics.precision_score(test_y, predict)
                #     recall = metrics.recall_score(test_y, predict)
                #     print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
                accuracy = metrics.accuracy_score(test_y, predict)
                print('accuracy: %.2f%%' % (100 * accuracy))
                c = 100 * accuracy
                aaa.append(c)
                bbb.append(c)
                aaa.append(trd)
                bbb.append(trd)
                aaa.append(ted)
                bbb.append(ted)
            data.append(aaa)
            bbbb.append(bbb)
        cgg = [(bbbb[0][i])/1 for i in range(len(bbbb[0]))]
        cgg.append(final_list[gh])
        data1.append(cgg)
    for ghg in range(len(data)):
        print(data[ghg])
        print('/n')
    for cccc in range(len(data1)):
        worksheet2.write_column(0, cccc, data1[cccc])
    workbook2.close()
    for ccccc in range(len(data)):
        worksheet1.write_column(0, ccccc, data[ccccc])
    workbook1.close()
                # joblib.dump(model, 'E:/study/hj/ma/model/densenet161R_' + classifier + '_' + str(gx+1) + '_model.m')

            # for i in range(len(aaa)):
            #     print(aaa[i])
            # data = aaa
            # bb = gx + gh*4
            # storFile(data, bb)







