from sklearn import metrics
import numpy as np
import os
import warnings
import catboost
from modAL import ActiveLearner
from modAL.uncertainty import entropy_sampling
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
import scipy.io as sio
from modAL.batch import uncertainty_batch_sampling
from functools import partial
from sklearn.linear_model import LogisticRegressionCV

BATCH_SIZE = 5
# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(X_train, y_train):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


# KNN Classifier
def knn_classifier(X_train, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(X_train, y_train):
    model = LogisticRegressionCV(penalty='l2')
    model.fit(X_train, y_train)
    return model


# Random Forest Classifier
def random_forest_classifier(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=68, random_state=2)
    model.fit(X_train, y_train)
    return model


# Decision Tree Classifier
def decision_tree_classifier(X_train, y_train):
    from sklearn import tree
    model = tree.DecisionTreeClassifier(random_state=2)
    model.fit(X_train, y_train)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(X_train, y_train):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200, random_state=2)
    model.fit(X_train, y_train)
    return model


# SVM Classifier
def svm_classifier(X_train, y_train):

    model = SVC(kernel='rbf', probability=True, random_state=2)
    model.fit(X_train, y_train)
    return model


def classifier_1(X_train, X_test, y_train, y_test):
    model_save_file = None
    model_save = {}
    # test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT']
    test_classifiers = ['KNN', 'LR', 'RF', 'DT']
    num = ['1', '2', '3', '4']
    classifiers = {
        # 'NB': naive_bayes_classifier,
        'KNN': knn_classifier,
        'LR': logistic_regression_classifier,
        'RF': random_forest_classifier,
        'DT': decision_tree_classifier,
    }

    print('reading training and testing data...')
    # X_train, y_train, X_test, y_test = read_data(data_file)

    is_binary_class = (len(np.unique(y_train)) == 2)
    for classifier, i in zip(test_classifiers, num):
        print('******************* %s ********************' % classifier)
        model = classifiers[classifier](X_train, y_train)
        predict = model.predict(X_test)
        predict_dict[classifier] = predict
        accuracy = metrics.accuracy_score(y_test, predict)
        print('accuracy: %.2f%%' % (100 * accuracy))

def classifier_NB(X_train, X_test, y_train, y_test):
    # global accuracy_dict
    print("******************* NB ********************")
    is_binary_class = (len(np.unique(y_train)) == 2)

    print('******************* %s ********************' % naive_bayes_classifier)
    model = naive_bayes_classifier(X_train, y_train)
    
    predict = model.predict(X_test)
    predict_dict['NB'] = predict

    accuracy = metrics.accuracy_score(y_test, predict)
    print('accuracy: %.2f%%' % (100 * accuracy))


def classifer_adaboost(X_train, X_test, y_train, y_test):
    train_time = []
    test_time = []
    # global accuracy_dict
    print("******************* adaboost ********************")
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8, min_samples_split=20, min_samples_leaf=5),
                               algorithm="SAMME",
                               n_estimators=500, learning_rate=0.7, random_state=2)
    
    model.fit(X_train, y_train)
    predict = model.predict(X_test)  # predict is holped to be equal to y_test
    predict_dict['adaboost'] = predict
    accuracy = metrics.accuracy_score(y_test, predict)
    print('accuracy: %.2f%%' % (100 * accuracy))



def classifier_GBDT(X_train, X_test, y_train, y_test):
    # global accuracy_dict
    true_dict = {}
    false_dict = {}
    model_save_file = None
    model_save = {}
    test_classifiers = ['GBDT']
    num = ['1']
    classifiers = {
        'GBDT': gradient_boosting_classifier
    }
    is_binary_class = (len(np.unique(y_train)) == 2)
    for classifier, i in zip(test_classifiers, num):
        model = classifiers[classifier](X_train, y_train)
        predict = model.predict(X_test)
        predict_dict[classifier] = predict

        accuracy = metrics.accuracy_score(y_test, predict)
        print('accuracy: %.2f%%' % (100 * accuracy))



def classifier_xgboost(X_train, X_test, y_train, y_test,tree_num):
    train_time = []
    test_time = []
    # global accuracy_dict
    print("******************* xgboost ********************")
    # start_time = time.time()
    clf = XGBClassifier(
        silent=0,  # 设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
        # nthread=4,# cpu 线程数 默认最大
        learning_rate=0.21,  # 如同学习率
        min_child_weight=2,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        max_depth=10,  # 构建树的深度，越大越容易过拟合
        gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
        subsample=0.9,  # 随机采样训练样本 训练实例的子采样比
        max_delta_step=0,  # 最大增量步长，我们允许每个树的权重估计。
        colsample_bytree=0.9,  # 生成树时进行的列采样
        reg_lambda=1.9,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        # reg_alpha=0,  # L1 正则项参数
        # scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
        objective='multi:softmax',  # 多分类的问题 指定学习任务和相应的学习目标，返回预测类别，不是概率
        num_class=2,  # 类别数，多分类与 multisoftmax 并用
        n_estimators=tree_num,  # 树的个数
        seed=1000  # 随机种子
        # eval_metric= 'auc'
    )
    clf.fit(X_train, y_train, eval_metric='auc')

    y_true, y_pred = y_test, clf.predict(X_test)
    if tree_num == 20:
        predict_dict['xgboost_20'] = y_pred
    elif tree_num == 40:
        predict_dict['xgboost_40'] = y_pred
    auc = metrics.accuracy_score(y_true, y_pred) * 100
    
    print("Accuracy : %.4g" % auc)
    # cost_time = time.time() - start_time


def classifier_catboost(X_train, X_test, y_train,y_test):
    print("******************* catboost ********************")

    model = catboost.CatBoostClassifier()

    model.fit(X_train,y_train)
    predict = model.predict(X_test)  # predict is holped to be equal to y_test
    accuracy = metrics.accuracy_score(y_test, predict)
    pro = model.predict_proba(X_test)
    return accuracy

def active_learning(X_train,X_test,y_train,classifier):
    from sklearn.ensemble import RandomForestClassifier
    preset_batch = partial(uncertainty_batch_sampling,n_instances = BATCH_SIZE)
    if classifier == "classifier1":
        learner = ActiveLearner(estimator = catboost.CatBoostClassifier(),query_strategy=preset_batch,X_training = X_train,y_training = y_train)
    elif classifier == "classifier2":
        learner = ActiveLearner(estimator = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8, min_samples_split=20, min_samples_leaf=5),algorithm="SAMME",n_estimators=500, learning_rate=0.7, random_state=2),query_strategy=preset_batch,X_training = X_train,y_training = y_train)
    elif classifier == "classifier3":
        learner = ActiveLearner(estimator = RandomForestClassifier(n_estimators=68, random_state=2),query_strategy=preset_batch,X_training = X_train,y_training = y_train)
    elif classifier == "classifier4":
        learner = ActiveLearner(estimator = LogisticRegressionCV(penalty='l2'),query_strategy=preset_batch,X_training = X_train,y_training = y_train)
    predict = learner.predict(X_test)
    # query for labels
    query_idx, query_inst = learner.query(X_test)

    return query_idx, query_inst,predict

import h5py
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # print(os.getcwd())
    """
    loading data
    """
    # Initialize classifcation
    feature_5 = h5py.File(r'/DATA/shihaowei/feature/5/dca/dca_sift_vgg19.mat', 'r')
    feature_3_5 = h5py.File(r'/DATA/shihaowei/feature/3-5/dca/dca_sift_vgg19.mat','r')
    sift_5_fea = sio.loadmat(r'/DATA/shihaowei/feature/5/ALLSiftVectors.mat')
    vgg19_5_fea = sio.loadmat(r'/DATA/shihaowei/feature/5/deepfashion_vgg19_1.mat')
    sift_3_5_fea = sio.loadmat(r'/DATA/shihaowei/feature/3-5/ALLSiftVectors.mat')
    vgg19_3_5_fea = sio.loadmat(r'/DATA/shihaowei/feature/3-5/deepfashion_vgg19_1.mat')
    label_5 =  sio.loadmat(r'/DATA/shihaowei/feature/5/Lable.mat')
    label_3_5 = sio.loadmat(r'/DATA/shihaowei/feature/3-5/Lable.mat')
    s_5_fea = np.array(sift_5_fea['countVectors'])
    vgg_5_fea = np.array(vgg19_5_fea['vgg19_1']).reshape(5238,4096)
    s_3_5_fea = np.array(sift_3_5_fea['countVectors'])
    vgg_3_5_fea = np.array(vgg19_3_5_fea['vgg19_1']).reshape(16025,4096)
    X_train = np.array(feature_5['all']).T
    X_test = np.array(feature_3_5['all']).T
    y_train = np.array(feature_5['train_label'])
    y_test = np.array(feature_3_5['train_label'])
    Label_5 = np.array(label_5['Lable'])
    Label_3_5 = np.array(label_3_5['Lable'])
    train_index = np.array(feature_5['rand_index'])
    test_index = np.array(feature_3_5['rand_index'])

#Sample Refinement
    predict_dict = {}
    index = []

    classifier_1(X_train, X_test, y_train, y_test)
    classifier_NB(X_train, X_test, y_train, y_test)  # NB need all the feature descriptor must be non-negative
    classifer_adaboost(X_train, X_test, y_train, y_test)
    classifier_GBDT(X_train, X_test, y_train, y_test)

    """
        this is only for xgboost
        """
    #
    # """
    # tree_num = [20,40,60,80,100,125,150,175,200,300,400,500,1000,2000]
    tree_num = [20,40]

    cost_time = []
    for item in tree_num:
        classifier_xgboost(X_train, X_test, y_train, y_test,item)
        print('tree num is : %d \n' % item)


    predict_KNN = predict_dict['KNN']
    predict_LR = predict_dict['LR']
    predict_RF = predict_dict['RF']
    predict_DT = predict_dict['DT']
    predict_NB = predict_dict['NB']
    predict_adaboost = predict_dict['adaboost']
    predict_GBDT = predict_dict['GBDT']
    predict_xgboost_20 = predict_dict['xgboost_20']
    predict_xgboost_40 = predict_dict['xgboost_40']

    for i in range(len(predict_DT)):
        if predict_KNN[i] == predict_LR[i] == predict_NB[i] == predict_GBDT[i] == predict_DT[i]:
            if predict_LR[i] == y_test[i]:
                # s_5_fea = np.vstack((s_5_fea,s_3_5_fea[int(test_index[i] - 1)]))
                # vgg_5_fea = np.vstack((vgg_5_fea,vgg_3_5_fea[int(test_index[i] - 1)]))
                # Label_5 = np.vstack((Label_5,Label_3_5[int(test_index[i] - 1)]))
                index.append(i)
    X_test = X_test[index]
    y_test = y_test[index]



    # active learning
    index_first_al = []
    index_second_al = []
    index_third_al = []
    index_fourth_al = []

    X_train_AL = X_train
    X_test_AL = X_test
    y_train_AL = y_train
    y_test_AL = y_test
    for i in range(900):
        #LR for activelearning
        query = list(active_learning(X_train_AL,X_test_AL,y_train_AL,"classifier1"))
        max_qu = max(query[0])
        max_index = np.where(max_qu)[0]
        if query[2][query[0][max_index]] == y_test[query[0][max_index]]:
            index_first_al.append(int(test_index[index[int(query[0][max_index])]] - 1))
            X_train_AL = np.vstack((X_train_AL,query[1][max_index]))
            y_train_AL = np.vstack((y_train_AL,y_test_AL[query[0][max_index]]))
            X_test_AL = np.delete(X_test_AL,query[0][max_index],axis=0)
            y_test_AL = np.delete(y_test_AL,query[0][max_index],axis=0)
        else:
            X_test_AL = np.delete(X_test_AL,query[0][max_index],axis=0)
            y_test_AL = np.delete(y_test_AL,query[0][max_index],axis=0)


    X_train_AL = X_train
    X_test_AL = X_test
    y_train_AL = y_train
    y_test_AL = y_test
    for i in range(900):
        #LR for activelearning
        query = list(active_learning(X_train_AL,X_test_AL,y_train_AL,"classifier2"))
        max_qu = max(query[0])
        max_index = np.where(max_qu)[0]
        if query[2][query[0][max_index]] == y_test[query[0][max_index]]:
            index_first_al.append(int(test_index[index[int(query[0][max_index])]] - 1))
            X_train_AL = np.vstack((X_train_AL,query[1][max_index]))
            y_train_AL = np.vstack((y_train_AL,y_test_AL[query[0][max_index]]))
            X_test_AL = np.delete(X_test_AL,query[0][max_index],axis=0)
            y_test_AL = np.delete(y_test_AL,query[0][max_index],axis=0)
        else:
            X_test_AL = np.delete(X_test_AL,query[0][max_index],axis=0)
            y_test_AL = np.delete(y_test_AL,query[0][max_index],axis=0)
    #
    #
    #
    # X_train_AL = X_train
    # X_test_AL = X_test
    # y_train_AL = y_train
    # y_test_AL = y_test
    # for i in range(900):
    #     #LR for activelearning
    #     query = list(active_learning(X_train_AL,X_test_AL,y_train_AL,"classifier3"))
    #     max_qu = max(query[0])
    #     max_index = np.where(max_qu)[0]
    #     if query[2][query[0][max_index]] == y_test[query[0][max_index]]:
    #         index_first_al.append(int(test_index[index[int(query[0][max_index])]] - 1))
    #         X_train_AL = np.vstack((X_train_AL,query[1][max_index]))
    #         y_train_AL = np.vstack((y_train_AL,y_test_AL[query[0][max_index]]))
    #         X_test_AL = np.delete(X_test_AL,query[0][max_index],axis=0)
    #         y_test_AL = np.delete(y_test_AL,query[0][max_index],axis=0)
    #     else:
    #         X_test_AL = np.delete(X_test_AL,query[0][max_index],axis=0)
    #         y_test_AL = np.delete(y_test_AL,query[0][max_index],axis=0)
    #
    #
    #
    #
    # X_train_AL = X_train
    # X_test_AL = X_test
    # y_train_AL = y_train
    # y_test_AL = y_test
    # for i in range(900):
    #     #LR for activelearning
    #     query = list(active_learning(X_train_AL,X_test_AL,y_train_AL,"classifier4"))
    #     max_qu = max(query[0])
    #     max_index = np.where(max_qu)[0]
    #     if query[2][query[0][max_index]] == y_test[query[0][max_index]]:
    #         index_first_al.append(int(test_index[index[int(query[0][max_index])]] - 1))
    #         X_train_AL = np.vstack((X_train_AL,query[1][max_index]))
    #         y_train_AL = np.vstack((y_train_AL,y_test_AL[query[0][max_index]]))
    #         X_test_AL = np.delete(X_test_AL,query[0][max_index],axis=0)
    #         y_test_AL = np.delete(y_test_AL,query[0][max_index],axis=0)
    #     else:
    #         X_test_AL = np.delete(X_test_AL,query[0][max_index],axis=0)
    #         y_test_AL = np.delete(y_test_AL,query[0][max_index],axis=0)
    
    index_add = []
    index_sum = []
    # index_add = list(set(index_first_al + index_second_al + index_third_al + index_fourth_al))

    from collections import Counter
    index_sum = index_first_al + index_second_al + index_third_al + index_fourth_al
    index_add = list(set(index_sum))
    # b = dict(Counter(index_sum))
    # index_add = [key for key,value in b.items() if value > 1]




    s_5_fea = np.vstack((s_5_fea,s_3_5_fea[index_add]))
    vgg_5_fea = np.vstack((vgg_5_fea,vgg_3_5_fea[index_add]))
    Label_5 = np.vstack((Label_5,Label_3_5[index_add]))
    print(len(index_add))
    sio.savemat(r"/DATA/shihaowei/feature/sr+af/cat+ada/sift_al.mat", {'sift':s_5_fea})
    sio.savemat(r"/DATA/shihaowei/feature/sr+af/cat+ada/vgg19_al.mat", {'vgg19':vgg_5_fea})
    sio.savemat(r"/DATA/shihaowei/feature/sr+af/cat+ada/label_al.mat", {'label':Label_5})

    
    
    
        
