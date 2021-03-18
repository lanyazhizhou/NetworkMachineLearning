# coding=utf-8
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import time
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.model_selection import train_test_split

def data_processing():
    # 读取数据
    data_dir = "./data/"
    raw_data_filename = data_dir + "kddcup.data.corrected"
    print("Loading raw data...")
    raw_data = pd.read_csv(raw_data_filename, header=None)
    raw_data=raw_data.sample(frac=0.3)

    '''
    将非数值型的数据转换为数值型数据
    0,tcp,http,SF,215,45076,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0,0,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,normal.
    '''
    print("Transforming data...")
    raw_data[1], protocols = pd.factorize(raw_data[1])  # factorize()方法将非数值型数据映射成数字，返回值是一个元组
    raw_data[2], services = pd.factorize(raw_data[2])
    raw_data[3], flags = pd.factorize(raw_data[3])
    raw_data[41], attacks = pd.factorize(raw_data[41])

    # 对原始数据进行切片，分离出特征和标签，第1~41列是特征，第42列是标签
    features = raw_data.iloc[:, :raw_data.shape[1] - 1]  # pandas中的iloc切片是完全基于位置的索引
    labels = raw_data.iloc[:, raw_data.shape[1] - 1:]

    # 将多维的标签转为一维的数组
    labels = labels.values.ravel()

    # 将数据分为训练集和测试集,并打印维数
    df = pd.DataFrame(features)
    X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=0.9, test_size=0.1)
    return X_train, X_test, y_train, y_test


def classify(input_vct, data_set):
    data_set_size = data_set.shape[0]
    diff_mat = np.tile(input_vct, (data_set_size, 1)) - data_set  # 扩充input_vct到与data_set同型并相减
    sq_diff_mat = diff_mat**2  # 矩阵中每个元素都平方
    distance = sq_diff_mat.sum(axis=1)**0.5  # 每行相加求和并开平方根
    return distance.min(axis=0)  # 返回最小距离


def file2mat(test_filename, para_num):
    """
    将表格存入矩阵，test_filename为表格路径，para_num为存入矩阵的列数
    返回目标矩阵，和矩阵每一行数据的类别
    """
    fr = open(test_filename)
    lines = fr.readlines()
    line_nums = len(lines)
    result_mat = np.zeros((line_nums, para_num))  # 创建line_nums行，para_num列的矩阵
    class_label = []
    for i in range(line_nums):
        line = lines[i].strip()
        item_mat = line.split(',')
        result_mat[i, :] = item_mat[0: para_num]
        class_label.append(item_mat[-1])  # 表格中最后一列正常1异常2的分类存入class_label
    fr.close()
    return result_mat, class_label


def roc(data_set):
    normal = 0
    data_set_size = data_set.shape[1]
    roc_rate = np.zeros((2, data_set_size))
    for i in range(data_set_size):
        if data_set[2][i] == 1:
            normal += 1
    abnormal = data_set_size - normal
    max_dis = data_set[1].max()
    for j in range(1000):
        threshold = max_dis / 1000 * j
        normal1 = 0
        abnormal1 = 0
        for k in range(data_set_size):
            if data_set[1][k] > threshold and data_set[2][k] == 1:
                normal1 += 1
            if data_set[1][k] > threshold and data_set[2][k] == 2:
                abnormal1 += 1
        roc_rate[0][j] = normal1 / normal  # 阈值以上正常点/全体正常的点
        roc_rate[1][j] = abnormal1 / abnormal  # 阈值以上异常点/全体异常点
    return roc_rate


def test(training_filename, test_filename):
    training_mat, training_label = file2mat(training_filename, 32)
    test_mat, test_label = file2mat(test_filename, 32)
    test_size = test_mat.shape[0]
    result = np.zeros((test_size, 3))
    for i in range(test_size):
        result[i] = i + 1, classify(test_mat[i], training_mat), test_label[i]  # 序号， 最小欧氏距离， 测试集数据类别
    result = np.transpose(result)  # 矩阵转置
    print("开始画图：")
    plt.figure(1)
    plt.scatter(result[0], result[1], c=result[2], edgecolors='None', s=1, alpha=1)
    # 图1 散点图：横轴为序号，纵轴为最小欧氏距离，点中心颜色根据测试集数据类别而定， 点外围无颜色，点大小为最小1，灰度为最大1
    roc_rate = roc(result)
    plt.figure(2)
    plt.scatter(roc_rate[0], roc_rate[1], edgecolors='None', s=1, alpha=1)
    # 图2 ROC曲线， 横轴误报率，即阈值以上正常点/全体正常的点；纵轴检测率，即阈值以上异常点/全体异常点
    plt.show()

def sklearn_knn(X_train,X_test,y_train,y_test):
    # 训练模型
    print("Training model...")
    clf = KNeighborsClassifier(n_neighbors=3)  #近邻参数设置成3个相邻
    trained_model = clf.fit(X_train, y_train)
    # print("Score:", trained_model.score(X_train, y_train))
    trainData_end = time.time()

    # 预测
    print("Predicting...")
    y_pred = clf.predict(X_test)

    # 绘制混淆矩阵图形
    # plotMatrix(attacks,y_test,y_pred)

    print("Computing performance metrics...")
    results = confusion_matrix(y_test, y_pred)
    error = zero_one_loss(y_test, y_pred)

    # 根据混淆矩阵求预测精度
    list_diag = np.diag(results)
    list_raw_sum = np.sum(results, axis=1)
    print("Predict accuracy: ", np.mean(list_diag) / np.mean(list_raw_sum))

if __name__ == "__main__":
    startTime = time.time()
    X_train, X_test, y_train, y_test=data_processing()
    sklearn_knn(X_train, X_test, y_train, y_test)
    endTime = time.time()
    print("总时间:%fs" % (startTime - endTime))
    # test('save_csv/shuzhi/total2_data_train.csv', 'save_csv/shuzhi/total2_data_test.csv')