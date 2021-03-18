import pandas as pd
import numpy as np
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.model_selection import train_test_split
from kdd99.PlotConfusionMatrix import PlotConfusionMatrix
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
# from DataProcessing import mergeData


# 绘制混淆矩阵
def plotMatrix(attacks, y_test, y_pred):
    # attacks是整个数据集的标签集合，但是切分测试集的时候，某些标签数量很少，可能会被去掉，这里要剔除掉这些标签
    y_test_set = set(y_test)
    y_test_list = list(y_test_set)
    attacks_test = []
    for i in range(0, len(y_test_set)):
        attacks_test.append(attacks[y_test_list[i]])
    p = PlotConfusionMatrix()
    p.prepareWork(attacks_test, y_test, y_pred)


# 绘制学习曲线
def plotLearningCurve(X_train, X_test, y_train, y_test):
    test = []
    train = []
    for i in range(20):
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=i + 1)
        train_model = clf.fit(X_train, y_train)
        score = train_model.score(X_test, y_test)
        # score1 = train_model.score(X_train, y_train)
        test.append(score)
        # train.append(score1)
    plt.figure(figsize=(20, 8), dpi=100)
    plt.plot(range(1, 21), test, color='green', label='training accuracy')
    # plt.plot(range(1, 21), train, color='red', label='test accuracy')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Tree Depth')
    plt.title("Learning Curve")
    plt.show()


# 进行网格搜索，寻找最优参数
def gridSearchCV(Xtrain, Ytrain):
    # gini_thresholds = np.linspace(0, 0.5, 20)  # 基尼系数的边界
    # entropy_thresholds = np.linespace(0, 1, 50)

    # 一串参数和这些参数对应的，我们希望网格搜索来搜索的参数的取值范围
    # parameters = {'splitter': ('best', 'random')
    #     , 'criterion': ("gini", "entropy")
    #     , "max_depth": [*range(1, 10)]
    #     , 'min_samples_leaf': [*range(1, 50, 5)]
    #     , 'min_impurity_decrease': [*gini_thresholds]
    #               }

    parameters = {
         'solver': ['lbfgs','sag','newton-cg','liblinear','saga']
    }
    clf = AdaBoostClassifier(n_estimators=10)   # 实例化Adaboost
    GS = GridSearchCV(clf, parameters, cv=10)  # 实例化网格搜索，cv指的是交叉验证
    GS.fit(Xtrain, Ytrain)

    print(GS.best_params_)  # 从我们输入的参数和参数取值的列表中，返回最佳组合

    print(GS.best_score_)  # 网格搜索后的模型的评判标准


load_start = time.time()

# 加载数据
raw_data_filename = "data/clearData/total_expend.csv"
print("Loading raw data...")
raw_data = pd.read_csv(raw_data_filename, header=None,low_memory=False)

# 随机抽取比例
# raw_data=raw_data.sample(frac=0.01)

# 查看标签数据
last_column_index = raw_data.shape[1] - 1
print("print data labels:")
print(raw_data[last_column_index].value_counts())
load_end = time.time()


# 非数值转换为数值型数据
print("Transforming data...")
raw_data[last_column_index], attacks = pd.factorize(raw_data[last_column_index], sort=True)

# 对原始数据进行切片，分离出特征和标签，第1~41列是特征，第42列是标签
features = raw_data.iloc[:, :raw_data.shape[1] - 1]  # pandas中的iloc切片是完全基于位置的索引
labels = raw_data.iloc[:, raw_data.shape[1] - 1:]

# 数据标准化
# features = preprocessing.scale(features)
# features = pd.DataFrame(features)

# 数据归一化
# min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# features = min_max_scaler.fit_transform(features)
# features = pd.DataFrame(features)

# 数据正则化
features = preprocessing.normalize(features, norm='l1')
features = pd.DataFrame(features)

# 将多维的标签转为一维的数组
labels = labels.values.ravel()

# 将数据分为训练集和测试集,并打印维数
df = pd.DataFrame(features)
X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=0.8, test_size=0.2, stratify=labels)
# print("X_train,y_train:", X_train.shape, y_train.shape)
# print("X_test,y_test:", X_test.shape, y_test.shape)

dataPro_end = time.time()

# 训练模型
print("Training model...")
clf = AdaBoostClassifier(n_estimators=10)
trained_model = clf.fit(X_train, y_train)
print("Score:", trained_model.score(X_train, y_train))
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
print("Predict accuracy of AdaBoost: ", np.mean(list_diag) / np.mean(list_raw_sum))

# sklearn2pmml(trained_model, "DecisionTree.pmml")

preData_end = time.time()
print("加载时间:%fs" %(load_end-load_start))
print("数据处理时间:%fs" %(dataPro_end-load_end))
print("训练模型时间:%fs" %(trainData_end-dataPro_end))
print("测试数据时间:%fs" %(preData_end-trainData_end))
print("总时间:%fs" %(preData_end-load_start))

# 绘制学习曲线
# plotLearningCurve(X_train, X_test, y_train, y_test)

# 网格搜索
# gridSearchCV(X_train,y_train)
