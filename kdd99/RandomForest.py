import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.model_selection import train_test_split
import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

'''
使用随机森林的方法训练模型，最后输出的预测精度是Predict accuracy:  0.9999632535697625
'''

# 读取数据
data_dir = "./data/"
raw_data_filename = data_dir + "totall.csv"
print("Loading raw data...")
raw_data = pd.read_csv(raw_data_filename, header=None)

'''
将非数值型的数据转换为数值型数据
0,tcp,http,SF,215,45076,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0,0,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,normal.
'''
print("Transforming data...")
raw_data[1], protocols = pd.factorize(raw_data[1],sort=True)  # factorize()方法将非数值型数据映射成数字，返回值是一个元组
raw_data[2], services = pd.factorize(raw_data[2],sort=True)
raw_data[3], flags = pd.factorize(raw_data[3],sort=True)
raw_data[41], attacks = pd.factorize(raw_data[41],sort=True)

# 对原始数据进行切片，分离出特征和标签，第1~41列是特征，第42列是标签
features = raw_data.iloc[:, :raw_data.shape[1] - 1]  # pandas中的iloc切片是完全基于位置的索引
labels = raw_data.iloc[:, raw_data.shape[1] - 1:]

# 将多维的标签转为一维的数组
labels = labels.values.ravel()

# 将数据分为训练集和测试集,并打印维数
df = pd.DataFrame(features)
X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=0.8, test_size=0.2)


# n_estimators:森林中树的个数;n_jobs:并行作业的数量，为-1时是处理器的核数; random_state:随机种子
print("Training model...")
clf = RandomForestClassifier(n_jobs=-1, random_state=3, n_estimators=100)
trained_model = clf.fit(X_train, y_train)
print("Score:", trained_model.score(X_train, y_train))


model = PMMLPipeline([('RandomForest', RandomForestClassifier(n_jobs=-1, random_state=3, n_estimators=100))])
model.fit(X_train,y_train)
sklearn2pmml(model, './save_model/RandomForest.pmml', with_repr=True)

# predicting
print("Predicting...")
y_pred = clf.predict(X_test)

print("Computing performance metrics...")
results = confusion_matrix(y_test, y_pred)
error = zero_one_loss(y_test, y_pred)

# 根据混淆矩阵求预测精度
list_diag = np.diag(results)
list_raw_sum = np.sum(results, axis=1)
print("Predict accuracy: ", np.mean(list_diag) / np.mean(list_raw_sum))
