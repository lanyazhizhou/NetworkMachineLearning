import pandas as pd
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn2pmml import sklearn2pmml


# 加载数据
raw_data_filename = "data/clearData/total_expend.csv"
print("Loading raw data...")
raw_data = pd.read_csv(raw_data_filename, header=None,low_memory=False)

# 随机抽取比例
# raw_data=raw_data.sample(frac=0.03)

# 将非数值型的数据转换为数值型数据
# print("Transforming data...")
raw_data[last_column_index], attacks = pd.factorize(raw_data[last_column_index], sort=True)
# 对原始数据进行切片，分离出特征和标签，第1~41列是特征，第42列是标签
features = raw_data.iloc[:, :raw_data.shape[1] - 1]  # pandas中的iloc切片是完全基于位置的索引
labels = raw_data.iloc[:, raw_data.shape[1] - 1:]
# 数据标准化
# features = preprocessing.scale(features)
# features = pd.DataFrame(features)
# 将多维的标签转为一维的数组
labels = labels.values.ravel()

# 将数据分为训练集和测试集,并打印维数
df = pd.DataFrame(features)
X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=0.8, test_size=0.2, stratify=labels)

pipeline = PMMLPipeline([("classifier", DecisionTreeClassifier(criterion='entropy', max_depth=12, min_samples_leaf=1, splitter="best"))])
pipeline.fit(X_train, y_train)
sklearn2pmml(pipeline, "data/pmml/DecisionTreeIris.pmml", with_repr = True)





