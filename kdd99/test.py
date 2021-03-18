import pandas as pd

# 加载数据
raw_data_filename = "./data/kddcup.data.corrected"
raw_data = pd.read_csv(raw_data_filename,header=None)

# 将非数值类型转换成数值类型
raw_data[41], attacks = pd.factorize(raw_data[41], sort=True)

# 对原始数据进行切片，分离出特征和标签，第1~41列是特征，第42列是标签
features = raw_data.iloc[:, :raw_data.shape[1] - 1]
labels = raw_data.iloc[:, raw_data.shape[1] - 1:]
labels = labels.values.ravel()
