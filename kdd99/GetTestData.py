import pandas as pd
from sklearn.model_selection import train_test_split

class GetTestData:
    def getTestData(self):
        # 读取数据  kddcup.data_10_percent_corrected,kddcup.data.corrected
        raw_data_filename = "data/test.csv"
        raw_data = pd.read_csv(raw_data_filename, header=None)

        '''
        将非数值型的数据转换为数值型数据
        0,tcp,http,SF,215,45076,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0,0,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,normal.
        '''
        # print("Transforming data...")
        raw_data[1], protocols = pd.factorize(raw_data[1],sort=True)  # factorize()方法将非数值型数据映射成数字，返回值是一个元组
        raw_data[2], services = pd.factorize(raw_data[2],sort=True)
        raw_data[3], flags = pd.factorize(raw_data[3],sort=True)
        raw_data[41], attacks = pd.factorize(raw_data[41],sort=True)

        # 对原始数据进行切片，分离出特征和标签，第1~41列是特征，第42列是标签
        features = raw_data.iloc[:, :raw_data.shape[1] - 1]  # pandas中的iloc切片是完全基于位置的索引
        labels = raw_data.iloc[:, raw_data.shape[1] - 1:]

        # 将多维的标签转为一维的数组
        labels = labels.values.ravel()

        return features,labels

    def createTestData(self):
        # 读取数据  kddcup.data_10_percent_corrected,kddcup.data.corrected
        data_dir = "./data/"
        raw_data_filename = data_dir + "test.csv"
        raw_data = pd.read_csv(raw_data_filename, header=None)
        test_data=raw_data.sample(frac=0.4)
        file = 'extend_data/test1.csv'
        test_data.to_csv(file, index=False, header=False)




# gt = GetTestData()
# gt.createTestData()

