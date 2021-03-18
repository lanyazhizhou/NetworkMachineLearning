import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle

def writeData(file):
    print("Loading raw data...")
    raw_data = pd.read_csv(file, header=None,low_memory=False)
    return raw_data

def lookData(raw_data):
    # 打印数据集的标签数据数量
    last_column_index = raw_data.shape[1] - 1
    print(raw_data[last_column_index].value_counts())
    # 取出数据集标签部分
    labels = raw_data.iloc[:, raw_data.shape[1] - 1:]
    # 多维数组转为以为数组
    labels = labels.values.ravel()
    label_set = set(labels)
    return label_set


# 将大的数据集根据标签特征分为num类（特征数量），存储到lists集合中
def separateData(raw_data):
    # dataframe数据转换为多维数组
    lists=raw_data.values.tolist()
    temp_lists=[]
    # 生成15个空的list集合，用来暂存生成的15种特征集
    for i in range(0,15):
        temp_lists.append([])
    # 得到raw_data的数据标签集合
    label_set = lookData(raw_data)
    # 将无序的数据标签集合转换为有序的list
    label_list = list(label_set)
    for i in range(0,len(lists)):
        # 得到所属标签的索引号
        data_index = label_list.index(lists[i][len(lists[0])-1])
        temp_lists[data_index].append(lists[i])
        if i%5000==0:
            print(i)
    saveData(temp_lists,'data/expendData/')
    return temp_lists

# 将lists分批保存到file文件路径下
def saveData(lists,file):
    label_set = lookData(raw_data)
    label_list = list(label_set)
    for i in range(0,len(lists)):
        save = pd.DataFrame(lists[i])
        file1 = file+label_list[i]+'.csv'
        save.to_csv(file1,index=False,header=False)

# lists存储着num类数据集，将数据集数量少的扩充到至少不少于5000条，然后存储起来。
def expendData(lists):
    totall_list = []
    for i in range(0,len(lists)):
        while len(lists[i])<5000:
            lists[i].extend(lists[i])
        print(i)
        totall_list.extend(lists[i])
    saveData(lists,'data/expendData/')
    save = pd.DataFrame(totall_list)
    file = 'data/expendData/totall_extend.csv'
    save.to_csv(file, index=False, header=False)


# 按行合并多个Dataframe数据(行数据)
def mergeData():
    monday = writeData("data\MachineLearningCVE\Monday-WorkingHours.pcap_ISCX.csv")
    monday = monday.drop([0]) #剔除第一行特征介绍
    friday1 = writeData("data\MachineLearningCVE\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    friday1 = friday1.drop([0])
    friday2 = writeData("data\MachineLearningCVE\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
    friday2 = friday2.drop([0])
    friday3 = writeData("data\MachineLearningCVE\Friday-WorkingHours-Morning.pcap_ISCX.csv")
    friday3 = friday3.drop([0])
    thursday1 = writeData("data\MachineLearningCVE\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
    thursday1 = thursday1.drop([0])
    thursday2 = writeData("data\MachineLearningCVE\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
    thursday2 = thursday2.drop([0])
    tuesday = writeData("data\MachineLearningCVE\Tuesday-WorkingHours.pcap_ISCX.csv")
    tuesday = tuesday.drop([0])
    wednesday = writeData("data\MachineLearningCVE\Wednesday-workingHours.pcap_ISCX.csv")
    wednesday = wednesday.drop([0])
    frame = [monday,friday1,friday2,friday3,thursday1,thursday2,tuesday,wednesday]
    result = pd.concat(frame)
    return result

# 将数据合并后，清除CIC-IDS数据集中的脏数据：含有Nan、Infiniti等数据的行数，然后保存为totall
def clearDirtyData():
    # df = mergeData()
    df = writeData('data/compared/totall_Flow3.csv')
    dropList = df[(df[20] == "Infinity") | (df[21] == "Infinity")].index.tolist()
    print(dropList)
    df = df.drop(dropList)
    df = df.drop([0])
    df = df.drop([0,1,2,3,5,6],axis=1)
    file = 'data/compared/totall_Flow1.csv'
    df.to_csv(file, index=False, header=False)
    return df

# 将raw_data的特征数据进行标准化，并保存
def biaozhunData(raw_data):
    dropList=clearDirtyData(raw_data)
    raw_data=raw_data.drop(dropList)
    features = raw_data.iloc[:, :raw_data.shape[1] - 1]  # pandas中的iloc切片是完全基于位置的索引
    labels = raw_data.iloc[:, raw_data.shape[1] - 1:]
    index = features.loc[0]
    features = features.drop([0])
    features = preprocessing.scale(features)
    save = pd.DataFrame(features)
    file = 'data/clearData/biaozhun.csv'
    save.to_csv(file, index=False, header=False)

# 将raw_data的特征数据进行归一化，并保存
def guiyiData(raw_data):
    dropList = clearDirtyData(raw_data)
    raw_data = raw_data.drop(dropList)
    features = raw_data.iloc[:, :raw_data.shape[1] - 1]  # pandas中的iloc切片是完全基于位置的索引
    features = features.drop([0])
    min_max_scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(0, 1))
    save = min_max_scaler.fit_transform(features)
    save = pd.DataFrame(save)
    file = 'data/clearData/guiyi.csv'
    save.to_csv(file, index=False, header=False)

# 将raw_data的特征数据进行正则化，并保存
def zhengzeData(raw_data):
    dropList = clearDirtyData(raw_data)
    raw_data = raw_data.drop(dropList)
    features = raw_data.iloc[:, :raw_data.shape[1] - 1]  # pandas中的iloc切片是完全基于位置的索引
    features = features.drop([0])
    save = preprocessing.normalize(features, norm='l1')
    save = pd.DataFrame(save)
    file = 'data/clearData/zhengze.csv'
    save.to_csv(file, index=False, header=False)

# 读取total数据，将每列特征数据分离开来，然后将标签少于5000条的数据扩充到5000以上，然后合并保存为totall_expend
def balanceData():
    file = 'data/clearData/total.csv'
    raw_data=writeData(file)
    lists=separateData(raw_data)
    expendData(lists)


