import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.model_selection import train_test_split
import pickle

def writeData(file):
    print("Loading raw data...")
    raw_data = pd.read_csv(file, header=None)
    return raw_data

def separateTrainAndTest(raw_data):
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
    print(labels)

    # 将数据分为训练集和测试集,并打印维数
    df = pd.DataFrame(features)
    X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=0.8, test_size=0.2)
    print("X_train,y_train:", X_train.shape, y_train.shape)
    print("X_test,y_test:", X_test.shape, y_test.shape)

def lookData(raw_data):
    labels = raw_data.iloc[:, raw_data.shape[1] - 1:]
    labels = labels.values.ravel()
    labels_list = list(labels)
    label_set = set(labels)
    print("The number of labels：",len(label_set))
    for i in label_set:
        print("%s的样本量是：%s"%(i,labels_list.count(i)))
    return label_set


# 将大的数据集根据标签特征分为23类，存储到lists集合中
def separateData(raw_data):
    lists=raw_data.values.tolist()
    temp_lists=[]
    # 生成23个空的list集合，用来暂存生成的23种特征集
    for i in range(0,23):
        temp_lists.append([])
    label_set = lookData(raw_data)
    label_list = list(label_set)
    for i in range(0,len(lists)):
        # 得到所属标签的索引号
        data_index = label_list.index(lists[i][41])
        temp_lists[data_index].append(lists[i])
        if i%5000==0:
            print(i)
    saveData(temp_lists,'save_csv/')
    return temp_lists

# 将lists分批保存到file文件路径下
def saveData(lists,file):
    label_set = lookData(raw_data)
    label_list = list(label_set)
    for i in range(0,len(lists)):
        # print(lists[i])
        save = pd.DataFrame(lists[i])
        file1 = file+label_list[i]+'csv'
        save.to_csv(file1,index=False,header=False)

# lists存储着23类数据集，将数据集数量少的扩充到至少不少于5000条，然后存储起来。
def expendData(lists):
    totall_list = []
    for i in range(0,len(lists)):
        while len(lists[i])<5000:
            lists[i].extend(lists[i])
        print(i)
        totall_list.extend(lists[i])
    saveData(lists,'extend_data/')
    save = pd.DataFrame(totall_list)
    file = 'extend_data/totall.csv'
    save.to_csv(file, index=False, header=False)

# 合并数据集
def mergeData(list1,list2):
    d1 = pd.DataFrame(list1)
    d2 = pd.DataFrame(list2)
    save=pd.merge(d1,d2)
    file = 'extend_data/test2.csv'
    save.to_csv(file, index=False, header=False)


def exchange(data,tum):
    tum = tuple(tum)
    print(tum)
    for i in range(0,len(data)):
        data[i] = tum.index(data[i])
        if i%5000==0:
            print(i)
    return data


# 将非数值型数据转换成数值型数据的转换单保存下来
def saveMuData(raw_data):
    '''
    将非数值型的数据转换为数值型数据
    0,tcp,http,SF,215,45076,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0,0,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,normal.
    '''
    print("Transforming data...")
    raw_data[1], protocols = pd.factorize(raw_data[1])  # factorize()方法将非数值型数据映射成数字，返回值是一个元组
    raw_data[2], services = pd.factorize(raw_data[2])
    raw_data[3], flags = pd.factorize(raw_data[3])
    raw_data[41], attacks = pd.factorize(raw_data[41])
    protocols = pd.DataFrame(protocols)
    file = 'save_csv/feishuzhi/protocols.csv'
    protocols.to_csv(file, index=False, header=False)
    services = pd.DataFrame(services)
    file = 'save_csv/feishuzhi/services.csv'
    services.to_csv(file, index=False, header=False)
    flags = pd.DataFrame(flags)
    file = 'save_csv/feishuzhi/flags.csv'
    flags.to_csv(file, index=False, header=False)
    attacks = pd.DataFrame(attacks)
    file = 'save_csv/feishuzhi/attacks.csv'
    attacks.to_csv(file, index=False, header=False)
    return protocols,services,flags,attacks


# 将非数值型数据转换成数值型数据的转换单保存下来
def saveNumericalData():
    file = "data/totall2.csv"
    raw_data = writeData(file)
    '''
    将非数值型的数据转换为数值型数据
    0,tcp,http,SF,215,45076,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0,0,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,normal.
    '''
    print("Transforming data...")
    raw_data[1], protocols = pd.factorize(raw_data[1])  # factorize()方法将非数值型数据映射成数字，返回值是一个元组
    raw_data[2], services = pd.factorize(raw_data[2])
    raw_data[3], flags = pd.factorize(raw_data[3])
    raw_data[41], attacks = pd.factorize(raw_data[41])
    test,train=train_test_split(raw_data,test_size=0.8,train_size=0.2)
    lookData(raw_data)
    lookData(test)
    lookData(train)
    file = 'save_csv/shuzhi/total2_data.csv'
    raw_data.to_csv(file, index=False, header=False)
    file = 'save_csv/shuzhi/total2_data_test.csv'
    test.to_csv(file, index=False, header=False)
    file = 'save_csv/shuzhi/total2_data_train.csv'
    train.to_csv(file, index=False, header=False)

# 读取数据  kddcup.data_10_percent_corrected,kddcup.data.corrected
# data_dir = "./extend_data/"
raw_data = writeData('extend_data/test1.csv')
# lookData(raw_data)
# raw_data = writeData(file)
# saveMuData(raw_data)
# saveNumericalData()


