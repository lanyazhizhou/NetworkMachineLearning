import pandas as pd

def writeData(file):
    print("Loading raw data...")
    raw_data = pd.read_csv(file, header=None,low_memory=False)
    return raw_data

def clear(df):
    dropList = df[(df[20] == "NaN")|(df[21] == "Infinity")].index.tolist()
    return dropList

def clear1(raw_data):
    raw_data = raw_data.drop([0])
    raw_data = raw_data.drop([0, 1, 2, 3, 5, 6],axis=1)
    file = 'data/compared/totall_Flow0.csv'
    raw_data.to_csv(file, index=False, header=False)

raw_data = writeData('data/clearData/total.pcap_Flow.csv')
list=clear(raw_data)
print(list)
raw_data = raw_data.drop(list)
clear1(raw_data)

# dropList = raw_data[(raw_data[20] == "NaN")|(raw_data[21] == "Infinity")].index.tolist()
# print(dropList)

# print(raw_data.shape[0])
# save = raw_data.drop(lists)

# file = 'data/compared/totall_Flow0.csv'
# save.to_csv(file, index=False, header=False)
