import pickle

with open('/Users/pc/withH5model/data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

print(data_dict.keys())
