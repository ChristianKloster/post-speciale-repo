import pandas as pd

train = pd.read_csv('../data/Ciao/ciao_DVD_25.txt')
test = pd.read_csv('../data/Ciao/ciao_DVD_75.txt')

train['rating'] = 1
test['rating'] = 1

train.to_csv('../data/Ciao/ciao_DVD_train_data_25_implicit.txt', index=None)
test.to_csv('../data/Ciao/ciao_DVD_test_data_75_implicit.txt', index=None)

print('test')
