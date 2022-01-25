import pandas as pd
import utils
import scipy.io

train = pd.read_csv('../data/epinion/train.csv')
test = pd.read_csv('../data/epinion/test.csv')

data = train.append(test)
data.columns = ['uid', 'iid', 'rating']

train, test = utils.train_test_split(data)

train.to_csv('../data/epinion/epinion_25_normal.csv', header=True, index=False)
test.to_csv('../data/epinion/epinion_75_normal.csv', header=True, index=False)

print('pd')
