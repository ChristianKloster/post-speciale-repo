import pandas as pd

from LRMF import utils

if __name__ == '__main__':
    train = pd.read_csv('data/EATNN_ciao/training_ciao_implicit_25_75.csv')
    train.to_csv('data/EATNN_ciao/training_ciao_implicit_25_75_without_ratings.csv', index=False,
                 columns=['uid', 'iid'])
    test = pd.read_csv('data/EATNN_ciao/testing_ciao_implicit_25_75.csv')
    test.to_csv('data/EATNN_ciao/testing_ciao_implicit_25_75_without_ratings.csv', index=False,
                 columns=['uid', 'iid'])


