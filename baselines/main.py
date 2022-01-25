import pickle
import pandas as pd
from operator import itemgetter
from baselines import BPR, BPR_utils, SBPR_theano
from baselines.QSBPR import QSBPR
from LRMF import *


def run_BPR():
    k_list = [1,2,3,4,5,6,7,8]

    train_data = pd.DataFrame(columns=['uid', 'iid', 'rating'])
    for k in k_list:
        load = pd.read_csv(f'../data/epinion/extreme_folds/epinion_extreme_fold_{k}.csv')
        train_data = train_data.append(load)
    train_data.to_csv(f'../data/epinion/train_8_1_1.csv', index=False)

    test = pd.read_csv(f'../data/epinion/extreme_folds/epinion_extreme_fold_{10}.csv')
    largest_iid = max(max(train_data.iid), max(test.iid))

    train_data, uid_raw_to_inner, iid_raw_to_inner = BPR_utils.load_data_from_csv(f'../data/epinion/train_8_1_1.csv')
    test_data, uid_raw_to_inner, iid_raw_to_inner = BPR_utils.load_data_from_csv(f'../data/epinion/extreme_folds/epinion_extreme_fold_{10}.csv',
                                                                                uid_raw_to_inner, iid_raw_to_inner)

    model = BPR.BPR(32, len(uid_raw_to_inner.keys()), len(iid_raw_to_inner.keys()), largest_iid)

    model.train(train_data)

    for j in [5, 10, 15, 50, 100]:
        ndcg, prec, recall = model.test_cold(test_data, k=j)
        print(f'NDCG@{j}: {ndcg}\tPrecision@{j}: {prec}\tRecall@{j}:{recall}')

def run_SBPR():
    # loading interaction data
    k_list = [1,2,3,4,5,6,7,8]

    train_data = pd.DataFrame(columns=['uid', 'iid', 'rating'])
    for k in k_list:
        load = pd.read_csv(f'../data/epinion/normal_folds/epinion_normal_fold_{k}.csv')
        train_data = train_data.append(load)
    train_data.to_csv(f'../data/epinion/normal_folds/epinion_normal_train_data.csv', index=False)

    test = pd.read_csv(f'../data/epinion/normal_folds/epinion_normal_fold_10.csv')
    largest_iid = max(max(train_data.iid), max(test.iid))

    train_data, uid_raw_to_inner, iid_raw_to_inner = BPR_utils.load_data_from_csv(f'../data/epinion/normal_folds/epinion_normal_train_data.csv')
    test_data, uid_raw_to_inner, iid_raw_to_inner = BPR_utils.load_data_from_csv(f'../data/epinion/normal_folds/epinion_normal_fold_10.csv',
                                                                                uid_raw_to_inner, iid_raw_to_inner)

    #social_data = pd.read_csv('../data/Ciao/new_trust.csv', header=[1,2,3])
    social_data = pd.read_csv('../data/epinion/normal_folds/epinion_normal_coratings.csv')


    #social_data.columns = ['uid', 'sid']
    #social_data.columns = ['uid', 'sid', 'trust']
    social_data.columns = ['trust', 'uid', 'sid']
    #social_data = social_data.drop(['co-ratings'], axis=1)
    #social_data = social_data.drop(['trust'], axis=1)

    user_count = social_data['uid'].unique()

    model = SBPR_theano.SBPR(10, len(uid_raw_to_inner.keys()), len(iid_raw_to_inner.keys()), largest_iid)

    model.train(train_data, social_data)

    for k in [5, 10, 15, 50, 100]:
        ndcg, prec, recall = model.test(test_data, k=k)
        print(f'NDCG@{k}: {ndcg}\tPrecision@{k}: {prec}\tRecall@{k}:{recall}')

def run_SBPR_random():
    # loading interaction data
    k_list = [1,2,3,4,5,6,7,8]

    train_data = pd.DataFrame(columns=['uid', 'iid', 'rating'])
    for k in k_list:
        load = pd.read_csv(f'../data/epinion/extreme_folds/epinion_extreme_fold_{k}.csv')
        train_data = train_data.append(load)
    train_data.to_csv(f'../data/epinion/extreme_folds/epinion_extreme_train_data.csv', index=False)

    test = pd.read_csv(f'../data/epinion/extreme_folds/epinion_extreme_fold_10.csv')
    largest_iid = max(max(train_data.iid), max(test.iid))

    train_data, uid_raw_to_inner, iid_raw_to_inner = BPR_utils.load_data_from_csv(f'../data/epinion/extreme_folds/epinion_extreme_train_data.csv')
    test_data, uid_raw_to_inner, iid_raw_to_inner = BPR_utils.load_data_from_csv(f'../data/epinion/extreme_folds/epinion_extreme_fold_10.csv',
                                                                                uid_raw_to_inner, iid_raw_to_inner)

    social_data = pd.read_csv('../data/epinion/random_trusts_power_ratings.csv')
    social_data.columns = ['uid', 'sid', 'trust']

    model = SBPR_theano.SBPR(10, len(uid_raw_to_inner.keys()), len(iid_raw_to_inner.keys()))

    model.train(train_data, social_data)

    for k in [5, 10, 15, 50, 100]:
        ndcg, prec, recall = model.test(test_data, k=k)
        print(f'NDCG@{k}: {ndcg}\tPrecision@{k}: {prec}\tRecall@{k}:{recall}')


if __name__ == '__main__':
    #run_BPR()
    run_SBPR()
    #run_SBPR_random()
