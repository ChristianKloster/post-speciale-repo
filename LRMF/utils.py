import os
import random
import time
from collections import defaultdict

import networkx as net
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix

DATA_ROOT = 'data/ciao'


def load_data(csv_file: str):
    filepath = os.path.join(DATA_ROOT, csv_file)
    data = pd.read_csv(filepath)

    return data


def build_colike_network(data: pd.DataFrame):
    '''
    Building the colike item-item directed graph. In the graph there is an edge between item1 and item2 with weight
    equal to how many users have coliked those two items. If no users colike two items, there is no edge.

    :rtype: net.DiGraph
    '''
    unique_iids = data['iid'].unique()

    G = net.DiGraph()
    summed_weight_dict = defaultdict(int)
    print('Building colike network')
    for iid_1 in tqdm(unique_iids):
        # uids for users who have consumed iid 1
        users_consumed_iid1 = list(data['uid'][data['iid'] == iid_1])
        # all data for uids who have consumed iid 1
        data_consumed_iid1 = data[data['uid'].isin(users_consumed_iid1)]
        # iids consumed by users who have consumed iid 1
        other_unique_iids = data_consumed_iid1['iid'].unique()
        for iid_2 in other_unique_iids:
            if iid_1 != iid_2:
                # Users who have consumed iid 2
                users_consumed_iid2 = list(data_consumed_iid1['uid'][data_consumed_iid1['iid'] == iid_2])
                # users who have both consumed iid1 and iid2
                colike_users = [u for u in users_consumed_iid1 if u in set(users_consumed_iid2)]
                colike_count = len(colike_users)
                if colike_count > 0:
                    G.add_edge(iid_1, iid_2, weight=colike_count)
                    # used for normalization later
                    summed_weight_dict[iid_1] += colike_count
    # Normalizing weights
    print('Normalizing weights...')
    for (u, v, weight) in G.edges(data='weight'):
        normalized_weight = weight / summed_weight_dict[u]
        G[u][v]['weight'] = normalized_weight

    return G


def build_interaction_matrix(data: pd.DataFrame, raw_2inner_uid: dict, raw_2inner_iid: dict):
    inner_uids = np.array([raw_2inner_uid[uid] for uid in data['uid']])
    inner_iids = np.array([raw_2inner_iid[iid] for iid in data['iid']])
    interactions = np.array(data['rating'], dtype=int)
    R = csr_matrix((interactions, (inner_uids, inner_iids)), shape=(len(raw_2inner_uid), len(raw_2inner_iid)))

    return R

def train_test_split_user(data: pd.DataFrame, test_size: float = 0.75):
    unique_uids = data['uid'].unique()
    random.shuffle(unique_uids, random.seed(2020))
    num_test_users = int(test_size*len(unique_uids))
    test_users, train_users = unique_uids[:num_test_users], unique_uids[num_test_users:]

    train = pd.DataFrame(columns=['uid', 'iid', 'rating'])
    test = pd.DataFrame(columns=['uid', 'iid', 'rating'])
    for uid in train_users:
        user_data = data[data['uid'] == uid]
        train = train.append(user_data)
    for uid in test_users:
        user_data = data[data['uid'] == uid]
        test = test.append(user_data)

    return train, test

def train_test_split(data: pd.DataFrame, test_size: float = 0.75):
    unique_uids = data['uid'].unique()

    # removing items with less than 5 interactions
    data = data.groupby('iid').filter(lambda iid: len(iid) >= 5)

    train = pd.DataFrame(columns=['uid', 'iid', 'rating'])
    test = pd.DataFrame(columns=['uid', 'iid', 'rating'])
    for uid in tqdm(unique_uids):
        # shuffled user data
        user_data = data[data['uid'] == uid].sample(frac=1, random_state=2020)
        # computing how many interactions to use as test
        num_interactions = len(user_data)
        test_interactions = round(num_interactions * test_size)
        # adding approx 75% of user's data to test and 25% of user's data to train
        train = train.append(user_data[test_interactions:])
        test = test.append(user_data[:test_interactions])

    return train, test


def build_id_dicts(data: pd.DataFrame):
    uids = data['uid'].unique()
    iids = data['iid'].unique()
    num_users = len(uids)
    num_items = len(iids)
    inner_2raw_uid = dict(zip(range(num_users), uids))
    raw_2inner_uid = dict(zip(uids, range(num_users)))
    inner_2raw_iid = dict(zip(range(num_items), iids))
    raw_2inner_iid = dict(zip(iids, range(num_items)))

    return inner_2raw_uid, raw_2inner_uid, inner_2raw_iid, raw_2inner_iid
