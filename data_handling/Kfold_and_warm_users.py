import pandas as pd
import random
import math
import numpy as np
import json
import ast
import utils
import ijson


#train = pd.read_csv('../data/epinion/epinion_25_normal.csv')
#test = pd.read_csv('../data/epinion/epinion_75_normal.csv')
data = pd.read_csv("../data/ML 10m/ratings.csv")
#data = data.drop(['Unnamed: 0'], axis=1)
#data = data.rename(columns={'work': 'iid', 'user': 'uid', 'stars': 'rating'})
"""
trust = pd.read_csv('../data/CiaoDVD/random_trust.csv')
trust = trust[['uid','sid','count']]
trust.to_csv('../data/CiaoDVD/random_trust.csv', index=False)
print('bp')
"""
def filter_warm_users():
    test_data = pd.read_csv(f'../data/epinion/normal_folds/epinion_normal_fold_{10}.csv')
    val_data = pd.read_csv(f'../data/epinion/normal_folds/epinion_normal_fold_{9}.csv')
    train = pd.DataFrame(columns=['uid', 'iid', 'rating'])
    for fold in [1, 2, 3, 4, 5, 6, 7, 8]:
        load = pd.read_csv(f'../data/epinion/normal_folds/epinion_normal_fold_{fold}.csv')
        train = train.append(load)

    uid_counts = train['uid'].value_counts()
    for user in test_data['uid']:
        try:
            if user in train['uid'] and uid_counts.iloc[user] > 10:
                if user in test_data['uid']:
                    test_data = test_data.drop(user)
                if user in val_data['uid']:
                    val_data = val_data.drop(user)
        except:
            print('a user was missing from traindata')
    test_data.to_csv(f'../data/epinion/normal_folds/test_without_warm.csv', index=False)
    val_data.to_csv(f'../data/epinion/normal_folds/val_without_warm.csv', index=False)

def kfold_normal(k=10):
    shuffled_df = data.sample(frac=1)
    folds = np.array_split(shuffled_df, k)

    for idx, fold in enumerate(folds):
        fold.to_csv(f'../data/ML 10m/ml10m_normal_fold_{idx+1}.csv', index=False)


def large_json_to_csv(k=10):
    outfile = pd.DataFrame(None, columns=['work', 'user', 'stars'])
    with open("../data/lthing_data/reviews.json", 'r', 1) as f:
        line_count = 0
        for idx, line in enumerate(f):
            line = eval(line)
            series_raw = pd.Series(line)
            series_trimmed = series_raw.drop(['flags', 'unixtime', 'nhelpful','time', 'comment'])

            outfile = outfile.append(series_trimmed, ignore_index=True)
            if idx % 1000 == 0:
                print(idx)
                if idx % 1000000 == 0:
                    outfile.to_csv(f'../data/Ciao/extreme_folds/ciao_extreme_fold_{idx+1}.csv', index=False)
                    outfile = pd.DataFrame(None, columns=['work', 'user', 'stars'])

        outfile.to_csv(f"../data/lthing_data/lthing_parts/lbthings_{line_count}.csv", index=False)
        print(line_count)

def kfold_extreme(k=10):
    unique_uids = data['uid'].unique()
    random.shuffle(unique_uids, random.seed(2020))

    ids = list(unique_uids)

    n = math.floor(len(ids) / k)
    partitions = [ids[i * n:(i + 1) * n] for i in range((len(ids) + n - 1) // n )]

    # adding leftover items partitions if a leftover partition exists
    if len(partitions) == k + 1:
        for idx, item in enumerate(partitions[k]):
            partitions[idx].append(item)

    for split in range(0, k):
        fold = data.loc[data['uid'].isin(partitions[split])]
        fold = fold.sort_values(['uid', 'iid'])
        fold.to_csv(f'../data/Ciao/extreme_folds/ciao_extreme_fold_{split+1}.csv', index=False)


def string_to_id():
    df = pd.read_csv('../data/lthing_data/lbthings.csv')
    sdf = pd.read_csv('../data/lthing_data/edges.txt', sep=' ', header=None)

    for user in sdf[0].unique():
        try:
            sdf[0] = sdf[0].replace(user, df['user_id'].loc[df['user'] == user].iloc[0])
        except IndexError:
            print(f'found no occurrence of {user}')

    sdf.to_csv(f"../data/lthing_data/id_edges_test.csv", index=False)

    for user in sdf[1].unique():
        try:
            sdf[1] = sdf[1].replace(user, df['user_id'].loc[df['user'] == user].iloc[0])
        except IndexError:
            print(f'found no occurrence of {user}')

    sdf.to_csv(f"../data/lthing_data/id_edges.csv", index=False)

    df['user_id'] = df.user.astype('category').cat.rename_categories(range(1, df.user.nunique()+1))
    df['work'] = df.work.astype('category').cat.rename_categories(range(1, df.work.nunique()+1))
    df['stars'] = df['stars'].fillna(0)
    print('bp')
    df.to_csv(f"../data/lthing_data/lbthings.csv", index=False)




#kfold_extreme()
kfold_normal()
#large_json_to_csv()
#string_to_id()
#filter_warm_users()
