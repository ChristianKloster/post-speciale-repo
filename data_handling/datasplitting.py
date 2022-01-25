import pandas as pd
import math
import random


def k_split(data, k: int) -> None:
    """
    Splitting some data into k splits. Splitting is performed such that all data for one user
    is only in 1 split (either test or train).

    :param k: how many splits to perform
    :param path_to_data: path of the data to split
    """
    data.columns = ['uid', 'iid', 'count']
    sorted_data = data.sort_values('user')

    users = sorted_data['user'].unique()
    # shuffling users
    random.shuffle(users)

    n_users = users.size
    users_in_split = math.floor(n_users / k)

    for split in range(k):
        # computing users to use in this split
        test_users = users[split * users_in_split: (split + 1) * users_in_split]
        train_users = [u
                       for u in users
                       if u not in test_users]
        # collecting data for the users in this split
        test_data = sorted_data[sorted_data['user'].isin(test_users)]
        train_data = sorted_data[sorted_data['user'].isin(train_users)]

        # saving data as csv file in the data directory
        test_data.to_csv(f'../data/ciao/tenkfold/{split + 1}.csv', index=None, header=True)
        train_data.to_csv(f'data/train{split + 1}.csv', index=None, header=True)


train = pd.read_csv('../data/CiaoDVD/ciao_DVD_75.txt')
test = pd.read_csv('../data/CiaoDVD/ciao_DVD_25.txt')
data = train.append(test)

k_split(data, 10)
