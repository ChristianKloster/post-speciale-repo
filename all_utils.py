from collections import defaultdict

import typing
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans


def to_ndarray(data: pd.DataFrame):
    """
    :param data: pandas dataframe with user, item, rating as 3 first columns
    :return: m x n numpy array with rating r in index u,i if u, i, r appeared in data. Otherwise 0. Also returns
    dictionaries mapping raw ids of users and items to inner.
    """
    users = data['user'].unique()
    items = data['item'].unique()
    n_users = len(users)
    n_items = len(items)
    raw_to_inner_users = dict(zip(users, range(n_users)))
    raw_to_inner_items = dict(zip(items, range(n_items)))

    result = np.zeros((n_users, n_items))
    for row in data.itertuples(index=False, name='row'):
        inner_uid = raw_to_inner_users[row.user]
        inner_iid = raw_to_inner_users[row.item]
        rating = row.rating

        result[inner_uid, inner_iid] = rating

    return result, raw_to_inner_users, raw_to_inner_items


def load_actuals(data: pd.DataFrame, raw2inner_id_users, raw2inner_id_items):
    n_items = len(raw2inner_id_items)
    n_users = len(raw2inner_id_users)
    actuals = np.zeros(shape=(n_users, n_items))
    for row in data.itertuples(index=False):
        raw_uid = row[0]
        raw_iid = row[1]
        rating = row[2]
        inner_uid = raw2inner_id_users[raw_uid]
        inner_iid = raw2inner_id_items[raw_iid]
        actuals[inner_uid][inner_iid] = rating

    return actuals





def cluster_items(xs: np.ndarray, k: int):
    """
    clusters **xs** into **k** clusters

    :param xs: data to cluster
    :param k: number of clusters
    :rtype: (np.ndarray[float], np.ndarray[int])
    :return: cluster centroids and labels of each x in **xs**
    """
    kmeans = KMeans(n_clusters=k).fit(xs)

    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    return centroids, labels


def build_dataset(filepath: str, header: bool = False):
    """
    Builds a dataset as list of tuples (user, item, rating)

    :param filepath: Path to file
    """

    if not header:
        data = pd.read_csv(filepath, sep='\t', header=None)
        data.columns = ['user', 'item', 'rating', 'timestamp']
    else:
        data = pd.read_csv(filepath, sep=',')

    # TODO: make generic for all types of data
    return list(data[['user', 'item', 'rating']].itertuples(index=False, name=None))


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result

    return timed


# Print iterations progress
def print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', decimals: int = 1,
                       length: int = 100, fill: str = '█', print_end: str = "\r"):
    """
    Call in a loop to create terminal progress bar
    :param iteration: current iteration
    :param total: total iterations
    :param prefix: prefix string
    :param suffix: suffix string
    :param decimals: positive number of decimals in percent complete
    :param length: character length of bar
    :param fill: bar fill character
    :param print_end: end character (e.g. "\r", "\r\n")
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()
