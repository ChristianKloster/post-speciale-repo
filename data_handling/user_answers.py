import pickle
from operator import itemgetter

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple

import utils


def create_item_answers_for_single_user(data: List[Tuple[int, int, int]], item_ids: List[int]) -> np.ndarray:
    rating_on_items = np.zeros(len(item_ids))
    for idx, item in enumerate(item_ids):
        rating = [r for (_, i, r) in data if i == (item - 1)] # minus 1 because 1 off
        if rating: # if rating exist
            rating_on_items[idx] = rating[0]

    return rating_on_items


def create_item_answers(filepath: str, item_ids: List[int]) -> Dict[int, np.ndarray]:
    data = utils.build_dataset(filepath)

    res = {}

    users = list(set(map(itemgetter(0), data))) # list of unique user ids
    for user in users:
        user_data = [(u, i, r) for (u, i, r) in data if u == user]

        answers = create_item_answers_for_single_user(user_data, item_ids)
        res[user] = answers

    return res


# TODO: optimize - it takes a long time to run for ML-100k
def create_genre_answers_for_single_user(data: pd.DataFrame, item_data: pd.DataFrame, genres: list):
    item_ids = list(data['item'])

    summed_ratings_dict = {}
    n_genre_ratings_dict = {}
    for genre in genres:
        summed_ratings_dict[genre] = 0
        n_genre_ratings_dict[genre] = 0

    for item in item_ids:
        row = item_data[item_data['movieid'] == item]
        rating = data[data['item'] == item].iloc[0, 2]

        for genre in genres:
            if int(row.loc[:, genre]) == 1:
                summed_ratings_dict[genre] += rating
                n_genre_ratings_dict[genre] += 1
    result = []
    for genre in genres:
        n_ratings = n_genre_ratings_dict[genre]
        summed_ratings = summed_ratings_dict[genre]
        genre_avg = float(summed_ratings) / n_ratings if n_ratings != 0 else 0
        result.append(genre_avg)

    return np.array(result)


def create_genre_averages(data_filepath: str, item_data_filepath: str, genres: pd.DataFrame):
    data = pd.read_csv(data_filepath, sep='\t', header=None)
    data.columns = ['user', 'item', 'rating', 'timestamp']

    item_data = pd.read_csv(item_data_filepath, sep='|', header=None, encoding='ISO-8859-1')
    item_data.columns = ['movieid', 'title', 'release_data', 'video_release_date',
                         'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                         'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                         'Thriller', 'War', 'Western']
    users = data['user'].unique()
    res = {}
    for user in users:
        user_data = data[data['user'] == user]

        user_answers = create_genre_answers_for_single_user(user_data, item_data, list(genres['genre']))
        res[user] = user_answers

    return res


def most_popular_item_ids(filepath: str) -> Union[np.ndarray, int]:
    data = utils.build_dataset(filepath)
    item_ids = list(set(map(itemgetter(1), data))) # unique item ids
    item_n_ratings = np.zeros(max(item_ids)) # 0's of length largest item idx

    for item in item_ids:
        ratings = [r for (_, i, r) in data if i == item]
        if ratings:
            item_n_ratings[item - 1] = len(ratings) # minus 1 because idx starts at 0

    return (-item_n_ratings).argsort()


if __name__ == '__main__':
    # genres = pd.read_csv('../ml-100k/u.genre', sep='|', header=None, skiprows=1)
    # genres.columns = ['genre', 'id']

    filepath = f'../ml-100k/u.data'

    mp_items = most_popular_item_ids(filepath)
    n_items = len(mp_items)
    ignore_n_items = round(n_items * 0.02)
    ignore_10_percent_items = round(n_items * 0.1)
    random98_items = mp_items[ignore_n_items:]
    random90_items = mp_items[ignore_10_percent_items:]
    np.random.shuffle(random98_items)
    np.random.shuffle(random90_items)
    selected_random98_items = random98_items[:18]
    selected_random90_items = random90_items[:18]
    random98_answers = create_item_answers(filepath, item_ids=selected_random98_items)
    random90_answers = create_item_answers(filepath, item_ids=selected_random90_items)

    # genre_answers = create_genre_averages(filepath, item_data_filepath='../ml-100k/u.item', genres=genres)

    # saving answers
    with open(f'../data/random90_items_answers.pickle', 'wb') as f:
        pickle.dump(random90_answers, f)
    with open(f'../data/random98_items_answers.pickle', 'wb') as f:
        pickle.dump(random98_answers, f)

