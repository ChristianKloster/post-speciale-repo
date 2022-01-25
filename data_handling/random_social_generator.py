import pandas as pd
import numpy as np
import random

def random_power_distribution():
    trust = pd.read_csv('../data/epinion/trust.csv', header=None)
    trust = trust.iloc[1: , :]
    trust_count = trust[0].value_counts(sort=False)
    shuffled_index = trust_count.sample(frac=1)
    shuffled_index.index = trust_count.index

    random_ratings = pd.DataFrame(data=None, columns=['uid', 'sid', 'trust'])
    random_ratings = pd.DataFrame(data=None, columns=['uid', 'sid'])

    for user in trust_count.index:
        trust_count.loc[user] = shuffled_index.loc[user]
        sample = random.sample(trust_count.index.drop(user).tolist(), shuffled_index.loc[user])
        for sid in sample:
            random_ratings = random_ratings.append(pd.DataFrame(data=[[user, sid, 1]], columns=['uid', 'sid', 'trust']))
    random_ratings.to_csv('../data/epinion/random_trusts_power_ratings.csv', index=False)

random_power_distribution()
