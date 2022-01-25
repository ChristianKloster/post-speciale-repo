from collections import defaultdict
import urllib, csv

def load_data_from_csv(csv_file, users_to_i = {}, items_to_i = {}):
    """
      Loads data from a CSV file located at `csv_file`
      where each line is of the form:
        user_id_1, item_id_1
        ...
        user_id_n, item_id_n
      Initial mappings from user and item identifiers
      to integers can be passed using `users_to_i`
      and `items_to_i` respectively.
      This function will return a data array consisting
      of (user, item) tuples, a mapping from user ids to integers
      and a mapping from item ids to integers.
    """
    raw_data = []
    with open(csv_file) as f:
        csvreader = csv.reader(f)
        # skipping first row (header)
        next(csvreader)
        for user, item, rating in csvreader:
            raw_data.append((user, item))
    return load_data_from_array(raw_data, users_to_i, items_to_i)

def load_data_from_array(array, users_to_i = {}, items_to_i = {}):
    """
      Loads data from an array of tuples of the form:
          (user_id, item_id)
      Initial mappings from user and item identifiers
      to integers can be passed using `users_to_i`
      and `items_to_i` respectively.
      This function will return a data array consisting
      of (user, item) tuples, a mapping from user ids to integers
      and a mapping from item ids to integers.
    """
    data = []
    if len(users_to_i.values()) > 0:
        u = max(users_to_i.values()) + 1
    else:
        u = 0
    if len(items_to_i.values()) > 0:
        i = max(items_to_i.values()) + 1
    else:
        i = 0
    for user, item in array:
        if not user in users_to_i:
            users_to_i[user] = u
            u += 1
        if not item in items_to_i:
            items_to_i[item] = i
            i += 1
        data.append((users_to_i[user], items_to_i[item]))
    return data, users_to_i, items_to_i
