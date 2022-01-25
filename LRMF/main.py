import networkx as nx
import numpy as np
import DivRank as dr
from utils import *

if __name__ == '__main__':
    data = load_data('tmp.csv')
    print('Building network')
    g = build_colike_network(data)
    print('Done with network')
    dr.divrank(g)