import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bpr = pd.read_csv('../data/lthing_data/lthing.csv', dtype=int)
bpr_correct = pd.read_csv('../data/CiaoDVD_done_right/extreme_folds/times_predicted_correctly_bpr.txt', dtype=int)

k_list = [1,2,3,4,5,6,7,8]

train_data = pd.DataFrame(columns=['uid', 'iid', 'rating'])
for k in k_list:
    load = pd.read_csv(f'../data/epinion/extreme_folds/epinion_extreme_fold_{k}.csv')
    train_data = train_data.append(load)

users = train_data['uid'].unique()
items = train_data['iid'].unique()


top_n = bpr_correct['0'].nlargest(100)
indexes = top_n.index.tolist()

most_popular = train_data['iid'].value_counts().iloc[indexes]
predictions = bpr.iloc[indexes]
correct = bpr_correct.iloc[indexes]



'''
#EATNN
EATNN_trust = [0.0327, 0.0400, 0.0467, 0.0664, 0.0779, 0.0138, 0.0104, 0.0094, 0.0058, 0.0040, 0.0453, 0.0675, 0.0910, 0.1777, 0.2399]
EATNN_random = [0.0319, 0.0404, 0.0470, 0.0660, 0.0784, 0.0133, 0.0107, 0.0094, 0.0057, 0.0041, 0.0442, 0.0694, 0.0926, 0.1761, 0.2428]
EATNN_co = [0.0337, 0.04212, 0.0481, 0.0688, 0.0808, 0.0138, 0.0111, 0.0096, 0.0059, 0.0042, 0.0444, 0.0707, 0.0916, 0.1845, 0.2493]

#QEATNN
QEATNN_trust = [0.0377, 0.0454, 0.0506, 0.0706, 0.0823, 0.0167, 0.0121, 0.0102, 0.0061, 0.00428, 0.0505, 0.0743, 0.0923, 0.1805, 0.2432]
QEATNN_random = [0.0352, 0.0443, 0.0496, 0.0687, 0.0818, 0.0150, 0.0120, 0.0100, 0.0059, 0.0043, 0.0462, 0.0732, 0.0919, 0.1759, 0.2471]
QEATNN_co = [0.0347, 0.0442, 0.04972, 0.0690, 0.0804, 0.0154, 0.0122, 0.01019, 0.0060, 0.0041, 0.0452, 0.0738, 0.0935, 0.1792, 0.2400]

#SBPR
SBPR_trust = [0.0237, 0.0324, 0.0377, 0.0540, 0.0680, 0.0084, 0.0076, 0.0068, 0.0042, 0.0033, 0.0321, 0.0581, 0.0770, 0.1508, 0.2316]
SBPR_random = [0.0146,  0.0200, 0.0227, 0.0385, 0.0515, 0.0062, 0.0055, 0.0047, 0.0034, 0.0028, 0.0221, 0.0380, 0.0477, 0.1202, 0.1949]
SBPR_co = [0.0258,  0.0324, 0.0372, 0.0557,  0.0695, 0.0091, 0.0071, 0.0063, 0.0043, 0.0033, 0.0366,  0.0558, 0.0730, 0.1560, 0.2349]

ndcg = ['ndcg5', 'ndcg10', 'ndcg15', 'ndcg50', 'ndcg100']
prec = ['precision5', 'precision10', 'precision15', 'precision50', 'precision100']
recall = ['recall5', 'recall10', 'recall15', 'recall50', 'recall100']

trust_height = EATNN_trust[11:15]
random_height = EATNN_random[11:15]
co_height = EATNN_co[11:15]
'''


bar_width = 0.25

r1 = np.arange(len(top_n))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

plt.bar(r1, most_popular, color='#2d82cc', width=bar_width, edgecolor='white', label='Popularity') #Blue
plt.bar(r2, predictions['0'], color='#2bb356', width=bar_width, edgecolor='white', label='Predicted') #Green
plt.bar(r3, correct['0'], color='#8c5a37', width=bar_width, edgecolor='white', label='Correctly Predicted') #Brown
#plt.xlabel()
plt.show()
plt.xticks()

