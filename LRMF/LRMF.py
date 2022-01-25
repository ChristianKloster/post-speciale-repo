import os
import sys
from collections import defaultdict

from tqdm import tqdm
from scipy.linalg import solve_sylvester
import scipy.sparse

import numpy as np

import evaluation.evaluation_v2 as eval2
import sys
import DivRank as dr
import Tree
import pandas as pd
import utils
import maxvol
from maxvol2 import py_rect_maxvol
from numpy.linalg import inv
from numpy.linalg import norm
import time
import pickle

def local_questions_vector(candidates: list, entity_embeddings: pd.DataFrame, max_length: int):
    questions, _ = py_rect_maxvol(entity_embeddings.loc[list(candidates)], maxK=max_length)
    questions = questions[:max_length].tolist()
    return [candidates[i] for i in questions]


class LRMF():
    def __init__(self, data: pd.DataFrame, num_global_questions: int,
                 num_local_questions: int, use_saved_data = False, alpha: float = 0.01, beta: float = 0.01,
                 embedding_size: int = 20, candidate_load: bool = False, num_candidate_items: int = 200):
        '''
        Skriv lige noget om at vi bruger det her paper som kan findes her
        '''
        self.num_candidate_items = num_candidate_items
        self.num_global_questions = num_global_questions
        self.num_local_questions = num_local_questions


        if use_saved_data:
            train_data = pd.DataFrame(columns=['uid', 'iid', 'rating'])
            k_list = [1,2,3,4,5,6,7,8]
            for k in k_list:
                load = pd.read_csv(f'../data/epinion/normal_folds/epinion_normal_fold_{fold}.csv')
                train_data = train_data.append(load)
            self.test_data = pd.read_csv(f'../data/epinion/normal_folds/epinion_normal_fold_10.csv')
        else:
            self.train_data = train_df
            self.test_data = test_df

        self.R = pd.pivot_table(self.train_data.astype(str).astype(int), values='rating', index='uid', columns='iid').fillna(0)
        self.test_R = pd.pivot_table(self.test_data.astype(str).astype(int), values='rating', index='uid', columns='iid').fillna(0)

        self.ratings = self.R.to_numpy()
        self.test_ratings = self.test_R.to_numpy()
        self.train_iids = list(self.R.columns)
        self.test_iids = list(self.test_R.columns)

        self.embedding_size = embedding_size
        self.num_users, self.num_items = self.ratings.shape
        self.users = self.R.index.tolist()
        self.items = self.R.columns.tolist()

        self.num_test_users = self.test_ratings.shape[0]

        self.V = pd.DataFrame(data=np.random.rand(embedding_size, self.num_items), columns=sorted(self.items), dtype='float')  # Item representations
        self.alpha = alpha
        self.beta = beta
        self.store_model('../data/epinion/normal_folds/LRMF_best_model_epinion_normal_folds.pkl')
        if candidate_load:
            with open(f'../data/epinion/normal_folds/LRMF_normal_candidate_set.csv', 'rb') as f:
                self.candidate_items = pickle.load(f)
        else:
            self.candidate_items = self._find_candidate_items()
            with open(f'../data/epinion/normal_folds/LRMF_normal_candidate_set.csv', 'wb') as f:
                pickle.dump(self.candidate_items, f)

    def fit(self, tol: float = 0.01, maxiters: int = 10):
        best_tree = None
        best_V = None
        best_loss = sys.maxsize
        best_ndcg = sys.maxsize
        best_ten = None
        best_fifty = None
        best_hundred = None

        ndcg_list = []
        loss = []
        for epoch in range(maxiters):
            #local_representatives = local_questions_vector(self.candidate_items, self.V.T, self.num_local_questions)

            tree = self._grow_tree(self.users, list(self.candidate_items), 0, [])

            self.V = self._learn_item_profiles(tree)

            epoch_loss = self._compute_loss(tree)

            loss.append(epoch_loss)
            #ten, fifty, hundred = self.evaluate(tree)
            #ten = self.evaluate(tree)
            print('bp')

            if epoch_loss < best_loss:
                best_epoch = epoch
                best_loss = epoch_loss
                best_tree = tree
               # best_ten = ten
               # best_fifty = fifty
               # best_hundred = hundred
                best_V = self.V

        self.tree = best_tree
        self.V = best_V
        self.best_ten = best_ten
        self.best_fifty = best_fifty
        self.best_hundred = best_hundred
        self.store_model('../data/epinion/normal_folds/LRMF_best_model_epinion_normal_folds.pkl')

    def _find_candidate_items(self):
        # building item-item colike network
        colike_graph = utils.build_colike_network(self.train_data)
        # computing divranks for each raw iid
        divranks = dr.divrank(colike_graph)
        # sorting candidates based on divrank score
        sorted_candidate_items = sorted(divranks, key=lambda n: divranks[n], reverse=True)
        # return number of candidate items wanted (200)
        print('candidate set done')
        return sorted_candidate_items[:self.num_candidate_items]

    def _grow_tree(self, users, candidates: list, depth: int, global_representatives: list):

        current_node = Tree.Node(users, None, None, None, None)
        best_question, like, dislike = None, None, None

        if depth == self.num_global_questions:
            local_representatives = local_questions_vector(candidates, self.V.T, self.num_local_questions)
            current_node.set_locals(local_representatives)

            current_node.set_globals(global_representatives)

            current_node.question = global_representatives[-1:]

            B = self._build_B(users, local_representatives, global_representatives)
            current_node.set_transformation(self._solve_sylvester(B, users))

        if depth < self.num_global_questions:
            # computes loss with equation 11 for each candidate item
            min_loss, best_question, best_locals, best_candidates = np.inf, None, [], []

            for item in tqdm(self.candidate_items, desc=f'[Selecting question at depth {depth} ]'):
                like, dislike = self._split_users(users, item)
                loss = 0

                for group in [like, dislike]:
                    rest_candidates = [i for i in self.candidate_items if not i == item]  # Inner candidates
                    local_questions = local_questions_vector(rest_candidates, self.V.T, self.num_local_questions)
                    loss += self._group_loss(group, global_representatives + [item], local_questions)

                if loss < min_loss:
                    min_loss = loss
                    best_question = item
                    best_locals = local_questions
                    best_candidates = [i for i in self.candidate_items if not i == item]

            g_r = list(global_representatives).copy()
            g_r.append(best_question)

            current_node.question = best_question
            current_node.set_locals(best_locals)

            U_like, U_dislike = self._split_users(users, best_question)

            if not U_like or not U_dislike:
                current_node.child = self._grow_tree(users, best_candidates, depth + 1, g_r)

            else:
                current_node.like = self._grow_tree(U_like, best_candidates, depth + 1, g_r)
                current_node.dislike = self._grow_tree(U_dislike, best_candidates, depth + 1, g_r)
        return current_node

    def _group_loss(self, users, global_representatives, local_representatives):
        B = self._build_B(users, local_representatives, global_representatives)
        T = self._solve_sylvester(B, users)

        Rg = self.R.loc[users]
        pred = B @ T @ self.V
        pred.index = Rg.index

        loss = ((Rg - pred) ** 2).values.sum()
        regularisation = T.sum() ** 2

        return loss + regularisation

    def _build_B(self, users, local_representatives, global_representatives):
        # B = [U1, U2, e]
        U1 = self.R.loc[users][global_representatives]
        U2 = self.R.loc[users][local_representatives]
        #U1 = self.ratings[users, :][:, global_representatives]
        #U2 = self.ratings[users, :][:, local_representatives]
        return np.hstack((U1, U2, np.ones(shape=(len(users), 1))))

    def _solve_sylvester(self, B, users):
        T = solve_sylvester(B.T @ B,
                            self.alpha * inv(self.V @ self.V.T),
                            B.T @ self.R.loc[users] @ self.V.T @ inv(self.V @ self.V.T))
        return T

    def _split_users(self, users, iid):
        if implicit:
            like = []
            for uid in users:
                if self.R.loc[uid][iid] >= 1:
                    like = like + [uid]
            dislike = list(set(users) - set(like))
        else:
            like = [uid for uid in users if self.R.loc[uid][iid] >= 4] # inner uids of users who like inner iid
            dislike = list(set(users) - set(like))  # inner iids of users who dislike inner iid

        return like, dislike

    def _learn_item_profiles(self, tree):
        S = pd.DataFrame(index=self.users, columns=range(self.embedding_size), dtype='float')

        for user in tqdm(self.users, desc="[Optimizing entity embeddings...]"):
            leaf = Tree.traverse_a_user(user, self.R.loc[user], tree)
            B = self._build_B([user], leaf.local_questions, leaf.global_questions)

            S.loc[user] = B @ leaf.transformation

        S = S.to_numpy()
        new_V = (inv(S.T @ S + self.beta * np.identity(self.embedding_size)) @ S.T @ self.R.to_numpy())
        return pd.DataFrame(data=new_V, columns=sorted(self.items))

    def _compute_loss(self, tree):
        if tree.is_leaf():
            B = self._build_B(tree.users, tree.local_questions, tree.global_questions)
            pred = B @ tree.transformation @ self.V
            Rg = self.R.loc[tree.users].to_numpy()
            return norm(Rg[Rg.nonzero()] - pred.to_numpy()[Rg.nonzero()]) + self.alpha * norm(tree.transformation)

        else:
            if tree.child == None:
                return self._compute_loss(tree.like) + self._compute_loss(tree.dislike)
            else:
                return self._compute_loss(tree.child)


    def evaluate(self, tree):
        item_profiles = pd.DataFrame(self.V.copy())

        actual = self.test_R.copy().to_numpy()

        users = list(self.test_R.index)
        user_profiles = pd.DataFrame(data=0, index=users, columns=range(20))

        u_a_empty = []


        for user in users:
            try:
                answers = self.R.loc[user]
            except KeyError:
                answers = self.test_R.loc[user]

            user_profiles.loc[user] = self.interview_new_user(answers, u_a_empty, tree)

        pred = user_profiles.to_numpy() @ item_profiles.to_numpy()

        actual = (actual != 0).astype(int)

        print(f'----- Computing metrics for k5')
        m5 = eval2.Metrics2(pred, actual, 5, 'ndcg, precision, recall').calculate()
        print(f'----- Computing metrics for k10')
        m10 = eval2.Metrics2(pred, actual, 10, 'ndcg, precision, recall').calculate()
        print(f'----- Computing metrics for k15')
        m15 = eval2.Metrics2(pred, actual, 15, 'ndcg, precision, recall').calculate()
        print(f'----- Computing metrics for k50')
        m50 = eval2.Metrics2(pred, actual, 50, 'ndcg, precision, recall').calculate()
        print(f'----- Computing metrics for k100')
        m100 = eval2.Metrics2(pred, actual, 100, 'ndcg, precision, recall').calculate()

        return m5, m10, m15, m50, m100

    def interview_new_user(self, actual_answers, user_answers, tree):
        if tree.is_leaf():
            # Have we asked all our local questions?
            if len(user_answers) < len(tree.global_questions) + len(tree.local_questions):
                for local_question in tree.local_questions:
                    # First try to exhaust the available answers
                    try:
                        answer = actual_answers[local_question]
                        u_a = user_answers.copy()
                        u_a.append(answer)
                        return self.interview_new_user(actual_answers, u_a, tree)

                    # If we cannot get an answer from the arguments, set answer to 0
                    except KeyError:
                        u_a = user_answers.copy()
                        u_a.append(0)
                        return self.interview_new_user(actual_answers, u_a, tree)

            # If we have asked all of our questions, return the transformed user vector
            else:
                user_vector = [a for a in user_answers]
                user_vector.append(1)  # Add bias
                return np.array(user_vector) @ tree.transformation

        # find answer to global question
        try:
            answer = actual_answers[tree.question]
        except KeyError:
            print(f'item {tree.question} was not found in test, setting answer to 0')
            answer = 0

        u_a = user_answers.copy()
        u_a.append(answer)

        if implicit:
            if tree.child == None:
                if answer >= 1:
                    return self.interview_new_user(actual_answers, u_a, tree.like)
                else:
                    return self.interview_new_user(actual_answers, u_a, tree.dislike)
            else:
                return self.interview_new_user(actual_answers, u_a, tree.child)

        else:
            if tree.child == None:
                if answer >= 4:
                    return self.interview_new_user(actual_answers, u_a, tree.like)
                else:
                    return self.interview_new_user(actual_answers, u_a, tree.dislike)
            else:
                return self.interview_new_user(actual_answers, u_a, tree.child)

    def store_model(self, file):
        DATA_ROOT = ''
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    def test_cold(self, tree, k=10):
        ndcg_0 = []
        ndcg_1 = []
        ndcg_2 = []
        ndcg_3 = []
        ndcg_4 = []
        ndcg_5 = []
        ndcg_6 = []
        ndcg_7 = []
        ndcg_8 = []
        ndcg_9 = []
        ndcg_10 = []

        users = [u for u in self.test_data['uid'].unique()]
        for user in tqdm(users):
            try:
                train_ratings = self.R.loc[user]
                train_length = train_ratings.astype(bool).sum(axis=0)
            except KeyError:
                train_length = 0

            if (train_length) > 10:
                continue

            u_a_empty = []
            user_profiles = self.interview_new_user(self.test_R.loc[user], u_a_empty, tree)

            actuals = self.test_R.loc[user]
            preds = user_profiles @ self.V.copy()
            top_k_items = preds.argsort()[-k:][::-1]

            tp = 1. / np.log2(np.arange(2, k + 2))
            dcg_relevances = [1 if actuals[p] > 0 else 0 for p in top_k_items]

            DCG = np.sum(dcg_relevances * tp)
            IDCG = tp[:min(len(actuals), k)].sum()

            if train_length == 10: ndcg_10.append(DCG/IDCG)
            if train_length == 9: ndcg_9.append(DCG/IDCG)
            if train_length == 8: ndcg_8.append(DCG/IDCG)
            if train_length == 7: ndcg_7.append(DCG/IDCG)
            if train_length == 6: ndcg_6.append(DCG/IDCG)
            if train_length == 5: ndcg_5.append(DCG/IDCG)
            if train_length == 4: ndcg_4.append(DCG/IDCG)
            if train_length == 3: ndcg_3.append(DCG/IDCG)
            if train_length == 2: ndcg_2.append(DCG/IDCG)
            if train_length == 1: ndcg_1.append(DCG/IDCG)
            if train_length == 0: ndcg_0.append(DCG/IDCG)

        return np.mean(ndcg_0), np.mean(ndcg_1), np.mean(ndcg_2), np.mean(ndcg_3), \
               np.mean(ndcg_4), np.mean(ndcg_5), np.mean(ndcg_6), np.mean(ndcg_7), \
               np.mean(ndcg_8), np.mean(ndcg_9), np.mean(ndcg_10)



def test_tree(users, items, depth):
    like, dislike, best_item = None, None, None

    if depth < 2 or not users:
        best_item = min(items)
        if 3 in users:
            best_item = 3
        split_idx = round(len(users) / 2)

        U_like = users[:split_idx]
        U_dislike = users[split_idx:]

        like = test_tree(U_like, items - {best_item}, depth + 1)
        dislike = test_tree(U_dislike, items - {best_item}, depth + 1)

    return Tree.Node(users, best_item, like, dislike)


if __name__ == '__main__':
    data = None
    implicit = True  # Set to true for consumption instead of explicit results
    global_questions = 3
    local_questions = 2

    ndcg5 = 0
    ndcg10 = 0
    ndcg15 = 0
    ndcg50 = 0
    ndcg100 = 0
    precision5 = 0
    precision10 = 0
    precision15 = 0
    precision50 = 0
    precision100 = 0
    recall5 = 0
    recall10 = 0
    recall15 = 0
    recall50 = 0
    recall100 = 0

    k_list = [1,2,3,4,5,6,7,8]
    train_df = pd.DataFrame(columns=['uid', 'iid', 'rating'])
    #train_df = pd.DataFrame(columns=['work', 'user', 'stars', 'user_id'])
    for fold in k_list:
        load = pd.read_csv(f'../data/epinion/normal_folds/epinion_normal_fold_{fold}.csv')
        #load.to_csv(f'../data/CiaoDVD_done_right/normal_folds/ciao_DVD_normal_fold_{fold}.csv')
        train_df = train_df.append(load)
        #Validation df does not matter here
    #train_df = train_df.drop(['user'], axis=1)
    #train_df = train_df.rename(columns= {'work': 'iid', 'stars': 'rating', 'user_id': 'uid'})
    train_df = train_df.fillna(0)
    train_df['rating'] = train_df['rating'].astype('int')
    train_df['uid'] = train_df['uid'].astype('int')
    test_df = pd.read_csv(f'../data/epinion/normal_folds/epinion_normal_fold_10.csv')
    lrmf = LRMF(train_df, global_questions, local_questions, candidate_load=False)
    lrmf.fit()
    five, ten, fifteen, fifty, hundred = lrmf.evaluate(lrmf.tree)
    print('its done!')

"""
    with open(f"../models/LRMF_best_extreme_model_epinion.pkl", 'rb') as f:
        lrmf = pickle.load(f)

    f5, t10, f15, f50, h100 = lrmf.evaluate(lrmf.tree)
    print(f5)
    print(t10)
    print(f15)
    print(f50)
    print(h100)







    f5, t10, f15, f50, h100 = lrmf.evaluate(lrmf.tree)
    print(f5)
    print(t10)
    print(f15)
    print(f50)
    print(h100)



   # zero, one, two, three, four, five, sixs, seven, eight, nine, ten = lrmf.test_cold(lrmf.tree)
   # print('bp')

    #print('bp')
    #for item in [f5, t10, f15, f50, h100]:
   #     print(str(item))
   #     print(f'ndcg: {item[2]}')
   #     print(f'prec: {item[0]}')
    #    print(f'recall: {item[1]}')

    #lrmf = LRMF(data, global_questions, local_questions, use_saved_data=True, candidate_load=True)
    #lrmf = LRMF(data, global_questions, local_questions, use_saved_data=True)
    #lrmf.fit()
 #   load data


        with open(f"../models/LRMF/ten_fold/best_model_fold_{k}.pkl", 'rb') as f:
            lrmf = pickle.load(f)

        f5, t10, f15, f50, h100 = lrmf.evaluate(lrmf.tree)
        ndcg5 += f5['ndcg']
        ndcg10 += t10['ndcg']
        ndcg15 += f15['ndcg']
        ndcg50 += f50['ndcg']
        ndcg100 += h100['ndcg']
        precision5 += f5['precision']
        precision10 += t10['precision']
        precision15 += f15['precision']
        precision50 += f50['precision']
        precision100 += h100['precision']
        recall5 += f5['recall']
        recall10 += t10['recall']
        recall15 += f15['recall']
        recall50 += f50['recall']
        recall100 += h100['recall']


    print(f'NDCG5{ndcg5/10}')
    print(f'NDCG10{ndcg10/10}')
    print(f'NDCG15{ndcg15/10}')
    print(f'NDCG50{ndcg50/10}')
    print(f'NDCG100{ndcg100/10}')
    print(f'PRECISION5{precision5/10}')
    print(f'PRECISION10{precision10/10}')
    print(f'PRECISION15{precision15/10}')
    print(f'PRECISION50{precision50/10}')
    print(f'PRECISION100{precision100/10}')
    print(f'RECALL5{recall5/10}')
    print(f'RECALL10{recall10/10}')
    print(f'RECALL15{recall15/10}')
    print(f'RECALL50{recall50/10}')
    print(f'RECALL100{recall100/10}')
"""

