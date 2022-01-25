import os
from collections import defaultdict

from tqdm import tqdm
from scipy.linalg import solve_sylvester
import scipy.sparse
import matplotlib.pyplot as plt
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

implicit = False

def local_questions_vector(candidates: list, entity_embeddings: np.ndarray, max_length: int):
    questions, _ = py_rect_maxvol(entity_embeddings[list(candidates)], maxK=max_length)
    return questions[:max_length].tolist()


class LRMF():
    def __init__(self, data: pd.DataFrame, num_global_questions: int,
                 num_local_questions: int, use_saved_data = False, alpha: float = 0.01, beta: float = 0.01,
                 embedding_size: int = 20, candidate_items: set = None, num_candidate_items: int = 200):
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
                load = pd.read_csv(f'../data/Ciao/normal_folds/ciao_normal_fold_{k}.csv')
                train_data = train_data.append(load)
            self.test_data = pd.read_csv(f'../data/Ciao/normal_folds/ciao_normal_fold_10.csv')
        else:
            self.train_data, self.test_data = utils.train_test_split_user(data)
            #self.train_data.to_csv('LRMF_data/each_movie/each_movie_train.txt', sep=',', index=False)
            #self.test_data.to_csv('LRMF_data/each_movie/each_movie_test.txt', sep=',', index=False)

        self.inner_2raw_uid, self.raw_2inner_uid, self.inner_2raw_iid, self.raw_2inner_iid = utils.build_id_dicts(data)


        self.train_data.iid = [self.raw_2inner_iid[iid] for iid in self.train_data.iid]
        self.test_data.iid = [self.raw_2inner_iid[iid] for iid in self.test_data.iid]
        ###--- Notice ---###

        #self.R = utils.build_interaction_matrix(self.train_data, self.raw_2inner_uid, self.raw_2inner_iid)
        #self.test_R = utils.build_interaction_matrix(self.test_data, self.raw_2inner_uid, self.raw_2inner_iid)

        self.R = pd.pivot_table(self.train_data, values='rating', index='uid', columns='iid').fillna(0)
        self.ratings = self.R.to_numpy()
        self.test_R = pd.pivot_table(self.test_data, values='rating', index='uid', columns='iid').fillna(0)
        self.test_ratings = self.test_R.to_numpy()
        self.train_iids = list(self.R.columns)
        self.test_iids = list(self.test_R.columns)

        self.embedding_size = embedding_size
        self.num_users, self.num_items = self.ratings.shape
        self.num_test_users = self.test_ratings.shape[0]

        self.V = np.random.rand(embedding_size, self.num_items)  # Item representations
        self.alpha = alpha
        self.beta = beta

        ###--- Notice ---###
        ### Make sure the candidate items are produced recently or you produce them again when using this option.
        if candidate_items is not None:
            self.candidate_items = candidate_items
        else:
            self.candidate_items = self._find_candidate_items()
            with open(f'../data/Ciao/normal_folds/LRMF_normal_candidate_set.csv', 'wb') as f:
                pickle.dump(self.candidate_items, f)

    def fit(self, tol: float = 0.01, maxiters: int = 5):

        users = [u for u in range(self.num_users)]
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
            maxvol_representatives, _ = maxvol.maxvol(self.V.T)

            tree = self._grow_tree(users, set(self.candidate_items), 0, maxvol_representatives, [])

            self.V = self._learn_item_profiles(tree)

            epoch_loss = self._compute_loss(tree)

            loss.append(epoch_loss)
            #ten, fifty, hundred = self.evaluate(tree)
            ten = self.evaluate(tree)
            print('bp')

            if epoch_loss < best_loss:
                best_epoch = epoch
                best_loss = epoch_loss
                best_tree = tree
                best_ten = ten
               # best_fifty = fifty
               # best_hundred = hundred
                best_V = self.V

        self.tree = best_tree
        self.V = best_V
        self.best_ten = best_ten
        self.best_fifty = best_fifty
        self.best_hundred = best_hundred
        self.store_model(f'../data/Ciao/normal_folds/LRMF_normal_best_model.pkl')

    def _find_candidate_items(self):
        # building item-item colike network
        # assumes that train_data.iid is inner_ids
        colike_graph = utils.build_colike_network(self.train_data)
        # computing divranks for each raw iid
        divranks = dr.divrank(colike_graph)
        # sorting raw iids based on their divrank score
        sorted_candidate_items = sorted(divranks, key=lambda n: divranks[n], reverse=True)
        # return the top num_candidate_items
        return sorted_candidate_items[:self.num_candidate_items]

    def _grow_tree(self, users, items: set, depth: int,
                   maxvol_iids: list, global_representatives: list):
        '''
        :param users: list of uids
        :param items: list of iids on candidate items
        :param depth: depth of tree
        :param global_representatives: items asked previously (defaults to None)
        :return: Tree
        '''
        current_node = Tree.Node(users, None, None, None, None)
        best_question, like, dislike = None, None, None

        local_representatives = local_questions_vector(self.candidate_items, self.V.T, self.num_local_questions)
        current_node.set_locals(local_representatives)  # set the best local questions to ask in this node
        current_node.set_globals(global_representatives)  # set the global questions asked to get here

        if depth == self.num_global_questions:
            current_node.question = current_node.global_questions[-1:]
            B = self._build_B(users, local_representatives, global_representatives)
            current_node.set_transformation(self._solve_sylvester(B, users))

        if depth < self.num_global_questions:
            # computes loss with equation 11 for each candidate item
            min_loss, best_question, best_locals = np.inf, None, []
            for item in tqdm(self.candidate_items, desc=f'[Selecting question at depth {depth} ]'):
                like, dislike = self._split_users(users, item)
                loss = 0

                # Some solutions uses a padding of -1 for the global questions here
                for group in [like, dislike]:
                    rest_candidates = [i for i in self.candidate_items if not i == item]
                    local_questions = local_questions_vector(rest_candidates, self.V.T, self.num_local_questions)
                    loss += self._group_loss(group, global_representatives, local_questions)

                if loss < min_loss:
                    min_loss = loss
                    best_question = item
                    best_locals = local_questions
            if best_question is None:
                print('Oh shit, this should not happen :(((')


            # update a copy of global questions with the new best item
            g_r = list(global_representatives).copy()
            g_r.append(best_question)
            current_node.question = best_question

            # notice, if one of the splits have no users, we cant build a transformation matrix.
            # therefore we don't split but still increase the amount of global questions asked.
            U_like, U_dislike = self._split_users(users, best_question)
            if not U_like or not U_dislike:
                print('could not split')
                print(current_node.question)
                current_node.child = self._grow_tree(users, items - {best_question}, depth + 1, maxvol_iids, g_r)

            else:
                # overrides the local questions.
                current_node.set_locals(best_locals)
                # append the best question to the global questions asked so far

                # calculate the transformation matrix
                # B = self._build_B(users, best_locals, g_r)
                # current_node.transformation = self._solve_sylvester(B, users)
                print(current_node.question)
                current_node.like = self._grow_tree(U_like, items - {best_question}, depth + 1, maxvol_iids,
                                   g_r)
                current_node.dislike = self._grow_tree(U_dislike, items - {best_question}, depth + 1, maxvol_iids,
                                      g_r)
        return current_node

    def _group_loss(self, users, global_representatives, local_representatives):
        B = self._build_B(users, local_representatives, global_representatives)
        T = self._solve_sylvester(B, users)

        Rg = self.ratings[users]
        pred = B @ T @ self.V
        loss = ((Rg - pred) ** 2).sum()
        regularisation = T.sum() ** 2

        return loss + regularisation

    def _build_B(self, users, local_representatives, global_representatives):
        # B = [U1, U2, e]
        U1 = self.ratings[users, :][:, global_representatives]
        U2 = self.ratings[users, :][:, local_representatives]
        return np.hstack((U1, U2, np.ones(shape=(len(users), 1))))

    def _solve_sylvester(self, B, users):
        T = solve_sylvester(B.T @ B,
                            self.alpha * inv(self.V @ self.V.T),
                            B.T @ self.ratings[users] @ self.V.T @ inv(self.V @ self.V.T))

        return T

    def _split_users(self, users, iid):
        if implicit:
            like = [uid for uid in users if self.ratings[uid, iid] == 1]
            dislike = list(set(users) - set(like))
        else:
            like = [uid for uid in users if self.ratings[uid, iid] >= 4] # inner uids of users who like inner iid
            dislike = list(set(users) - set(like))  # inner iids of users who dislike inner iid

        return like, dislike

    def _learn_item_profiles(self, tree):
        S = np.zeros(shape=(self.num_users, self.embedding_size))

        for user in tqdm(range(self.num_users), desc="[Optimizing entity embeddings...]"):
            leaf = Tree.traverse_a_user(user, self.ratings[user], tree)
            B = self._build_B([user], leaf.local_questions, leaf.global_questions)

            try:
                S[user] = B @ leaf.transformation
            except ValueError:
                print('Arrhhh shit, here we go again')

        return (inv(S.T @ S + self.beta * np.identity(self.embedding_size)) @ S.T @ self.R).to_numpy()

    def _compute_loss(self, tree):
        if tree.is_leaf():
            B = self._build_B(tree.users, tree.local_questions, tree.global_questions)
            pred = B @ tree.transformation @ self.V
            Rg = self.ratings[tree.users]
            return norm(Rg[Rg.nonzero()] - pred[Rg.nonzero()]) + self.alpha * norm(tree.transformation)

        else:
            if tree.child == None:
                return self._compute_loss(tree.like) + self._compute_loss(tree.dislike)
            else:
                return self._compute_loss(tree.child)


    def evaluate(self, tree):
        item_profiles = self.V.T

        test_users = range(self.num_test_users)
        user_profiles = pd.DataFrame(data=0, index=range(self.num_test_users), columns=range(20))

        u_a_empty = []
        for user in test_users:
            user_profiles.iloc[user] = self.interview_new_user(self.test_ratings[user], u_a_empty, tree)

        pred = user_profiles @ item_profiles.T

        print(f'----- Computing metrics for k10')
        m10 = eval2.Metrics2(np.array(pred), self.test_ratings, 10, 'ndcg,precision,recall').calculate()
       # print(f'----- Computing metrics for k50')
       # m50 = eval2.Metrics2(np.array(pred), self.test_ratings, 50, 'ndcg,precision,recall').calculate()
       # print(f'----- Computing metrics for k100')
       # m100 = eval2.Metrics2(np.array(pred), self.test_ratings, 100, 'ndcg,precision,recall').calculate()

        return m10#, m50, m100

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

                    # If we cannot get an answer from the arguments, return the question
                    except IndexError:
                        return local_question

            # If we have asked all of our questions, return the transformed user vector
            else:
                user_vector = [a for a in user_answers]
                user_vector.append(1)  # Add bias
                return np.array(user_vector) @ tree.transformation

        # find answer to global question
        try:
            answer = actual_answers[tree.question]
        except IndexError:
            print(f'item {tree.question} was not found in test, setting answer to 0')
            answer = 0

        u_a = user_answers.copy()
        u_a.append(answer)

        if implicit:
            if tree.child == None:
                if answer == 1:
                    return self.interview_new_user(actual_answers, u_a, tree.like)
                if answer == 0:
                    return self.interview_new_user(actual_answers, u_a, tree.dislike)
            else:
                return self.interview_new_user(actual_answers, u_a, tree.child)

        else:
            if tree.child == None:
                if answer >= 3:
                    return self.interview_new_user(actual_answers, u_a, tree.like)
                if answer < 3:
                    return self.interview_new_user(actual_answers, u_a, tree.dislike)
            else:
                return self.interview_new_user(actual_answers, u_a, tree.child)

    def store_model(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)



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

    #data = pd.read_csv('LRMF_data/each_movie/eachmovie_triple', sep='\s+', names=['iid', 'uid', 'rating'])
    """
    data = pd.read_csv('../LRMF/ciao/ratings.csv')


    "trust = pd.read_csv('../LRMF/LRMF_data/ciao_explicit/new_trust.csv')

    inner_uids = list(range(data.uid.unique().shape[0]))
    raw_uids = sorted(list(data.uid.unique()))
    raw_2_inner_uid = dict(zip(raw_uids, inner_uids))

    #data.uid = [raw_2_inner_uid[uid] for uid in data.uid]

    trust.uid = [raw_2_inner_uid[uid] for uid in trust.uid]
    trust.sid = [raw_2_inner_uid[uid] for uid in trust.sid]
    trust.to_csv('LRMF_data/ciao_explicit/trust_extreme.csv', index=False)
    """

    """
    raw_iids = sorted(list(data.iid.unique()))
    inner_iids = list(range(data.iid.unique().shape[0]))
    raw_2_inner_iid = dict(zip(raw_iids, inner_iids))

    data.iid = [raw_2_inner_iid[iid] for iid in data.iid]

    train_data, test_data = utils.train_test_split(data)

    train_data = train_data.astype(int)
    test_data = test_data.astype(int)

    train_iids = train_data.iid.unique()
    test_iids = test_data.iid.unique()
    test_item_not_in_train = list(np.setdiff1d(train_iids, test_iids))

    test_data = test_data[~test_data.iid.isin(test_item_not_in_train)]
    print('bp')
    
    train_data.to_csv('LRMF_data/ciao_implicit/training_ciao_implicit_25_75.csv', index=False)
    test_data.to_csv('LRMF_data/ciao_implicit/testing_ciao_implicit_25_75.csv', index=False)
    #trust.to_csv('LRMF_data/each_movie/socials_updated_ids.csv', index=False)

    train_data = pd.read_csv('LRMF_data/each_movie/training_each_movie_25_75.csv')
    test_data = pd.read_csv('LRMF_data/each_movie/testing_each_movie_25_75.csv')

    train_iids = train_data.iid.unique()
    test_iids = test_data.iid.unique()

    raw_iids = sorted(list(train_iids))
    inner_iids = list(range(train_iids.shape[0]))
    raw_2_inner_iid = dict(zip(raw_iids, inner_iids))

    train_data.iid = [raw_2_inner_iid[iid] for iid in train_data.iid]
    test_data.iid = [raw_2_inner_iid[iid] for iid in test_data.iid]

    train_data.to_csv('LRMF_data/each_movie/training_updated_each_movie_25_75.csv', index=False)
    test_data.to_csv('LRMF_data/each_movie/testing_updated_each_movie_25_75.csv', index=False)
  


    train_data = pd.read_csv('LRMF_data/ciao_explicit/training_data_ciao_explicit_25_75.csv')
    test_data = pd.read_csv('LRMF_data/ciao_explicit/testing_data_ciao_explicit_25_75.csv')

    occurences = train_data.uid.value_counts()
    occurences_2 = occurences.value_counts().sort_index()

    indexes = [index for index in occurences_2.index.values.tolist() if index > 10]

    extreme_users = np.setdiff1d(test_data.uid.unique(), train_data.uid.unique()).size
    ten_plus = 0
    for index in indexes:
        ten_plus += occurences_2.loc[index]

    top_ten = occurences_2[:11]
    top_ten.loc[11] = ten_plus

    names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '10+']

    top_ten = pd.concat([pd.Series([extreme_users]), top_ten])


    plot = top_ten.plot(kind='bar')
    plot.set_xticklabels(names)
    plot.set_title('Ciao Explicit   EachMovie', size='24')
    plt.xlabel('Interactions', size='20')
    plt.ylabel('Users', size='20')
    #plt.ylim((0, 22500))

    plt.show()
    print('bp')

    print('hvad fanden??')
    """

    """
    trust = pd.read_csv('LRMF_data/ciao_explicit/raw_trust_with_removed_friendsships.csv')

    inner_uids = list(range(data.uid.unique().shape[0]))
    raw_uids = sorted(list(data.uid.unique()))
    raw_2_inner_uid = dict(zip(raw_uids, inner_uids))

    data.uid = [raw_2_inner_uid[uid] for uid in data.uid]

    trust.uid = [raw_2_inner_uid[uid] for uid in trust.uid]
    trust.sid = [raw_2_inner_uid[uid] for uid in trust.sid]
    trust.to_csv('LRMF_data/ciao_explicit/raw_trust_with_removed_friendships_updated_ids.csv', index=False)

    raw_iids = sorted(list(data.iid.unique()))
    inner_iids = list(range(data.iid.unique().shape[0]))
    raw_2_inner_iid = dict(zip(raw_iids, inner_iids))

    data.iid = [raw_2_inner_iid[iid] for iid in data.iid]

    train_data, test_data = utils.train_test_split(data)

    train_data = train_data.astype(int)
    test_data = test_data.astype(int)

    train_iids = train_data.iid.unique()
    test_iids = test_data.iid.unique()
    test_item_not_in_train = list(np.setdiff1d(train_iids, test_iids))

    if len(test_item_not_in_train) == 0:
        train_data.to_csv('LRMF_data/ciao_explicit/training_data_ciao_explicit_25_75.csv', index=False)
        test_data.to_csv('LRMF_data/ciao_explicit/testing_data_ciao_explicit_25_75.csv', index=False)
    """
