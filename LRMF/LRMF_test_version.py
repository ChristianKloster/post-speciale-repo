import os
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
                train_data = pd.read_csv('data/each_movie_train_70_30.csv', sep=',')
                self.train_data = train_data.drop(train_data.columns[0], axis=1)
                test_data = pd.read_csv('data/each_movie_test_70_30.csv', sep=',')
                self.test_data = test_data.drop(test_data.columns[0], axis=1)
        else:
            self.train_data, self.test_data = utils.train_test_split(data)

        self.inner_2raw_uid, self.raw_2inner_uid, self.inner_2raw_iid, self.raw_2inner_iid = utils.build_id_dicts(data)
        #self.R = utils.build_interaction_matrix(self.train_data, self.raw_2inner_uid, self.raw_2inner_iid)
        #self.test_R = utils.build_interaction_matrix(self.test_data, self.raw_2inner_uid, self.raw_2inner_iid)

        self.R = pd.pivot_table(self.train_data, values='count', index='uid', columns='iid').fillna(0)
        self.ratings = self.R.to_numpy().astype('int')
        self.test_R = pd.pivot_table(self.test_data, values='count', index='uid', columns='iid').fillna(0)
        self.test_ratings = self.test_R.to_numpy().astype('int')
        self.train_iids = list(self.R.columns)
        self.test_iids = list(self.test_R.columns)

        self.embedding_size = embedding_size
        self.num_users, self.num_items = self.ratings.shape
        self.num_test_users = self.test_ratings.shape[0]

        self.V = np.random.rand(embedding_size, self.num_items)  # Item representations
        self.alpha = alpha
        self.beta = beta

        if candidate_items is not None:
            self.candidate_items = candidate_items
        else:
            self.candidate_items = self._find_candidate_items()
            with open('data/candidate_ciao_exp_25-75.pkl', 'wb') as f:
                pickle.dump(self.candidate_items, f)

    def fit(self, tol: float = 0.01, maxiters: int = 10):
        users = [u for u in range(self.ratings.shape[0])]

        for epoch in range(4, 5):
        #    maxvol_representatives, _ = maxvol.maxvol(self.V.T)

            with open(f'models/each_movie_tree{epoch}.pkl', 'rb') as f:
                tree = pickle.load(f)
        #    tree = self._grow_tree(users, self.candidate_items, 0, maxvol_representatives, [])
        #    with open(f'models/each_movie_tree{epoch}.pkl', 'wb') as f:
        #        pickle.dump(tree, f)

        #    self.V = self._learn_item_profiles(tree)
        #    with open(f'models/each_movie_items{epoch}.pkl', 'wb') as f:
        #        pickle.dump(self.V, f)
            with open(f'models/each_movie_items{epoch}.pkl', 'rb') as f:
                self.V = pickle.load(f)

            m10, m50, m100 = self.evaluate(tree)
            print('bp')

    def _find_candidate_items(self):
        # building item-item colike network
        colike_graph = utils.build_colike_network(self.train_data)
        # computing divranks for each raw iid
        divranks = dr.divrank(colike_graph)
        # sorting raw iids based on their divrank score
        sorted_candidate_items = sorted(divranks, key=lambda n: divranks[n], reverse=True)
        # taking top num_candidate_items
        raw_candidate_iids = sorted_candidate_items[:self.num_candidate_items]
        # translating raw iids to inner iids
        inner_candidate_iids = [self.raw_2inner_iid[iid] for iid in raw_candidate_iids]
        return set(inner_candidate_iids)

    def _grow_tree(self, users, items: set, depth: int,
                   maxvol_iids: list, global_representatives: list):
        '''
        :param users: list of uids
        :param items: list of iids on candidate items
        :param depth: depth of tree
        :param global_representatives: items asked previously (defaults to None)
        :return: Tree
        '''
        current_node = Tree.Node(users, None, None, None)
        best_question, like, dislike = None, None, None

        local_representatives = local_questions_vector(candidates, self.V.T, self.num_local_questions)
        current_node.set_locals(local_representatives)
        current_node.set_globals(global_representatives)

        if depth == self.num_global_questions:
            B = self._build_B(users, local_representatives, global_representatives)
            current_node.set_transformation(self._solve_sylvester(B, users))

        if depth < self.num_global_questions:
            # computes loss with equation 11 for each candidate item
            min_loss, best_question, best_locals = np.inf, None, []
            for item in tqdm(candidates, desc=f'[Selecting question at depth {depth} ]'):
                like, dislike = self._split_users(users, item)
                loss = 0

                # Some solutions uses a padding of -1 for the global questions here
                for group in [like, dislike]:
                    rest_candidates = [i for i in candidates if not i == item]
                    local_questions = local_questions_vector(rest_candidates, self.V.T, self.num_local_questions)
                    loss += self._group_loss(group, global_representatives, local_questions)

                if loss < min_loss:
                    min_loss = loss
                    best_question = item
                    best_locals = local_questions
            # set current nodes question and best local questions.
            current_node.question = best_question
            current_node.set_locals(best_locals)

            # append the best question to the global questions asked so far
            g_r = list(global_representatives).copy()
            g_r.append(best_question)
            current_node.set_globals(g_r)

            # calculate the transformation matrix
            B = self._build_B(users, best_locals, g_r)
            current_node.transformation = self._solve_sylvester(B, users)

            U_like, U_dislike = self._split_users(users, best_question)
            if not U_like or not U_dislike:
                return current_node

            current_node.like = self._grow_tree(U_like, items - {best_question}, depth + 1, maxvol_iids,
                                   g_r)

            current_node.dislike = self._grow_tree(U_dislike, items - {best_question}, depth + 1, maxvol_iids,
                                      g_r)
        return current_node


    def _evaluate_eq11(self, users, global_representatives, local_representatives):
        B = self._build_B(users, local_representatives, global_representatives)
        T = self._solve_sylvester(B, users)

        Rg = self.ratings[users]
        pred = B @ T @ self.V
        return norm(Rg[Rg.nonzero()] - pred[Rg.nonzero()]) + self.alpha * norm(T)

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
        like = [uid for uid in users if self.ratings[uid, iid] >= 4] # inner uids of users who like inner iid
        dislike = list(set(users) - set(like))  # inner iids of users who dislike inner iid

        return like, dislike

    def _learn_item_profiles(self, tree):
        S = np.zeros(shape=(self.num_users, self.embedding_size))

        for user in tqdm(range(self.num_users), desc="[Optimizing entity embeddings...]"):
            leaf = Tree.traverse_a_user(user, self.ratings, tree)
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
            return self._compute_loss(tree.like) + self._compute_loss(tree.dislike)

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
        print(f'----- Computing metrics for k50')
        m50 = eval2.Metrics2(np.array(pred), self.test_ratings, 50, 'ndcg,precision,recall').calculate()
        print(f'----- Computing metrics for k100')
        m100 = eval2.Metrics2(np.array(pred), self.test_ratings, 100, 'ndcg,precision,recall').calculate()

        return m10, m50, m100

        #return ndcg10 / n_test_users, precision10 / n_test_users, recall10 / n_test_users

    def interview_new_user(self, actual_answers, user_answers, tree):
        if tree.is_leaf():
            # Have we asked all our local questions?
            if len(user_answers) < global_questions + local_questions:
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
        answer = actual_answers[tree.question]

        u_a = user_answers.copy()
        u_a.append(answer)
        if answer >= 4:
            return self.interview_new_user(actual_answers, u_a, tree.like)
        if answer < 4:
            return self.interview_new_user(actual_answers, u_a, tree.dislike)

    def store_model(self, file):
        DATA_ROOT = 'models'
        with open(os.path.join(DATA_ROOT, file), 'wb') as f:
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
    data = pd.read_csv('../data/ciao_explicit_preprocessed/new_ratings.csv')
    with open(f'data/candidate_items_eachmovie.txt', 'rb') as f:
        candidates = pickle.load(f)
    global_questions = 3
    local_questions = 2
    lrmf = LRMF(data, global_questions, local_questions, use_saved_data=True, candidate_items=candidates)
    lrmf.fit()




