import os
import sys
import time

import pandas as pd
import scipy.sparse
import tensorflow.compat.v1 as tf
import numpy as np
from tqdm import tqdm

import utils

tf.disable_v2_behavior()

DATA_ROOT = 'data/ciao_from_them'

def load_data(csv_file):
    data = pd.read_csv(csv_file)
    return data


train_data = pd.read_csv('../data/ciao_explicit_preprocessed/training_data_ciao_explicit_25_75.csv')
test_data = pd.read_csv('../data/ciao_explicit_preprocessed/testing_data_ciao_explicit_25_75.csv')
social_data = load_data('../data/ciao_explicit_preprocessed/raw_trust_with_removed_friendships_updated_ids.csv')
n_users = max(len(train_data['uid'].unique()), len(test_data['uid'].unique()))
n_items = max(len(train_data['iid'].unique()), len(test_data['iid'].unique()))


def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()

class EATNN:
    """
    Reproduction of paper: "An Eicient Adaptive Transfer Neural Network for Social-aware Recommendation"
    Authors: Chong Chen, Min Zhang, Chenyang Wang, Weizhi Ma, Minming Li, Shaoping Ma
    url: http://www.thuir.cn/group/~mzhang/publications/SIGIR2019ChenC.pdf
    """
    def __init__(self, n_users, n_items, max_item_pu, max_friend_pu, embedding_size = 64, attention_size = 32):
        """
        Constructs object of EATNN class
        :param n_users: Number of users
        :param n_items: Number of items
        TODO: what is this?
        :param max_item_pu: Max number of items for all users
        TODO: what is this?
        :param max_friend_pu: Max number of friendships for all users
        :param embedding_size: Size of embeddings (d in paper). Defaults to 64
        :param attention_size: Size of output from attention network (k in paper). Defaults to 32
        """
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_size = embedding_size
        self.attention_size = attention_size
        self.max_item_pu = max_item_pu
        self.max_friend_pu = max_friend_pu
        # weighting for entries in R
        self.weight_r = 0.1
        # weighting for entries in X
        self.weight_x = 0.1
        # parameter to adjust weight of social interactions
        self.mu = 0.1
        # regularization terms
        self.lambda_bilinear = [1e-1, 1e-2, 1e-3]

    def _create_variables(self):
        """
        Creates/instantiates variables for model such that we can look them up when learning and making predictions
        """
        # Weights to map user embeddings into embeddings representing shared knowledge in item and social domains (u^c in paper)
        self.uw_c = tf.Variable(tf.random.truncated_normal(shape=[self.n_users, self.embedding_size],
                                                                mean=0.0, stddev=0.01, dtype = tf.float32, name='uc'))
        # Weights to map user embeddings into embeddings representing preferences to items (u^i in paper)
        self.uw_i = tf.Variable(tf.random.truncated_normal(shape=[self.n_users, self.embedding_size],
                                                              mean=0.0, stddev=0.01, dtype=tf.float32, name='ui'))
        # Weights to map user embeddings into embeddings preferences to items (u^s in paper)
        self.uw_s = tf.Variable(tf.random.truncated_normal(shape=[self.n_users, self.embedding_size],
                                                                mean=0.0, stddev=0.01, dtype=tf.float32, name='us'))
        # Embeddings for items (q in paper)
        self.Q = tf.Variable(tf.random.truncated_normal(shape=[self.n_items + 1, self.embedding_size],
                                                        mean=0.0, stddev=0.01, dtype=tf.float32, name='Q'))
        # Embeddings for social interactions (g in paper)
        self.G = tf.Variable(tf.random.truncated_normal(shape=[self.n_users + 1, self.embedding_size],
                                                        mean=0.0, stddev=0.01, dtype=tf.float32, name='G'))

        # Weights used in item domain prediction layer
        self.H_i = tf.Variable(tf.constant(0.01, shape=[self.embedding_size, 1]), name='hi')

        # Weights used in social domain prediction layer
        self.H_s = tf.Variable(tf.constant(0.01, shape=[self.embedding_size, 1]), name='hf')

        # Item domain attention network parameters
        self.W_alpha = tf.Variable(
            tf.random.truncated_normal(shape=[self.embedding_size, self.attention_size], mean=0.0, stddev=tf.sqrt(
                tf.divide(2.0, self.attention_size + self.embedding_size))), dtype=tf.float32, name='Walpha')
        self.B_alpha = tf.Variable(tf.constant(0.00, shape=[self.attention_size]), name='Balpha') # 1 x k
        self.H_alpha = tf.Variable(tf.constant(0.01, shape=[self.attention_size, 1], name='Halpha')) # k x 1

        # Social domain attention network parameters
        self.W_beta = tf.Variable(
            tf.random.truncated_normal(shape=[self.embedding_size, self.attention_size], mean=0.0, stddev=tf.sqrt(
                tf.divide(2.0, self.attention_size + self.embedding_size))), dtype=tf.float32, name='Wbeta')
        self.B_beta = tf.Variable(tf.constant(0.00, shape=[self.attention_size]), name='Bbeta')  # 1 x k
        self.H_beta = tf.Variable(tf.constant(0.01, shape=[self.attention_size, 1], name='Hbeta'))  # k x 1

    def _create_placeholders(self):
        self.input_u = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_i = tf.placeholder(tf.int32, [None, 1], name='input_iid')

        self.input_ur = tf.placeholder(tf.int32, [None, self.max_item_pu], name="input_ur")
        self.input_uf = tf.placeholder(tf.int32, [None, self.max_friend_pu], name="input_ur")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def _item_attentive_transfer(self):
        """
        Item attentive transfer (eq. 5)
        """
        item_w = tf.exp(tf.matmul(tf.nn.relu(tf.matmul(self.u_i, self.W_alpha) + self.B_alpha), self.H_alpha))
        common_w = tf.exp(tf.matmul(tf.nn.relu(tf.matmul(self.u_c, self.W_alpha) + self.B_alpha), self.H_alpha))
        A_item = tf.divide(item_w, item_w + common_w)
        A_common = 1.0 - A_item
        P_iu = A_item * self.u_i + A_common * self.u_c

        return P_iu, item_w

    def _social_attentive_transfer(self):
        """
        Social attentive transfer (eq. 5)
        """
        social_w = tf.exp(tf.matmul(tf.nn.relu(tf.matmul(self.u_s, self.W_beta) + self.B_beta), self.H_beta))
        common_w = tf.exp(tf.matmul(tf.nn.relu(tf.matmul(self.u_c, self.W_beta) + self.B_beta), self.H_beta))
        A_social = tf.divide(social_w, social_w + common_w)
        A_common = 1.0 - A_social
        P_su = A_social * self.u_s + A_common * self.u_c

        return P_su, social_w

    def _create_inference(self):
        """
        Inference used for learning model parameters
        """
        # Mapped embeddings for users (u^c, u^i and u^s)
        self.u_c = tf.nn.embedding_lookup(self.uw_c, self.input_u)
        self.u_c = tf.reshape(self.u_c, [-1, self.embedding_size])
        self.u_i = tf.nn.embedding_lookup(self.uw_i, self.input_u)
        self.u_i = tf.reshape(self.u_i, [-1, self.embedding_size])
        self.u_s = tf.nn.embedding_lookup(self.uw_s, self.input_u)
        self.u_s = tf.reshape(self.u_s, [-1, self.embedding_size])

        # Attentive transferred embeddings for users (p^I_u and p^S_u)
        self.P_iu, self.item_w = self._item_attentive_transfer()
        self.P_su, self.social_w = self._social_attentive_transfer()
        # adding dropout on transferred embeddings to avoid overfitting
        self.P_iu = tf.nn.dropout(self.P_iu, self.dropout_keep_prob)
        self.P_su = tf.nn.dropout(self.P_su, self.dropout_keep_prob)

        # Looking up item embeddings from data
        self.pos_item = tf.nn.embedding_lookup(self.Q, self.input_ur)
        # Items used for this inference
        self.pos_n_ratings = tf.cast(tf.not_equal(self.input_ur, self.n_items), 'float32')
        # Performing matrix multiplication to obtain item embeddings for this inference
        self.pos_item = tf.einsum('ab,abc->abc', self.pos_n_ratings, self.pos_item)
        # Transferred embeddings for items multiplied with item embeddings
        self.pos_r = tf.einsum('ac,abc->abc', self.P_iu, self.pos_item)
        # Need to multiply with H_i as well
        self.pos_r = tf.einsum('ajk,kl->ajl', self.pos_r, self.H_i)
        self.pos_r = tf.reshape(self.pos_r, [-1, max_item_pu])

        # Social embeddings lookup
        self.pos_friend = tf.nn.embedding_lookup(self.G, self.input_uf)
        # Social interactions used for this inference
        self.pos_n_friends = tf.cast(tf.not_equal(self.input_uf, self.n_users), 'float32')
        # Obtaining embeddings for socials used in this inference
        self.pos_friend = tf.einsum('ab,abc->abc', self.pos_n_friends, self.pos_friend)
        # Multiplying with social attentive transferred user embeddings
        self.pos_f = tf.einsum('ac,abc->abc', self.P_su, self.pos_friend)
        # Need to multiply with H_s as well
        self.pos_f = tf.einsum('abc,cd->abd', self.pos_friend, self.H_s)
        self.pos_f = tf.reshape(self.pos_f, [-1, max_social_pu])

    def _prediction(self):
        """
        Computes prediction for R_uv (eq. 6 in paper)
        """
        # Computing dot product p^I_u dot q_v
        dot = tf.einsum('ac,bc->abc', self.P_iu, self.Q)
        # Multiplying with H_i
        pred = tf.einsum('abc,cd->abd', dot, self.H_i)
        pred = tf.reshape(pred, [-1, self.n_items + 1])
        return pred

    def _create_loss(self):
        # Loss for item domain (eq. 13)
        self.loss_item = self.weight_r * tf.reduce_sum(
            tf.reduce_sum(
                tf.reduce_sum(
                    tf.einsum('ab,ac->abc', self.Q, self.Q), 0) *
                    tf.reduce_sum(tf.einsum('ab,ac->abc', self.P_iu, self.P_iu), 0) *
                    tf.matmul(self.H_i, self.H_i, transpose_b=True), 0), 0)
        self.loss_item += tf.reduce_sum((1.0 - self.weight_r) * tf.square(self.pos_r) - 2.0 * self.pos_r)

        # Loss for social domain (eq. 14)
        self.loss_social = self.weight_x * tf.reduce_sum(
            tf.reduce_sum(
                tf.reduce_sum(
                    tf.einsum('ab,ac->abc', self.G, self.G), 0) *
                    tf.reduce_sum(tf.einsum('ab,ac->abc', self.P_su, self.P_su), 0) *
                    tf.matmul(self.H_s, self.H_s, transpose_b=True), 0), 0)
        self.loss_social += tf.reduce_sum((1.0 - self.weight_x) * tf.square(self.pos_f) - 2.0 * self.pos_f)

        # adding l2 regularization on social and item attentive user embeddings
        self.l2_loss = tf.nn.l2_loss(self.P_iu + self.P_su)
        # l2 regularization on item domain specific model parameters
        self.l2_loss_item = tf.nn.l2_loss(self.W_alpha) + tf.nn.l2_loss(self.B_alpha) + tf.nn.l2_loss(self.H_alpha)
        # l2 regularization on social domain specific model parameters
        self.l2_loss_social = tf.nn.l2_loss(self.W_beta) + tf.nn.l2_loss(self.B_beta) + tf.nn.l2_loss(self.H_beta)

        # adding everything together
        self.loss = self.loss_item + self.mu * self.loss_social \
                    + self.lambda_bilinear[0] * self.l2_loss \
                    + self.lambda_bilinear[1] * self.l2_loss_item \
                    + self.lambda_bilinear[2] * self.l2_loss_social

    def build_graph(self):
        """
        Builds the tensorflow graph, i.e. the model
        """
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self.prediction = self._prediction()


def train_step(u_batch, i_batch, s_batch):
    """
    A single training step
    """

    feed_dict = {
        model.input_u: u_batch,
        model.input_ur: i_batch,
        model.input_uf: s_batch,
        model.dropout_keep_prob: 0.3,
    }
    _, loss, wi, ws = sess.run(
        [train_op, model.loss, model.item_w, model.social_w], feed_dict)
    return loss, wi, ws


def eval_coldstart(testset: dict, train_r, test_r):
    """
    Evaluates the model (recall and ndcg)
    """
    test_users = np.fromiter(testset.keys(), dtype=int)
    test_users_reshaped = test_users[:, np.newaxis]
    # groups to hold user idx with ratings. g0 = users with 0 ratings
    g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10 = [], [], [], [], [], [], [], [], [], [], []
    # filling groups
    for user in tqdm(test_users, desc='Grouping users'):
        num_ratings = np.sum(train_r[user])
        if num_ratings > 10: continue
        if num_ratings == 10: g10.append(user)
        if num_ratings == 9: g9.append(user)
        if num_ratings == 8: g8.append(user)
        if num_ratings == 7: g7.append(user)
        if num_ratings == 6: g6.append(user)
        if num_ratings == 5: g5.append(user)
        if num_ratings == 4: g4.append(user)
        if num_ratings == 3: g3.append(user)
        if num_ratings == 2: g2.append(user)
        if num_ratings == 1: g1.append(user)
        if num_ratings == 0: g0.append(user)

    ndcg_list = []
    for group in tqdm([g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10], desc='Computing NDCG@10 for groups'):
        group_reshaped = np.array(group)[:, np.newaxis]
        n_batches = int(len(group) / 128) + 1
        ndcg = []
        for batch_num in range(n_batches):
            start_index = batch_num * 128
            end_index = min((batch_num + 1) * 128, len(group))
            batch = group_reshaped[start_index:end_index]

            if batch.size == 0: continue

            n_batch_users = end_index - start_index

            feed_dict = {
                model.input_u: batch,
                model.dropout_keep_prob: 1.0,
            }

            predictions = sess.run(model.prediction, feed_dict)
            predictions = np.array(predictions)
            # 1 too many predictions. Maybe because of adding 1 extra item embedding earlier.
            predictions = np.delete(predictions, -1, axis=1)

            u_b = group[start_index:end_index]
            # Finding the items we should not evaluate since they're used during training
            idx = np.zeros_like(predictions, dtype=bool)
            idx[train_r[u_b].nonzero()] = True
            # Setting predictions for those items to minus inf such that we will not evaluate them
            predictions[idx] = -np.inf

            idx_top_k_items = np.argpartition(-predictions, 10, 1)
            top_k_predictions = predictions[np.arange(n_batch_users)[:, np.newaxis], idx_top_k_items[:, :10]]
            idx_part = np.argsort(-top_k_predictions, axis=1)
            idx_topk = idx_top_k_items[np.arange(end_index - start_index)[:, np.newaxis], idx_part]

            tp = 1. / np.log2(np.arange(2, 12))
            test_batch = test_r[u_b]

            DCG = (test_batch[np.arange(n_batch_users)[:, np.newaxis], idx_topk].toarray() * tp).sum(axis=1)
            IDCG = np.array([(tp[:min(n, 10)]).sum()
                             for n in test_batch.getnnz(axis=1)])

            ndcg.append(np.mean(DCG / IDCG))

        ndcg_list.append(np.mean(ndcg))
    return ndcg_list

def print_metrics(ndcgs, precisions, recalls):
    ks = [1, 2, 5] #10, 50, 100
    for i, k in enumerate(ks):
        print(f'NDCG@{k}: {ndcgs[i]} \t Precision@{k}: {precisions[i]} \t Recall@{k}: {recalls[i]}')


def get_train_instances(train_r_set: dict, train_s_set: dict):
    user_train, item_train, social_train = [], [], []
    for user in train_r_set.keys():
        user_train.append(user)
        item_train.append(train_r_set[user])
        social_train.append(train_s_set[user])

    user_train = np.array(user_train)
    item_train = np.array(item_train)
    social_train = np.array(social_train)
    user_train = user_train[:, np.newaxis]

    return user_train, item_train, social_train


if __name__ == '__main__':
    random_seed = 2020

    u_train = np.array(train_data['uid'], dtype=np.int32)
    u_test = np.array(test_data['uid'], dtype=np.int32)
    i_train = np.array(train_data['iid'], dtype=np.int32)
    i_test = np.array(test_data['iid'], dtype=np.int32)
    u_friend = np.array(social_data['uid'], dtype=np.int32)
    v_friend = np.array(social_data['sid'], dtype=np.int32)

    n_train_users = np.ones(len(u_train))
    train_r = scipy.sparse.csr_matrix((n_train_users, (u_train, i_train)), dtype=np.int16, shape=(n_users, n_items))
    n_test_users = np.ones(len(u_test))
    test_r = scipy.sparse.csr_matrix((n_test_users, (u_test, i_test)), dtype=np.int16, shape=(n_users, n_items))

    # Building test set
    test_set = {}
    for u in range(len(u_test)):
        user = u_test[u]
        if user in test_set:
            test_set[user].append(i_test[u])
        else:
            test_set[user] = [i_test[u]]

    # Building training set for items
    train_set = {}
    max_item_pu = 0
    for u in range(len(u_train)):
        if u_train[u] in train_set:
            train_set[u_train[u]].append(i_train[u])
        else:
            train_set[u_train[u]] = [i_train[u]]
    for u in train_set.keys():
        if len(train_set[u]) > max_item_pu:
            max_item_pu = len(train_set[u])
    for i in train_set.keys():
        while len(train_set[i]) < max_item_pu:
            train_set[i].append(n_items)

    # Building training for social domain
    train_f_set = {}
    max_social_pu = 0
    for i in range(len(u_friend)):
        if u_friend[i] in train_f_set:
            train_f_set[u_friend[i]].append(v_friend[i])
        else:
            train_f_set[u_friend[i]] = [v_friend[i]]
    for i in train_f_set.keys():
        if len(train_f_set[i]) > max_social_pu:
            max_social_pu = len(train_f_set[i])

    for i in train_set.keys():
        if not i in train_f_set:
            train_f_set[i] = [n_users]
        while len(train_f_set[i]) < max_social_pu:
            train_f_set[i].append(n_users)

    batch_size = 128
    with tf.Graph().as_default():
        tf.set_random_seed(random_seed)

        session_conf = tf.ConfigProto()
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = EATNN(n_users, n_items, max_item_pu, max_social_pu)
            model.build_graph()

            optimizer = tf.train.AdagradOptimizer(learning_rate=0.05, initial_accumulator_value=1e-8).minimize(model.loss)
            train_op = optimizer

            sess.run(tf.global_variables_initializer())

            user_train, item_train, friend_train = get_train_instances(train_set, train_f_set)

            for epoch in range(100):
                print(f'Epoch: {epoch}')
                start_t = _writeline_and_time('\tUpdating...')

                shuffled_indices = np.random.permutation(np.arange(len(user_train)))
                user_train = user_train[shuffled_indices]
                item_train = item_train[shuffled_indices]
                friend_train = friend_train[shuffled_indices]

                n_batches = int(len(user_train) / batch_size)
                loss = 0.0
                for batch_num in range(n_batches):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, len(user_train))

                    u_batch = user_train[start_index:end_index]
                    i_batch = item_train[start_index:end_index]
                    f_batch = friend_train[start_index:end_index]

                    batch_loss, wi, wf = train_step(u_batch, i_batch, f_batch)
                    loss += batch_loss
                print('\r\tUpdating: time=%.2f'
                      % (time.time() - start_t))

                ndcgs = eval_coldstart(test_set, train_r, test_r)
                print(ndcgs)
                print(f'sum:{np.sum(ndcgs)}')
