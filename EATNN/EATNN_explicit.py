import os
import pickle
import sys
import time

from LRMF import LRMF
import pandas as pd
import scipy.sparse
import tensorflow.compat.v1 as tf
import numpy as np

import Tree
import utils

tf.disable_v2_behavior()

def load_data(csv_file):
    data = pd.read_csv(csv_file)
    return data

train = pd.DataFrame(columns=['uid', 'iid', 'rating'])
#val = pd.DataFrame(columns=['uid', 'iid', 'rating'])
#test = pd.DataFrame(columns=['uid', 'iid', 'rating'])

for fold in [1,2,3,4,5,6,7,8]:
    load = pd.read_csv(f'../data/ciaoDVD_done_right/normal_folds/ciao_DVD_normal_fold_{fold}.csv')
    train = train.append(load)

val = pd.read_csv(f'../data/ciaoDVD_done_right/normal_folds/val_without_warm.csv')
test = pd.read_csv(f'../data/ciaoDVD_done_right/normal_folds/test_without_warm.csv')

data = train.append(val.append(test))
#data = pd.read_csv('../data/ciaoDVD_done_right/normal_folds')

#data = pd.read_csv('../LRMF/', sep='\s+', names=['iid', 'uid', 'rating'])
unique_uid = data['uid'].unique()
unique_iid = data['iid'].unique()
num_users = len(unique_uid)
num_items = len(unique_iid)

#test_data, train_data = utils.train_test_split(data)
inner_2raw_uid, raw_2inner_uid, inner_2raw_iid, raw_2inner_iid = utils.build_id_dicts(data)

#train_data = pd.read_csv('../LRMF/LRMF_data/ciao_explicit/exp_ciao_extreme_train_70_30.csv', sep=',')
#test_data = pd.read_csv('../LRMF/LRMF_data/ciao_explicit/exp_ciao_extreme_test_70_30.csv', sep=',')
train_data = train
test_data = test

train_data.iid = [raw_2inner_iid[iid] for iid in train_data.iid]
train_data.uid = [raw_2inner_uid[uid] for uid in train_data.uid]

test_data.iid = [raw_2inner_iid[iid] for iid in test_data.iid]
test_data.uid = [raw_2inner_uid[uid] for uid in test_data.uid]

social_data = pd.read_csv('../data/CiaoDVD_done_right/normal_folds/ciao_DVD_normal_coratings.csv')

interaction_matrix = pd.pivot_table(data, values='rating', index='uid', columns='iid').fillna(0).to_numpy()
#train_data_interaction_matrix = pd.pivot_table(train_data, values='rating', index='uid', columns='iid').fillna(0).to_numpy()
#test_data_interaction_matrix = pd.pivot_table(test_data, values='rating', index='uid', columns='iid').fillna(0).to_numpy()

with open('../models/LRMF/ten_fold/best_model_normal_8_1_1.pkl', 'rb') as f:
    lrmf = pickle.load(f)
tree = lrmf.tree


def writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()


class EATNN:
    """
    Reproduction of paper: "An Efficient Adaptive Transfer Neural Network for Social-aware Recommendation"
    Authors: Chong Chen, Min Zhang, Chenyang Wang, Weizhi Ma, Minming Li, Shaoping Ma
    url: http://www.thuir.cn/group/~mzhang/publications/SIGIR2019ChenC.pdf
    """

    def __init__(self, n_users, n_items, max_questions, max_items, max_friends, embedding_size=64, attention_size=32):
        """
        Constructs object of EATNN class

        :param n_users: Number of users
        :param n_items: Number of items
        :param max_questions: Max number of questions for all users
        :param max_items: Max number of items for all users
        :param max_friends: Max number of friendships for all users
        :param embedding_size: Size of embeddings (d in paper). Defaults to 64
        :param attention_size: Size of output from attention network (k in paper). Defaults to 32
        """
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_size = embedding_size
        self.attention_size = attention_size
        self.max_items = max_items
        self.max_friends = max_friends
        self.max_questions = max_questions
        # weighting for entries in R
        self.weight_r = 0.1
        # weighting for entries in X
        self.weight_x = 0.1
        # weighting for entries in Q
        self.weight_q = 0.1
        # parameter to adjust weight of social interactions
        self.mu = 0.1
        # regularization terms
        self.lambda_bilinear = [1e-3, 1e-1, 1e-2, 1e-1]

    def _create_variables(self):
        """
        Creates/instantiates variables for model such that we can look them up when learning and making predictions
        """
        # Weights to map uid into embeddings representing shared knowledge in item and social domains (u^c in paper)
        self.uw_c = tf.Variable(tf.random.truncated_normal(shape=[self.n_users, self.embedding_size],
                                                           mean=0.0, stddev=0.01, dtype=tf.float32, name='uc'))
        # Weights to map uid into embeddings representing preferences to items (u^i in paper)
        self.uw_i = tf.Variable(tf.random.truncated_normal(shape=[self.n_users, self.embedding_size],
                                                           mean=0.0, stddev=0.01, dtype=tf.float32, name='ui'))
        # Weights to map uid into embeddings preferences to items (u^s in paper)
        self.uw_s = tf.Variable(tf.random.truncated_normal(shape=[self.n_users, self.embedding_size],
                                                           mean=0.0, stddev=0.01, dtype=tf.float32, name='us'))
        # Weights to map uid into embeddings for questionnaires
        self.uw_q = tf.Variable(tf.random.truncated_normal(shape=[self.n_users, self.embedding_size],
                                                           mean=0.0, stddev=0.01, dtype=tf.float32, name='uq'))
        # TODO: figure out why they +1 in sizes
        # Embeddings for items (M x D)
        self.Q = tf.Variable(tf.random.truncated_normal(shape=[self.n_items + 1, self.embedding_size],
                                                        mean=0.0, stddev=0.01, dtype=tf.float32, name='Q'))
        # Embeddings for social interactions (N x D)
        self.G = tf.Variable(tf.random.truncated_normal(shape=[self.n_users + 1, self.embedding_size],
                                                        mean=0.0, stddev=0.01, dtype=tf.float32, name='G'))
        # Embeddings for questionnaires (M x D)
        self.V = tf.Variable(tf.random.truncated_normal(shape=[self.n_items + 1, self.embedding_size],
                                                        mean=0.0, stddev=0.01, dtype=tf.float32, name='V'))

        # Weights used in item domain prediction layer
        self.H_i = tf.Variable(tf.constant(0.01, shape=[self.embedding_size, 1]), name='hi')
        # Weights used in social domain prediction layer
        self.H_s = tf.Variable(tf.constant(0.01, shape=[self.embedding_size, 1]), name='hf')
        # Weights used in questionnaire domain prediction layer
        self.H_q = tf.Variable(tf.constant(0.01, shape=[self.embedding_size, 1]), name='hq')

        # Item domain attention network parameters
        self.W_item = tf.Variable(
            tf.random.truncated_normal(shape=[self.embedding_size, self.attention_size], mean=0.0, stddev=tf.sqrt(
                tf.divide(2.0, self.attention_size + self.embedding_size))), dtype=tf.float32, name='Witem')
        self.B_item = tf.Variable(tf.constant(0.00, shape=[self.attention_size]), name='Bitem')  # 1 x k
        self.H_item = tf.Variable(tf.constant(0.01, shape=[self.attention_size, 1], name='Hitem'))  # k x 1

        # Social domain attention network parameters
        self.W_social = tf.Variable(
            tf.random.truncated_normal(shape=[self.embedding_size, self.attention_size], mean=0.0, stddev=tf.sqrt(
                tf.divide(2.0, self.attention_size + self.embedding_size))), dtype=tf.float32, name='Wsocial')
        self.B_social = tf.Variable(tf.constant(0.00, shape=[self.attention_size]), name='Bsocial')  # 1 x k
        self.H_social = tf.Variable(tf.constant(0.01, shape=[self.attention_size, 1], name='Hsocial'))  # k x 1

        # Questionnaire domain attention network parameters
        self.W_question = tf.Variable(
            tf.random.truncated_normal(shape=[self.embedding_size, self.attention_size], mean=0.0, stddev=tf.sqrt(
                tf.divide(2.0, self.attention_size + self.embedding_size))), dtype=tf.float32, name='Wquestion')
        self.B_question = tf.Variable(tf.constant(0.00, shape=[self.attention_size]), name='Bquestion')  # 1 x k
        self.H_question = tf.Variable(tf.constant(0.01, shape=[self.attention_size, 1], name='Hquestion'))  # k x 1

    def _create_placeholders(self):
        self.input_u = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_i = tf.placeholder(tf.int32, [None, 1], name='input_iid')

        self.input_ui = tf.placeholder(tf.int32, [None, self.max_items], name="input_ui")
        self.input_ur = tf.placeholder(tf.float32, [None, self.max_items], name="input_ur")
        self.input_uf = tf.placeholder(tf.int32, [None, self.max_friends], name="input_uf")
        self.input_uq = tf.placeholder(tf.int32, [None, self.max_questions], name='input_uq')

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def _item_attentive_transfer(self):
        """
        Item attentive transfer (eq. 5)
        """
        # eq. 3 for computing attentions
        item_attention = tf.exp(tf.matmul(tf.nn.relu(tf.matmul(self.u_i, self.W_item) + self.B_item), self.H_item))
        common_attention = tf.exp(tf.matmul(tf.nn.relu(tf.matmul(self.u_c, self.W_item) + self.B_item), self.H_item))
        # eq. 4 for computing weights
        item_weight = tf.divide(item_attention, item_attention + common_attention)
        common_weight = 1.0 - item_weight
        # eq. 5 for computing transferred user embedding
        user_embedding = item_weight * self.u_i + common_weight * self.u_c

        # returning user embedding, so that we can make predictions. Item weight such that we can analyse,
        # when it's low or high
        return user_embedding, item_weight

    def _social_attentive_transfer(self):
        """
        Social attentive transfer (eq. 5)
        """
        # eq. 3 for computing attentions
        social_attention = tf.exp(
            tf.matmul(tf.nn.relu(tf.matmul(self.u_s, self.W_social) + self.B_social), self.H_social))
        common_attention = tf.exp(
            tf.matmul(tf.nn.relu(tf.matmul(self.u_c, self.W_social) + self.B_social), self.H_social))
        # eq. 4 for computing weights
        social_weight = tf.divide(social_attention, social_attention + common_attention)
        common_weight = 1.0 - social_weight
        # eq. 5 for computing transferred user embedding
        user_embedding = social_weight * self.u_s + common_weight * self.u_c

        # returning user embedding, so that we can make predictions. Social weight such that we can analyse,
        # when it's low or high
        return user_embedding, social_weight

    def _questionnaire_attentive_transfer(self):
        """
        Transferring knowledge from questionnaires. Our contribution.
        Works the same way, as the attentive networks in paper.
        """
        # eq. 3 for computing attentions
        q_attention = tf.exp(
            tf.matmul(tf.nn.relu(tf.matmul(self.u_q, self.W_question) + self.B_question), self.H_question))
        common_attention = tf.exp(
            tf.matmul(tf.nn.relu(tf.matmul(self.u_c, self.W_question) + self.B_question), self.H_question))
        # eq. 4 for computing weights
        q_weight = tf.divide(q_attention, q_attention + common_attention)
        common_weight = 1.0 - q_weight
        # eq. 5 for computing transferred user embedding
        user_embedding = q_weight * self.u_q + common_weight * self.u_c

        # returning user embedding, so that we can make predictions. Social weight such that we can analyse,
        # when it's low or high
        return user_embedding, q_weight

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
        # Our contribution (haven't added u^c2 because I figured, if we use the same u^c, we essentially use
        # shared knowledge between all domains. Maybe that is favorable?
        # TODO: We could test with only sharing between item domain and questionnaire domain.
        self.u_q = tf.nn.embedding_lookup(self.uw_q, self.input_u)
        self.u_q = tf.reshape(self.u_q, [-1, self.embedding_size])

        # Attentive transferred embeddings for users (p^I_u and p^S_u)
        self.P_iu, self.item_w = self._item_attentive_transfer()
        self.P_su, self.social_w = self._social_attentive_transfer()
        # Our contribution
        self.P_qu, self.question_w = self._questionnaire_attentive_transfer()

        # adding dropout on transferred embeddings to avoid overfitting
        self.P_iu = tf.nn.dropout(self.P_iu, self.dropout_keep_prob)
        self.P_su = tf.nn.dropout(self.P_su, self.dropout_keep_prob)
        self.P_qu = tf.nn.dropout(self.P_qu, self.dropout_keep_prob)

        # Looking up item embeddings from data
        self.pos_item = tf.nn.embedding_lookup(self.Q, self.input_ui)
        # Items used for this inference
        self.pos_n_ratings = tf.cast(tf.not_equal(self.input_ui, self.n_items), 'float32')
        # Performing matrix multiplication to obtain item embeddings for this inference
        self.pos_item = tf.einsum('ab,abc->abc', self.pos_n_ratings, self.pos_item)
        # Transferred embeddings for items multiplied with item embeddings
        self.pos_r = tf.einsum('ac,abc->abc', self.P_iu, self.pos_item)
        # Need to multiply with H_i as well
        self.pos_r = tf.einsum('ajk,kl->ajl', self.pos_r, self.H_i)
        self.pos_r = tf.reshape(self.pos_r, [-1, max_items])

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
        self.pos_f = tf.reshape(self.pos_f, [-1, max_friends])

        # Questionnaire embeddings lookup
        self.pos_questions = tf.nn.embedding_lookup(self.V, self.input_uq)
        # Answered questions for this inference
        self.pos_n_questions = tf.cast(tf.not_equal(self.input_uq, self.n_items), 'float32')
        # Obtaining embeddings for questions used in this inference
        self.pos_questions = tf.einsum('ab,abc->abc', self.pos_n_questions, self.pos_questions)
        # Multiplying with question attentive transferred user embeddings
        self.pos_q = tf.einsum('ac,abc->abc', self.P_qu, self.pos_questions)
        # Need to multiply with H_q as well
        self.pos_q = tf.einsum('abc,cd->abd', self.pos_questions, self.H_q)
        self.pos_q = tf.reshape(self.pos_q, [-1, self.max_questions])

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
        # Squared error for items we have rating for (where clause)
        zero = tf.constant(0, dtype=tf.float32)
        where = tf.not_equal(self.input_ur, zero)
        self.loss_item = tf.reduce_sum(tf.square(self.input_ur[where] - self.pos_r[where]))

        # Loss for social domain (eq. 14)
        self.loss_social = self.weight_x * tf.reduce_sum(
            tf.reduce_sum(
                tf.reduce_sum(
                    tf.einsum('ab,ac->abc', self.G, self.G), 0) *
                tf.reduce_sum(tf.einsum('ab,ac->abc', self.P_su, self.P_su), 0) *
                tf.matmul(self.H_s, self.H_s, transpose_b=True), 0), 0)
        self.loss_social += tf.reduce_sum((1.0 - self.weight_x) * tf.square(self.pos_f) - 2.0 * self.pos_f)

        # TODO: consider this loss for questionnaire
        self.loss_question = self.weight_q * tf.reduce_sum(
            tf.reduce_sum(
                tf.reduce_sum(
                    tf.einsum('ab,ac->abc', self.V, self.V), 0) *
                tf.reduce_sum(tf.einsum('ab,ac->abc', self.P_qu, self.P_qu), 0) *
                tf.matmul(self.H_q, self.H_q, transpose_b=True), 0), 0)
        self.loss_item += tf.reduce_sum((1.0 - self.weight_q) * tf.square(self.pos_q) - 2.0 * self.pos_q)

        # adding l2 regularization on social, item and questionnaire attentive user embeddings
        self.l2_loss = tf.nn.l2_loss(self.P_iu + self.P_su + self.P_qu)
        # l2 regularization on item domain model parameters
        self.l2_loss_item = tf.nn.l2_loss(self.W_item) + tf.nn.l2_loss(self.B_item) + tf.nn.l2_loss(self.H_item)
        # l2 regularization on social domain model parameters
        self.l2_loss_social = tf.nn.l2_loss(self.W_social) + tf.nn.l2_loss(self.B_social) + tf.nn.l2_loss(self.H_social)
        # l2 regularization on questionnaire domain model parameters
        self.l2_loss_question = tf.nn.l2_loss(self.W_question) + tf.nn.l2_loss(self.B_question) + tf.nn.l2_loss(
            self.H_question)

        # adding everything together
        self.loss = self.loss_item + self.mu * self.loss_social + self.mu * self.loss_question \
                    + self.lambda_bilinear[0] * self.l2_loss \
                    + self.lambda_bilinear[1] * self.l2_loss_item \
                    + self.lambda_bilinear[2] * self.l2_loss_social \
                    + self.lambda_bilinear[3] * self.l2_loss_question

    def build_graph(self):
        """
        Builds the tensorflow graph, i.e. the model
        """
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self.prediction = self._prediction()


def train_step(u_batch, i_batch, r_batch, s_batch, q_batch):
    """
    A single training step
    """

    feed_dict = {
        model.input_u: u_batch,
        model.input_ui: i_batch,
        model.input_ur: r_batch,
        model.input_uf: s_batch,
        model.input_uq: q_batch,
        model.dropout_keep_prob: 0.1,
    }
    _, loss, wi, ws, wq = sess.run(
        [train_op, model.loss, model.item_w, model.social_w, model.question_w], feed_dict)
    return loss, wi, ws, wq


def get_train_instances(train_i_set: dict, train_r_set: dict, train_s_set: dict, train_q_set: dict):
    user_train, rating_train, item_train, social_train, question_train = [], [], [], [], []
    for user in train_i_set.keys():
        user_train.append(user)
        rating_train.append(train_r_set[user])
        item_train.append(train_i_set[user])
        social_train.append(train_s_set[user])
        question_train.append(train_q_set[user])

    user_train = np.array(user_train)
    item_train = np.array(item_train)
    rating_train = np.array(rating_train)
    social_train = np.array(social_train)
    question_train = np.array(question_train)
    user_train = user_train[:, np.newaxis]

    return user_train, item_train, rating_train, social_train, question_train


def eval_step(testset: dict, train_r, test_r, batch_size: int):
    """
    Evaluates the model (recall and ndcg)
    """
    test_users = np.fromiter(testset.keys(), dtype=int)
    n_test_users = len(test_users)
    test_users_reshaped = test_users[:, np.newaxis]

    n_batches = int(n_test_users / batch_size) + 1

    recall10 = []
    recall50 = []
    recall100 = []
    precision10 = []
    precision50 = []
    precision100 = []
    ndcg10 = []
    ndcg50 = []
    ndcg100 = []

    for batch_num in range(n_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, n_test_users)
        u_batch = test_users_reshaped[start_idx:end_idx]

        n_batch_users = end_idx - start_idx

        feed_dict = {
            model.input_u: u_batch,
            model.dropout_keep_prob: 1.0,
        }

        predictions = sess.run(model.prediction, feed_dict)

        u_b = test_users[start_idx:end_idx]
        predictions = np.array(predictions)
        # 1 too many predictions. Maybe because of adding 1 extra item embedding earlier.
        predictions = np.delete(predictions, -1, axis=1)

        # Finding the items we should not evaluate since they're used during training
        idx = np.zeros_like(predictions, dtype=bool)
        idx[train_r[u_b].nonzero()] = True
        # Setting predictions for those items to minus inf such that we will not evaluate them
        predictions[idx] = -np.inf

        # Computing recall and precision
        recall = []
        precision = []
        ndcg = []
        for k in [10, 50, 100]:
            idx_topk_items = np.argpartition(-predictions, k, 1)
            bin_predictions = np.zeros_like(predictions, dtype=bool)
            bin_predictions[np.arange(n_batch_users)[:, np.newaxis], idx_topk_items[:, :k]] = True

            bin_true = np.zeros_like(predictions, dtype=bool)
            bin_true[test_r[u_b].nonzero()] = True

            recommended_and_relevant = (np.logical_and(bin_true, bin_predictions).sum(axis=1)).astype(np.float32)
            n_rel = bin_true.sum(axis=1)
            # Replacing zeros with ones s.t. we avoid division by 0
            n_rel[n_rel == 0] = 1
            local_recall = recommended_and_relevant / n_rel
            recall.append(local_recall)
            precision.append(recommended_and_relevant / k)

            idx_top_k_items = np.argpartition(-predictions, k, 1)
            top_k_predictions = predictions[np.arange(n_batch_users)[:, np.newaxis], idx_top_k_items[:, :k]]
            idx_part = np.argsort(-top_k_predictions, axis=1)
            idx_topk = idx_topk_items[np.arange(end_idx - start_idx)[:, np.newaxis], idx_part]

            tp = 1. / np.log2(np.arange(2, k + 2))
            test_batch = test_r[u_b]

            r_b = test_batch[np.arange(n_batch_users)[:, np.newaxis], idx_topk].toarray()
            DCG = ((r_b != 0).astype(int) * tp).sum(axis=1)
            IDCG = np.array([(tp[:min(n, k)]).sum()
                             for n in test_batch.getnnz(axis=1)])
            ndcg.append(DCG / IDCG)

        recall10.append(recall[0])
        recall50.append(recall[1])
        recall100.append(recall[2])
        precision10.append(precision[0])
        precision50.append(precision[1])
        precision100.append(precision[2])
        ndcg10.append(ndcg[0])
        ndcg50.append(ndcg[1])
        ndcg100.append(ndcg[2])

    recall10 = np.mean(np.hstack(recall10))
    recall50 = np.mean(np.hstack(recall50))
    recall100 = np.mean(np.hstack(recall100))
    precision10 = np.mean(np.hstack(precision10))
    precision50 = np.mean(np.hstack(precision50))
    precision100 = np.mean(np.hstack(precision100))
    ndcg10 = np.mean(np.hstack(ndcg10))
    ndcg50 = np.mean(np.hstack(ndcg50))
    ndcg100 = np.mean(np.hstack(ndcg100))

    print_metrics([ndcg10, ndcg50, ndcg100], [precision10, precision50, precision100], [recall10, recall50, recall100])


def print_metrics(ndcgs, precisions, recalls):
    ks = [10, 50, 100]
    for i, k in enumerate(ks):
        print(f'NDCG@{k}: {ndcgs[i]} \t Precision@{k}: {precisions[i]} \t Recall@{k}: {recalls[i]}')


def preprocess_data(u_train, u_test, r_train, r_test, i_train, i_test, u_friend, v_friend):
    # Building interactions test set
    test_set = {}
    for u in range(len(u_test)):
        user = u_test[u]
        if user in test_set:
            test_set[user].append(i_test[u])
        else:
            test_set[user] = [i_test[u]]

    # Building training set for questionnaire
    train_q_set = {}
    max_questions = 0
    for u in np.unique(u_train):
        leaf = Tree.traverse_a_user(u, interaction_matrix, tree)
        train_q_set[u] = leaf.global_questions + leaf.local_questions
    # making sure all inputs are of same size
    for u in train_q_set.keys():
        if len(train_q_set[u]) > max_questions:
            max_questions = len(train_q_set[u])
    for u in train_q_set.keys():
        while len(train_q_set[u]) < max_questions:
            train_q_set[u].append(num_items)

    # Building training set for interactions
    train_set = {}
    max_items = 0
    for u in range(len(u_train)):
        # Not considering items asked as question
        user = u_train[u]
        if i_train[u] in train_q_set[user]:
            continue
        if u_train[u] in train_set:
            train_set[u_train[u]].append(i_train[u])
        else:
            train_set[u_train[u]] = [i_train[u]]
    # making sure all inputs are of same size
    for u in train_set.keys():
        if len(train_set[u]) > max_items:
            max_items = len(train_set[u])
    for u in train_set.keys():
        while len(train_set[u]) < max_items:
            train_set[u].append(num_items)

    # Building training set for ratings
    train_r_set = {}
    for u in range(len(u_train)):
        user = u_train[u]
        # Not considering ratings on items asked as question
        if i_train[u] in train_q_set[user]:
            continue
        if u_train[u] in train_r_set:
            train_r_set[user].append(r_train[u])
        else:
            train_r_set[user] = [r_train[u]]
    # making sure all inputs are of same size
    for u in train_set.keys():
        while len(train_r_set[u]) < max_items:
            train_r_set[u].append(0)

    # Building training set for social domain
    train_f_set = {}
    max_friends = 0
    for i in range(len(u_friend)):
        if u_friend[i] in train_f_set:
            train_f_set[u_friend[i]].append(v_friend[i])
        else:
            train_f_set[u_friend[i]] = [v_friend[i]]
    for i in train_f_set.keys():
        if len(train_f_set[i]) > max_friends:
            max_friends = len(train_f_set[i])
    for i in train_set.keys():
        if not i in train_f_set:
            train_f_set[i] = [num_users]
        while len(train_f_set[i]) < max_friends:
            train_f_set[i].append(num_users)

    return test_set, train_set, train_r_set, train_f_set, train_q_set, max_questions, max_items, max_friends


if __name__ == '__main__':
    random_seed = 2020

    u_train = np.array(train_data['uid'], dtype=np.int32)
    u_test = np.array(test_data['uid'], dtype=np.int32)
    r_train = np.array(train_data['rating'], dtype=np.int32)
    r_test = np.array(test_data['rating'], dtype=np.int32)
    i_train = np.array(train_data['iid'], dtype=np.int32)
    i_test = np.array(test_data['iid'], dtype=np.int32)
    u_friend = np.array(social_data['uid'], dtype=np.int32)
    v_friend = np.array(social_data['sid'], dtype=np.int32)

    n_train_users = np.ones(len(u_train))
    train_r = scipy.sparse.csr_matrix((r_train, (u_train, i_train)), dtype=np.int16, shape=(num_users, num_items))
    n_test_users = np.ones(len(u_test))
    test_r = scipy.sparse.csr_matrix((r_test, (u_test, i_test)), dtype=np.int16, shape=(num_users, num_items))

    test_set, train_set, train_r_set, train_f_set, train_q_set, max_questions, max_items, max_friends = \
        preprocess_data(u_train, u_test, r_train, r_test, i_train, i_test, u_friend, v_friend)

    batch_size = 128
    with tf.Graph().as_default():
        tf.set_random_seed(random_seed)

        session_conf = tf.ConfigProto()
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = EATNN(num_users, num_items, max_questions, max_items, max_friends)
            model.build_graph()

            optimizer = tf.train.AdamOptimizer().minimize(model.loss)
            train_op = optimizer

            sess.run(tf.global_variables_initializer())

            user_train, item_train, rating_train, friend_train, question_train = get_train_instances(train_set, train_r_set,
                                                                                       train_f_set, train_q_set)

            for epoch in range(100):
                print(f'Epoch: {epoch}')
                start_t = writeline_and_time('\tUpdating...')

                shuffled_indices = np.random.permutation(np.arange(len(user_train)))
                user_train = user_train[shuffled_indices]
                item_train = item_train[shuffled_indices]
                rating_train = rating_train[shuffled_indices]
                friend_train = friend_train[shuffled_indices]
                question_train = question_train[shuffled_indices]

                n_batches = int(len(user_train) / batch_size)
                loss = 0.0
                for batch_num in range(n_batches):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, len(user_train))

                    u_batch = user_train[start_index:end_index]
                    i_batch = item_train[start_index:end_index]
                    r_batch = rating_train[start_index:end_index]
                    f_batch = friend_train[start_index:end_index]
                    q_batch = question_train[start_index:end_index]

                    batch_loss, wi, wf, wq = train_step(u_batch, i_batch, r_batch, f_batch, q_batch)
                    loss += batch_loss
                print('\r\tUpdating: time=%.2f'
                      % (time.time() - start_t))

                eval_step(test_set, train_r, test_r, batch_size)

