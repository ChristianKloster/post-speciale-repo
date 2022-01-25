# theano-bpr
#
# Copyright (c) 2014 British Broadcasting Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import theano, numpy
import theano.tensor as T
import time
import sys
import os
from tqdm import tqdm

os.environ['THEANO_FLAGS'] = 'device=gpu'
from collections import defaultdict


class SBPR(object):
    def __init__(self, rank, n_users, n_items, largest_iid, lambda_u=0.015, lambda_v=0.025, lambda_b=0.01,
                 learning_rate=0.05):
        '''
          Creates a new object for training and testing a Bayesian
          Personalised Ranking (BPR) Matrix Factorisation
          model, as described by Rendle et al. in:

            http://arxiv.org/abs/1205.2618

          This model tries to predict a ranking of items for each user
          from a viewing history.
          It's also used in a variety of other use-cases, such
          as matrix completion, link prediction and tag recommendation.

          `rank` is the number of latent features in the matrix
          factorisation model.

          `n_users` is the number of users and `n_items` is the
          number of items.

          The regularisation parameters can be overridden using
          `lambda_u`, `lambda_i` and `lambda_j`. They correspond
          to each three types of updates.

          The learning rate can be overridden using `learning_rate`.

          This object uses the Theano library for training the model, meaning
          it can run on a GPU through CUDA. To make sure your Theano
          install is using the GPU, see:

            http://deeplearning.net/software/theano/tutorial/using_gpu.html

          When running on CPU, we recommend using OpenBLAS.

            http://www.openblas.net/

          Example use (10 latent dimensions, 100 users, 50 items) for
          training:

          >>> from baselines import BPR
          >>> bpr = BPR(10, 100, 50)
          >>> from numpy.random import randint
          >>> train_data = zip(randint(100, size=1000), randint(50, size=1000))
          >>> bpr.train(train_data, social_data)

          This object also has a method for testing, which will return
          the Area Under Curve for a test set.

          >>> test_data = zip(randint(100, size=1000), randint(50, size=1000))
          >>> bpr.test(test_data)

          (This should give an AUC of around 0.5 as the training and
          testing set are chosen at random)
        '''
        self._rank = rank
        self._n_users = n_users
        self._n_items = n_items
        self.largest_iid = largest_iid
        self._lambda_u = lambda_u
        self._lambda_v = lambda_v
        self._lambda_bias = lambda_b
        self._learning_rate = learning_rate
        self._train_users = set()
        self._train_items = set()
        self._train_dict = {}
        self._configure_theano()
        self._generate_train_model_function()

    def _configure_theano(self):
        """
          Configures Theano to run in fast mode
          and using 32-bit floats.
        """
        theano.config.mode = 'FAST_RUN'
        theano.config.floatX = 'float32'

    def _generate_train_model_function(self):
        """
          Generates the train model function in Theano.
          This is a straight port of the objective function
          described in the BPR paper.

          We want to learn a matrix factorisation

            U = W.H^T

          where U is the user-item matrix, W is a user-factor
          matrix and H is an item-factor matrix, so that
          it maximises the differences between
          W[u,:].H[i,:]^T and W[u,:].H[k,:]^T and
          W[u,:].H[k,:]^T and W[u,:].H[j,:]^T
          where `i` is a positive item
          (one the user `u` has watched), `k` is a social positive item
          (one the friends of `u` has watched) and `j` a negative item
          (one the user `u` and friends of `u` hasn't watched).
        """
        u = T.lvector('u')
        i = T.lvector('i')
        k = T.lvector('k')
        j = T.lvector('j')

        self.W = theano.shared(numpy.random.random((self._n_users, self._rank)).astype('float32'), name='W')
        self.H = theano.shared(numpy.random.random((self._n_items, self._rank)).astype('float32'), name='H')

        self.B = theano.shared(numpy.zeros(self._n_items).astype('float32'), name='B')

        x_ui = T.dot(self.W[u], self.H[i].T).diagonal() + self.B[i]
        x_uk = T.dot(self.W[u], self.H[k].T).diagonal() + self.B[k]
        x_uj = T.dot(self.W[u], self.H[j].T).diagonal() + self.B[j]

        obj = T.sum(T.log(T.nnet.sigmoid(x_ui - x_uk)) + T.log(T.nnet.sigmoid(x_uk - x_uj)) -
                    self._lambda_u * (self.W[u] ** 2).sum(axis=1) -  # user regularization
                    self._lambda_v * (self.H[i] ** 2 + self.H[k] ** 2 + self.H[j] ** 2).sum(axis=1) - # item regularization
                    self._lambda_bias * (self.B[i] ** 2 + self.B[k] ** 2 + self.B[j])) # bias regularization
        cost = -obj

        obj_without_social = T.sum(T.log(T.nnet.sigmoid(x_ui - x_uj)) -
                                   self._lambda_u * (self.W[u] ** 2).sum(axis=1) - # user regularization
                                   self._lambda_v * (self.H[i] ** 2 + self.H[j] ** 2).sum(axis=1) - # item regularization
                                   self._lambda_bias * (self.B[i] ** 2 + self.B[j]))  # bias regularization

        cost_without_social = -obj_without_social

        g_cost_W_without_social = T.grad(cost=cost_without_social, wrt=self.W)
        g_cost_H_without_social = T.grad(cost=cost_without_social, wrt=self.H)
        g_cost_B_without_social = T.grad(cost=cost_without_social, wrt=self.B)

        updates_without_social = [(self.W, self.W - self._learning_rate * g_cost_W_without_social),
                                  (self.H, self.H - self._learning_rate * g_cost_H_without_social),
                                  (self.B, self.B - self._learning_rate * g_cost_B_without_social)]

        g_cost_W = T.grad(cost=cost, wrt=self.W)
        g_cost_H = T.grad(cost=cost, wrt=self.H)
        g_cost_B = T.grad(cost=cost, wrt=self.B)

        updates = [(self.W, self.W - self._learning_rate * g_cost_W),
                   (self.H, self.H - self._learning_rate * g_cost_H),
                   (self.B, self.B - self._learning_rate * g_cost_B)]

        self.train_model = theano.function(inputs=[u, i, k, j], outputs=cost, updates=updates)
        self.train_model_without_social = theano.function(inputs=[u, i, j],
                                                          outputs=cost_without_social,
                                                          updates=updates_without_social)

    def train(self, train_data, social_data, epochs=100):
        """
          Trains the SBPR Matrix Factorisation model using Stochastic
          Gradient Descent over `train_data`.

          `train_data` is an array of (user_index, item_index) tuples.
          `social_data` is an array of (user_index, friend_index) tuples.

          We first create a set of random samples from `train_data` for
          training, of size `epochs` * size of `train_data`.

          We then iterate through the resulting training samples 1-by-1
          and run one iteration of gradient descent for the sample.
        """
        self._train_dict, self._train_users, self._train_items = self._data_to_dict(train_data)
        self._social_dict = self._generate_social_dict(social_data)
        n_sgd_samples = len(train_data) * epochs
        sgd_users, sgd_pos_items, sgd_neg_items, sgd_social_pos_items = self._uniform_user_sampling(n_sgd_samples)
        i = 0
        # number of iterations
        for epoch in tqdm(range(epochs), desc='Training'):
            # iterating over number of training samples
            for sample in range(len(train_data)):
                user = sgd_users[i:i+1]
                pos_item = sgd_pos_items[i:i+1]
                social_pos_item = sgd_social_pos_items[i:i+1]
                neg_item = sgd_neg_items[i:i+1]

                if social_pos_item != [-1]:
                    self.train_model(user, pos_item, social_pos_item, neg_item)
                else:
                    self.train_model_without_social(user, pos_item, neg_item)
                i += 1

    def _uniform_user_sampling(self, n_samples):
        """
          Creates `n_samples` random samples from training data for performing Stochastic
          Gradient Descent. We start by uniformly sampling users, and then sample a positive and a negative
          item for each user sample.
        """
        sys.stderr.write("Generating %s random training samples\n" % str(n_samples))
        sgd_users = numpy.array(list(self._train_users))[
            numpy.random.randint(len(list(self._train_users)), size=n_samples)]
        sgd_pos_items, sgd_neg_items, sgd_social_pos_items = [], [], []
        for sgd_user in tqdm(sgd_users, desc='Generating random samples'):
            # taking positive sample
            pos_item = self._train_dict[sgd_user][numpy.random.randint(len(self._train_dict[sgd_user]))]
            sgd_pos_items.append(pos_item)
            # taking social positive sample
            if self._social_dict[sgd_user]:
                social_pos_item = self._social_dict[sgd_user][numpy.random.randint(len(self._social_dict[sgd_user]))]
            else:
                social_pos_item = -1
            sgd_social_pos_items.append(social_pos_item)
            # taking negative sample
            neg_item = numpy.random.randint(self._n_items)
            # ... which cannot be a positive sample or social sample
            while neg_item in self._train_dict[sgd_user] or neg_item in self._social_dict[sgd_user]:
                neg_item = numpy.random.randint(self._n_items)
            sgd_neg_items.append(neg_item)

        return sgd_users, sgd_pos_items, sgd_neg_items, sgd_social_pos_items

    def predictions(self, user_index):
        """
          Computes item predictions for `user_index`.
          Returns an array of prediction values for each item
          in the dataset.
        """
        w = self.W.get_value()
        h = self.H.get_value()
        b = self.B.get_value()
        user_vector = w[user_index, :]
        return user_vector.dot(h.T) + b

    def prediction(self, user_index, item_index):
        """
          Predicts the preference of a given `user_index`
          for a given `item_index`.
        """
        return self.predictions(user_index)[item_index]

    def top_predictions(self, user_index, topn=10):
        """
          Returns the item indices of the top predictions
          for `user_index`. The number of predictions to return
          can be set via `topn`.
          This won't return any of the items associated with `user_index`
          in the training set.
        """
        return [
                   item_index for item_index in numpy.argsort(self.predictions(user_index))
                   if item_index not in self._train_dict[user_index]
               ][::-1][:topn]

    def test(self, test_data, k=10):
        """
          Computes the NDCG@k, Precision@k and Recall@k on `test_data`.

          `test_data` is an array of (user_index, item_index) tuples.
        """
        times_predicted = [0] * self.largest_iid
        times_predicted_correctly = [0] * self.largest_iid

        test_dict, test_users, test_items = self._data_to_dict(test_data)
        ndcg_values = []
        prec_values = []
        recall_values = []
        for user in tqdm(test_dict.keys(), desc=f'Testing with k={k}'):
            actuals = test_dict[user]
            preds = self.top_predictions(user, topn=k)
            for item in preds:
                times_predicted[item] += 1
                if item in actuals:
                    times_predicted_correctly[item] += 1

            n_rec_and_rel = len(set(preds) & set(actuals))
            n_rel = len(actuals)

            precision = n_rec_and_rel / k
            recall = n_rec_and_rel / n_rel

            tp = 1. / numpy.log2(numpy.arange(2, k + 2))
            dcg_relevances = [1 if p in actuals else 0 for p in preds]

            DCG = numpy.sum(dcg_relevances * tp)
            IDCG = tp[:min(len(actuals), k)].sum()

            ndcg_values.append(DCG/IDCG)
            prec_values.append(precision)
            recall_values.append(recall)

        #with open('../data/CiaoDVD_done_right/extreme_folds/times_predicted_sbpr.txt', 'w') as f:
        #    for item in times_predicted:
        #        f.write("%s\n" % item)
        #with open('../data/CiaoDVD_done_right/extreme_folds/times_predicted_correctly_sbpr.txt', 'w') as f:
        #    for item in times_predicted_correctly:
         #       f.write("%s\n" % item)

        return numpy.mean(ndcg_values), numpy.mean(prec_values), numpy.mean(recall_values)

    def test_cold(self, test_data, k=10):
        test_dict, test_users, test_items = self._data_to_dict(test_data)

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
        for user in tqdm(test_dict.keys(), desc=f'Testing with k={k}'):
            train_ratings = self._train_dict[user]
            # not considering users with more than 10 ratings, i.e. warm users
            if len(train_ratings) > 10: continue

            actuals = test_dict[user]
            preds = self.top_predictions(user, topn=k)

            tp = 1. / numpy.log2(numpy.arange(2, k + 2))
            dcg_relevances = [1 if p in actuals else 0 for p in preds]

            DCG = numpy.sum(dcg_relevances * tp)
            IDCG = tp[:min(len(actuals), k)].sum()

            if len(train_ratings) == 10: ndcg_10.append(DCG / IDCG)
            if len(train_ratings) == 9: ndcg_9.append(DCG / IDCG)
            if len(train_ratings) == 8: ndcg_8.append(DCG / IDCG)
            if len(train_ratings) == 7: ndcg_7.append(DCG / IDCG)
            if len(train_ratings) == 6: ndcg_6.append(DCG / IDCG)
            if len(train_ratings) == 5: ndcg_5.append(DCG / IDCG)
            if len(train_ratings) == 4: ndcg_4.append(DCG / IDCG)
            if len(train_ratings) == 3: ndcg_3.append(DCG / IDCG)
            if len(train_ratings) == 2: ndcg_2.append(DCG / IDCG)
            if len(train_ratings) == 1: ndcg_1.append(DCG / IDCG)
            if len(train_ratings) == 0: ndcg_0.append(DCG / IDCG)

        return numpy.mean(ndcg_0), numpy.mean(ndcg_1), numpy.mean(ndcg_2), numpy.mean(ndcg_3), \
               numpy.mean(ndcg_4), numpy.mean(ndcg_5), numpy.mean(ndcg_6), numpy.mean(ndcg_7), \
               numpy.mean(ndcg_8), numpy.mean(ndcg_9), numpy.mean(ndcg_10)

    def _data_to_dict(self, data):
        data_dict = defaultdict(list)
        items = set()
        for (user, item) in data:
            data_dict[user].append(item)
            items.add(item)
        return data_dict, set(data_dict.keys()), items

    def _generate_social_dict(self, social_data):
        social_dict = defaultdict(list)
        for user in tqdm(self._train_users, desc='Generating social data'):
            social_dict[user] = []
            friends = list(social_data.query('uid == @user')['sid'])
            if len(friends):
                social_pos_items = []
                # finding friends' rated items
                for friend in friends:
                    social_pos_items += self._train_dict[friend]
                # removing items user has rated himself
                pos_items = self._train_dict[user]
                social_pos_items = list(set(social_pos_items) - set(pos_items))
                # checking if friends have rated minimum 1 item the user hasn't himself
                if len(social_pos_items):
                    social_dict[user] = social_pos_items
        return social_dict


