#User - user CF 
from __future__ import print_function
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

class uuCF:
    def __init__(self, y_data, k, cosine_func = cosine_similarity):
        self.y_data = y_data
        self.k = k
        self.sim_func = cosine_func
        self.Ybar = None 
        self.n_users = int(np.max(self.y_data[:, 0])) + 1
        self.n_items = int(np.max(self.y_data[:, 1])) + 1

    def fit(self):
        users = self.y_data[:, 0]
        self.Ybar = self.y_data.copy()
        self.mu = np.zeros((self.n_users,))
        for n in range(self.n_users):
            ids = np.where(users == n)[0].astype(np.int32)
            item_ids = self.y_data[ids, 1]
            ratings = self.y_data[ids, 2]
            self.mu[n] = np.mean(ratings) if ids.size > 0 else 0 
            self.Ybar[ids, 2] = ratings - self.mu[n]

        #From the ratings matrix as a sparse matrix 
        self.Ybar = sparse.coo_matrix((self.Ybar[:, 2], 
                                      (self.Ybar[:, 1], self.Ybar[:, 0])),
                                      (self.n_items, self.n_users)).tocsr()
        self.S = self.sim_func(self.Ybar.T, self.Ybar.T)

    def predict(self, u, i):
        ids = np.where(self.y_data[:, 1] == u)[0].astype(np.int32)
        users_rated_i = (self.y_data[ids,0]).astype(np.int32)
        sim = self.S[u, users_rated_i]
        nns = np.argsort(sim)[-self.k:]
        nearest_s = sim[nns]
        r = self.Ybar[i, users_rated_i[nns]]
        eps = 1e-8
        return (r * nearest_s).sum()/(np.abs(nearest_s).sum() + eps) + self.mu[u]

