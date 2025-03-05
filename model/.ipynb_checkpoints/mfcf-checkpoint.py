#Maxtrix factorization CF
from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

class MF(object):
    def __init__(self, Y, K, lam = 0.1, Xinit = None, Winit = None, 
                lr = 0.5, max_iter = 1000, print_every = 100):
        self.Y = Y
        self.K = K
        self.lam = lam 
        self.lr = lr
        self.max_iter = max_iter
        self.print_every = print_every
        self.n_users = int(np.max(Y[:, 0])) + 1
        self.n_items = int(np.max(Y[:, 1])) + 1
        self.n_ratings = Y.shape[0]
        self.X = np.random.randn(self.n_items, K) if Xinit is None else Xinit
        self.W = np.random.randn(K, self.n_users) if Winit is None else Winit
        self.b = np.random.rand(self.n_items)
        self.d = np.random.rand(self.n_users)

    def loss(self):
        L = 0
        for i in range(self.n_ratings):
            n, m, rating = int(self.Y[i, 0]), int(self.Y[i, 1]), self.Y[i, 2]
            L += 0.5 * (self.X[m].dot(self.W[:, n]) + self.b[m] + self.d[n] - rating)**2
        L /= self.n_ratings
        return L + 0.5 * self.lam *(np.sum(self.X**2) + np.sum(self.W**2))

    def updateXb(self):
        for m in range(self.n_items):
            ids = np.where(self.Y[:, 1] == m)[0]
            users_ids, ratings = self.Y[ids, 0].astype(np.int32), self.Y[ids,2]
            Wm, dm = self.W[:, users_ids], self.d[users_ids]
            for i in range(30):
                xm = self.X[m]
                error = xm.dot(Wm) + self.b[m] + dm - ratings
                grad_xm = error.dot(Wm.T)/self.n_ratings + self.lam * xm
                grad_bm = np.sum(error)/self.n_ratings
                #Update
                self.X[m] -= self.lr * grad_xm
                self.b[m] -= self.lr * grad_bm

    def updateWd(self):
        for n in range(self.n_users):
            ids = np.where(self.Y[:, 0] == n)[0]
            items_ids, ratings = self.Y[ids, 1].astype(np.int32), self.Y[ids, 2]
            Xn, bn = self.X[items_ids], self.b[items_ids] 
            for i in range(30):
                Wn = self.W[:, n]
                error = Xn.dot(Wn) + bn + self.d[n] - ratings
                grad_Wn = Xn.T.dot(error)/self.n_ratings + self.lam * Wn 
                grad_dn = np.sum(error)/self.n_ratings
                #update
                self.W[:, n] -= self.lr * grad_Wn.reshape(-1)
                self.d[n] -= self.lr * grad_dn

    def evaluate_RMSE(self, rate_test):
        n_tests = rate_test.shape[0]
        se = 0
        for n in range(n_tests):
            pred = self.predict(rate_test[n, 0], rate_test[n, 1])
            se += (pred - rate_test[n, 2])**2 
        rmse = np.sqrt(se/n_tests)
        return rmse

    def fit(self):
        for it in range(self.max_iter):
            self.updateWd()
            self.updateXb()
            if (it + 1) % self.print_every == 0:
                rmse = self.evaluate_RMSE(self.Y)
                print(f"iter: {it+1}, loss: {self.loss()}, RMSE train: {rmse}")

    def predict(self, u, i):
        '''
        predict the rating of user u for item i 
        '''
        u, i = int(u), int(i)
        pred = self.X[i, :].dot(self.W[:, u]) + self.b[i] + self.d[u]
        return max(0, min(5, pred))
    