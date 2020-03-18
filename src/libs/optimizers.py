# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

from functools import partial
import numpy as np
from scipy import optimize
from sklearn.metrics import f1_score


class OptimizersF1(object):
    def __init__(self):
        self.coef_ = []

    def _kappa_loss(self, coef, X, y):
        """
        y_hat = argmax(coef*X, axis=-1)
        :param coef: (1D array) weights
        :param X: (2D array) logits
        :param y: (1D array) label
        :return: -f1
        """
        X_p = np.copy(X)
        X_p = coef * X_p
        ll = f1_score(y, np.argmax(X_p, axis=-1), average='macro')

        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        # initial_coef = np.array([1. for _ in range(len(set(y)))])
        # initial_coef = np.array([1. for _ in range(3)])
        initial_coef = np.random.normal(size=(3,))
        self.coef_ = optimize.minimize(fun=loss_partial, x0=initial_coef, method='nelder-mead')

    def predict(self, X, y):
        X_p = np.copy(X)
        X_p = self.coef_.x * X_p
        return f1_score(y, np.argmax(X_p, axis=-1), average='macro')

    def coefficients(self):
        return self.coef_.x
