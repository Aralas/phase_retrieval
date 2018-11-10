# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:Algorithms.py
@time:2018/11/914:38



Algorithms:
    GD_PR: Gradient Descent Phase Retrieval (Wirtinger Flow)
    N_PR: Newton Phase Retrieval
    GN_PR: Gaussian Newton Phase Retrieval
    SP_PR: Subspace Pursuit Phase Retrieval
    HTP_PR: Hard Thresholding Pursuit Phase Retrieval
    IHT_PR: Iterative Hard Thresholding Phase Retrieval
    OMP_PR: Orthogonal Matching Pursuit Phase Retrieval
"""

import numpy as np


class PhaseRetrieval(object):
    def __init__(self, x, A, y, z, k, epsilon, initializer, max_iter, searcher):
        self.x = x
        self.A = A
        self.y = y
        self.z = z
        self.m, self.n = A.shape
        self.k = k
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.initializer = initializer
        self.searcher = searcher

    def reconstruct_error(self, x0):
        # solve for solution:  alpha * x = x0
        xt = self.x.transpose()
        alpha = np.dot(xt, x0) / np.dot(xt, x)
        x = alpha * self.x
        error = np.linalg.norm(x0 - x, 2) / np.linalg.norm(x, 2)
        return error

    def measurement_error(self, x0):
        y0 = abs(self.A.dot(x0))
        error = np.linalg.norm(y0 - self.y, 2)
        return error


class GD_PR(PhaseRetrieval):

    def __init__(self, x, A, y, z, k, epsilon, max_iter, initializer, searcher):
        PhaseRetrieval.__init__(self, x, A, y, z, k, epsilon, max_iter, initializer, searcher)

    def solver(self):
        x0 = self.initializer.get_initialization(self.y, self.A)    # 还没定义
        recon_error = [self.reconstruct_error(x0)]
        meas_error = [self.measurement_error(x0)]
        for iteration in range(self.max_iter):
            x0 = self.searcher.searching()
            recon_error.append(self.reconstruct_error(x0))
            meas_error.append(self.measurement_error(x0))
            if recon_error[-1] < self.epsilon or meas_error[-1] < self.epsilon:
                return recon_error, meas_error



class N_PR(PhaseRetrieval):
    def solver(self):
        for iteration in range(self.max_iter):
            pass


class SP_PR(PhaseRetrieval):
    def solver(self):
        pass


class HTP_PR(PhaseRetrieval):
    def solver(self):
        pass


class IHT_PR(PhaseRetrieval):
    def solver(self):
        pass


class OMP_PR(PhaseRetrieval):
    def solver(self):
        pass
