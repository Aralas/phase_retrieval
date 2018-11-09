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
    def __init__(self, x, A, y, z, k, trial_num, initializer, optimizer):
        self.x = x
        self.A = A
        self.y = y
        self.z = z
        (self.m, self.n) = A.shape
        self.k = k
        self.trial_num = trial_num
        self.initializer = initializer
        self.optimizer = optimizer

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
    def solver(self):
        for iteration in range(self.trial_num):
            pass


class N_PR(PhaseRetrieval):
    def solver(self):
        for iteration in range(self.trial_num):
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
