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
import Initialization
import SearchingDirection
import math


class PhaseRetrieval(object):
    def __init__(self, x, A, y, z, param):
        self.x = x
        self.A = A
        self.y = y
        self.z = z
        self.m, self.n = A.shape
        self.param = param
        # self.k = param.k
        # self.epsilon = param.epsilon
        # self.max_iter = param.max_iter
        # self.initializer = param.initializer
        # self.searcher = param.searcher
        # self.step_chooser = param.step_chooser
        self.loss_func = SearchingDirection.LossFunction(A, y, z)

    def reconstruct_error(self, x0):
        # solve for solution:  alpha * x = x0
        xt = self.x.transpose()
        alpha = np.dot(xt, x0) / np.dot(xt, self.x)
        if alpha == 0:
            alpha = 1
        x = alpha * self.x
        error = np.linalg.norm(x0 - x, 2) / np.linalg.norm(x, 2)
        return error

    def measurement_error(self, x0):
        y0 = abs(self.A.dot(x0))
        error = np.linalg.norm(y0 - self.y, 2)
        return error

    def select_initialization(self, initializer):
        init_object = Initialization.Initialization(self.A, self.y, self.param.k, self.param.data_type,
                                                    self.param.isComplex)
        if initializer in ['init_random', 'init_spectral', 'init_optimal_spectral']:
            init_func = getattr(init_object, initializer)
            return init_func
        else:
            print('There is no such initializer %s' % initializer)

    def select_step_chooser(self, step_chooser):
        step_object = SearchingDirection.StepChooser(self.loss_func, self.param.k, self.param.step_value)
        if step_chooser in ['backtracking_line_search', 'constant_step']:
            step_chooser_func = getattr(step_object, step_chooser)
            return step_chooser_func
        else:
            print('There is no such step_chooser %s' % step_chooser)

    def select_searcher(self, searcher):
        searcher_object = SearchingDirection.Searcher(self.A, self.y, self.z)
        if searcher in ['gradient_descent', 'newton', 'guassian_newton', 'steepest_descent', 'coordinate_descent']:
            searcher_func = getattr(searcher_object, searcher)
            return searcher_func
        else:
            print('There is no such searcher %s' % searcher)


class GD_PR(PhaseRetrieval):

    def __init__(self, x, A, y, z, param):
        PhaseRetrieval.__init__(self, x, A, y, z, param)

    def solver(self):
        init_func = self.select_initialization(self.param.initializer)
        step_func = self.select_step_chooser(self.param.step_chooser)
        searcher_func = self.select_searcher(self.param.searcher)

        x0 = init_func()
        recon_error = [self.reconstruct_error(x0)]
        meas_error = [self.measurement_error(x0)]

        success = False
        for iteration in range(self.param.max_iter):
            x0 = searcher_func(x0, step_func)
            if self.param.k < self.param.n:
                x_sort_index = abs(x0).argsort(axis=0)
                x0[x_sort_index[0:(self.param.n - self.param.k), 0]] = 0
            recon_error.append(self.reconstruct_error(x0))
            meas_error.append(self.measurement_error(x0))
            if min(recon_error[-1], meas_error[-1]) < self.param.epsilon:
                success = True
                break
        return recon_error, meas_error, iteration, success

class N_PR(PhaseRetrieval):

    def __init__(self, x, A, y, z, param):
        PhaseRetrieval.__init__(self, x, A, y, z, param)

    def solver(self):
        init_func = self.select_initialization(self.param.initializer)
        step_func = self.select_step_chooser(self.param.step_chooser)
        searcher_func = self.select_searcher(self.param.searcher)

        x0 = init_func()
        recon_error = [self.reconstruct_error(x0)]
        meas_error = [self.measurement_error(x0)]

        success = False
        for iteration in range(self.param.max_iter):
            x0 = searcher_func(x0, step_func)
            if self.param.k < self.param.n:
                x_sort_index = abs(x0).argsort(axis=0)
                x0[x_sort_index[0:(self.param.n - self.param.k), 0]] = 0
            recon_error.append(self.reconstruct_error(x0))
            meas_error.append(self.measurement_error(x0))
            if min(recon_error[-1], meas_error[-1]) < self.param.epsilon:
                success = True
                break
        return recon_error, meas_error, iteration, success

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
