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
import Projection


class PhaseRetrieval(object):
    def __init__(self, x, A, y, z, param):
        self.x = x
        self.A = A
        self.y = y
        self.z = z
        self.m, self.n = A.shape
        self.param = param
        # self.loss_func = Projection.LossFunction(self.x, A, y, z, param)

    def select_initialization(self, initializer):
        init_object = Initialization.Initialization(self.A, self.z, self.param.k, self.param.data_type,
                                                    self.param.isComplex)
        if initializer in ['init_random', 'init_constant', 'init_spectral', 'init_optimal_spectral']:
            init_func = getattr(init_object, initializer)
            return init_func
        else:
            print('There is no such initializer %s' % initializer)

    def select_step_chooser(self, step_chooser):
        step_object = Projection.StepChooser(self.param.k, self.param.step_value)
        if step_chooser in ['backtracking_line_search', 'constant_step']:
            step_chooser_func = getattr(step_object, step_chooser)
            return step_chooser_func
        else:
            print('There is no such step_chooser %s' % step_chooser)

    def select_projection_method(self, projection, support):
        projection_object = Projection.ProjectionMethod(support, self.x, self.A, self.y, self.z, self.param)
        if projection in ['gradient_descent', 'newton', 'gauss_newton', 'steepest_descent', 'coordinate_descent']:
            projection_func = getattr(projection_object, projection)
            return projection_func
        else:
            print('There is no such projection method %s' % projection)


class GD_PR(PhaseRetrieval):

    def __init__(self, x, A, y, z, param):
        PhaseRetrieval.__init__(self, x, A, y, z, param)

    def solver(self):
        init_func = self.select_initialization(self.param.initializer)
        step_func = self.select_step_chooser(self.param.step_chooser)
        # projection_func = self.select_projection_method(self.param.projection, np.array(range(self.n)))
        projection_object = Projection.ProjectionMethod(np.array(range(self.n)), self.x, self.A, self.y, self.z,
                                                        self.param)
        x0 = init_func()
        x_hat, recon_error, meas_error, iteration, success = projection_object.gradient_descent(x0, step_func,
                                                                                                truncated=True)
        return recon_error, meas_error, iteration, success


class N_PR(PhaseRetrieval):

    def __init__(self, x, A, y, z, param):
        PhaseRetrieval.__init__(self, x, A, y, z, param)

    def solver(self):
        init_func = self.select_initialization(self.param.initializer)
        step_func = self.select_step_chooser(self.param.step_chooser)
        # projection_func = self.select_projection_method(self.param.projection, np.array(range(self.n)))
        projection_object = Projection.ProjectionMethod(np.array(range(self.n)), self.x, self.A, self.y, self.z,
                                                        self.param)
        x0 = init_func()
        x_hat, recon_error, meas_error, iteration, success = projection_object.newton(x0, step_func, truncated=True)
        return recon_error, meas_error, iteration, success


class GN_PR(PhaseRetrieval):

    def __init__(self, x, A, y, z, param):
        PhaseRetrieval.__init__(self, x, A, y, z, param)

    def solver(self):
        init_func = self.select_initialization(self.param.initializer)
        step_func = self.select_step_chooser(self.param.step_chooser)
        # projection_func = self.select_projection_method(self.param.projection, np.array(range(self.n)))
        projection_object = Projection.ProjectionMethod(np.array(range(self.n)), self.x, self.A, self.y, self.z,
                                                        self.param)
        x0 = init_func()
        x_hat, recon_error, meas_error, iteration, success = projection_object.gauss_newton(x0, step_func,
                                                                                            truncated=True)
        return recon_error, meas_error, iteration, success


class SP_PR(PhaseRetrieval):

    def __init__(self, x, A, y, z, param):
        PhaseRetrieval.__init__(self, x, A, y, z, param)

    def get_projection(self, support, x0, step_func, truncated):
        projection_func = self.select_projection_method(self.param.projection, support)
        x_hat, recon_error, meas_error, iteration, success = projection_func(x0, step_func, truncated)
        return x_hat, recon_error, meas_error, iteration, success

    def gradient_f(self, x_hat):
        z_hat = self.A.dot(x_hat) ** 2
        b = z_hat - self.z
        grad = 2 / len(self.A) * np.dot(np.dot(self.A.transpose(), b * self.A), x_hat)
        return grad

    def solver(self):
        init_func = self.select_initialization(self.param.initializer)
        step_func = self.select_step_chooser(self.param.step_chooser)
        x0 = init_func()
        grad = self.gradient_f(x0)
        sort_grad_index = np.argsort(-abs(grad), axis=0)
        T0 = sort_grad_index[0:self.param.k, :].reshape(self.param.k)
        x0, recon_error, meas_error, iteration, success = self.get_projection(T0, x0[T0], step_func, truncated=False)

        for iteration_alg in range(self.param.max_iter):
            grad = self.gradient_f(x0)
            sort_grad_index = np.argsort(-abs(grad), axis=0)
            T1 = sort_grad_index[0:self.param.k, :].reshape(self.param.k)
            T_tilde = np.union1d(T0, T1)
            x_tilde, recon_error, meas_error, iteration, success = self.get_projection(T_tilde, x0[T_tilde], step_func,
                                                                                       truncated=False)

            sort_x_index = np.argsort(-abs(x_tilde), axis=0)
            T0 = sort_x_index[0:self.param.k, :].reshape(self.param.k)
            x0, recon_error, meas_error, iteration, success = self.get_projection(T0, x_tilde[T0], step_func,
                                                                                  truncated=False)
            if success:
                break
        return recon_error, meas_error, iteration_alg, success


class OMP_PR(PhaseRetrieval):

    def __init__(self, x, A, y, z, param):
        PhaseRetrieval.__init__(self, x, A, y, z, param)

    def get_projection(self, support, x0, step_func, truncated):
        projection_func = self.select_projection_method(self.param.projection, support)
        x_hat, recon_error, meas_error, iteration, success = projection_func(x0, step_func, truncated)
        return x_hat, recon_error, meas_error, iteration, success

    def gradient_f(self, x_hat):
        z_hat = self.A.dot(x_hat) ** 2
        b = z_hat - self.z
        grad = 2 / len(self.A) * np.dot(np.dot(self.A.transpose(), b * self.A), x_hat)
        return grad

    def solver(self):
        init_func = self.select_initialization(self.param.initializer)
        step_func = self.select_step_chooser(self.param.step_chooser)
        x0 = init_func()
        index_set = []

        for iteration_alg in range(self.param.k):
            grad = self.gradient_f(x0)
            grad[index_set] = 0
            sort_grad_index = np.argsort(-abs(grad), axis=0).reshape(self.n)
            index_set.append(sort_grad_index[0])
            x0, recon_error, meas_error, iteration, success = self.get_projection(index_set, x0[index_set], step_func,
                                                                                  truncated=False)
            if success:
                break
        return recon_error, meas_error, iteration_alg, success


class HTP_PR(PhaseRetrieval):
    def solver(self):
        pass


class IHT_PR(PhaseRetrieval):
    def solver(self):
        pass
