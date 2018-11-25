# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:ProjectionProjection.py
@time:2018/11/914:37
"""
import numpy as np
import math


class StepChooser(object):

    def __init__(self, k, step_value):
        self.alpha = 0.3
        self.beta = 0.8
        self.tau = 1000
        self.k = k
        self.step_value = step_value

    # def generate_x(self, x, k, step, delta_x):
    #     x_new = x + step * delta_x
    #     if k < len(x):
    #         x_sort_index = abs(x_new).argsort(axis=0)
    #         x_new[x_sort_index[0:(len(x) - k), 0]] = 0
    #     return x_new

    # def backtracking_line_search(self, x_hat, delta_x):
    #     step = 1
    #     # while self.loss_func.f(self.generate_x(x_hat, self.k, step, delta_x)) >= (
    #     #         self.loss_func.f(x_hat) + self.alpha * step * np.dot(self.loss_func.gradient(x_hat).transpose(),
    #     #                                                              delta_x)):
    #     while self.loss_func.f(x + step * delta_x) >= (
    #             self.loss_func.f(x_hat) + self.alpha * step * np.dot(self.loss_func.gradient(x_hat).transpose(),
    #                                                                  delta_x)):
    #         step = step * self.beta
    #     return step

    def constant_step(self, x_hat, delta_x):
        return self.step_value


class LossFunction(object):
    # x_hat is truncated by support

    def __init__(self, support, x, A, y, z, param):
        self.support = support
        self.x = x
        self.A = A
        self.A_trunc = A[:, support]
        self.y = y
        self.z = z
        self.m, self.n = A.shape
        self.k = param.k
        self.max_iter = param.max_iter
        self.epsilon = param.epsilon

    def f(self, x_hat):
        loss = np.mean((self.z - self.A_trunc.dot(x_hat)) ** 2) / 2
        return loss

    def gradient(self, x_hat):
        z_hat = np.dot(self.A_trunc, x_hat) ** 2
        b = z_hat - self.z
        # func = lambda i: b[i] * np.dot(self.A[i].reshape(-1, 1), self.A[i].reshape(1, -1))
        # grad = 2 * np.dot(np.mean([func(i) for i in range(len(self.A))], axis=0), x_hat)
        grad = 2 / self.m * np.dot(np.dot(self.A_trunc.transpose(), b * self.A_trunc), x_hat)
        return grad

    def hessian(self, x_hat):
        b = 3 * (np.dot(self.A_trunc, x_hat) ** 2) - self.z
        hess = np.dot(self.A_trunc.transpose(), self.A_trunc * b)
        return hess

    def get_full_x(self, x_hat):
        x0 = np.zeros((self.n, 1))
        x0[self.support] = x_hat
        return x0

    def reconstruct_error(self, x_hat):
        # solve for solution:  alpha * x = x0
        x0 = self.get_full_x(x_hat)
        xt = self.x.transpose()
        alpha = np.dot(xt, x0) / np.dot(xt, self.x)
        if alpha == 0:
            alpha = 1
        x = alpha * self.x
        error = np.linalg.norm(x0 - x, 2) / np.linalg.norm(x, 2)
        return error

    def measurement_error(self, x_hat):
        x0 = self.get_full_x(x_hat)
        y0 = abs(self.A.dot(x0))
        error = np.linalg.norm(y0 - self.y, 2) / np.linalg.norm(self.y, 2)
        return error


class ProjectionMethod(LossFunction):

    def __init__(self, support, x, A, y, z, param):
        LossFunction.__init__(self, support, x, A, y, z, param)

    def gradient_descent(self, x0, step_func, truncated):
        recon_error = [self.reconstruct_error(x0)]
        meas_error = [self.measurement_error(x0)]
        success = False
        x_hat = x0
        for iteration in range(self.max_iter):
            delta_x = -1 * self.gradient(x_hat)
            step = step_func(x_hat, delta_x)
            x_hat = x_hat + step * delta_x
            if truncated:
                x_sort_index = abs(x_hat).argsort(axis=0)
                x_hat[x_sort_index[0:(self.n - self.k), 0]] = 0
            recon_error.append(self.reconstruct_error(x_hat))
            meas_error.append(self.measurement_error(x_hat))
            if min(recon_error[-1], meas_error[-1]) < self.epsilon:
                success = True
                break
        x_hat = self.get_full_x(x_hat)
        return x_hat, recon_error, meas_error, iteration, success



    def newton(self, x0, step_func, truncated):
        recon_error = [self.reconstruct_error(x0)]
        meas_error = [self.measurement_error(x0)]
        success = False
        x_hat = x0
        for iteration in range(self.max_iter):
            hess = self.hessian(x_hat)
            grad = self.gradient(x_hat)
            if np.linalg.det(hess) != 0:
                delta_x = -1 * np.dot(np.array(np.mat(hess).I), grad)
            else:
                delta_x = -1 * grad
            step = step_func(x_hat, delta_x)
            x_hat = x_hat + step * delta_x
            if truncated:
                x_sort_index = abs(x_hat).argsort(axis=0)
                x_hat[x_sort_index[0:(self.n - self.k), 0]] = 0
            recon_error.append(self.reconstruct_error(x_hat))
            meas_error.append(self.measurement_error(x_hat))
            if x_hat[0] == 0:
                print()
            if np.linalg.norm(delta_x, 2) * step< 0.0001:
                break
            if min(recon_error[-1], meas_error[-1]) < self.epsilon:
                success = True
                break
        x_hat = self.get_full_x(x_hat)
        return x_hat, recon_error, meas_error, iteration, success

    # def gauss_newton(self, x_hat, step_func, support):
    #     As = self.A[:, support]
    #     xs = x_hat[support]
    #     y_hat = np.dot(As, xs)
    #     J = 2 * y_hat * self.A
    #     return x_new

    def steepest_descent(self, x_hat, delta_x, step):
        pass

    def coordinate_descent(self, x_hat, delta_x, step):
        pass

# class GradientDescent(LossFunction):
#
#     def __init__(self, A, y, z, step_chooser):
#         LossFunction.__init__(self, A, y, z)
#         self.step_chooser = step_chooser
#
#     def searching(self, x_hat, iter):
#         delta_x = -1 * self.gradient(x_hat)
#         step = self.step_chooser.choose_step(delta_x, iter)
#         x_new = x_hat + step * delta_x
#         return x_new
#
#
# class Newton(LossFunction):
#
#     def searching(self):
#         pass
#
#
# class GuassianNewton(LossFunction):
#
#     def searching(self):
#         pass
#
#
# class SteepestDescent(LossFunction):
#
#     def searching(self):
#         pass
#
#
# class CoordinateDescent(LossFunction):
#
#     def searching(self):
#         pass


# class BacktrackingLineSearch(object):
#
#     def __init__(self, loss_func, tau=330, alpha=0.3, beta=0.8):
#         self.alpha = alpha
#         self.beta = beta
#         self.loss_func = loss_func
#
#     def choose_step(self, x_hat, delta_x, iter):
#         step = 1
#         while self.loss_func.f(x_hat + step * delta_x) >= (
#                 self.loss_func.f(x_hat) + self.alpha * step * np.dot(self.loss_func.gradient(x_hat).transpose(),
#                                                                      delta_x)):
#             step = step * self.beta
#         return step
#
#
# class StepDecline(object):
#
#     def __init__(self, loss_func, tau=330, alpha=0.3, beta=0.8):
#         self.tau = tau
#
#     def choose_step(self, x_hat, delta_x, iter):
#         return min(0.1, (1 - math.exp(-1 * iter / self.tau)) / 2)
