# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:SearchingDirection.py
@time:2018/11/914:37
"""
import numpy as np
import math


class LossFunction(object):

    def __init__(self, A, y, z):
        self.A = A
        self.y = y
        self.z = z

    def f(self, x_hat):
        loss = np.mean((self.z - self.A.dot(x_hat)) ** 2) / 2
        return loss

    def gradient(self, x_hat):
        z_hat = self.A.dot(x_hat) ** 2
        b = z_hat - self.z
        # func = lambda i: b[i] * np.dot(self.A[i].reshape(-1, 1), self.A[i].reshape(1, -1))
        # grad = 2 * np.dot(np.mean([func(i) for i in range(len(self.A))], axis=0), x_hat)
        grad = 2 / len(self.A) * np.dot(np.dot(self.A.transpose(), b * self.A), x_hat)
        return grad

    def hessian(self, x_hat):
        b = 3 * (self.A.dot(x_hat) ** 2) - self.z
        hess = np.dot(self.A.transpose(), self.A * b)
        return hess

class StepChooser(object):

    def __init__(self, loss_func):
        self.alpha = 0.3
        self.beta = 0.8
        self.loss_func = loss_func
        self.tau = 1000

    def generate_x(self, x, k, step, delta_x):
        x_new = x + step * delta_x
        if k < len(x):
            x_sort_index = abs(x_new).argsort(axis=0)
            x_new[x_sort_index[0:(len(x) - k), 0]] = 0
        return x_new

    def backtracking_line_search(self, x_hat, delta_x, iter, k):
        step = 1
        while self.loss_func.f(self.generate_x(x_hat, k, step, delta_x)) >= (
                self.loss_func.f(x_hat) + self.alpha * step * np.dot(self.loss_func.gradient(x_hat).transpose(),
                                                                     delta_x)):
            step = step * self.beta
        return step

    def constant_step(self, x_hat, delta_x, iter, k):
        # return min(0.001, (1 - math.exp(-1 * iter / self.tau)) / 2)
        return 0.001





class Searcher(LossFunction):

    def __init__(self, A, y, z):
        LossFunction.__init__(self, A, y, z)

    def gradient_descent(self, x_hat, step_func, iteration, k):
        delta_x = -1 * self.gradient(x_hat)
        step = step_func(x_hat, delta_x, iteration, k)
        x_new = x_hat + step * delta_x
        return x_new

    def newton(self, x_hat, step_func, iteration, k):
        hess = self.hessian(x_hat)
        grad = self.gradient(x_hat)
        if np.linalg.det(hess) != 0:
            delta_x = -1 * np.dot(np.array(np.mat(hess).I), grad)
        else:
            delta_x = -1 * grad
        step = step_func(x_hat, delta_x, iteration, k)
        x_new = x_hat + step * delta_x
        return x_new

    def gaussian_newton(self, x_hat, delta_x, step):
        pass

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
