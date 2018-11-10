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
        func = lambda i: b[i] * np.dot(self.A[i].reshape(-1, 1), self.A[i].reshape(1, -1))
        grad = 2 * np.dot(np.mean([func(i) for i in range(len(self.A))], axis=0), x_hat)
        return grad


class StepChooser(object):

    def __init__(self, loss_func):
        self.alpha = 0.3
        self.beta = 0.8
        self.loss_func = loss_func
        self.tau = 330

    def backtracking_line_search(self, x_hat, delta_x, iter):
        step = 1
        while self.loss_func.f(x_hat + step * delta_x) >= (
                self.loss_func.f(x_hat) + self.alpha * step * np.dot(self.loss_func.gradient(x_hat).transpose(),
                                                                     delta_x)):
            step = step * self.beta
        return step

    def step_decline(self, x_hat, delta_x, iter):
        return min(0.1, (1 - math.exp(-1 * iter / self.tau)) / 2)


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


class Searcher(LossFunction):

    def __init__(self, A, y, z):
        LossFunction.__init__(self, A, y, z)

    def gradient_descent(self, x_hat, delta_x, step):
        x_new = x_hat + step * delta_x
        return x_new

    def newton(self, x_hat, delta_x, step):
        pass

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
