# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:HyperParameter.py
@time:2018/11/914:59

This script will set parameters used for Phase Retrieval test.

Parameters:
    x: 1D signal
    n: length of signal
    m: length of measurements
    k: sparsity (x is non-sparse if k=n)
    A: linear measurement matrix of size (n, m)
    y: measurements y=Ax
    z: squared value of y
    isComplex: control whether x is complex-value
    trial_num: number of experiments for each algorithm

    algorithm:
            GD_PR: Gradient Descent Phase Retrieval (Wirtinger Flow)
            N_PR: Newton Phase Retrieval
            GN_PR: Gaussian Newton Phase Retrieval
            SP_PR: Subspace Pursuit Phase Retrieval
            HTP_PR: Hard Thresholding Pursuit Phase Retrieval
            IHT_PR: Iterative Hard Thresholding Phase Retrieval
            OMP_PR: Orthogonal Matching Pursuit Phase Retrieval

    data_type:
            'Gaussian' 1D random Gaussian vector
            'digital' 1D 0-1 vector (k is the number of 1)

    searcher:
            GradientDescent
            Newton
            GuassianNewton
            SteepestDescent
            CoordinateDescent

    step_chooser:
            BacktrackingLineSearch
            StepDecline

    initializer:


"""


class ParameterSetting(object):

    def __init__(self, n, m, k, epsilon, isComplex, trial_num, max_iter, algorithm, data_type, step_chooser, searcher, initializer):
        self.n = n
        self.m = m
        self.k = k
        self.epsilon = epsilon
        self.isComplex = isComplex
        self.max_iter = max_iter
        self.trial_num = trial_num
        self.algorithm = algorithm
        self.data_type = data_type
        self.step_chooser = step_chooser
        self.searcher = searcher
        self.initializer = initializer


param_setting = ParameterSetting(n=10, m=20, k=10, epsilon=0.001, isComplex=False, trial_num=1, max_iter=3000,
                                 algorithm='GD_PR', step_chooser='step_decline', data_type='Gaussian',
                                 searcher='gradient_descent', initializer='init_random')
