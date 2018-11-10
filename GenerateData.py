# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:GenerateData.py
@time:2018/11/914:50

This script will generate 1D signal for Phase Retrieval
based on hyper parameters defined in HyperParameter.py.

Parameters:
    x: 1D signal
    n: length of signal
    m: length of measurements
    k: sparsity (x is non-sparse if k=n)
    A: linear measurement matrix of size nxm
    y: measurements y=Ax
    z: squared value of y
    seed: random seed

    isComplex: control whether x is complex-value
    Options: compared algorithms
    trial_num: number of experiments for each algorithm
    data_type:  'Gaussian' 1D random Gaussian vector
                'digital' 1D 0-1 vector (k is the number of 1)
"""

import HyperParameter as hp
import numpy as np
import random

param = hp.param_setting


def generate_data(seed):
    random.seed(seed)
    np.random.seed(seed)
    if param.data_type == 'Gaussian':
        x = np.random.randn(param.n, 1) + np.random.randn(param.n, 1) * (1j) * param.isComplex
    elif param.data_type == 'digital':
        x = np.ones(param.n, 1)
    else:
        print('There is no such type of data: %s' % param.data_type)
    if param.k < param.n:
        indices = random.sample(range(param.n), param.n - param.k)
        x[indices] = 0
    A = np.random.randn(param.m, param.n) + np.random.randn(param.m, param.n) * (1j) * param.isComplex
    y = abs(A.dot(x))
    z = np.power(y, 2)
    return x, A, y, z
