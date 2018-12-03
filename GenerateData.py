# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:GenerateData.py
@time:2018/11/914:50

This script will generate 1D signal for Phase Retrieval based on hyper parameters defined in RunTest.py.

Parameters:
    n: length of signal
    m: length of measurements
    k: sparsity (x is non-sparse if k=n)
    seed: random seed
    isComplex: control whether x is complex-value
    data_type:  'Gaussian' 1D random Gaussian vector
                'digital' 1D 0-1 vector (k is the number of 1)

outputs:
    x: 1D objective signal of length-n
    A: linear measurement matrix of size (m, n)
    y: linear measurements y = Ax
    z: squared value of y
"""

import numpy as np
import random


def generate_data(seed, param):
    random.seed(seed)
    np.random.seed(seed)
    if param.data_type == 'Gaussian':
        x = np.random.randn(param.n, 1)
    elif param.data_type == 'digital':
        x = np.ones(param.n, 1)
    else:
        print('There is no such type of data: %s' % param.data_type)
    if param.k < param.n:
        indices = random.sample(range(param.n), param.n - param.k)
        x[indices] = 0
    A = np.random.randn(param.m, param.n)
    if param.isComplex:
        x = x + np.random.randn(param.n, 1) * (1j)
        A = A + np.random.randn(param.m, param.n) * (1j)
    y = abs(A.dot(x))
    z = np.power(y, 2)
    return x, A, y, z
