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

n = hp.n
m = hp.m
k = hp.k

isComplex = hp.isComplex
data_type = hp.data_type


def generateData(seed):
    random.seed(seed)
    np.random.seed(seed)
    if data_type == 'Gaussian':
        x = np.random.randn(n, 1) + np.random.randn(n, 1)*(1j)*isComplex
    elif data_type == 'digital':
        x = np.ones(n, 1)
    if k != n:
        indices = random.sample(range(n), n - k)
        x[indices] = 0
    A = np.random.randn(m, n) + np.random.randn(m, n)*(1j)*isComplex
    y = A.dot(x)
    z = np.power(y, 2)
    return (x, A, y, z)
