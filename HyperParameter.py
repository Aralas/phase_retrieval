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



# Parameters
n = 100
m = 200
k = 10
isComplex = True
trial_num = 500
Options = ['Newton']
data_type = 'Gaussian'
