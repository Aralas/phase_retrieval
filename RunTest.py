# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:RunTest.py
@time:2018/11/914:58

This script will set parameters used for Phase Retrieval and run experiments.

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
            gradient_descent
            newton
            guassian_newton
            steepest_descent
            coordinate_descent

    step_chooser:
            backtracking_line_search
            constant_step

    initializer:
            init_random

"""

import GenerateData as GD
import SearchingDirection as SD
import time
import Algorithms


class ParameterSetting(object):

    def __init__(self, n, m, k, epsilon, step_value, isComplex, trial_num, max_iter, algorithm, data_type,
                 step_chooser, searcher, initializer):
        self.n = n
        self.m = m
        self.k = k
        self.epsilon = epsilon
        self.step_value = step_value
        self.isComplex = isComplex
        self.max_iter = max_iter
        self.trial_num = trial_num
        self.algorithm = algorithm
        self.data_type = data_type
        self.step_chooser = step_chooser
        self.searcher = searcher
        self.initializer = initializer


def select_algorithm(algorithm):
    if algorithm in ['GD_PR', 'N_PR', 'GN_PR', 'SP_PR', 'HTP_PR', 'IHT_PR', 'OMP_PR']:
        alg_module_class = getattr(Algorithms, algorithm)
        return alg_module_class
    else:
        print('There is no such algorithm %s' % algorithm)


def run_experiment(param):
    success_exp = 0
    start_time_all = time.time()
    for experiment_i in range(param.trial_num):
        start_time = time.time()
        seed = experiment_i
        x, A, y, z = GD.generate_data(seed, param)
        alg_class = select_algorithm(param.algorithm)
        alg_object = alg_class(x, A, y, z, param)
        reconstruct_error, measurement_error, iteration, success = alg_object.solver()
        success_exp += success
        end_time = time.time()
        print('experiment: %d, success_rate: %f, recon_error: %f, meas_error: %f, iteration: %d, time: %f' % (
            experiment_i, success_exp / (experiment_i + 1), reconstruct_error[-1], measurement_error[-1], iteration,
            end_time - start_time))
    end_time_all = time.time()
    print('time for %d experiments is %f, success rate is %f' % (
        param.trial_num, end_time_all - start_time_all, success_exp / param.trial_num))
    record.write(str(param.n) + ',' + str(param.m) + ',' + str(param.k) + ',' + str(param.step_value) + ',' + \
                 str(success_exp / param.trial_num) + ',' + str(end_time_all - start_time_all) + '\n')
    record.flush()


record = open('record.txt', 'a+')
# record.write('n, m, k, step, success rate, time\n')


for k in [6, 8, 10]:
    for step_value in [0.02, 0.015, 0.008, 0.005]:
        print('*' * 10, k, step_value, '*' * 10)
        param_setting = ParameterSetting(n=100, m=400, k=k, epsilon=0.001, step_value=step_value,
                                         isComplex=False, trial_num=500, max_iter=3000, algorithm='GD_PR',
                                         step_chooser='constant_step', data_type='Gaussian',
                                         searcher='gradient_descent', initializer='init_random')
        run_experiment(param_setting)

for k in [15, 20]:
    for step_value in [0.006, 0.004, 0.002]:
        print('*' * 10, k, step_value, '*' * 10)
        param_setting = ParameterSetting(n=100, m=400, k=k, epsilon=0.001, step_value=step_value,
                                         isComplex=False, trial_num=500, max_iter=3000, algorithm='GD_PR',
                                         step_chooser='constant_step', data_type='Gaussian',
                                         searcher='gradient_descent', initializer='init_random')
        run_experiment(param_setting)

for k in [25, 30]:
    for step_value in [0.005, 0.004, 0.0025, 0.002]:
        print('*' * 10, k, step_value, '*' * 10)
        param_setting = ParameterSetting(n=100, m=400, k=k, epsilon=0.001, step_value=step_value,
                                         isComplex=False, trial_num=500, max_iter=3000, algorithm='GD_PR',
                                         step_chooser='constant_step', data_type='Gaussian',
                                         searcher='gradient_descent', initializer='init_random')
        run_experiment(param_setting)

for k in [40, 50, 60]:
    for step_value in [0.002, 0.0015, 0.0008, 0.0005]:
        print('*' * 10, k, step_value, '*' * 10)
        param_setting = ParameterSetting(n=100, m=400, k=k, epsilon=0.001, step_value=step_value,
                                         isComplex=False, trial_num=500, max_iter=3000, algorithm='GD_PR',
                                         step_chooser='constant_step', data_type='Gaussian',
                                         searcher='gradient_descent', initializer='init_random')
        run_experiment(param_setting)



for k in [70, 80, 90, 100]:
    for step_value in [0.0015, 0.0008, 0.0006, 0.0004]:
        print('*' * 10, k, step_value, '*' * 10)
        param_setting = ParameterSetting(n=100, m=400, k=k, epsilon=0.001, step_value=step_value,
                                         isComplex=False, trial_num=500, max_iter=3000, algorithm='GD_PR',
                                         step_chooser='constant_step', data_type='Gaussian',
                                         searcher='gradient_descent', initializer='init_random')
        run_experiment(param_setting)

record.close()
