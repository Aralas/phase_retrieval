# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:RunTest.py
@time:2018/11/914:58

This script will set parameters used for Phase Retrieval and run experiments.

Parameters:
    n: length of signal
    m: length of measurements
    k: sparsity (x is non-sparse if k=n)
    epsilon: stop condition
    step_value: the step to update estimation in projection step
    isComplex: control whether x is complex
    trial_num: number of experiments for each algorithm
    max_iter: max number of iterations for projec†ion

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

    step_chooser:
            backtracking_line_search
            constant_step

    initializer:
            init_random
            init_spectral

"""

import GenerateData as GD
import time
import signal
import Algorithms


class ParameterSetting(object):

    def __init__(self, n, m, k, epsilon, step_value, isComplex, trial_num, max_iter, algorithm, data_type,
                 step_chooser, projection, initializer):
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
        self.projection = projection
        self.initializer = initializer


def select_algorithm(algorithm):
    if algorithm in ['GD_PR', 'N_PR', 'GN_PR', 'SP_PR', 'HTP_PR', 'IHT_PR', 'OMP_PR', 'OMP_PR_2']:
        alg_module_class = getattr(Algorithms, algorithm)
        return alg_module_class
    else:
        print('There is no such algorithm %s' % algorithm)

# raise error to the experiment after fixed time
def handler(signum, frame):
    raise AssertionError


def run_experiment(param):
    success_exp = 0
    start_time_all = time.time()
    for experiment_i in range(param.trial_num):
    # try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(100)
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
    # except:
        print('experiment: %d, success_rate: %f' % (experiment_i, success_exp / (experiment_i + 1)))

    end_time_all = time.time()
    print('time for %d experiments is %f, success rate is %f' % (
        param.trial_num, end_time_all - start_time_all, success_exp / param.trial_num))
    record.write(str(param.n) + ',' + str(param.m) + ',' + str(param.k) + ',' + str(param.step_value) + ',' + \
                 str(success_exp / param.trial_num) + ',' + str(end_time_all - start_time_all) + '\n')
    record.flush()


record = open('record_OMP_2_GN_k.txt', 'a+')
# record.write('n, m, k, step, success rate, time\n')
for step_value in [1]:
    record.write('\n')
    for k in [6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        for m in [400, 350, 300, 250, 200, 150, 100]:
            print('*' * 10, 'k %d, m %d, step %f' % (k, m, step_value), '*' * 10)
            param_setting = ParameterSetting(n=100, m=m, k=k, epsilon=0.001, step_value=step_value,
                                             isComplex=False, trial_num=500, max_iter=1000, algorithm='OMP_PR',
                                             step_chooser='constant_step', data_type='Gaussian',
                                             projection='gauss_newton', initializer='init_spectral')
            run_experiment(param_setting)
record.close()
