# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:RunTest.py
@time:2018/11/914:58
"""

import GenerateData as GD
import HyperParameter as hp
import importlib
import SearchingDirection as SD
import time

param = hp.param_setting


def select_algorithm(algorithm):
    alg_module = importlib.import_module('.', 'Algorithms')
    if algorithm in ['GD_PR', 'N_PR', 'GN_PR', 'SP_PR', 'HTP_PR', 'IHT_PR', 'OMP_PR']:
        alg_module_class = getattr(alg_module, algorithm)
        return alg_module_class
    else:
        print('There is no such algorithm %s' % algorithm)


success_exp = 0
start_time_all = time.time()
for experiment_i in range(param.trial_num):
    start_time = time.time()
    seed = experiment_i
    x, A, y, z = GD.generate_data(seed)
    loss_func_object = SD.LossFunction(A, y, z)
    alg_class = select_algorithm(param.algorithm)
    alg_object = alg_class(x, A, y, z, param.k, param.epsilon, param.max_iter, param.initializer, param.searcher,
                           param.step_chooser)
    reconstruct_error, measurement_error, iteration, success = alg_object.solver()
    success_exp += success
    end_time = time.time()
    print('experiment: %d, success_rate: %f, recon_error: %f, meas_error: %f, iteration: %d, time: %f' % (
        experiment_i, success_exp / (experiment_i + 1), reconstruct_error[-1], measurement_error[-1], iteration,
        end_time - start_time))
end_time_all = time.time()

print('time for %d experiments is %f, success rate is %f' % (param.trial_num, end_time_all - start_time_all, success_exp / param.trial_num))
