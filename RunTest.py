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

param = hp.param_setting


def select_algorithm(algorithm):
    alg_module = importlib.import_module('Algorithm')
    if algorithm in ['GD_PR', 'N_PR', 'GN_PR', 'SP_PR', 'HTP_PR', 'IHT_PR', 'OMP_PR']:
        alg_module_class = getattr(alg_module, algorithm)
        return alg_module_class
    else:
        print('There is no such algorithm %s' % algorithm)


for experiment_i in range(param.trial_num):
    seed = experiment_i
    x, A, y, z = GD.generate_data(seed)
    loss_func_object = SD.LossFunction(A, y, z)
    alg_class = select_algorithm(param.algorithm)
    alg_object = alg_class(x, A, y, z, param.k, param.epsilon, param.max_iter, param.initializer, param.searcher,
                           param.step_chooser)
    reconstruct_error, measurement_error = alg_object.solver()
