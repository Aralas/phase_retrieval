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


def select_step_chooser(step_chooser):
    step_module = importlib.import_module('SearchingDirection')
    if step_chooser in ['BacktrackingLineSearch', 'StepDecline']:
        step_module_class = getattr(step_module, step_chooser)
        return step_module_class
    else:
        print('There is no such step_chooser %s' % step_chooser)


def select_searcher(searcher):
    searcher_module = importlib.import_module('SearchingDirection')
    if searcher in ['GradientDescent', 'Newton', 'GuassianNewton', 'SteepestDescent', 'CoordinateDescent']:
        searcher_module_class = getattr(searcher_module, searcher)
        return searcher_module_class
    else:
        print('There is no such searcher %s' % searcher)


def select_algorithm(algorithm):
    alg_module = importlib.import_module('Algorithm')
    if algorithm in ['GD_PR', 'N_PR', 'GN_PR', 'SP_PR', 'HTP_PR', 'IHT_PR', 'OMP_PR']:
        alg_module_class = getattr(alg_module, algorithm)
        return alg_module_class
    else:
        print('There is no such algorithm %s' % algorithm)


x, A, y, z = GD.generate_data(0)
loss_func = SD.LossFunction(A, y, z)
step_class = select_step_chooser(param.step_chooser)
searcher_class = select_searcher(param.searcher)
alg_class = select_algorithm(param.algorithm)

step_object = step_class()

alg_object = alg_class(x, A, y, z, param.k, param.epsilon, param.max_iter, 'initializer', searcher)

reconstruct_error, measurement_error = alg_object.solver()