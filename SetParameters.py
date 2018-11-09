# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:SetParameters.py
@time:2018/11/914:59

This script will set parameters used for Phase Retrieval test.



                        Recovery Problem
This script creates a complex-valued random Gaussian signal. Measurements of
the signal are then obtained by applying a linear operator to the signal, and
computing the magnitude (i.e., removing the phase) of the results.

                      Measurement Operator
Measurement are obtained using a linear operator, called 'A', that contains
random Gaussian entries.

                     The Recovery Algorithm
First, the recovery options should be given to the 'Options' class, which
manages them and applies the default cases for each algorithm or initializer,
in case of not being explicitly provided by the user.
All the information about the problem is collectted in the 'Retrieval' class and
the image is recovered by calling its method solve_phase_retrieval().

For more details, see the Phasepack user guide.

Based on MATLAB implementation by Rohan Chandra, Ziyuan Zhong, Justin Hontz,
Val McCulloch, Christoph Studer & Tom Goldstein.
Copyright (c) University of Maryland, 2017.
Python version of the phasepack module by Juan M. Bujjamer.
University of Buenos Aires, 2018.
"""



# Parameters
n = 100
m = 200
isComplex = True
k = 10






