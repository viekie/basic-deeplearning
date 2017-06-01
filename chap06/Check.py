#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-06-01 14:55:23
import RecurrentLayer
import IdentityActivator
import numpy


def gradient_check():
    error_function = lambda o: o.sum()
    rl = RecurrentLayer(3, 2, IdentityActivator(), 0.001)
    x, d = data_set()
    rl.forward(x[0])
    rl.forward(x[1])

    sensitity_array = np.ones(rl.state_list[-1].shape,
                              dtype=np.float64)
    rl.backward(sensitity_array, IdentityActivator())
