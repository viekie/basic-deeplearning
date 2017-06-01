#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-06-01 13:49:09
import numpy as np


class elements_wise_op(object):
    @staticmethod
    def element_wise_op(array, op):
        for i in np.nditer(array, op_flags=['readwrite']):
            i[...] = op(i)
