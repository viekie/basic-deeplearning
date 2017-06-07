#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-06-07 08:34:07
import numpy as np


class SigmoidActivator(object):
    def forward(self, input):
        return 1.0 / (1.0 + np.exp(-input))

    def backward(self, output):
        return output * (1 - output)
