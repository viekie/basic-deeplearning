#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-05-23 15:17:41
import numpy as np


class SigmoidActivator(object):
    def forward(self, input):
        return 1.0 / (1.0 + np.exp(-input))

    def backward(self, input):
        return input * (1 - input)
