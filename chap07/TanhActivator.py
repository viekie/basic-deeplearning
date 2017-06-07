#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-06-07 08:38:17
import numpy as np


class TanhAvtivator(object):
    def forward(self, input):
        return -1.0 + 2.0 / (1.0 + np.exp(-2 * input))

    def backward(self, input):
        return 1 - input * input
