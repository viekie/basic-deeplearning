#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-05-27 09:23:04
import numpy as np


class Filter(object):
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
        self.weights = np.random.uniform(-0.0001, 0.0001,
                                         (depth, height, width))
        self.bais = 0.0
        self.weights_gradient = np.zeros(self.weights.shape)
        self.bais_gradient = 0.0

    def get_weights(self):
        return self.weights

    def get_bais(self):
        return self.bais

    def update(self, learning_rate):
        self.weights += learning_rate * self.weights_gradient
        self.bais += learning_rate * self.bais_gradient

    def __str__(self):
        return 'weight: %s, bais: %s' % (self.weights, self.bais)
