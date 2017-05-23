#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-05-23 15:02:49
import numpy as np


class FullConnectLayer(object):
    def __init__(self, input_size, output_size, activator):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator

        self.weights = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        self.bais = np.zeros((output_size, 1))
        self.output = np.zeros((output_size, 1))

    def forward(self, input_vec):
        self.input = input_vec
        self.output = self.activator.forward(
            np.dot(self.weights, input_vec) + self.bais
        )

    def backward(self, delta_vec):
        # 反向计算delta, a * (1-a) * sum(w * sigma)
        self.delta = self.activator.backward(self.input) * np.dot(
            self.weights.T, delta_vec)
        # 计算梯度 delta * x
        self.weights_grad = np.dot(delta_vec, self.input.T)
        self.bais_grad = delta_vec

    def update_weights(self, rate):
        # 更新权重
        self.weights += rate * self.weights_grad
        self.bais += rate * self.bais_grad
