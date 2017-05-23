#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-05-23 15:25:25
import FullConnectLayer
import SigmoidActivator


class Network(object):
    def __init__(self, layers):
        self.layers = []
        '''
        定义整个网络
        '''
        for i in range(len(layers)):
            self.layers.append(
                FullConnectLayer(
                    layers[i],
                    layers[i + 1],
                    SigmoidActivator()))

    def predict(self, sample):
        # 预测sample的输出值
        input = sample
        # 逐层递推的计算，最终获取输出向量
        for i in range(len(self.layers)):
            self.layers[i].forward(input)
            input = self.layers[i].output
        return input

    def train(self, datasets, labels, rate, iteration):
        for i in range(iteration):
            for d in range(len(labels)):
                self.train_one_sample(datasets[d], labels[d], rate)

    def train_one_sample(self, sample, label, rate):
        # 预测输出结果
        self.predict(sample)
        # 计算梯度
        self.calc_gradient(label)
        # 更新权重
        self.update_weights(rate)

    def calc_gradient(self, label):
        # 因为最后一层梯度计算方式不同，所以单独计算最后一层的梯度
        output_delta = self.layers[-1].activator.backward(
            self.layers[-1].output) * (label - self.layers[-1].output)
        # 更新其他层的梯度
        for layer in self.layers[::-1]:
            layer.backward(output_delta)
            output_delta = layer.delta

    def update_weights(self, rate):
        # 更新weights和bais 矩阵
        for layer in self.layers:
            layer.update_weights()
