#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-05-27 22:59:44
import numpy as np


class MaxPoolingLayer(object):
    def __init__(self, input_width, input_height, channel_number,
                 filter_width, filter_height, stride):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride
        self.output_widht = (input_width - filter_width) / stride + 1
        self.output_height = (input_height - filter_height) / stride + 1
        self.output_array = np.zeros((channel_number,
                                      self.output_height, self.output_widht))

    def forward(self, input_array):
        '''
        前向计算maxpooling之后的值
        '''
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_widht):
                    self.output_array[d][i][j] = \
                        (self.get_patch(input_array[d], i, j,
                                        self.filter_width,
                                        self.filter_width, self.stride).max())

    def backward(self, input_array, sensitivity_array):
        '''
        反向计算pooling层的梯度，最大值对应的梯度为１* 损失值
        其他的单元梯度为０
        '''
        self.delta_array = np.zeros(input_array.shape)
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_widht):
                    patch_array = \
                        self.get_patch(input_array[d], i, j,
                                       self.filter_width,
                                       self.filter_height, self.stride)
                    k, l = self.get_max_index(patch_array)
                    self.delta_array[d, i * self.stride + k,
                                     j * self.stride + l] = \
                        sensitivity_array[d, i, j]

    def get_patch(self, input_array, i, j, stride, height, width):
        '''
        获取需要被卷积的单元, 针对２Ｄ和３Ｄ分别进行获取
        '''
        ret_array = []
        nd = input_array.ndim
        if nd == 3:
            sd = input_array.shape[0]
            for d in range(sd):
                ret_array.append(self.get_sub_array(input_array[d], i, j,
                                                    stride, height, width))
        else:
            ret_array = self.get_sub_array(input_array,
                                           i, j, stride, height, width)
        return ret_array

    def get_max_index(self, patch_array):
        '''
        获取最大值行列号
        '''
        max_i = 0
        max_j = 0
        temp = patch_array[0][0]
        for i in range(patch_array.shape[0]):
            for j in range(patch_array.shape[1]):
                if patch_array[i][j] >= temp:
                    max_i = i
                    max_j = j
        return max_i, max_j
