#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-05-27 08:35:06
import numpy as np
import Filter


class ConvolutionalLayer(object):
    def __init__(self, input_width, input_height, channel_number,
                 filter_width, filter_height, filter_number, zero_padding,
                 stride, activator, learning_rate):
        # 初始化长、宽、深度、filter的长、宽和深度、激活函数和学习速率
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filer_height = filter_height
        self.filter_number = filter_number
        self.filters = []
        self.zero_padding = zero_padding
        self.stride = stride
        self.activator = activator
        self.learning_rate = learning_rate
        # 计算输出结果的宽度
        self.output_width = \
            ConvolutionalLayer.calc_output_size(self.input_width,
                                                self.filter_width,
                                                self.zero_padding,
                                                self.stride)
        # 计算输出结果的高度
        self.output_height = \
            ConvolutionalLayer.calc_output_size(self.input_height,
                                                self.filter_height,
                                                self.zero_padding,
                                                self.stride)
        # 定义生成矩阵
        self.output_array = \
            np.zeros((self.filter_number, self.output_height,
                      self.output_width))
        # 定义filter
        for i in range(channel_number):
            self.filters.append(Filter(self.filter_width, self.filer_height,
                                       self.channel_number))

    @staticmethod
    def calc_output_size(input_size, filter_size, padding_size, stride):
        '''
        计算输出大小
        '''
        return (input_size + 2 * padding_size - filter_size) / stride + 1

    def forward(self, input_array):
        '''
        向前计算输出值
        '''
        self.input_array = input_array
        # 先进行padding
        self.padding_input_array = self.padding(input_array, self.zero_padding)
        # 逐个进行卷积计算
        for f in self.filters:
            self.convolution(self.padding_input_array, f.get_weights(),
                             f.get_bais(), self.stride, self.output_array[f])
        # 对卷积的结果进行Relu函数操作
        self.element_wise_op(self.output_array, self.activator.forward)

    def elements_wise_op(self, input_array, f):
        for i in np.nditer(input_array, op_flags=['readwrite']):
            i[...] = f(input_array[i])

    def convolution(self, input_array, kernel_array, bais,
                    stride, output_array):
        # 获取输出和卷积核的大小
        output_width = output_array[1]
        output_height = output_array[0]
        kernel_width = kernel_array[-1]
        kernel_height = kernel_array[-2]

        # 逐个计算卷积结果
        for i in range(output_height):
            for j in range(output_width):
                output_array[i][j] = \
                    (self.get_patch(input, i, j, stride,
                                    kernel_height, kernel_width) *
                     kernel_array).sum() + bais

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

    def get_sub_array(self, input_array, i, j, stride, height, width, nd):
        '''
        获取子矩阵
        '''
        row_start = i * stride
        col_start = j * stride
        return input_array[row_start: row_start + height, col_start: col_start]

    def padding(self, input_array, zp):
        '''
        对input_array进行padding
        '''
        if zp == 0:
            return input_array
        else:
            if input_array.ndim == 3:
                input_width = input_array.shape[2]
                input_height = input_array.shape[1]
                input_depth = input_array.shape[0]
                # 初始化要被返回的padding结果
                padding_array = np.zeros((input_depth, input_height + 2 * zp,
                                          input_width + 2 * zp))
                # 对padding中原本已经存在的元素进行copy
                padding_array[:,
                              zp: zp + input_height,
                              zp + input_width] = input_array
                return padding_array
            elif input_array.ndim == 2:
                input_width = input_array.shape[1]
                input_height = input_array.shape[0]
                padding_array = np.zeros((input_height, input_width))
                padding_array[zp: zp + input_height,
                              zp: zp + input_width] = input_array
                return padding_array

    def bp_sensitivity_map(self, sensitivity_array, activator):
        # 扩展为步长为１
        expanded_array = \
            self.expand_sensitivity_map(sensitivity_array)
        expanded_width = expanded_array.shape[2]
        # 计算需要padding 的大小
        zp = (self.input_width + self.filter_width - 1 - expanded_width) / 2
        # 执行padding
        padded_array = self.padding(expanded_array, zp)
        # 创建存放梯度的数组
        self.delta_array = self.create_delta_array()
        # 每一个filter都作用于sensitivity map,然后对相应的
        # filter对应的结果进行求和
        for f in range(self.filter_number):
            filter = self.filters[f]
            # 权重矩阵180度旋转
            rotate_weights = np.array(map(lambda i: np.rot90(i, 2),
                                          filter.get_weights()))
            # 创建临时梯度矩阵
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                # 更新每一个channel对应的梯度矩阵
                self.convolution(padded_array[f], rotate_weights,
                                 delta_array[d], 1, 0)
            # 将每个filter求出的梯度矩阵进行各自求和
            self.delta_array += delta_array
        # 生成输入向量转换为np的array
        derivative_array = np.array(self.input_array)
        # 将输入向量求导数矩阵
        self.elements_wise_op(derivative_array, self.activator.backward)
        # 求出梯度
        self.delta_array *= derivative_array

    def expand_sensitivity_map(self, sensitivity_array):
        depth = sensitivity_array.shape[0]
        # 按照步长为１,计算卷积结果行、列数
        expand_width = (self.input_width -
                        self.filter_width + 2 * self.zero_padding + 1)
        expand_height = (self.input_height -
                         self.filter_width + 2 * self.zero_padding + 1)
        # 生成扩展后矩阵大小
        expand_array = np.zeros((depth, expand_height, expand_width))
        # 对相应位置赋值, 其他扩展的位置为０
        for i in range(self.output_height):
            for j in range(self.output_width):
                step_i = i * self.stride
                step_j = j * self.stride
                expand_array[:, step_i, step_j] = sensitivity_array[:, i, j]
        return expand_array

    def create_delta_array(self):
        return np.zeros((self.channel_number, self.input_height,
                         self.input_width))

    def bp_gradient(self, sensitivity_array):
        # 按照步长为1 扩展sensitivity_array
        expand_array = self.expand_sensitivity_map(sensitivity_array)
        for f in range(self.filter_number):
            filter = filter[f]
            for d in range(filter.weights.shape[0]):
                # 将扩展后的sensitivity_array和input进行卷积,实际就是为了求梯度
                self.convolution(self.padding_input_array[d],
                                 expand_array,
                                 filter.weights_gradient[d], 1, 0)
            filter.bias_grad = expand_array[f].sum()

    def update(self):
        '''
        filter进行梯度下降更新
        '''
        for filter in self.filters:
            filter.update(self.learning_rate)
