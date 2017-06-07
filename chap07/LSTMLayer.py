#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-06-07 08:49:44
import numpy as np
import SigmoidActivator
import TanhActivator


class LSTMLayer(object):
    def __int__(self, input_width, state_width, learning_rate):
        '''
        根据LSTM前向计算和方向传播算法，我们需要初始化一系列矩阵和向量。
        这些矩阵和向量有两类用途，一类是用于保存模型参数，如Wf...。另一类
        是保存各种中间计算结果，以便于反向传播算法使用，如ht等。
        '''
        self.input_width = input_width
        self.state_width = state_width
        self.learning_rate = learning_rate
        # 门的激活函数
        self.gate_activator = SigmoidActivator()
        # 输出的激活函数
        self.output_activator = TanhActivator()
        # 时刻计数
        self.times = 0
        # 各个时刻的单元状态向量C
        self.c_list = self.init_state_vec()
        # 各个时刻的输出向量H
        self.h_list = self.init_state_vec()
        # 各个时刻的遗忘门F
        self.f_list = self.init_state_vec()
        # 各个时刻的输出门Ｉ
        self.i_list = self.init_state_vec()
        # 各个时刻的输出门O
        self.o_list = self.init_state_vec()
        # 各个时刻的即时状态C~
        self.ct_list = self.init_state_vec()
        # 遗忘门权重矩阵Wfh, Wfx, 偏置bf
        self.Wfh, self.Wfx, self.bf = self.init_weight_mat()
        # 输出门权重矩阵Wih, Wix, 偏置bi
        self.Wih, self.wix, self.bi = self.init_weight_mat()
        # 输出门权重矩阵Woh, Wox, 偏置bo
        self.Woh, self.Wox, self.bo = self.init_weight_mat()
        # 单元状态权重矩阵Wch, Wcx, 偏置bc
        self.Wch, self.Wcx, self.bc = self.init_weight_mat()

    def init_state_vec(self):
        state_vec_list = []
        state_vec_list.append(np.zeros(self.state_width, 1))
        return state_vec_list

    def init_weight_mat(self):
        Wh = np.random.uniform(-0.0001, 0.0001,
                               (self.state_width, self.state_width))
        Wx = np.random.uniform(-0.0001, 0.0001,
                               (self.state_width, self.input_width))
        b = np.zeros((self.state_width, 1))
        return Wh, Wx, b

    def forward(self, x):
        self.times += 1
        # 计算遗忘门
        fg = self.calc_gate(x, self.Wfx, self.Wfh, self.bf,
                            self.gate_activator)
        self.f_list.append(fg)
        # 计算输入门
        ig = self.calc_gate(x, self.Wix, self.Wih, self.bi,
                            self.gate_activator)
        self.i_list.append(ig)
        # 计算输出门
        og = self.calc_gate(x, self.Wox, self.Woh, self.bo,
                            self.gate_activator)
        self.o_list.append(og)
        # 计算即时状态
        ct = self.calc_gate(x, self.Wcx, self.Wch, self.bc,
                            self.output_activator)
        self.ct_list.append(ct)
        # 计算单元状态
        c = fg * self.c_list[self.times - 1] + ig * ct
        self.c_list.append(c)
        # 计算输出
        h = og * self.output_activator.forward(c)
        self.h_list.append(h)

    def calc_gate(self, x, Wx, Wh, b, activator):
        h = self.h_list[self.times - 1]
        net = np.dot(Wh, h) + np.dot(Wx, x) + b
        gate = activator.forward(net)
        return gate

    def backward(self, x, delta_h, activator):
        '''
        采用反向传播算法,实现LSTM训练
        '''
        self.calc_delta(delta_h, activator)
        self.calc_graient(x)

    def calc_delta(self, delta_h, activator):
        '''
        计算误差项目
        '''
        # 初始化各个时刻的误差项
        # 输出误差
        self.delta_h_list = self.init_delta()
        # 输出门误差
        self.delta_o_list = self.init_delta()
        # 输入门误差
        self.delta_i_list = self.init_delta()
        # 遗忘门误差
        self.delta_f_list = self.init_delta()
        # 即时输出误差
        self.delta_ct_list = self.init_delta()

        self.delta_h_list[-1] = delta_h
        for k in range(self.times, 0, -1):
            self.calc_delta_k(k)

    def init_delta(self):
        '''
        初始化误差
        '''
        delta_list = []
        for i in range(self.times + 1):
            delta_list.append(np.zeros((self.state_width, 1)))
        return delta_list

    def calc_delta_k(self, k):
        '''
        计算k时刻的损失
        '''
        # 获取k时刻向前计算结果
        ig = self.i_list[k]
        og = self.o_list[k]
        fg = self.f_list[k]
        ct = self.ct_list[k]
        c = self.c_list[k]
        c_prev = self.c_list[k - 1]
        tanh_c = self.output_activator.forward(c)
        delta_k = self.delta_h_list[k]

        delta_o = delta_k * tanh_c * \
            self.gate_activator.backward(og)

        delta_f = delta_k * og * (1 - tanh_c * tanh_c) * c_prev * \
            self.gate_activator.backward(fg)

        delta_i = delta_k * og * (1 - tanh_c * tanh_c) * ct * \
            self.gate_activator.backward(ig)

        delta_ct = delta_k * og * (1 - tanh_c * tanh_c) * ig * \
            self.output_activator.backward(ct)

        delta_h_prev = (np.dot(delta_o.transpose(), self.Woh) +
                        np.dot(delta_i.transpose(), self.Wih) +
                        np.dot(delta_f.transpose(), self.Wfh) +
                        np.dot(delta_ct.transpose(), self.Wch)).transpose()
        self.delta_h_list[k - 1] = delta_h_prev
        self.delta_f_list[k] = delta_f
        self.delta_i_list[k] = delta_i
        self.delta_o_list[k] = delta_o
        self.delta_ct_list[k] = delta_ct

    def calc_gradient(self, x):
        self.Wfh_grad, self.Wfx_grad, \
            self.bf_grad = self.init_weight_gradient_mat()

        self.Wih_grad, self.Wix_grad, \
            self.bi_grad = self.init_weight_gradient_mat()

        self.Woh_grad, self.Wox_grad, \
            self.bo_grad = self.init_weight_gradient_mat()

        self.Wch_grad, self.Wcx_grad, \
            self.bc_grad = self.init_weight_gradient_mat()

        for t in range(self.times, 0, -1):
            (Wfh_grad, bf_grad,
             Wih_grad, bi_grad,
             Woh_grad, bo_grad,
             Wch_grad, bc_grad) = (self.calc_gradient_t(t))

            self.Wfh_grad += Wfh_grad
            self.bf_grad += bf_grad
            self.Wih_grad += Wih_grad
            self.bi_grad += bi_grad
            self.Woh_grad += Woh_grad
            self.bo_grad += bo_grad
            self.Wch_grad += Wch_grad
            self.bc_grad += bc_grad

        xt = x.transpose()
        self.Wfx_grad = np.dot(self.delta_f_list[-1], xt)
        self.Wix_grad = np.dot(self.delta_i_list[-1], xt)
        self.Wox_grad = np.dot(self.delta_o_list[-1], xt)
        self.Wcx_grad = np.dot(self.delta_ct_list[-1], xt)

    def init_weight_gradient_mat(self):
        Wh_grad = np.zeros((self.state_width, self.state_width))
        Wx_grad = np.zeros((self.state_width, self.input_width))
        b_grad = np.zeros((self.state_width, 1))
        return Wh_grad, Wx_grad, b_grad

    def calc_gradient_t(self, t):
        h_prev = self.h_list[t - 1].transpose()
        Wfh_grad = np.dot(self.delta_f_list[t], h_prev)
        bf_grad = self.delta_f_list[t]

        Wih_grad = np.dot(self.delta_i_list[t], h_prev)
        bi_grad = self.delta_f_list[t]

        Woh_grad = np.dot(self.delta_o_list[t], h_prev)
        bo_grad = self.delta_f_list[t]

        Wch_grad = np.dot(self.delta_ct_list[t], h_prev)
        bc_grad = self.delta_ct_list[t]

        return Wfh_grad, bf_grad, Wih_grad, bi_grad, \
            Woh_grad, bo_grad, Wch_grad, bc_grad

    def update(self):
        self.Wfh -= self.learning_rate * self.Wfh_grad
        self.Wfx -= self.learning_rate * self.Wfx_grad
        self.bf -= self.learning_rate * self.bf_grad
        self.Wih -= self.learning_rate * self.Wih_grad
        self.Wix -= self.learning_rate * self.Wix_grad
        self.bi -= self.learning_rate * self.bi_grad
        self.Woh -= self.learning_rate * self.Woh_grad
        self.Wox -= self.learning_rate * self.Wox_grad
        self.bo -= self.learning_rate * self.bo_grad
        self.Wch -= self.learning_rate * self.Wch_grad
        self.Wcx -= self.learning_rate * self.Wcx_grad
        self.bc -= self.learning_rate * self.bc_grad
