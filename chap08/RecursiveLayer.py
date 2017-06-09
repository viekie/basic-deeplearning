#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-06-09 14:39:06
import numpy as np
import TreeNode


class RecursiveLayer(object):
    def __init__(self, node_width, child_count, activator, learning_rate):
        '''
        初始化构造方法
        '''
        # 每个节点的向量维度, 也就是N维向量
        self.node_width = node_width
        # 子节点数量
        self.child_count = child_count
        self.activator = activator
        self.learning_rate = learning_rate
        # 初始化权重
        self.W = np.random.uniform(-0.0001, 0.0001,
                                   (node_width, node_width * child_count))
        self.b = np.zeros((node_width, 1))
        self.root = None

    def forward(self, *children):
        '''
        前向计算
        '''
        # 将多个子节点合并成一个向量
        children_data = self.concatenate(children)
        # 计算父节点数值
        parent_data = self.activator.forward(np.dot(self.W,
                                                    children_data) + self.b)
        # 通过调用forward计算，一直变更root节点，最后形成一棵树
        self.root = TreeNode(parent_data, children, children_data)

    def concatenate(self, tree_nodes):
        '''
        将数据拼接成一个一维向量
        '''
        concat = np.zeros((0, 1))
        for node in tree_nodes:
            concat = np.concatenate((concat, node.data))
        return concat

    def backward(self, parent_delta):
        '''
        反向误差传播
        '''
        self.calc_delta(parent_delta, self.root)
        self.W_grad, self.b_grad = self.calc_gradient(self.root)

    def calc_delta(self, parent_delta, parent):
        '''
        计算误差增量
        '''
        parent.delta = parent_delta
        if parent.children:
            # 计算子节点的误差
            children_delta = np.dot(self.W.T, parent_delta) * \
                (self.activator.backward(parent.children_data))
            # 因为所有子节点误差都在一个向量里面，只要切分了，就够了
            slices = [(i, i * self.node_width, (i + 1) * self.node_width)
                      for i in range(self.child_count)]
            # 遍历所有子节点，计算子节点的误差增量
            for s in slices:
                self.calc_delta(children_delta[s[1]: s[2]],
                                parent.children[s[0]])

    def calc_gradient(self, parent):
        '''
        计算梯度
        '''
        W_grad = np.zeros((self.node_width,
                           self.node_width * self.child_count))
        b_grad = np.zeros((self.node_width, 1))

        if not parent.children:
            return W_grad, b_grad
        # 计算对权重的梯度
        parent.W_grad = np.dot(parent.delta, parent.children_data.T)
        parent.b_grad = parent.delta
        # 对梯度进行累计求和
        W_grad += parent.W_grad
        b_grad += parent.b_grad
        # 计算子节点的梯度
        for child in parent.children:
            W, b = self.calc_gradient(child)
            W_grad += W
            b_grad += b
        return W_grad, b_grad

    def update(self):
        '''
        更新梯度
        '''
        self.W -= self.learning_rate * self.W_grad
        self.b -= self.learning_rate * self.b_grad
