#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-05-19 08:17:03
import random


class Connection(object):
    def __init__(self, downstream_node, upstream_node):
        self.downstream_node = downstream_node
        self.upstream_node = upstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0

    def calc_gradient(self):
        '''
        计算connection的weight做准备
        此处算法相当于公式里面的 delta * xi
        delta就是downstream计算出来的delta
        xi相当于connection的upstream的output
        '''
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def update_weight(self, rate):
        self.calc_gradient()
        self.weight += rate * self.gradient

    def __str__(self):
        downstream_info = "downstream: %s\r\n" % self.downstream_node
        upstream_info = 'upstream: %s\r\n' % self.upstream_node
        weight_info = 'weight: %f\r\n' % self.weight
        gradient_info = 'gradient: %f\r\n' % self.gradient

        return downstream_info + upstream_info + weight_info + gradient_info
