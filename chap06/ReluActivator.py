#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-06-01 10:27:22


class ReluActivator(object):
    def forward(self, input):
        return max(0, input)

    def backward(self, input):
        return 1 if input > 0 else 0
