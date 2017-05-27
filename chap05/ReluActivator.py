#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-05-27 09:52:48


class ReluActivator(object):
    def forward(self, input):
        return max(0, input)

    def backward(self, output):
        return 1 if output > 0 else 0
