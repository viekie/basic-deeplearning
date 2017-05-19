#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-05-19 15:36:33
import struct


class Loader(object):
    def __init__(self, path, count):
        self.path = path
        self.count = count

    def get_file_content(self):
        with open(self.path, 'rb') as f:
            content = f.read()
        return content

    def to_int(self, byte):
        return struct.unpack('B', byte)[0]
