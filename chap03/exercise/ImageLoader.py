#!/usr/bin/env python
# -*- coding:utf8 -*-
# Power by viekie. 2017-05-19 15:43:51
import Loader


class ImageCloader(Loader):
    def get_picture(self, content, index):
        start = index * 28 * 28 + 16
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(self.to_int(content[start + i * 28 + j]))
        return picture

    def get_sample(self, picture):
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    def load(self):
        content = self.get_file_content()
        data_set = []
        for index in range(self.count):
            data_set.append(self.get_sample(self.get_picture(content, index)))
        return data_set
