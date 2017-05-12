#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# Power by viekie. 2016-06-30 00:57:45


class Perception(object):

    def __init__(self, domain_num, activitor):
        '''
        感知器的构造函数，一个感知器模型包含以下几个部分：
        1、权重 也就是w
        2、截距 也就是b
        3、激活函数
        '''
        self.weights = [0.0 for _ in range(domain_num)]
        self.bais = 0.0
        self.activitor = activitor

    def __str__(self):
        return 'weights: %s, bais: %s' % (self.weights, self.bais)

    def predict(self, input_vectors):
        '''
        定义预测函数，y = w * x + b
        '''
        return self.activitor(
            reduce(lambda a, b: a + b,
                   map(lambda (x, w): x * w,
                       zip(input_vectors, self.weights)),
                   self.bais))

    def train(self, samples_attr, samples_label, iteration, rate):

        '''
        定义训练函数, 对输入数据进行iteratiron次迭代
        '''
        for i in range(iteration):
            self.updateModelWeights(samples_attr, samples_label, rate)

    def updateModelWeights(self, samples_attr, samples_label, rate):
        for x_vector, label in (zip(samples_attr, samples_label)):
            output = self.predict(x_vector)
            self.update(x_vector, label, output, rate)

    def update(self, input_vector, label, output, rate):
        '''
        更新权重
        '''
        delta = label - output
        self.weights = map(lambda (w, x): w + x * delta * rate,
                           zip(self.weights, input_vector)
                           )

        self.bais += rate * delta


def activitor(perception_result):
    '''
    定义激活函数
    '''
    return 1 if perception_result > 0 else 0


def prepare_training_data():
    '''
    准备训练数据，进行and操作
    '''
    input_vec = [[0, 0], [0, 1], [1, 0], [1, 1]]
    label = [0, 1, 1, 1]
    return input_vec, label


def train_perception(input_num, iter, rate):
    p = Perception(input_num, activitor)
    input_vec, label = prepare_training_data()
    p.train(input_vec, label, iter, rate)
    return p


if __name__ == '__main__':
    p = train_perception(2, 10, 0.1)
    print p
    print '1 or 1 is %s', p.predict([1, 1])
    print '1 or 0 is %s', p.predict([1, 0])
    print '0 or 1 is %s', p.predict([0, 1])
    print '0 or 0 is %s', p.predict([0, 0])
