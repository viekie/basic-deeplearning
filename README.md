# Learn Deeplearning with NO Basic Knowledge
## Chapter 01 Perception
本算法实现了最简单的感知器算法，可以实现and or not等基本计算，但是感知器模型对与线性不可分的数据是无能为力的。例如，感知器算法是无法计算异或操作。
## Chapter 02 Linear Regression
本算法实现了线性回归算法。如果假设数据误差符合高斯分布，那么对数据进行最大似然估计，进行最优化求值和采用均方损失函数，进行最优化求值得到的是相同的结果，都是最小二乘估计。
## Chapter 03 NeuralNetwork
本算法实现了全链接神经网络，网络大体被分为NeuralNetwork、Layer、Connections、Node、Connection几个类，以及校验工具箱ToolBox。</br>
Network:整个神经网络的整体框架,包含若干Layer,主要进行训练，预测。</br>
Layer:神经网络的某一层，包含若干节点。</br>
Connections:所有的Connection的集合。</br>
Connection:链接各个Node。</br>
Node:神经网络中的各节点。</br>
excerise:里面包含以Yann的手写识别数据库中的数字识别为样本，进行测试
