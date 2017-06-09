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
excerise:里面包含以Yann的手写识别数据库中的数字识别为样本，进行测试。
## Chapter 04 Vector Programming
本算法根据向量化编程的思路，对上一节的全连接神经网络进行重写。本节代码中使用了numpy等类库。
## Chapter 05 Convolutional Neural Network
本章介绍了卷积神经网络，通常卷积神经网络会和全连接神经网络混合使用，前面若干层分别是不同尺寸，不同功能的卷积核，进行卷积计算，然后伴随若干个全连接神经网络，进行softmax分类。</br>
此处的BP算法，根据输出值，按照全连接网络里面的反向传播算法，进行梯度下降。到达卷积层时，按照本章的pooling和convolutionalLayer里面定义的方法进行反向传播。以全连接层计算出的损失作为卷基层的input_array,进行后续计算。具体推到，参照image文件夹下的图片。
## Chapter 06 Recurrent Neural Network
本章介绍了循环神经网络，每一个层共享一组Ｗ和Ｕ权值。假设当前层处于第t时刻的第l层，想要前向计算O(t,l+1)输出，则计算等于W*O(t-1, l+1) + U*O(t, l)。在同一层，共享一组Ｗ和Ｕ权重。具体推到公式，参见image文件见下的图片</br>
## Chapter 07 LSTM
本章介绍了LSTM算法，通过增加遗忘门，输出门和输出门进行优化循环神经网络，以避免梯度消失或者梯度爆炸问题。具体数学证明见Image下。</br>
## Chapter 08 Recursive Neural Network
本章介绍了递归神经网络，递归神经网络在一定程度上解决了语法树的问题。公式推到见Image文件夹.a
---

##Reference
https://www.zybuluo.com/hanbingtao/note/433855
https://www.zybuluo.com/hanbingtao/note/448086
https://www.zybuluo.com/hanbingtao/note/476663
https://www.zybuluo.com/hanbingtao/note/485480
https://www.zybuluo.com/hanbingtao/note/541458
https://www.zybuluo.com/hanbingtao/note/581764
https://www.zybuluo.com/hanbingtao/note/626300

##Concat and Email
email: viekiedu@gmail.com
id: viekie
