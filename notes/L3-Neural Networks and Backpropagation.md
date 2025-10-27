---
typora-root-url: ./fig
---

# **L3: Neural Networks and Backpropagation**

### activation functions

![image-20251020153619386](/ch3-activationfunc.png)

一个基本的神经网络：

1、Define the network 

2、Forward pass

3、Calculate the analytical gradients

4、Gradient descent 

 **more neurons = more capacity** also 过拟合 就像knn里的k=1的情况

Do not use size of neural network as a regularizer. **Use stronger regularization** instead

因为神经元越多学习能力越强，但同时训练所需要的时间也越多。为了平衡二者，我们常做的是在目前神经元数量基础上尽可能去拟合（用正则化相关参数进行调节），发挥目前神经网络最强的能力。

Problem: How to compute gradients?

1-(Bad) Idea: Derive gradients on paper

2- Better Idea: Computational graphs + Backpropagation

### Backpropagation

求解损失函数对每个权重的偏导数（梯度），然后通过梯度下降更新参数。

**前向传播**:输入数据从输入层逐层计算至输出层，得到预测值

$\hat{y} = f(x; \theta)$

**反向传播**：从输出层开始，**逆向计算每个层的梯度**，逐层传递误差

$\frac{\partial L}{\partial w_{l}} = \frac{\partial L}{\partial a_{l}} \cdot \frac{\partial a_{l}}{\partial z_{l}} \cdot\frac{\partial z_{l}}{\partial w_{l}}$

![image-20251020170539112](/ch3-bp.png)

![image-20251020171907717](/ch3-pytorch-multiply-api.png)

#### Backprop with Vectors even matrix!

Loss L still a scalar!

![image-20251020172425298](/ch3-bp-matrix.png)