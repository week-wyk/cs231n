---
typora-root-url: ./fig
---

# **L2: Regularization and Optimization**

loss function

### regularization

prefer simpler model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            奥卡姆剃刀

加上$\lambda R(W)$，训练数据更差，测试数据更好

L1:很多0

L2:更加分散但非零

softmax可以将一组浮点数转换为相对概率



### optimization

### gradient

数值法：根据导数定义式来求解，每次取一个尽量小的h即可，对于不可微的函数也可以用h来求$f'(x_0) = \lim_{h \to 0} \frac{f(x_0 + h) - f(x_0)}{h}$

Numerical gradient: approximate, slow, easy to write 

解析法：根据现有公式直接计算 Analytic gradient: exact, fast, error-prone

In practice: Always use analytic gradient, but check  implementation with numerical gradient. This is called a  **gradient check**

gradient descent

用误差小于一定值来停止/预期的迭代的次数来停止

随机梯度下降SGD

每次随机采样1batch（数字自定义），来计算梯度

GD problem：

1、Very **slow progress** along shallow dimension, jitter along steep direction

2、局部最小值/saddle point

![ch2-saddlepoint](/ch2-saddlepoint.jpg)

3、SGD问题： 因为是二次采样的，有噪声

#### 如何解决问题？

1、SGD+momentum

相当于用二阶导进行迭代

![ch2-addmomentum](/ch2-addmomentum.png)

![ch2-SGD_Momentum](/ch2-SGD_Momentum.jpg)

:warning: 为什么加和减一样我还没有理解

2、RMSProp

![image-20251020130430751](/ch2-RMSPROP.png)

在梯度小的时候多走几步，增加有效步数，在梯度大的时候减少有效步数。



3、adam

**![image-20251020130633994](/ch2-Adam.png)**

AdamW: Adam Variant with Weight Decay

![image-20251020131027002](/ch2-AdamW.png)

#### 关于Learning rate

1、Step: Reduce learning rate at a few fixed  points. E.g. for ResNets, multiply LR by 0.1  after epochs 30, 60, and 90.

2、Cosine ： $\alpha_t = 1/2 \alpha_0 (1-t/T)$

3、 linear $ \alpha_t = \alpha_0 (1-t/T)$

4、 inverse sqrt $\alpha_t = \alpha_0 /\sqrt t$

5、 linear warmup 

High initial learning rates can make loss  explode; linearly increasing learning rate  from 0 over the first ~5,000 iterations can  prevent this. 

Empirical rule of thumb: If you increase the  batch size by N, also **scale the initial  learning rate by N**

### 二阶优化

 缺点：Hessian has O(N^2) elements Inverting takes O(N^3) N = (Tens or Hundreds of) Millions



In practice:

- **Adam(W)**is a good default choice in many cases; it  often works ok even with constant learning rate
- **SGD+Momentum** can outperform Adam but may  require more tuning of LR and schedule
- If you can **afford to do** full batch updates then look  beyond 1st order optimization (2nd order and  beyond)
