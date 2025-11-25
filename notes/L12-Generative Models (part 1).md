---
typora-root-url: ./fig
---

# L12: Generative Models (part 1)

### **Supervised vs Unsupervised Learning**

- Supervised Learning 

Data: (x, y) x is data, y is label 

Goal: Learn **a function to map** x -> y

- Unsupervised Learning 

Data: x Just data, no labels! 

Goal: **Learn hidden structure** in data	

### **Generative vs Discriminative Models**

- Discriminative Model: Learn a probability  distribution p(y|x)

  No way to handle unreasonable inputs; must  give a label distribution for all possible inputs

  对于特定的输入有不同的分配，输入在竞争

  最常规

- Generative Model: Learn a probability  distribution p(x)

   All possible imagescompete for probability mass

  任何可能的都在竞争概率的分配

  无用

- Conditional Generative  Model: Learn p(x|y)

   Each possible label induces a  competition across all possible images

  最有用

  - Assign labels while **rejecting outliers** 

  - Sample to **generate data** from labels

通过Bayes’ Rule关联

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

后两个在一些paper里有混淆

### Why Generative Models?

- Modeling ambiguity: If there are many possible  outputs x for an input y, we want to model P(x | y) 
- Text to Image: Produce output image x from input text y

## Autoregressive Models

Assume x is a sequence

Use the chain rule of probability RNN/ masked Transformer

文字处理很自然，因为这就是离散且1d序列

图像怎么办？

对每个像素的三个通道进行连接 认为是一个序列

问题：训练太贵了，一张图片会很长

## Variational Autoencoders (VAEs)

VAE 引入了一个隐变量 (Latent Variable) $z$（比如代表图片的“风格”、“角度”等抽象特征）。
在 VAE 的设定里，生成数据的边际似然概率 $p_\theta(x)$ 是通过对 $z$ 积分得到的：
$$p_\theta(x) = \int p_\theta(x|z) p(z) \, dz$$

问题：概率不可直接计算得到 z有无穷可能

解决方法： optimize a lower bound on the density



## (Non-Variational) Autoencoders

 Idea: Unsupervised method for learning to extract features z from inputs x, without labels

feature z将被压缩，维度很低，所以本质上是如何在极度受限的条件下，依然能保持信息不丢失

![image-20251125203307408](/ch12-autoencoder.png)

如果要生成东西，使用decoder

$z$ 确实是特征的总结。但是，只有当你“手里有一张具体的图”时，编码器才能帮你总结出这个 $z$。$z$ 对于特定的 $x$ 就是一个特定的、死板的数值。

只有decoder的时候如何确定z呢？

## Variational Autoencoders (VAEs)

**核心目标**：VAE 是自编码器的概率版本，主要做两件事：1) 从原始数据中学习潜在特征 $z$；2) 从模型中采样以生成新数据 。

假设训练数据 $x$ 是由未观察到的潜在因子 $z$（如属性、方向等）生成的。

训练完成后，生成新数据的步骤是：先从先验分布 $p(z)$（通常假设为**简单的**高斯分布）中采样 $z$，再从条件分布 $p(x|z)$ 中采样 $x$。 

训练难题与解决思路

**最大似然估计的困难**：我们想通过最大似然法训练模型，但由于没观测到 $z$，必须计算边缘概率 $p(x) = \int p(x|z)p(z)dz$ 。

问题：虽然可以用解码器计算 $p(x|z)$，但对所有可能的 $z$ 进行积分是不可计算的 (intractable) 。

尝试用贝叶斯公式 $p(x) = p(x|z)p(z) / p(z|x)$，但分母中的后验概率 $p(z|x)$ 同样无法计算。

**解决方案**：训练另一个神经网络（编码器）$q_{\phi}(z|x)$ 来近似真实的后验分布 $p_{\theta}(z|x)$，并联合训练编码器和解码器 。

![image-20251125212754111](/ch12-VAE.png)

神经网络不直接输出数据，而是输出概率分布的参数（如均值 $\mu$ 和方差 $\Sigma$）。

###  ELBO

整理可得：

$$\log p_\theta(x) = E_{z \sim q}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z)) + D_{KL}(q_\phi(z|x) || p_\theta(z|x))$$

我们要最大化左边的 $\log p_\theta(x)$，现在看看右边这三项能不能算：

$E[\log p_\theta(x|z)]$：可以通过解码器计算，衡量生成的 $x$ 和原图有多像。 

$D_{KL}(q||p(z))$：编码器输出分布与标准正态分布的距离。对于高斯分布，这有闭式解，可以计算 。

$D_{KL}(q||p_\theta(z|x))$：**无法计算**。因为这里面包含了未知的真实后验 $p_\theta(z|x)$ 8。

既然第三项算不出来，我们就利用 KL 散度的性质：**KL 散度永远非负 ($D_{KL} \ge 0$)**。如果我们直接丢掉第三项（因为它 $\ge 0$），等式就会变成**不等式**：

$$\log p_\theta(x) \ge E_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))$$

**这就是 ELBO (Evidence Lower Bound)。**

最终的 VAE 训练目标不是直接最大化似然（做不到），而是最大化这个 ELBO：

$$\text{ELBO} = \underbrace{E_{z \sim q}[\log p_\theta(x|z)]}_{\text{重建损失 (Reconstruction)}} - \underbrace{D_{KL}(q_\phi(z|x) || p(z))}_{\text{正则项 (Regularization)}}$$

最大化ELBO的意义：

1、对于**重建损失**，越大越好，希望生成的图像越像原图越好。

2、对于**正则项**越小越好，希望隐变量分布越接近标准正态分布越好。

### VAE 训练

**Step 1 **：将输入 $x$ 传入编码器，得到 $z$ 的分布参数

**Step 2 **：计算 KL 散度，强迫编码器的输出接近标准正态分布 $N(0, I)$

**Step 3 (重参数化技巧)**：为了能进行反向传播，不能直接采样，而是通过 $z = \mu + \epsilon \odot \Sigma$ （其中 $\epsilon \sim N(0,I)$）的方式采样 $z$。

**Step 4 **：将采样的 $z$ 传入解码器，得到预测的数据均值。

**Step 5 **：计算预测均值与原始输入 $x$ 的 L2 距离。



重建损失希望方差趋近于 0 以精确重建，而先验损失希望方差趋近于 1 以符合高斯分布，这两者在训练中相互制衡。