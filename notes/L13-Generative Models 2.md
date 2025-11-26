---
typora-root-url: ./fig
---

# **L13: Generative Models 2**

## Generative Adversarial Networks(GANs)

**Idea**: Introduce a latent variable z with simple prior p(z) (e.g. unit Gaussian)

Sample 𝑧 ∼ 𝑝(𝑧) and pass to a **Generator Network** $x = G(z)$

Then x is a sample from the **Generator distribution** $p_G$. Want $p_G = p_{data}$!

**假设固定生成器 $G$ (Imagine fixing $G$)：**

$$\min_G \max_D \Big( \underbrace{E_{x \sim p_{data}}[\log D(x)]}_{\text{Discriminator wants } D(x)=1 \text{ for real data}} + \underbrace{E_{z \sim p(z)}[\log(1 - D(G(z)))]}_{\text{Discriminator wants } D(x)=0 \text{ for fake data}} \Big)$$

![image-20251126143840954](/ch13-GANgoal.png)训练时：gradient decent：

$ D = D+\alpha_D dV/dD$， $ G = G - \alpha_G dV / dG$

困难：GAN 的最终目标是达到博弈论中的纳什均衡（两个人都没法再让对方更惨）。但基于梯度下降（Gradient Descent）的优化算法是为**寻找最小值**设计的，而不是为**寻找平衡点**设计的。这导致训练过程经常出现**震荡（Oscillation）**，Loss 忽高忽低，始终不收敛。

理论上我们知道有最优解 $p_G = p_{data}$

但实际上 Neural nets with **fixed capacity** may not be able to represent optimal D and G

![image-20251126145326266](/ch13-GAN-Problemandsolution.png)

### GAN Architectures: DC-GAN

DC-GAN was the first GAN architecture that worked on non-toy data

### GAN Architectures: StyleGAN

![image-20251126150009914](/ch13-StyleGAN.png)

### 特点：Latent Space Interpolation

Given latent vectors $z_0$ and $z_1$ , we can **interpolate** between them:

$z_t = t z_0 + (1-t) z_1, x_t = G(z_t)$

The resulting image $x_t$ smoothly interpolate between samples

效果类似于短视频界输入几张照片，ai生成照片间变化的视频。

### 优劣

Pros**:**

- Simple formulation

- Very good image quality

Cons:

- No loss curve to look at

- Unstable training
- Hard to scale to big models + data

## Diffusion Models

![image-20251126150853093](/ch13-DiffusionModelIntuition.png)

![image-20251126151755734](/ch13-DM1.png)

```python
for x in dataset:
	z = torch.randn_like(x) 
    t = random.uniform(0, 1) 
    xt = (1 - t) * x + t * z  
    v = model(xt,t)
	loss = (z - x - v).square().sum()
```

### Rectified Flow: Sampling

```python
sample = torch.randn(x_shape)
for t in torch.linspace(1, 0, num_steps):
	v = model(sample,t)
	sample = sample -  v / num_steps
```

本质上就是在学习两个分布之间的关系

### Conditional Rectified Flow

![image-20251126155645380](/ch13-Conditional.png)

### Classifier-Free Guidance (CFG)

Randomly drop y during training.

Now the same model is conditional and unconditional!

training:

```pyt
for (x,y) in dataset:
	z = torch.randn_like(x) 
    t = random.uniform(0, 1) 
    xt = (1 - t) * x + t * z 
    if random.random() < 0.5: y = y_null
    v = model(xt, y, t)
	loss = (z - x - v).square().sum()
```

$$v^{cfg} = v^{y} + w \cdot (v^{y} - v^{\emptyset})$$

**$w$ (Guidance Scale)**：这是你在 WebUI 里调的那个系数（通常是 7.0 左右）。

- 如果 $w=0$，就是普通的有条件生成，模型可能不太理会你的 prompt。
- **$w > 0$**：**我们在人为地“夸大”条件的影响**。就像把原本微弱的“画猫”指令，通过减去“画普通图”的倾向，强行放大，让模型死死地盯着条件 $y$ 走。

![image-20251126160243347](/ch13-CFG1.png)

黄色起点为data平均值，意为与条件无关

sampling:

```pyth
y = user_input()
sample = torch.randn(x_shape)
for t in torch.linspace(1, 0, num_steps):
	vy = model(sample, y, t)
	v0 = model(sample, y_null, t)
	v = (1+w) * vy - w * v0
	sample = sample -  v / num_steps
```

缺点：Doubles the cost of sampling



”Classifier-Free” 因为它**显式地去掉了那个独立的分类器模型**。

一个处在中间状态的图像（比如加了一半噪的图）可能来自红色的**三角形数据**，也可能来自红色的**正方形数据**。模型需要预测一个速度向量 $v$，把紫色点推向真实数据。既然模型不知道它到底来自三角形还是正方形，为了让 Loss 最小，它只能预测这两个可能性的**平均值**（即图中中间那条绿色的箭头）。这被称为“条件期望”。

Full noise (t=1) is easy: optimal v is mean of $p_{data}$

No noise (t=0) is easy: optimal v is mean of $p_{noise}$

Middle noise is hardest, most ambiguous

But we give equal weight to all noise levels!

**Solution**: Use a non-uniform noise schedule

#### noise schedule

![image-20251126163457991](/ch13-noiseschedule.png)

传统的做法是 `t = random.uniform(0, 1)`，这意味着 $t=0.1$（很简单）和 $t=0.5$（很难）被选中的概率是一样的。

Logit-Normal Sampling 的步骤是：

1. **先生成一个正态分布的数**：$y \sim \mathcal{N}(\mu, \sigma^2)$。这是一个两头低、中间高的钟形曲线。

2. 映射到 [0, 1]：通过 Sigmoid 函数把 $y$ 压到 0 到 1 之间：

   $$t = \text{sigmoid}(y) = \frac{1}{1 + e^{-y}}$$

如果一个变量 $t$ 的 **Logit 变换**（即 $\log(\frac{t}{1-t})$，也就是 Sigmoid 的反函数）服从正态分布（Normal Distribution），那么 $t$ 就服从 Logit-Normal 分布。



问题：一张 $64 \times 64$ 的图只有 **4096** 个像素。一张 $1024 \times 1024$ 的图有 **100 多万** 个像素。像素数量增加了 **256 倍**。如果你直接在像素上做 Diffusion 或 Flow Matching，模型每一步都要预测 100 万个像素的噪声。这会导致**训练时间无限拉长**，显存（VRAM）瞬间爆炸。

### Latent Diffusion Models (LDMs)

所以我们提出了LDMs

先通过VAE获得隐藏层 再对隐藏层用Diffusion Model

问题：Decoder outputs often blurry

VAEs（变分自编码器）在训练时，通常使用的是 **L2 Loss (Mean Squared Error, MSE)** 或 L1 Loss 来衡量“重构图片”和“原图”像不像。**L2 Loss 的特性**：它痛恨“巨大的错误”，但能容忍“普遍的微小偏差”。

所以Add a discriminator!

解决模糊问题，我们在 VAE 的解码器后面加了一个 **Discriminator（判别器，来自 GAN）**。判别器不仅仅比较像素值，而是看纹理的真实感。

Modern LDM pipelines use **VAE** + **GAN** + **diffusion**!

### Diffusion Transformer (DiT)

![image-20251126170836307](/ch13-DIT.png)

Text-to-Image、Text-to-Video都是用这个

### Diffusion Distillation

During sampling we need to run the diffusion model many times (~30 – 50 for rectified flow)

This is really **slow!**

**Solution**: **distillation** algorithms reduce the number of steps (sometimes all the way to 1), can also bake in CFG

### Generalized Diffusion

核心公式如下：

$$x_t = a(t)x + b(t)z$$

**$x$**：真实图片（信号）。**$z$**：随机噪声。**$a(t)$ 和 $b(t)$**：两个随时间变化的系数（调节信号和噪声的比例） 。

只要改变 $a(t)$ 和 $b(t)$ 的定义就能得到完全不同的模型：

1. **Rectified Flow (整流流)** ：

​	设置：$a(t) = 1-t$, $b(t) = t$。这是最直观的**线性插值**。从纯图 ($t=0$) 匀速变成纯噪 ($t=1$)，路径是一条直线。

2. **Variance Preserving (VP) (方差保持)** ：

​	设置：$a(t) = \sqrt{\sigma(t)}$, $b(t) = \sqrt{1-\sigma(t)}$（通常满足 $a^2+b^2=1$）。这是经典的 **DDPM/Stable Diffusion** 的做法。无论怎么加噪，整体的能量（方差）保持为 1 。

3. **Variance Exploding (VE) (方差爆炸)** ：

​	设置：$a(t) = 1$, $b(t) = \sigma(t)$。信号不衰减，但噪声越来越大，直到把信号完全淹没 。

确定了怎么加噪，接下来要确定模型去学什么（损失函数）。列出了三种常见的预测目标 $y_{gt}$：

1. **x-prediction ($y_{gt} = x$)** ：直接让模型猜**原图**长什么样。在噪声很大时很难猜准。

2. $\epsilon$-prediction ($y_{gt} = z$)：让模型猜**噪声**长什么样。这是 DDPM 的经典做法。

3. **v-prediction ($y_{gt} = b(t)z - a(t)x$)** ：让模型猜**速度向量**（Velocity）。这是 **Rectified Flow** 和 **Stable Diffusion 2.0/3.0** 的核心。它结合了 $x$ 和 $z$ 的信息，通常被认为是最稳定的预测目标。

同一个扩散模型在数学上可以有三种不同的解释方式（也就是所谓 "Formalism"）：

1. **Latent Variable Model** ：类似 VAE。有一个前向加噪过程 $q(x_t|x_{t-1})$ 和一个学习出来的后向去噪过程 $p_\theta(x_{t-1}|x_t)$ 。

   **优化目标**最大化变分下界 (Variational Lower Bound, ELBO)。

2. **Score Function**：定义为对数概率密度的梯度 $s(x) = \nabla_x \log p(x)$ 。它就像一个**向量场**，指向数据密度最高的地方（即“最像真图”的方向）。模型就是在学习这个向量场。类似梯度上升

3. **随机微分方程 (SDE)** ：看作连续物理过程。把离散的时间步 $t$ 变成连续的时间，用微分方程 $dx = f(x,t)dt + g(t)dw$ 来描述数据的演变。可以利用现成的 SDE 求解器来进行采样。
