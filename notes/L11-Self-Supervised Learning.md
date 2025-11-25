---
typora-root-url: ./fig
---

# L11: Self-Supervised Learning

 What is the problem withlarge-scale training?

- We need a lot of **labeled data**

 Is there a way we can train neural networks without  the need for huge manually labeled datasets?

so Self-Supervised Learning

主要是：Transfer Learning and Downstream Task Performance
Assess the utility of the learned representations by transferring them to a downstream supervised task.

![image-20251125084137310](/ch11-self-supervisedlearningmethod.png)

## **Pretext Task**

###  rotation prediction

The model learns to  predict which rotation  is applied (4-way  **classification**)

###  “jigsaw puzzle”

将图片切片，用特定的方式[^1]打乱，预测图片顺序，预测这64个可能的序列是哪个。

为什么64可以涵盖所有：这64个序列是经过算法**特意挑选**出来的，遵循以下两个原则：**最大化汉明距离（Hamming Distance）：** 这意味着这64种排列方式彼此之间**差异巨大**。才是*有效的学习*

[^1]:**作为分类标签（Classification Labels）：** 模型并不是在做一个回归任务去预测坐标，而是在做**分类任务**。 我们将这64种特定的打乱方式编号为 Class 0 到 Class 63。输入：一张被打乱成“第5号排列方式”的图。输出：模型预测“这是第5类”。

### image completion: predict missing pixels (inpainting) 

masked 预测遮住的地方的图案

**adversarial learning**：预测得到的和真实值做对比，对预测值做改进

![image-20251125085617710](/ch11-adversariallearning.png)

### colorization

颜色给一个通道，预测另外两个

**Split-brain Autoencoder Idea: cross-channel predictions**

![image-20251125090047044](/ch11-crosschannelprediction.png)

有个这个还可以给视频未来帧预测颜色，相同的东西应该有相同的颜色，代表可以追踪视频的元素

![image-20251125090615119](/ch11-colorvideos_prediction.png)

### Masked Auto Encoders (MAE)

random masked 论文有对比，这个效果好

不对称（asymmetrical autoencoderdesign）：因为输入数据量的不对称，导致了模型设计的巨大差异：

**Encoder：非常重（Heavy）。** 因为它是用来提取特征的，我们希望它很深、很宽（比如 ViT-Large/Huge）。虽然它很重，但因为它只需要处理25%的数据，所以计算量依然可控。mask的部分不输入

**Decoder：非常轻（Lightweight）。** 通常很浅（比如只有8层，且宽度较窄）。它的任务仅仅是根据Encoder给的高级语义特征，把像素还原出来（Pixel reconstruction）。

Merges the encoder outputs with the shared mask tokens in previously masked places.

The  MSE (mean squared error loss) inthe pixel space between the  input image and the reconstructed image is adopted.  Loss is only computed for masked patches.

![image-20251125092707289](/ch11-linearProbing-FullFinetuning.png)

### Problems: 

1 coming up with individual pretext tasks is tedious, and 2) the  learned representations may not be general.

##  Contrastive representation learning

分类：让同类靠近，不同类分开

 **InfoNCEloss**：

$$L = -\log \frac{\exp(s(x, x^+))}{\exp(s(x, x^+)) + \sum \text{所有负样本的相似度}}$$

**Mutual Information（互信息）**

$$MI[f(x), f(x^+)] - \log(N) \ge -L$$

最小化这个 Loss，等价于**最大化**输入 $x$ 和其特征 $f(x)$ 之间的**互信息下界（Lower Bound）**。Loss 的作用就是通过增加干扰项（$N$），逼迫模型去榨干数据里的每一滴互信息。N越大，bound越紧。

### SimCLR: A Simple Framework for Contrastive Learning

![image-20251125095528064](/ch11-SimCLR.png)

**$x$ 是原图。****$t$ 和 $t'$ 是两个不同的增强操作（Augmentation）。** 比如：左边是“随机裁剪”，右边是“高斯模糊+变色”。**$\tilde{x}_i$ 和 $\tilde{x}_j$：** 这两张图虽然长得不一样，但**源头都是同一个 $x$**。数据增强： 即使它们长得不一样，模型也要认出它们是“本是同根生”，要把它们拉近（Maximize agreement）。

**$f(\cdot)$ 是编码器（ResNet）：** 负责提取特征，输出 $h$（Representation）。这是我们**真正想要的东西**，训练完后拿去下游任务用的。

**$g(\cdot)$ 是projection head（MLP层）：** 负责把特征映射到另一个空间，输出 $z$。**Loss 是在这个 $z$ 上算的，而不是在 $h$ 上算的。**

>  A possible explanation: 
>
> - contrastive learning objective may discard  useful information for downstream tasks 
> - representation space z is trained to be  invariant to data transformation.  
> - by leveraging the projection head g(ᐧ), **more  information can be preserved** in the h  representation space

左边的公式 $s(u, v)$ 用来衡量两个向量 $z_i$ 和 $z_j$ 指向的方向是否一致。在特征空间里，**方向一致 = 语义相同**。

![image-20251125095625741](/ch11-SimCLRalgorithm.png)

伪代码

直接认为除了那张同源生成的图片都是负样本，所以SimCLR **肯定用 Minibatch**，但它为了效果好，要求这个 Minibatch **超级大**（通常要几千）。论文配图看2048以上为佳。

### Momentum Contrastive Learning (MoCo)

优点：使用队列，降低batch_size

MoCo 让你用 **32 的 Batch Size**，享受 **65536 个负样本**的红利。它彻底解放了显存。

![image-20251125100959933](/ch11-MoCo.png)

### Contrastive Predictive Coding (CPC)



### DINO: Self-Distillation with No Labels

DINO v2用了更大数据训练
