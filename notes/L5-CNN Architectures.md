---
typora-root-url: ./fig
---

# **Lecture 5: CNN Architectures**

## Normalization Layer: LayerNorm

![image-20251111105731056](/ch5-LayerNorm.png)

**$\mu, \sigma$: $N \times 1$ (标准化是沿着 $D$ 轴进行)**

- $\mu$（均值）和 $\sigma$（标准差）的计算是沿着**特征维度 $D$** 进行的。
- **$\rightarrow$ Statistics calculated per batch (统计数据是按批次计算的):**
  - **对于批次中的** ***每一个\*** **样本 (N)，我们独立计算其 $D$ 个特征的均值和标准差。**
  - 因此，每个样本都有自己的一套 $\mu$ 和 $\sigma$，总共得到 $N$ 组，维度为 $N \times 1$。
  - **核心区别：** 这意味着 LayerNorm 的统计信息与批次大小 $N$ **无关**。即使批次很小，标准化结果也很稳定。

**$\gamma, \beta$: $1 \times D$ (可学习参数)**

- **$\rightarrow$ Learned parameters applied to each sample (可学习参数应用于每个样本):**
  - $\gamma$（缩放）和 $\beta$（平移）是可学习的，它们的数量与**特征维度 $D$** 相同。
  - 这 $D$ 个参数会应用于批次中**所有 $N$ 个样本**的对应特征上。

## other4Norm

![image-20251111110106345](/ch5-other4Norm.png)

### 1批标准化 (Batch Norm)



- **蓝色区域：** 涵盖了**所有 $N$ 个样本**，以及所有空间维度 ($H, W$)。
- **工作方式：**
  - 它沿着批次轴 $N$ 进行标准化。
  - **针对每一个通道 $C$**，独立计算所有样本在这个通道上的 $\mu$ 和 $\sigma$。
- **关键点：** 它强烈依赖于**批次大小 $N$**。如果 $N$ 很小，计算出的 $\mu$ 和 $\sigma$ 误差较大，效果会变差。
- **适用场景：** 批次足够大的视觉任务（如图像识别）。



### 2. 层标准化 (Layer Norm)



- **蓝色区域：** 涵盖了**所有通道 $C$** 和所有空间维度 ($H, W$)，但**仅限于一个样本**。
- **工作方式：**
  - 它沿着通道轴 $C$ 和空间轴 $H, W$ 进行标准化。
  - **针对批次中的** ***每一个\*** **样本 $N$**，计算它所有特征的 $\mu$ 和 $\sigma$。
- **关键点：** **与批次大小 $N$ 无关**。即使批次很小，标准化结果也很稳定。
- **适用场景：** 序列数据模型（如 RNN、Transformer），因为序列的长度多变，且 Batch Norm 不适用。



### 3. 实例标准化 (Instance Norm)



- **蓝色区域：** 仅涵盖了**一个样本 $N$** 和**一个通道 $C$**，但涵盖了所有空间维度 ($H, W$)。
- **工作方式：**
  - 它沿着空间轴 $H, W$ 进行标准化。
  - **针对批次中的** ***每一个\*** **样本 $N$** 和 ***每一个\*** **通道 $C$**，计算其对应的 $\mu$ 和 $\sigma$。
- **关键点：** 它消除了单个图像中通道特征的**风格差异**（例如颜色、对比度），同时保留了特征的结构信息。
- **适用场景：** 图像风格迁移 (Style Transfer)，因为风格信息很大程度上包含在激活的均值和方差中。



### 4. 组标准化 (Group Norm)



- **蓝色区域：** 涵盖了**一个样本 $N$** 和所有空间维度 ($H, W$)，但在**通道维度 $C$ 上进行分组**。
- **工作方式：**
  - 它是 Batch Norm 和 Layer Norm 的折衷。它将通道数 $C$ **分成 $G$ 个组**（例如 $G=32$）。
  - **针对批次中的** ***每一个\*** **样本 $N$**，分别计算**每组通道**内的 $\mu$ 和 $\sigma$。
- **关键点：** 它不依赖于批次 $N$，同时比 Layer Norm 更能保留通道间的差异信息（因为它没有将所有 $C$ 合并）。
- **适用场景：** 批次大小 $N$ 较小，或模型较大无法使用大批次的视觉任务。



## Regularization: Dropout 

In each forward pass, randomly set some neurons to zero Probability of dropping is a hyperparameter; 0.5 is common

Forces the network to have a redundant representation; Prevents co-adaptation[^1] of features

[^1]:这是指在训练过程中，神经元之间发展出**强烈的相互依赖关系**。例如，神经元 A 专门处理输入数据的左半部分，神经元 B 专门处理右半部分，它们必须同时存在才能正确识别一个物体。这会导致网络在训练集上表现非常好，但在遇到新数据时，这种脆弱的依赖关系很容易失效，造成过拟合。

Another interpretation: Dropout is training a large ensemble of  models (that share parameters). [^2]

[^2]:每一个子网络都可以被视为一个独立的模型。Dropout 实际上是在同时训练一个由这些子模型组成的庞大“集成”。这些子网络都共享了原始网络的大部分权重参数。这种共享使得训练单个大模型（通过 Dropout）比独立训练数千个小模型要高效得多。

### scale at test time!

**让测试阶段网络的平均输出分布与训练阶段一致。**

方法1：Scale at **train** time

在训练时，为了保持输出期望一致，可以把保留的神经元除以保留概率p。这样期望不变所以测试时不需要再缩放。

方法2：Scale at **test** time

训练时不做缩放，直接随机丢节点；到了测试时再整体乘上保留率 p



## Activation Functions

![image-20251111144714579](/ch5-activationFunction.png)



## case study

AlexNet

VGGNet

![image-20251111145542618](/C:/Users/wu_yike/AppData/Roaming/Typora/typora-user-images/image-20251111145542618.png)

deeper layers

但是增加层数不是万全的办法

 Hypothesis: the problem is an optimization problem,  deeper models are harder to optimize

不是说层数更多不好 而是说明层数更多的解更难找反而会不如一开始。要想让这个复杂的非线性函数 $H(x)$ 完美地拟合 $x$（即 $H(x) = x$），对于梯度优化器来说是**非常难**的一件事。优化器倾向于找到其他更容易拟合的解，这些解往往比恒等映射差，导致性能下降（退化）。

所以提出ResNet

一个函数趋近一个数字比一个函数要容易实现

![image-20251111150907116](/ch5-ResNet.png)

![image-20251111155243685](/ch5-ResNet-all.png)

## Weight Initialization

当初始值过小，这种“变小”的效应在多层网络中会**累积**。当激活值都趋近于 0 时，在**反向传播**过程中，计算的梯度也会非常小，甚至趋近于 0。这被称为**梯度消失问题 (Vanishing Gradient)**

当初始权重过大，同理会变得越来越大。

![image-20251111155934123](/ch5-kaimingInitalization.png)

## Data Preprocessing

center and scale  for each channel

## Data augmentation

将一张图片进行翻转平移 裁切 调色 遮挡 

正则化之一



**如果数据不够怎么办？**

![image-20251111161214380](/ch5-transferlearning.png)

## **模型训练基本方法**

### Step 1: Check initial loss (检查初始损失)



- **操作：** 在开始训练之前，用一个小的、未经优化的数据集运行模型的前向传播，计算损失函数（Loss）。
- **目的：** 确保模型和损失函数的设置是**正确**的。
  - **正确性检查：** 初始损失值应该与随机初始化下预期的损失值接近（例如，对于一个 10 类别的 Softmax 分类器，初始损失 $-\ln(1/10) \approx 2.3$）。
  - **如果初始损失异常（例如为 0 或 NaN），** 说明代码中存在 Bug（如数据归一化错误、损失函数实现错误）。



### Step 2: Overfit a small sample (让小样本过拟合)



- **操作：** 仅使用**非常小的训练集子集**（例如 20-50 个样本），关闭所有正则化（如 Dropout），并使用一个**高学习率**。
- **目的：** 证明模型**有能力**学习并拟合数据。
  - **如果模型能够快速达到 100% 的训练准确率（即损失降到 0）：** 证明模型的容量足够大，并且代码是正确无误的。
  - **如果连小样本都无法过拟合：** 说明网络结构或优化器设置可能存在严重问题。



### Step 3: Find LR that makes loss go down (找到能让损失下降的学习率)



- **操作：** 在全量数据集上进行训练，但仍然关闭正则化。**从小到大**尝试不同的学习率（Learning Rate, LR）。
- **目的：** 找到一个**合理的学习率范围**。
  - **太小：** 损失下降缓慢。
  - **太大：** 损失会立即变为 `NaN`（因为数值溢出）或大幅震荡。
  - **合理范围：** 找到一个能让损失值在最初几次迭代中**稳定下降**，且下降速度合理的学习率。这个值通常在 $10^{-6}$ 到 $10^{-1}$ 之间。



### Step 4: Coarse grid of hyperparams, train for $\sim$1-5 epochs (粗略网格搜索超参数，训练 1-5 个周期)



- **操作：** 使用在 Step 3 中找到的合理学习率，开始探索其他超参数（如正则化强度、学习率衰减等）。
  - **粗略网格搜索：** 快速测试大量超参数组合，每种组合只训练很短时间（1-5 个 epoch）。
- **目的：** **快速排除**那些明显表现不佳的超参数组合，圈定表现最好的几个区域。



### Step 5: Refine grid, train longer (精细网格搜索，训练更久)



- **操作：** 针对 Step 4 中表现最好的超参数区域，进行更**密集、更细致**的搜索，并让模型训练**更长**的时间（直到收敛）。
- **目的：** 找到最终的最优超参数组合。



### Step 6: Look at loss and accuracy curves (查看损失和准确率曲线)



- **操作：** 训练结束后，仔细检查训练集/验证集上的**损失曲线**和**准确率曲线**。
- **目的：** **诊断**模型的问题并做出最终评估。
  - **如果训练损失远低于验证损失：** 模型**过拟合**，需要增加正则化（Dropout, Data Augmentation）。
  - **如果两条损失曲线都很高：** 模型**欠拟合**，可能需要增加模型容量（更多层/神经元）或训练更久。

### step 7: GOTO step 5

tips:

![image-20251111162125891](/ch5-randomsearchisbetter.png)
