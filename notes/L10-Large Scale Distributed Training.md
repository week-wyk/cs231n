---
typora-root-url: ./fig
---

# **L10: Large Scale Distributed Training**

## GPU = Graphics Processing Unit

![image-20251121134245270](/ch10-GPUinside.png)

Mixed precision: 16-bit（用于矩阵乘法） / 32-bit（用于矩阵加法）

![image-20251121134953619](/ch10-gpucluster.png)

Google: Tensor Processing Units (TPUs)

## How to train on lots of GPUs

###  Data Parallelism (DP)

![image-20251121135916561](/ch10-dp.png)

在集群层面需要额外写代码完成沟通和传递的并行。

反向传播是从模型的**最后一层**（Output Layer）向**第一层**（Input Layer）计算梯度的。

当 GPU 刚刚算完**最后一层**的梯度时，**倒数第二层**的梯度还在计算中。此时，**最后一层**的梯度数据已经静止不变了（Ready）。我们不需要等前面所有层的梯度都算出来，就可以立刻把**最后一层**的梯度拿去在所有 GPU 间做平均（All-Reduce）。

但需要将所有参数都更新到一个gpu，对gpu显存大小有要求

 Solution: Split model weights across GPUs

### Fully Sharded Data Parallelism (FSPD)

![image-20251121151757871](/ch10-FSPD.png)

用通信换显存，再用流水线把通信时间抢回来

### Hybrid Sharded Data Parallel (HSDP)

把最费流量的“切分参数”操作，限制在**单机内部（一个 Node）**。 而在**多机之间**，只做最简单的梯度同步（DDP 模式），避免慢速网络拖后腿。

### Activation Checkpointing

Recompute activations  during the backward pass

用时间来换存储（空间）

problem : $N^2$ compute is bad!

 Don’t recompute everything;  save a checkpoint every C layers

 C checkpoints: $O(N^2/C)$ compute, $O(C)$ memory

 Problem: Lots of knobs to tune! How should we set them? 

Solution: **Maximize Model Flops Utilization (MFU)**

### FLOPS

全称是 **Fl**oating-point **Op**erations **P**er **S**econd（每秒浮点运算次数）。

**FLOPS (全大写 S):** 指 **“速度” (Rate/Speed)**。

- 含义：每秒能算多少次。
- 对象：**硬件 (GPU)**。
- 类比：这就好比汽车的 **“时速 (km/h)”**。
- *例句：这块 A100 GPU 的算力是 312 TFLOPS。*

**FLOPs (小写 s):** 指 **“总量” (Count/Quantity)**。

- 含义：这个模型一共需要算多少次加减乘除才能跑完。
- 对象：**模型 (Model)**。
- 类比：这就好比目的地的 **“距离 (km)”**。
- *例句：训练一遍 ResNet-50 需要 4 GFLOPs 的计算量。*

$$\text{训练时间} = \frac{\text{模型总计算量 (FLOPs)}}{\text{硬件实际算力 (FLOPS)}}$$

*(注意：实际算力通常只能达到理论峰值的 30%-50%，也就是常说的 MFU - Model FLOPS Utilization)*

在深度学习中，我们通常接触的单位是 **TFLOPS** (万亿次)（ **1 TFLOPS** = $10^{12}$ 次/秒）或 **PFLOPS** (千万亿次)。（**1 PFLOPS** = $10^{15}$ 次/秒）

**RTX 4090:** 大约 **82 TFLOPS**。

**NVIDIA H100:** 大约 **989 TFLOPS** (接近 1 PFLOPS)。

### Hardware FLOPs Utilization (HFU)

硬件利用率

problem: HFU does not account for  activation checkpointing or “helper”  computation like data augmentation,  optimizer, preprocessing

我们关注有效的计算，所以提出了MFU

### Model FLOPs Utilization (MFU)

![image-20251121154350610](/ch10-MFU.png)

 MFU >30% is good, >40% is excellent

## Context Parallelism (CP)

**Ring Attention** 

1. **Q 不动：** 每个 GPU 守着自己那一小段 Sequence 的 Query ($Q$)。
2. **K, V 转圈：** 把 Key 和 Value ($K, V$) 当作传送带上的货物。
3. **计算：** 货物传到我这里，我就算一下我的 $Q$ 和这个 $K$ 的 Attention，算完把货物传给下一个 GPU。

- **牛逼处：** 理论上可以支持无限长的序列（Near-Infinite Context）。
- **麻烦处：** 实现很难（Hardest to parallelize），因为要一边算一边传（overlap），代码写起来很痛苦。

**Ulysses (下图) —— “切头法”**

这是 DeepSpeed 提出的 **Ulysses（尤利西斯）** 算法，思路完全不同。它利用了 Transformer 的 **多头注意力 (Multi-Head Attention)** 机制。

1. **初始状态（切序列）：** 比如有 100 个 Token，2 个头 (Head A, Head B)。
   - GPU 1 拿前 50 个 Token（包含 Head A & B）。
   - GPU 2 拿后 50 个 Token（包含 Head A & B）。
2. **大挪移 (All-to-All)：** 这是一个关键的通信步骤（图中虽然没画箭头，但隐含了）。大家交换数据！
   - **GPU 1 说：** “我把后 50 个 Token 的 Head A 要过来，把前 50 个 Token 的 Head B 扔出去。”
3. **中间状态（切头）：** 交换完后：
   - **GPU 1：** 拥有 **所有 100 个 Token**，但只负责 **Head A**。
   - **GPU 2：** 拥有 **所有 100 个 Token**，但只负责 **Head B**。
4. **计算：** 因为 GPU 1 现在拥有 Head A 的所有 Token，它就可以直接算标准的 Attention，**完全不需要和其他 GPU 说话了！**
5. **挪移回去：** 算完后，再把数据换回来。

- **牛逼处：** 实现简单，算 Attention 的时候不需要通信，可以使用极其高效的 FlashAttention 算子。
- **死穴：** **并行度受限于 Head 的数量。** 如果你有 8 个 GPU，但模型只有 4 个 Head，这招就废了（不够分）。而 Ring Attention 没有这个限制。

## Pipeline Parallelism (PP)

![image-20251121160100020](/ch10-pipelineParallelism.png)

bubble越小效率越高

可以使用多个batch的数据同时进行流水线，会缩小bubble

## Tensor Parallelism (TP)

![image-20251121162117224](/ch10-TP.png)

用在transformer层特别好用，可以减少通信次数

 