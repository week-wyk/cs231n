---
typora-root-url: ./fig
---

# L7: Attention and Transformers

## 从RNN出发

![image-20251116194452223](/ch7-RNNSwithAttention.png)

我们不需要在训练数据里手动指定“正确的注意力权重”（No supervision）。我们只需要保证整个注意力计算过程在数学上是“平滑可导”的（All differentiable）。这样，当模型在最终翻译上犯错时，“反向传播”算法（Backprop）就能自动地、“逆流而上”地去修正那些参数，从而“教会”注意力机制在正确的时间去关注正确的词。

## 提炼出Attention

Attention 是一种通用的“操作”（general operator）。它允许一个“查询向量”（Query）去“关注”（Attend to）一系列“数据向量”（Data），并从中提取出一个“输出向量”（Output）。

![image-20251116225420671](/ch7-AttentionLayer.png)

换个形式，整理后见下

![image-20251116225756698](/ch7-AttentionLayer-main.png)

## self-Attention layer

: Q 、K 和 V来自**同一个**序列

Cross-Attention Layer主要特点：Q 来自**一个**序列，而 K 和 V 来自**另一个**序列。



![image-20251116230542358](/ch7-SelfAttentionLayer.png)

Self-Attention is **permutation equivariant**（输入交换位置，则输出也对应交换位置）

因为 Self-Attention 在计算**任何一个**词的输出时（比如 `y_i`），它会**“平等地”**看待所有**其他**词。它本质上是将输入视为一个**“集合”（Set）**，而不是一个“序列”（Sequence）。

这就是为什么 Transformer 必须引入“位置编码”（Positional Encodings）



### Masked Self-Attention Layer

Don’t let vectors “look ahead” in the sequence

不应该看到的置为-∞，softmax后变为0

 Used for language modeling  where you want to predict the  next word



### Multiheaded Self-Attention Layer

H independent  self-attention layers  (called **heads**), each  with their own weights

![image-20251116231646318](/ch7-Multiheaded Self-Attention Layer.png)

In practice, compute  all H heads in parallel  using **batched matrix  multiply operations**.

### **Self-Attention is Four Matrix Multiplies!**

 Q: How much **compute** does this take  as the number of vectors N increases? 

A: O(N^2)

 Q: How much **memory** does this take  as the number of vectors N increases? 

A: O(N^2)

![image-20251116232135164](/ch7-AL-Time-Memory.png)

**Flash Attention 的天才之处在于“融合”与“重计算”：**

1. **加载小块：** Flash Attention（作为一个融合的GPU“核函数”）把一小块 `Q_i` 和一小块 `K_j`, `V_j` 加载到**极快的 SRAM** 中。
2. **SRAM 内部计算：** 它在**SRAM内部**快速计算出这一小块的注意力 `A_ij`。
3. **立即使用并丢弃：** 它**立即**用这个小 `A_ij` 去乘以小 `V_j`，得到一小块**最终结果**，然后**立马把 `A_ij` 丢掉**。
4. **循环：** 它不断重复这个过程（加载下一小块 `K_{j+1}`, `V_{j+1}`），并在SRAM中维护一个“正在累加”的最终输出 `Y`。

空间复杂度 $O(N)$，时间复杂度 $O(N^2 *d)$

## Compared

![image-20251116232756919](/ch7-threewayscompared.png)

结果发现Attention的缺点是计算复杂度大 但对于模型训练而言，这可以通过使用很多显卡解决，相比其他方法的缺点是最容易解决的，于是得到：

**Attention is All You Need**

## Transformer

![image-20251116233721541](/ch7-transformerblock.png)

 A Transformer is just **a stack of**  identical Transformer blocks!



### 应用

#### LLM

![image-20251116233917773](/ch7-TransformerforLLM.png)

#### ViT

![image-20251116234140067](/ch7-TransformerforViT.png)

 16x16 conv with stride  16, 3 input channels, D  output channels is same as Flatten and apply a linear  transform 768 => D

Transformer（如 BERT 或 GPT）天生是为**序列**（Sequences）设计的，比如一句话（单词的序列）。但一张图像是**像素网格**，不是序列。

ViT 的天才之处在于它如何**把一张图像“翻译”成一个序列**

### 变体

#### Pre-Norm Transformer

![image-20251116234704754](/ch7-PreNormTransformer.png)



#### 2 

Replace Layer Normalization  with Root-Mean-Square  Normalization (RMSNorm)

#### 3 SwiGLU MLP

![image-20251116234853694](/ch7-SwiGLU_MLP_Transformer.png)

#### 4 Mixture of Experts (MoE)

与其训练大而全的模型，不如分散为多个小而精的专家模型

 Learn E separate sets of MLP weights in  each block; each MLP is an expert W1: [D x 4D] => [E x D x 4D] W2: [4D x D] => [E x 4D x D] Each token gets routed to A < E of the  experts. These are the active experts.

**稀疏性 (Sparsity)** 就体现在 `A < E`。Gating Network会为每个 Token **只**挑选 `A` 个它认为“最合适”的专家去处理。

 Increases params by E, But only increases compute by A

因为一个 Token **只**被 `A` 个专家处理（其他 `E-A` 个专家在“休息”），所以你的**实际计算成本**（FLOPs，即速度）只增加了 `A` 倍。