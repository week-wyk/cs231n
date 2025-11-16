---
typora-root-url: ./fig
---

# **Recurrent Neural Networks**

![image-20251114130549157](/ch6-initialization-ln.png)



## hidden state

![image-20251115224201670](/ch6-hiddenstate.png)

![image-20251115224302469](/ch6-RNNoutput.png)

1. **输入**  $x_t$ “ 更新”了 RNN 的**记忆 $h_t$**。
2. RNN 再“查看”自己**最新的记忆 $h_t$**，来决定要**输出 $y_t$**。

输入 $x_t$ 并不能“跳过” $h_t$ 去直接影响 $y_t$；它必须先被 $h_t$ “吸收”和“理解”，然后再由 $h_t$ 来决定输出。

![image-20251116000124600](/ch6-(Vanilla) Recurrent Neural Network.png)

### ComputationalGraph

![image-20251115232313679](/ch6-ComputationalGraphManytoMany.png)

![image-20251115232633926](/ch6-ComputationalGraphOnetoMany_0.png)

![image-20251115232714341](/ch6-ComputationalGraphOnetoMany_yn-1.png)



## 反向传播

随着时间，数值会很难算

于是：Truncated Backpropagation through time

Run forward and backward  through chunks of the  sequence instead of whole  sequence

![image-20251115234129751](/ch6-TBPTT.png)



## **词嵌入**

正常的模型训练需要：

输入字符——input layer(One-hot) —— hidden layer——output layer

但当字符范围很大 太稀疏 (Sparse) **无意义 (No meaning)：** 这种编码是“孤立”的。

所以提出词嵌入，即加一个嵌入层Embedding layer

![image-20251115235754063](/ch6-embeddinglayer.png)

Embedding Layer 本质上就是 RNN 的第一个权重矩阵 $W_{xh}$

当我们使用 Embedding Layer 时，虽然它在**理论上**代表了那个权重矩阵 $W_{xh}$，但在**实践中**，我们用一个**等效且高效**的“查表”操作，**代替（省去）**了那个完整的矩阵乘法步骤。

在同一个 RNN 中，不同的神经元（cells）[^1]在训练过程中，会自发地学会专门跟踪输入序列中的不同模式（比如“是否在注释里”、“是否在引号里”）。

[^1]: 一个“RNN 神经元”（或 单元/细胞）指的就是 $h_t$ 向量中的一个“数字” (a single element)/一个维度

![image-20251116004330215](/ch6-multilayerRNN.png)

## RNN tradeoffs 

- RNN Advantages: 
  - Can process any length of the input **(no context length)**
  - Computation for step t can (in theory) use information from many steps back  
  - Model size does not increase for longer input  
  - The same weights are applied on every timestep, so there is **symmetry**(对称) in  how inputs are processed.  

- RNN Disadvantages:  
  - Recurrent computation is **slow**  
  - In practice, difficult to access information **from many steps back** 



## 实际应用:Image Captioning

![image-20251116001326742](/ch6-CNN+RNN.png)

![image-20251116001646152](/ch6-CNNconnectedtoRNN.png)

在 RNN 计算**每一步**的新记忆 $h_t$ 时，它**不仅**要看 $x_t$ 和 $h_{t-1}$，还**必须**看一眼那个来自 CNN 的图像总结 $v$。



## LSTM

![image-20251116004800640](/ch6-ProblemofRNNBP.png)

 Largest singular value > 1:  Exploding gradients 

Largest singular value < 1: Vanishing gradients

![image-20251116005106960](/ch6-LSTM.png)

 Uninterrupted gradient flow

#### Do LSTMs solve the vanishing gradient problem?

 LSTM doesn’t guarantee that there is no vanishing/exploding gradient, but it  does provide an easier way for the model to learn long-distance dependencies



![image-20251116005310984](/ch6-LSTMsimilartoResNet.png)

higway Net 是ResNet的前驱。

它为每一层引入了两个**“门控”**：

1. **Transform Gate ( $T$ 门 / 转换门):** 一个“开关”，决定**“转换”**多少当前层的输入。
2. **Carry Gate ( $C$ 门 / 携带门):** 另一个“开关”，决定**“直接携带”**多少上一层的原始输入。

**如果 $T(x) \approx 1$（转换门“全开”）：**

- $y \approx H(x)$
- 这时网络就退化成了一个**普通网络**，信息必须被“转换”。

**如果 $T(x) \approx 0$（转换门“关闭”）：**

- $y \approx x$
- 这就是“**高速公路**”！
- **输入 $x$ 几乎原封不动地流向了下一层**，完全跳过了 $H(x)$ 的复杂处理。
