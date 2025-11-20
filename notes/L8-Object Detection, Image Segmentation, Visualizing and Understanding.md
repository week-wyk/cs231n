---
typora-root-url: ./fig
---

# L8: Object Detection, Image Segmentation, Visualizing and Understanding

 伴随着CV Task的发展

##  Classification

和图形无关，只需出一个概率分布就可以判断

## Semantic Segmentation

Sliding Window

滑动窗口然后对这个patch进行CNN，得到这个patch代表什么

问题：效率太低 Not  reusing shared features between  overlapping patches

于是替代方案：对整个图片进行CN 再分割

问题：CNN后图片大小会变小，深度增加，分割需要和原图大小一致

于是替代方案：升采样回到原来大小

### In-Network upsampling: “Unpooling”

![image-20251119235143669](/ch8-unpooling2.png)

![image-20251119235234339](/ch8-maxunpooling.png)

另一种理解的方法：

**降采样和升采样都可以看作卷积得到**

比如一个大小4x4的图和一个3x3的filter 做stride 2 pad1的卷积会让卷积后的大小变为2x2

同理可以将2x2的图用3x3的filter加回去 得到一个升采样的效果

![image-20251120000305963](/ch8-transposedConvolution.png)



组合起来这就是Fully Convolutional（FCN）

![image-20251120143518361](/ch8-unet.png)

copy and crop:

经过多次卷积和池化，图像越来越小。虽然提取到了高级特征（比如“这是个细胞”），但**丢失了空间位置信息**。

**右边（Decoder）**：通过上采样（Up-conv）把图像放大。但单纯放大后的图像往往是模糊的，边界不清晰。

**Copy 的作用**：直接把左边**同层级**的高分辨率特征图“复制”过来，拼接到右边的特征图上。这就好比作画时，把原来的**高清晰度线稿**直接叠在模糊的上色图上，让边界瞬间清晰起来。

在数学操作上，这不是简单的数值相加（Add），而是 **通道拼接（Concatenate）**。

- **操作：** 假设右边的特征图是 `[B, 64, H, W]`（模糊，有语义），左边的特征图是 `[B, 64, H, W]`（清晰，有细节）。
- **结果：** 拼起来变成 `[B, 128, H, W]`。



## Object  Detection

### Object Detection: Single Object (Classification + Localization)

![image-20251120144726173](/ch8-multitaskloss.png)

### Multiple Objects

输出数量不确定

方法1：对每个patch单独cnn

计算成本太大

所以

##### Selective Search

先找到可能有物体的区域备选，原理见下：

一开始，它并不找物体，而是利用一种基于图的分割算法（Graph-based Segmentation），把整张图片切成成千上万个**极小的、颜色相近的碎片**（这被称为 Superpixels，超像素）。

这是核心步骤。算法开始进行多轮迭代，像拼图一样把相邻的小块粘起来：它遍历所有相邻的小碎片，计算它们的**“相似度”**。找到最相似的两个邻居，把它们合并成一个大一点的块（Region）。新的块产生后，继续和它现在的邻居比对，再次合并……一直重复，直到整张图合并成一个巨大的块为止。

为了保证找到的区域是“Blobby”（像物体的团块）而不是杂乱的噪点，它设计了四种合并策略，只要满足其中一种就可能被合并：

1. **颜色相似（Color）：** 颜色直方图很像（比如都是黑色的毛）。
2. **纹理相似（Texture）：** 纹理梯度很像（比如都是条纹状）。
3. **大小优先（Size）：** 优先合并小块，防止大块吞噬一切，保证能检测到小物体。
4. **吻合度（Fill）：** 如果块 A 把块 B 包围了（像填空一样），那就把它们合体，避免中间有空洞。

在这个不断的合并过程中，每一次合并产生的**中间产物**（Region），都会被画一个外接矩形（Bounding Box）。 这就生成了那 2000 个候选框（Region Proposals）。

于是得到

#### R-CNN（Regions with CNN features）

![image-20251120150606454](/ch8-slow-R-CNN.png)

Warped image regions: 切出来的图有大有小，长宽不一，但后面的 CNN 网络（ConvNet）只吃固定大小（比如 224x224）。不管切出来的图长什么样，强制缩放、拉伸成统一大小的正方形图片。

**Bbox Regressor（边界框回归器）**：它会思考：Selective Search 一开始画的那个框可能不够准（比如切到了猫耳朵）。它输出 4 个修正值 $(dx, dy, dw, dh)$，告诉框应该往左移一点、变宽一点，从而把物体包得更紧密。

但也要进行2000次单独的CNN，还是太慢了

于是提出

#### FAST R-CNN

![image-20251120152703471](/ch8-fastRCNN.png)

##### Region Proposal Network

替代Selective Search的区域划分方法

**RPN 的做法（Anchors 锚框）：** **“先射箭再画靶”**。

RPN 在特征图的**每一个像素点**上，都预先放置了 **$k$ 个固定大小和比例的虚构框**（称为 Anchors）。通常是 9 个：3 种面积 $\times$ 3 种长宽比（1:1, 1:2, 2:1）。RPN 并不是在“画框”，而是在**判断这 9 个预设的框，哪一个和真实物体最像**。

对于特征图上的每一个点，RPN 输出两组数据：

- **分类分数 (2k scores)：** 这 $k$ 个锚框是物体的概率（Yes/No）。
- **回归坐标 (4k coordinates)：** 这 $k$ 个锚框应该怎么修整 $(dx, dy, dw, dh)$ 才能套住物体。

#### YOLO (You Only Look Once)

虽然 YOLO v1 当时在精度上不如 Faster R-CNN（特别是对小物体检测很差），但它快。

YOLO 将输入图像划分成一个 $S \times S$ 的网格（比如 $7 \times 7 = 49$ 个格子）。

**核心规则：** 如果一个物体（比如一条狗）的**中心点**落在了第 3 行第 3 列的格子里，那么**只有这个格子**负责检测这条狗。其他 48 个格子就算切到了狗的尾巴或头，也不许管，只许看戏（或者预测背景）。

每个格子不仅要盯着自己的地盘，还要一次性预测出所有信息。网络最后的输出是一个巨大的**张量 (Tensor)**。

以 YOLO v1 为例，每个格子要输出一个长度为 30 的向量，包含三部分信息：

1. **边界框 (Bounding Boxes)：**每个格子尝试预测 $B$ 个框（通常 $B=2$）。

   每个框包含 5 个数值：$x, y, w, h$ (位置) + $Confidence$ (置信度，即“我觉得这里有东西的概率” $\times$ “我觉得框得准不准 IoU”)。

   *这就占了 $2 \times 5 = 10$ 个数。*

2. **类别概率 (Class Probabilities)：**这个格子里的物体属于哪一类？（比如 20 个类别：猫、狗、车...）。*这就占了 $20$ 个数。*

网络一口气吐出了 $7 \times 7 \times 2 = 98$ 个预测框。大部分框的置信度很低（背景），直接扔掉。

**NMS (Non-Maximum Suppression，非极大值抑制)：** 这是一个筛选算法，通过计算 IoU[^1]，把重叠度高但置信度不够高的“跟风框”删掉，只保留最好的那个。

[^1]:IoU (Intersection over Union)：$$\text{IoU} = \frac{\text{Area of Overlap} \text{ (交集)}}{\text{Area of Union} \text{ (并集)}}$$ IoU 越大，说明两个框重叠得越厉害。NMS 的阈值（Threshold）通常设为 0.5 或 0.7。

#### Object Detection with Transformers: DETR

![image-20251120170936334](/ch8-DETR.png)

## Instance  Segmentation

### Mask R-CNN

在原来基础上加了一路mask 对特征进行处理，每个类别都生成一张mask图



## 可视化

### Which pixels matter: Saliency via Backprop

目标：找关键像素（此时模型已经训练好了，权重不再变动）。

固定权重，计算分类分数（比如“狗”的得分）相对于**输入图片像素**的梯度。$\frac{\partial Score}{\partial Image\_Pixels}$

如果计算出某个像素的梯度很大，就说明这个像素稍微变动一点点，最后的“狗”的分数就会剧烈波动。这证明：这个像素对分类结果非常重要（Saliency 高）。

通过计算“输出分数”对“输入像素”的梯度，我们可以直接高亮出图片中决定分类结果的关键部位，从而**验证模型学到的特征是否合理**。

### Class Activation Mapping (CAM)

![image-20251120193829560](/ch8-CAM.png)

问题：分辨率太低

### Gradient-Weighted Class Activation Mapping (Grad-CAM)

$$
\text{热力图} = \text{ReLU} \Big( \sum_k \underbrace{ (\text{权重 } w_k) }_{\text{CAM的做法}} \times A^k \Big)
$$

$$
\text{热力图} = \text{ReLU} \Big( \sum_k \underbrace{ (\text{梯度平均值 } \alpha_k) }_{\text{Grad-CAM的做法}} \times A^k \Big)
$$

A是指某张特征图

![image-20251120195312445](/ch8-Grad-CAM-example.png)