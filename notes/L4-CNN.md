---
typora-root-url: ./fig
---

# **Lecture 5: Image Classification with CNNs**

Image Features

dot product应该被看做template match?

### convolution layer

多张图片组成一个batch一起处理很常见

![image-20251020223558552](/ch4-convolution-layer.png)

但单纯的卷积依旧是线性计算

so add activation functions，先卷积再relu

卷积会使尺寸缩减：In general Input: W Filter: K Output: W – K + 1

 Solution: Add  **padding** around  the input before  sliding the filter  （In general Input: W Filter: K Padding: P Output: W –K + 1 + 2P；Common setting: P = (K –1) / 2）

#### Receptive Fields

![image-20251020224934593](/ch4-receptive-fields.png)

**Stride**d Convolution 步长

 In general: Input: W Filter: K Padding: P Stride: S Output: (W –K + 2P) / S + 1

![image-20251020225259525](/ch4-convolution-summary.png)

### Pooling Layer

downsampling

提高模型的泛化能力和提取稳健特征

有最大pooling/平均 /……

#### Hyperparameters

- Kernel Size 

- Stride 

- Pooling function