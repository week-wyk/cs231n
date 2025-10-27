---
typora-root-url: ./fig
---

# L1-**Image Classification with Linear Classifiers**

![ch1_L1_L2_distance](/ch1_L1_L2_distance.png)

![ch1_set_hyperparameters_1](/ch1_set_hyperparameters_1.png)

![ch1_set_hyperparameters_2](/ch1_set_hyperparameters_2.png)

### K-Nearest Neighbors Summary
In image classification we start with a training set of images and labels, and mustpredict labels on the test set
The K-Nearest Neighbors classifier predicts labels based on the K nearest training examples
Distance metric and K are **hyperparameters**
Choose hyperparameters using the validation set

Only run on the test set once at the very end!

### linear classification

核心：区分边界为线性或者超平面

![ch1_hardcases_LC](/ch1_hardcases_LC.png)

#### Softmax Classifier (Multinomial Logistic Regression)

取指数，算加权，使得输出范围为0-1，总和为1。

![ch1_softmaxclassifier](/ch1_softmaxclassifier.png)

#### Kullback-Leibler散度（KL散度）

$$D_{KL}(P||Q) = ∑_y P(y) log [P(y)/Q(y)]$$

描述两个分布的差异，非负，完全一致时散度为0.

#### 交叉熵Cross Entropy

$$H(P,Q)=H(p)+D_{KL}(P||Q) = -\sum_{i=1}^n p_i \log q_i$$

使用 $Q$编码$P $的平均码长。一般认为P为真实的分布，Q为预测的分布。

**熵** $H(P)$

$$H(P)=-\sum_{i=1}^n p_i \log p_i$$

$ P $ 的固有不确定性

交叉熵是分类任务的核心损失函数