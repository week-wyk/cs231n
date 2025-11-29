---
typora-root-url: ./fig
---

# L15:  Multi-Modal Foundation Models

**Always see with foundation models:**

- general /robust to many different tasks

**Often see with foundation models:**

\- Large # params

\- Large amount of data

\- Self-supervised pre-training objective

## CLIP

连接文字与图片！

![image-20251128103953667](/ch15-Clip.png)

非常强

**Out of the box classification**(No fine-tuning) 很强

Clever trick: we can create a classifier using the text encoder! Create a vector representation for *each* category!

短语比单个单词训练效果更好 +~5%

换个奇怪的测试集AC不变

CLIP performance is great also on graphic images , sketches, adversarial datasets.

### Why does CLIP perform so well?

Scale!

参数足够多，训练数据集足够大（400M）

### CoCa

![image-20251128002735675](/ch15-CoCa.png)

证明了不需要分别训练一个 CLIP 和一个 Captioner，只要架构设计得当（把 Decoder 拆开用），一个模型就能把“看图选词”和“看图说话”全包了。

### PROS AND CONS

#### Advantages of CLIP-style models

1. Dot product is super efficient

a. Easy to train (enables scaling)

b. Fast inference, e.g., retrieval over 5B images

2. Open-vocabulary (zero-shot generalization)
3. Can be chained with other models (CuPL)

#### Disadvantages

1. Rely too heavily on batch size to learn concepts过小学不到精细定义，但放大也不一定就能消除一些错例（a mug in some grass” and “some grass in a mug），不一定在同一个batch

2. Image-level captions are insufficient supervision

3. You can’t know everything in your 5B dataset

   CLIP 并不是真的“看懂”了世界，它只是“背下了”互联网上 50 亿张照片的样子。一旦你让它看它没背过的东西（新事物、罕见物体、逻辑组合），它那看似强大的 Zero-Shot 能力就会瞬间崩塌。它没有举一反三的智力，它只有海量的记忆。

## LLaVA

**images and text as input, and then output text**

add visual information to the LLM

Which image tokens work best here?

The CLIP encoder is a good option!

![image-20251128104230223](/ch16-ClipTokenChoice.png)

在 CLIP 或 ViT 的训练中，最后一层通常经过了一个 **Projection Head（投影层）**。它的任务是把几千维的丰富特征，强行压缩或映射到一个特定的空间，以便和文本向量算内积（Contrastive Loss）。在这个过程中，**大量的几何细节、空间位置信息、颜色纹理**会被丢弃，因为为了做“连连看”匹配，模型只需要关注“语义”（这是一只猫），而不需要关注“猫在左上角”或“猫毛的纹理”。

$L-1$ 层的特征虽然已经具备了高级语义（知道是猫），但还没有被最后一层的 Loss 强行压缩，保留了更多的 **Spatial Information（空间信息）**。

![image-20251128104532223](/ch15-LLaVA.png)

### Flamingo: 在LLaVA的基础上修改网络

![image-20251128104853595](/ch15-Flamingo.png)

Perceiver sampler converts variable sized image tokens to **fixed sized** ones

#### Flamingo gated cross-attention

![image-20251128105039999](/ch15-Flamingo gated cross-attention.png)

cross-attention: 让文本去“查询”图片。比如文本读到“狗”字时，去图片里找哪里有狗。

masked: 分割long text，变成图片与特定描述

![image-20251128105450655](/ch15-Flamingo masked attention.png)

Flaming enables in-context learning: 告知行为模式例子 可以学习到模式。



比如千问，There are open-weight models but they are all **distilled from GPT** 我们并不知道如何训练类似gpt的模型，只有open ai的员工知道。

#### How do we close the gap without relying on proprietary models?

讲者的实验室自研一个模型：Molmo。完全开源且效果仅仅略逊于gpt-4o

Never bet against open-source software!

Molmo是如何做到的？

Internet data is incidental. Human annotated data is intentional

PixMo data is intentional. 非常具体，对于一些很显然的关系也会有详细描述，比如A在B左边。这些信息是不会出现在internet上的。高质量的训练数据很重要。

Questions designed to extract meaningful visual information from annotators

• What is the image at first glance?

• What are the objects and their counts?

• What does the text say?

• What are the positions of the objects?

• What subtle details are noticeable?

• What is in the background?

• What is the style and color?

对于一些比如指向任务做得很好。

还可以做模型串联

## SAM: Segment Anything Model

![image-20251128112357043](/ch15-SAM.png)输入图片和promt 用于分割图像里的物体，输出分割好的图片。

输入的Mask 是之前预测的mask（Previous Prediction）用于修正。

## chaining

问题：What happens when a model is asked to classify a concept it has never seen?

### CuPL (CUstomized Prompts via Language models)

![image-20251128112723806](/ch15-CuPL.png)

### VisProg (visual programming)

与其训练新模型，Write a python script with the models **you have**
