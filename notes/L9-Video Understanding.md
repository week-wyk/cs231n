---
typora-root-url: ./fig
---

# L9: Video Understanding

Video Classification

Problem: Videos are big 

Raw video: Long, high FPS

 Training: Train model to classify short clips with low FPS

 Testing: Run model on different clips, average predictions

## Single-Frame CNN

train normal 2D CNN to classify video frames **independently**!

 Often **a very strong baseline** for video classification

##  Late Fusion (with FC layers)

![image-20251120201945723](/ch9-LateFusion-FC.png)

FC缺点：feature map会增大

![image-20251120202020124](/ch9-LateFusion-Pooling.png)

pooling缺点: 关键信息可能丢失

Late Fusion缺点: 经过很多的处理，低级 运动细节在提取的时候丢失了

so

##  Early Fusion (2D CNN

![image-20251120232207026](/ch9-EarlyFusion2D.png)

## 3D CNN

![image-20251120223805709](/ch9-3Dconv.png)

![image-20251120232318434](/ch9-3DCNN.png)

![image-20251120231948765](/ch9-EarlyFusionvsLFvs3DCNN.png)

## C3D: The VGG of 3D CNNs

 3x3x3 conv is very  expensive!

We can easily recognize actions using only motion information

## Measuring Motion: Optical Flow

 Tells where each pixel will  move in the **next frame**: F(x, y) = (dx, dy) It+1(x+dx, y+dy) = It(x, y)

![](/ch9-Two-StreamNetworks.png)

动作更有用

 So far all our temporal CNNs only model local  motion between frames in very short clips of ~2-5  seconds. What about long-term structure?

## Modeling long-term temporal structure: 

### RNN

Each depends on  two inputs: 

1. Same layer,  previous timestep 

2. Prev layer,  same timestep

![image-20251120234841063](/ch9-RNN-video.png)

 Problem: RNNs are slow for long  sequences (can’t be parallelized)

### Self-Attention: Spatio-Temporal Self-Attention (Nonlocal Block)

![image-20251120235555517](/ch9-NonlocalBlock.png)

不是很理解，是指将transformer拓展到3D吗？

### Inflating 2D Networks to 3D (I3D)

 Can use weights of 2D conv to  initialize 3D conv: copy Kt times in  space and divide by Kt This gives the same result as 2D conv  given “constant” video input

**copy the weight and do the inflation**



视频仍大有可为



还可以结合音频



长视频理解：选取最重要的部分进行分析