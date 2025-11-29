---
typora-root-url: ./fig
---

# **L14: 3D Vision**

## **Explicit Surface **显式

* Sampling Is Easy

- **Inside/Outside Test Hard**

### Point cloud：点云

**only points**, no connectivity

Points with orientation are called **surfels**

问题：Difficult to draw in undersampled regions

### **Polygonal Meshes** ：**Polygon**网格

**Mesh Upsampling - Subdivision**

via interpolation

**Mesh Downsampling - Simplification**

缺点：Non-parametric

## **Implicit Surface** 隐式

- Sampling Can Be Hard

- Inside/Outside Tests Easy

### **Parametric Representation**几何

用数学式子表示线/面

#### **Bézier Curves**

![image-20251127204158733](/ch14-BezierCurve.png)

![image-20251127204224851](/ch14-BezierSurface.png)

Algebraic Surfaces 难出现复杂的图形

方法1：把形状理解为基本几何图形的boolean 运算

方法2：用**Distance Functions**构造

数学上是用一个平滑最小函数 (Smooth Minimum, smin) 来替代生硬的 $\min$。我们通常使用类似这样的公式（以多项式平滑Smooth Minimum为例）：

$$\text{smin}(a, b, k) = \min(a, b) - \frac{(\max(k - |a - b|, 0))^2}{4k}$$

$a, b$: 两个物体的SDF距离值。

$k$: **融合半径**（你可以理解为液体的“表面张力”或粘稠度）。

- 当 $k=0$ 时，退化为硬边并集。
- 当 $k$ 越大，融合的范围越宽，过渡越平滑（如图片右侧所示）。

![image-20251127210220370](/ch14-DistanceFunction.png)

### **Level Set Methods**

如何把上一张图中优雅的数学概念（SDF/隐式曲面）**落地到计算机的数据结构中**。隐式曲面的离散存储形式

**解决方案**：既然写不出公式，那就**采样**。把空间划分为网格（Grid），存下每个点的数值。

**网格（The Grid）**：类似于图像的像素矩阵。**蓝色区域（负值）**表示物体内部。**红色区域（正值）**表示物体外部。

**黑线（Surface）**：即 $f(x) = 0$ 的位置。

计算机扫描网格，一旦发现相邻两个格子一个正（+0.10）、一个负（-0.05），就知道表面肯定穿过了这两个点之间。通过**插值（Interpolation）**，可以精确定位出 0 点的具体位置，从而画出平滑的曲线。

Level Set ($f=0$)

### **Related Representation: Voxels** 体像素

从之前level set的空间网格概念一样，与平面的像素类似

总结：

![image-20251127211215206](/ch14-ShapeRepresentation.png)

明白了表示方法，接下来就是获取数据

方法略过。

## 如何训练？

### **Multi-View CNN**

把3d认为是多个角度的2d照片，因为图像处理已经很成熟了。

![image-20251127211629712](/ch14-Multi-ViewCNN.png)

Indeed gives good performance

### **Pixels -> Voxels**: 3D Conv Deep Belief Networks (CDBN)

将CNN 拓展到3d

### **3D-GANs**



### Visual Object Networks (Geometry + Rendering)



### **Octave Tree Representations**

在空白或者和周围一致的地方填充大voxels，在接近边缘使用小voxels，提高利用率

![image-20251127212610216](/ch14-OctreeGN.png)

有没有别的方式？不如换一个3d表示形式！

### **PointNet: Learning on Point Clouds**

核心问题：点是无序的。用数学上的对称函数解决了顺序问题

输入点坐标 输出点坐标

Simplest form: directly aggregate all points with a symmetric operator

Just discovers simple **extreme/aggregate** properties of the geometry.

#### 如何计算loss（两个点云差异）？

![image-20251127222507723](/ch14-PointDistanceLossFunc.png)

第一种是两个总体里面找最近（懒惰），第二种是最近的一对一匹配，很难大规模实现。

**Non-Parametric -> Parametric**

可以直接学习那个表示3d物体边界的函数！于是

### **Parametric Decoder: AtlasNet**

它学习如何把一个二维平面“扭曲”变形放置到三维空间中。因为输入的 $(u, v)$ 是连续的平面，所以输出的 3D 点也就自然形成了一个平滑连续的曲面。

如果物体很复杂（比如一个圆环或者有把手的杯子），用**一张**纸（一个 2D 平面）去包它是包不住的，强行包会导致撕裂或无限拉伸。

**解决方案**：用 **K 个** 不同的 2D 平面（Patches）。

每个 MLP（或同一个 MLP 的不同调用）负责把一个小方块变形为物体表面的一部分。这就像地球是球体（3D），我们用很多页平面地图（2D Chart）来拼凑出整个地球表面。

与其先建模表面再判断里外，不如直接学习一个点在形状里面还是外面，于是

### Occupancy Networks

输入坐标 $\rightarrow$ 输出 $0/1$

但 $0/1$ 不好训练（需要3D真值），也不够好看

### NeRF

空间坐标 $(x, y, z)$ + 观看角度 $(\theta, \phi)$ ——颜色 $(R, G, B)$ + 密度 $\sigma$。

现实世界中，同一个点从不同角度看，颜色是不一样的（比如镜面反光）

这就是所谓的 **Ray Marching（光线步进）**：假设相机是眼睛，从相机中心穿过屏幕上的一个像素，向场景里发射一条射线（Ray）。在这条射线上每隔一段距离取一个点。把这些点的坐标 $(x,y,z)$ 和方向扔给 MLP，MLP 告诉你每个点有多“浓”（$\sigma$）以及是什么颜色。利用下面的公式，把这些点的颜色叠加起来，算出这个像素最终的颜色。

公式 $C \approx \sum T_i \alpha_i c_i$ 实际上是在模拟**光线穿过一团彩色烟雾**的过程。

我们拆解一下这个公式，它由三部分组成：

- **$c_i$ (Color)**：当前采样点 $i$ 自身的颜色。

- **$\alpha_i$ (Opacity)**：当前采样点 $i$ 的**不透明度**。取值$[0, 1]$。比尔-朗伯定律

  $$\alpha_i = 1 - \exp(-\sigma_i \delta_i )$$

  **$\delta_i$ (Delta)**即当前采样点 $i$ 和下一个采样点 $i+1$ 之间的距离。

- **$T_i$ (Transmittance)**：**透射率**。公式 $T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$ 是一个累乘。物理含义是：**光线到达点 $i$ 之前，还剩多少能量没有被前面的物体挡住。**

训练逻辑：

1. MLP 瞎猜颜色和密度。
2. 用公式算出像素颜色 $C_{pred}$。
3. 拿真实照片里的像素 $C_{gt}$ 对比，算 Loss。
4. 反向传播误差可以通过 $C$ 传回 $\alpha$ 和 $c$，再传回 $\sigma$，最后更新 MLP 的权重。

缺点：**NeRF parameterizes scenes densely, at every point in space.**浪费计算，慢

而**Gaussian splatting parameterizes the scene sparsely, only where density is nonzero.**

### **3DGS**： 3D Gaussian Splatting

用3d Gaussians 高斯球，控制它是圆的、扁的、还是长条的，以及它的旋转角度（姿态）。这让它能极好地贴合墙面、栏杆等薄结构。整个场景就是由 几百万个这样的半透明椭球 叠加在一起组成，训练用场景的投影

效率更高。











