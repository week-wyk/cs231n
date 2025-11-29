---
typora-root-url: ./fig
---

# **L16: Robot Learning**

Problems where an **agent **performs **actions** in the **environment**, and receives **rewards**

**Goal**: Learn how to take actions that maximize reward

物理交互

Problem formulation：**Goal：**  + **State:** + **Action:**  + **Reward:** 

## 特点：

Robot vision is embodied, active, and environmentally situated.

**Embodied:** Robots have physical bodies and experience the world directly. Their actions are part of a dynamic with the world and have **immediate feedback** on their own sensation.

**Active:** Robots are active perceivers. It knows why it wishes to sense, and chooses what to perceive, and determines how, when and where to achieve that perception.

**Situated:** Robots are **situated in the world**. They do not deal with abstract descriptions, but with the “here” and “now” of the world directly influencing the behavior of the system.

## Reinforcement Learning

RL trains **agents** that interact with an **environment** and learn to maximize **reward** **(trial and error)**

### Reinforcement Learning vs Supervised Learning

- Rewards and state transitions may be random

- Reward $r_t$ may not directly depend on action $a_t$

- Can’t backprop through world; can’t compute $dr_t /da_t$

- **Nonstationary**: What the agent experiences depends on how it acts



一些算法：Q Learning/ SAC / PPO/......

### Q Learning :DQN (Deep Q Network)

![image-20251128181225392](/ch16-DQN.png)

$Q(s, a)$ 是一个函数，它回答的问题是：

> “在当前状态 $s$（State）下，如果我采取动作 $a$（Action），**长期来看**我能获得多少总分？”

**传统 Q-Learning**：用一个巨大的**表格**（Q-Table）来存这些值。比如：`状态=看到球在左边`, `动作=向左`, `Q值=100`。

**Deep Q-Learning (这张图)**：Atari 游戏的状态（像素组合）太多了，表格存不下。所以，用一个神经网络（CNN）估算这个 Q 值。

输入：游戏画面。输出：每个动作对应的 Q 值。

利用 贝尔曼方程 (Bellman Equation)：

$$Q(s, a) \approx r + \gamma \max_{a'} Q(s', a')$$

今天的预测值，应该等于“刚才拿到的即时奖励 $r$” + “打折后的未来最高预期奖励”。

**Loss Function**：计算“神经网络当前的预测”和“贝尔曼方程算出的真实目标”之间的 **MSE Loss**，然后反向传播更新权重 $\theta$。

why stack of last 4 frames ？

如果你只看**一张**静态截图，你只知道球在哪里，但不知道球是**正在往上飞**还是**往下掉**。叠加 4 帧画面，神经网络就能“看”出物体的**速度**和**方向**（运动信息）。这就是把“画面”变成了完整的“状态 ($s_t$)”。

### pros and cons

RL 优势：允许进行广泛和全面的探索与世界互动，可以发现比人类更优的策略

![image-20251128181821566](/ch16-ProblemofRL.png)

## Model Learning & Model-Based Planning

use **planning** through the model to make decisions

Model might not be accurate enough.

1. Execute the first action 
2. Obtain new state
3. Re-optimize the action sequence using gradient descent 

Key: GPU for parallel sampling / gradient descent

## **Imitation Learning**

### Behavior Cloning (BC)

问题：一旦出现误差 误差会累计越来越大，无法调整



### Inverse Reinforcement Learning (IRL)

这是为了解决“奖励函数太难写”的问题。

**输入 (Input)**：**环境** + **专家行为 (Expert Behavior)**。

例如：你不需要告诉机器怎么开车得分，你直接给它看 100 小时老司机开车的录像。

**输出 (Output)**：**奖励函数 (Rewards)**。

机器观察老司机的行为，试图推导出你是基于什么样的价值观（Reward Function）在行动。



### Implicit Behavior Cloning (IBC)

引入能量函数 (Energy-Based Model, EBM)

训练一个**能量函数 (Energy Function)** $E(s, a)$。

**输入**当前状态 $s$ + 任意一个动作 $a$。**输出**：这个动作有多“离谱”（能量值）。能量越低，越像专家。

本质上是把**回归问题 (Regression)** [^1]变成了一个**能量建模问题 (Energy Modeling)**。

[^1]: 输出连续值。 classification是输出特定选项



### Diffusion Policies

本质上就是在学习和利用一个 Gradient Field（梯度场）

![image-20251128200719383](/ch16-DiffusionPolicies.png)

输入：当前观测 $O_t$ (图片/雷达) + 随机高斯噪声。

过程：去噪网络 (1D U-Net 或 Transformer) 逐步减去噪声。

输出：一段 $T \times D$ 的**动作轨迹**（比如未来 16 步的机械臂关节角度）。

#### Receding Horizon Control

现在是在 $t$ 时刻，模型一口气生成未来 $16$ 步的动作序列$[a_t, a_{t+1}, ..., a_{t+16}]$。**只执行**前 $8$ 步。执行完第 8 步后，我在 $t+8$ 时刻**重新观测**，再生成新的未来 $16$ 步。

平稳

## Robotic Foundation Models

![image-20251128201237583](/ch16-robotic-foundationModel.png)

Pi-Zero（开源）为例子

![image-20251128201741925](/ch16-Pi-Zero.png)

post-training（后训练） 本质上是Supervised Fine-Tuning

测试主要针对三类，见图右侧。

## 关于evaluation

Evaluation is primarily conducted in the **real world**

- Real-world evaluation is costly and noisy

  > “We have large enough budget such that we can still make progress.”

- **Weak correlation** between training loss and real-world success rate.



未来的趋势是build foundation world model（能够理解世界，而不是只学习做特定任务）

目前存在的一些问题：Current foundation models are not tailored for embodied agents