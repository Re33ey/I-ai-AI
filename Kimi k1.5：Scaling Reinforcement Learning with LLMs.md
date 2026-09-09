# Kimi k1.5：Scaling Reinforcement Learning with LLMs 技术解析

> 论文：**Kimi k1.5: Scaling Reinforcement Learning with LLMs**
> 团队：Kimi Team / Moonshot AI
> arXiv：2501.12599
> 最新论文版本：v4，2025 年 6 月更新。([arXiv][1])

---

# 一、背景

## 1.1 大模型 Scaling 遇到的问题

过去几年，大语言模型的能力提升基本遵循一条非常清晰的路线：

```text
更多参数
   +
更多训练数据
   +
更多训练计算量
   ↓
更强模型
```

也就是经典的 **Scaling Law**。

预训练阶段主要通过：

```text
Next Token Prediction
```

学习海量文本中的统计规律。

例如：

```text
输入：

The capital of France is

预测：

Paris
```

不断扩大：

```text
Model Size
Data Size
Training FLOPs
```

通常都能继续提高模型能力。

但是问题逐渐出现：

> **高质量预训练数据并不是无限的。**

Kimi k1.5 论文因此提出：

```text
Pretraining Scaling
        ↓
受限于已有数据

RL Scaling
        ↓
模型可以主动探索
        ↓
从 Reward 中继续产生学习信号
```

也就是说，强化学习给 LLM 提供了一个新的 Scaling 维度：

> **让模型通过探索产生新的“训练经验”。**

这也是论文标题：

> **Scaling Reinforcement Learning with LLMs**

真正想讨论的问题。([arXiv][2])

---

# 二、论文目标

Kimi k1.5 想回答的核心问题可以浓缩成一句话：

> **能不能像 Scaling Pretraining 一样 Scaling Reinforcement Learning？**

其中最关键的问题有三个。

第一：

```text
模型推理得越久
是不是就可能推理得越好？
```

第二：

```text
如果允许模型生成几十 K token 的 CoT
RL 怎么训练？
```

第三：

```text
Long-CoT 很强
但是推理成本很高
能不能把长推理能力压缩到短推理模型？
```

Kimi k1.5 最终给出的核心答案是：

```text
Long Context
+
Long-CoT
+
Reinforcement Learning
+
高效 Rollout
+
改进 Policy Optimization
```

并认为：

> **Context Length 本身可以成为 RL Scaling 的一个重要维度。**

论文将 RL 上下文扩展到 **128K**，并观察到随着可用推理长度增加，模型能力仍然持续提升。([GitHub][3])

---

# 三、Kimi k1.5 总体训练架构

Kimi k1.5 并不是直接拿一个 Base Model 开始 RL。

完整流程大致为：

```text
                Pretraining
                     │
                     ▼
              Vanilla SFT
                     │
                     ▼
              Long-CoT SFT
                     │
                     ▼
          Reinforcement Learning
                     │
             ┌───────┴────────┐
             │                │
             ▼                ▼
        Long-CoT Model    Long2Short
                              │
                              ▼
                       Short-CoT Model
```

论文明确给出的训练阶段包括：

```text
1. Pretraining

2. Vanilla SFT

3. Long-CoT SFT

4. Reinforcement Learning

5. Long2Short
```

其中整篇论文的重点并不是 Pretraining，而是：

```text
Long-CoT
+
RL Scaling
```

([arXiv][1])

---

# 四、核心思想

理解 Kimi k1.5，最重要的是抓住四件事情。

```text
Kimi k1.5
│
├── 1. Long Context Scaling
│
├── 2. Long-CoT Reinforcement Learning
│
├── 3. Improved Policy Optimization
│
└── 4. Long2Short
```

其中又以第一条最关键。

---

# 五、Long Context Scaling

## 5.1 什么是 Long-CoT

普通模型回答一道数学题可能是：

```text
问题
 ↓
思考几步
 ↓
答案
```

Long-CoT 则可能是：

```text
问题
 ↓
分析问题
 ↓
尝试方法 A
 ↓
发现问题
 ↓
重新思考
 ↓
尝试方法 B
 ↓
检查结果
 ↓
发现错误
 ↓
回溯
 ↓
重新计算
 ↓
验证
 ↓
答案
```

模型实际上在生成过程中执行了一种：

```text
Planning
Reflection
Backtracking
Correction
Exploration
```

即：

```text
规划
反思
回退
修正
探索
```

Long-CoT SFT 的 warm-up 数据也专门强调这些思维模式。([arXiv][1])

---

# 六、为什么“更长的 CoT”可能让模型更聪明

传统 Transformer 推理通常可以理解为：

```text
问题
 ↓
Token 1
 ↓
Token 2
 ↓
Token 3
 ↓
...
 ↓
答案
```

每生成一个新 Token，模型都可以重新读取之前的上下文。

因此如果模型生成了：

```text
10000 tokens
```

那么这些 token 并不只是“废话”。

它们相当于模型的：

```text
外部工作记忆
```

模型可以利用已经生成的内容继续：

```text
重新检查
重新规划
探索其他路线
修正错误
```

于是：

```text
Context Length ↑
       ↓
允许的 CoT Length ↑
       ↓
潜在搜索步骤 ↑
       ↓
探索空间 ↑
       ↓
复杂问题解决能力 ↑
```

Kimi k1.5 的一个重要观察正是：

> **随着 RL 推理 Context Length 增长，模型性能仍然能够继续提升。**

论文最高把 RL context 扩展到了：

```text
128K tokens
```

([GitHub][3])

---

# 七、一个非常重要的思想：用 CoT 替代显式搜索树

传统复杂 reasoning 方法可能使用：

```text
Monte Carlo Tree Search
```

例如：

```text
             Problem
                │
        ┌───────┼───────┐
        A       B       C
       / \     / \     / \
      ...     ...     ...
```

需要显式：

```text
Search Tree
+
Critic / Value Model
+
Node Evaluation
+
Backtracking
```

论文认为，Long-CoT 可以把这种搜索过程某种程度上“摊平”到一个序列中。

例如：

```text
Problem

↓
尝试 A

↓
发现 A 不行

↓
回溯

↓
尝试 B

↓
发现 B 部分正确

↓
修正

↓
得到答案
```

于是原来的：

```text
Tree Search
```

变成：

```text
Sequential Search
```

或者可以理解成：

```text
隐式 Search Tree
        ↓
编码进一个超长 CoT
```

这也是为什么 Kimi k1.5 强调：

> 可以不依赖显式 MCTS、独立 Value Function 和 Process Reward Model，同样训练出强大的推理能力。([GitHub][3])

---

# 八、RL Prompt 数据设计

很多人容易把 RL 的重点只放在：

```text
PPO
GRPO
Reward
```

但 Kimi k1.5 特别强调：

> **RL Prompt Set 本身非常重要。**

作者总结高质量 RL Prompt 有三个条件。

| 特性                    | 含义       |
| --------------------- | -------- |
| Diverse Coverage      | 数据领域足够丰富 |
| Balanced Difficulty   | 难度需要合理分布 |
| Accurate Evaluability | 答案必须可靠验证 |

数据覆盖：

```text
STEM
Coding
General Reasoning
Competition Problems
Vision + Text
```

([arXiv][1])

---

# 九、如何评估题目难度

这是 Kimi k1.5 一个很实用的工程技巧。

对于一个问题：

```text
Question
```

让当前 SFT Model：

```text
采样 10 次
```

例如：

```text
Result：

✓
×
×
✓
×
×
×
×
×
×
```

那么：

```text
Pass Rate = 2 / 10 = 20%
```

于是可以把：

```text
Pass Rate
```

作为当前模型视角下的：

```text
Difficulty
```

近似指标。

也就是说：

```text
Pass Rate 高
     ↓
问题简单

Pass Rate 低
     ↓
问题困难
```

这种方法的优点是：

> 难度不是人工定义，而是**相对于当前模型能力动态定义**。

([arXiv][1])

---

# 十、防止 Reward Hacking

这是整套 RL 系统特别关键的一步。

比如一道题：

```text
Question：

复杂推理过程……

Answer:

A
```

模型可能根本不会做题，但随机猜：

```text
A
```

刚好猜对。

Verifier 就会认为：

```text
Reward = 1
```

于是：

```text
错误推理
+
正确最终答案
=
正奖励
```

这就是典型：

```text
Reward Hacking
```

所以 Kimi 团队主动过滤一些很容易 hack 的题型，例如：

```text
Multiple Choice
True / False
部分 Proof Problems
```

对于一般 QA，他们甚至做了一个简单测试：

```text
不给模型 CoT
       ↓
直接猜答案 N 次
       ↓
如果能轻易猜对
       ↓
删除这个 Prompt
```

实验中：

```text
N = 8
```

论文认为这能够去掉大量容易被 Reward Hack 的 Prompt。([arXiv][1])

---

# 十一、Long-CoT SFT

直接 RL 一个普通模型有一个问题：

```text
模型根本不知道怎么 long thinking
```

所以 Kimi k1.5 首先进行一个：

```text
Long-CoT Warm-up
```

流程：

```text
Prompt
   ↓
生成高质量 Long-CoT
   ↓
验证
   ↓
Long-CoT Dataset
   ↓
Lightweight SFT
```

这些 CoT 特别包含：

```text
Planning
Evaluation
Reflection
Exploration
```

目的是让模型先学会：

> **怎么思考。**

再通过 RL 学会：

> **什么样的思考方式更容易获得 Reward。**

因此：

```text
Long-CoT SFT
      ↓
获得 reasoning 起点

RL
      ↓
强化有效 reasoning strategy
```

([arXiv][1])

---

# 十二、Kimi k1.5 的 RL 问题建模

假设数据：

$$
D=\{(x_i,y_i^*)\}
$$

其中：

```text
x = Question

y* = Ground Truth Answer
```

模型生成：

```text
z = Chain of Thought

y = Final Answer
```

于是完整轨迹：

```text
x
↓
z1
↓
z2
↓
z3
↓
...
↓
zm
↓
y
```

策略模型可以表示为：

$$
\pi_\theta(y,z|x)
$$

RL 的目标就是：

```text
让模型产生：

更好的 z
+
更正确的 y
```

最终得到更高 Reward。([arXiv][1])

---

# 十三、Kimi k1.5 使用什么 RL 算法？

这里特别值得注意。

Kimi k1.5 **不是 DeepSeek-R1 那套 GRPO**。

它采用：

> **Online Policy Mirror Descent 的一个变体。**

论文的核心优化目标可以简化理解为：

$$
\max_\theta
\mathbb{E}[r]
-
\tau KL(\pi_\theta||\pi_{\theta_i})
$$

可以拆成两部分：

```text
Reward
-
Policy Change Penalty
```

即：

```text
让模型获得更高 Reward
          │
          ▼
但又不能一次改变得太猛
```

因此引入：

```text
KL Regularization
```

约束新模型不要距离旧策略：

```text
π_old
```

太远。

---

# 十四、为什么需要 KL？

假设旧模型：

```text
Question

→ Reasoning A  40%
→ Reasoning B  35%
→ Reasoning C  25%
```

一次 RL 后突然变成：

```text
A 1%
B 1%
C 98%
```

即使 C 在当前 batch reward 很高，这种巨大变化也可能造成：

```text
Policy Collapse
Overfitting
Training Instability
```

所以 RL 通常需要：

```text
提高 Reward
+
控制 Policy Update 大小
```

也就是：

$$
Reward-\lambda KL
$$

本质上：

> **既要进步，又不能步子迈得太大。**

---

# 十五、它和 PPO / GRPO 有什么区别？

可以先这样理解：

| 方法        | 核心                                                     |
| --------- | ------------------------------------------------------ |
| PPO       | Policy + Critic/Value + Advantage                      |
| GRPO      | Group Reward → Relative Advantage                      |
| Kimi k1.5 | Online Mirror Descent + Reward + Policy Regularization |

Kimi k1.5 特别强调：

```text
不使用 Value Network
```

原因也非常有意思。

---

# 十六、为什么 Kimi k1.5 不使用 Value Model？

传统 RL：

```text
State
 ↓
Value Model
 ↓
估计当前状态价值
 ↓
进行 Credit Assignment
```

但 Long-CoT 中可能出现一个问题。

例如：

```text
z1：正确

z2：正确

z3：犯错

z4：发现错误

z5：回溯

z6：重新推理

z7：得到正确答案
```

传统 Value Model 可能认为：

```text
z3
↓
Value 很差
↓
应该惩罚
```

但从最终结果看：

```text
犯错
↓
发现错误
↓
自我纠正
```

本身可能恰恰是一种非常重要的 reasoning 能力。

因此 Kimi 团队认为：

> 对 Long-CoT 来说，传统逐状态 Credit Assignment 未必总是理想的。

模型需要被允许：

```text
Trial
↓
Error
↓
Reflection
↓
Recovery
```

论文因此没有引入独立 Value Network。

---

# 十七、Reward 设计

可以把 Kimi k1.5 的奖励粗略理解成：

```text
Reward
=
Correctness Reward
+
Length Reward
+
其他任务 Reward
```

其中数学和代码任务拥有相对可靠的可验证反馈。

例如 Coding：

```text
模型生成代码
      ↓
Code Sandbox
      ↓
运行 Test Cases
      ↓
Pass / Fail
      ↓
Reward
```

这实际上属于今天常说的：

```text
RLVR
=
Reinforcement Learning
with Verifiable Rewards
```

---

# 十八、Length Penalty

Long-CoT 有一个很现实的问题：

```text
模型越来越会“想”
        ↓
CoT 越来越长
        ↓
Token 越来越多
```

甚至出现：

```text
Overthinking
```

例如一道很简单的问题：

```text
2 + 3 = ?
```

模型却输出：

```text
首先我们分析加法……
换一个角度……
让我验证一下……
我们再重新考虑……
```

这显然非常浪费。

所以 Kimi k1.5 引入：

```text
Length Penalty
```

原则大致是：

```text
正确答案：

越短
奖励越高

错误答案：

越长
惩罚越重
```

论文还发现过早加入 Length Penalty 会影响 RL 学习，因此采用：

```text
训练早期
↓
允许模型充分探索

训练后期
↓
逐渐约束长度
```

([arXiv][1])

这个设计非常关键，因为存在一个天然矛盾：

```text
Reasoning Quality
      VS
Token Efficiency
```

---

# 十九、Sampling Strategy

RL 的计算资源非常宝贵。

假设一道题模型：

```text
成功率 = 100%
```

继续采样其实价值很低。

另一道题：

```text
成功率 = 0.01%
```

可能又太难，100 次 rollout 都没有正样本。

最有学习价值的其实通常是：

```text
模型“差一点会”的问题。
```

所以 Kimi k1.5 使用两种策略。

---

## 19.1 Curriculum Sampling

课程学习：

```text
Easy
 ↓
Medium
 ↓
Hard
```

训练初期：

```text
不要疯狂做超难题
```

因为模型还没有能力找到正确 trajectory。

这样：

```text
大量 GPU
↓
大量错误 rollout
↓
Reward ≈ 0
```

训练效率很低。

所以先学习：

```text
相对容易的问题
```

再逐渐进入：

```text
Hard Problems
```

([arXiv][1])

---

# 二十、Prioritized Sampling

第二种方法：

```text
Success Rate = si
```

采样概率大致：

$$
P_i \propto 1-s_i
$$

也就是说：

```text
成功率越低
↓
越容易被采样
```

例如：

| Question | Success Rate | Sampling Priority |
| -------- | -----------: | ----------------: |
| A        |          95% |                 低 |
| B        |          70% |                 中 |
| C        |          30% |                 高 |
| D        |          10% |                很高 |

这样计算资源会集中到模型的：

> **薄弱区域。**

([arXiv][1])

---

# 二十一、Partial Rollout：Kimi k1.5 最重要的工程创新之一

Long-CoT 最大的问题不是概念，而是：

> **太贵。**

假设一次 rollout：

```text
128K Tokens
```

如果每次 RL iteration 都重新：

```text
Question
↓
Token 1
↓
...
↓
Token 128000
```

成本极高。

所以 Kimi k1.5 提出：

# Partial Rollout

---

## 21.1 普通 Rollout

传统：

```text
Iteration N

Prompt
  ↓
────────────────────
完整生成 100K Tokens
────────────────────
  ↓
Reward
```

假如生成非常慢：

```text
整个 batch
```

都可能等它。

---

# 二十二、Partial Rollout 原理

给每一次 rollout 一个：

```text
Fixed Token Budget
```

例如：

```text
16K tokens
```

如果完整 CoT 是：

```text
80K
```

那么：

```text
Iteration 1

Prompt
 ↓
Token 1 → 16K
 ↓
暂停
 ↓
存 Replay Buffer
```

下一轮：

```text
Iteration 2

复用前面 16K
 ↓
生成 16K → 32K
 ↓
暂停
```

继续：

```text
Iteration 3
32K → 48K

Iteration 4
48K → 64K

Iteration 5
64K → 80K
```

最终才完成 trajectory。

论文中明确指出，旧 segment 可以直接从 Replay Buffer 复用，只有当前 iteration 的部分需要继续计算。([arXiv][1])

---

# 二十三、为什么 Partial Rollout 很重要

没有 Partial Rollout：

```text
长样本
───────────────────────── 120 秒

短样本
──── 10 秒

短样本
──── 8 秒
```

所有 GPU 可能都在：

```text
等那个 Long Sequence
```

导致：

```text
Straggler Problem
```

有 Partial Rollout：

```text
统一控制每次生成预算

Long trajectory
↓
切片

Short trajectory
↓
正常完成
```

于是：

```text
GPU utilization ↑

Iteration latency ↓

Long-CoT scalability ↑
```

论文还允许 rollout workers 异步处理不同 trajectory，从而避免某个超长请求长期占据计算资源。([arXiv][1])

---

# 二十四、Replay Buffer 的作用

Partial Rollout 必须配合：

```text
Replay Buffer
```

整体结构可以理解成：

```text
                Master
                   │
          ┌────────┴────────┐
          │                 │
   Rollout Workers     Reward Service
          │                 │
          └──────┬──────────┘
                 ▼
           Replay Buffer
                 │
                 ▼
          Trainer Workers
                 │
                 ▼
           Policy Model
```

Replay Buffer 保存：

```text
完整 trajectory

以及

未完成 partial trajectory
```

下一 iteration 就可以继续使用。

论文的 RL 系统采用：

```text
Rollout Phase
      ↓
Replay Buffer
      ↓
Training Phase
      ↓
Update Policy
      ↓
下一轮 Rollout
```

的迭代式同步框架。([arXiv][1])

---

# 二十五、Kimi k1.5 RL 系统架构

论文中的完整 RL infrastructure 大致包括：

```text
                      Master
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
 Rollout Workers    Reward Models    Code Sandbox
        │
        ▼
 Replay Buffer
        │
        ▼
 Trainer Workers
        │
        ▼
   Policy Model
        │
        ▼
Reference Policy
```

Master 负责协调：

```text
Rollout
Reward Evaluation
Replay Buffer
Trainer
```

([arXiv][1])

---

# 二十六、训练和推理如何共享 GPU

RL for LLMs 有一个非常特殊的问题。

一次 RL iteration 包括：

```text
Inference / Rollout
        ↓
Training
        ↓
Inference / Rollout
        ↓
Training
```

但：

```text
Training Framework
```

和：

```text
Inference Framework
```

通常完全不同。

例如 Kimi 使用：

```text
Megatron
```

做训练；

使用：

```text
vLLM
```

做 rollout inference。

如果分别部署 GPU：

```text
Training GPU
Inference GPU
```

那么会出现大量：

```text
Idle GPU
```

---

# 二十七、Hybrid Deployment

Kimi k1.5 采用：

```text
Megatron
+
vLLM
```

共用 GPU。

逻辑：

```text
Training Phase

Megatron
████████████

vLLM
Idle
```

训练结束：

```text
Megatron
Offload

↓ 权重传输

vLLM
Onload
```

然后：

```text
Rollout Phase

Megatron
Idle

vLLM
████████████
```

Rollout 完成：

```text
vLLM terminate/offload
↓
Megatron onload
↓
继续训练
```

Kimi 团队报告其系统可实现训练到推理阶段切换低于约一分钟，反向切换约十秒。([arXiv][1])

---

# 二十八、为什么 Long Context 可以视为 RL Scaling Law

这其实是论文最值得思考的地方。

传统 Scaling：

```text
Model Parameters ↑
Data ↑
Compute ↑
```

Kimi 提出另一个轴：

```text
RL Context Length ↑
```

意味着：

```text
允许更多 reasoning tokens
        ↓
允许更多 search steps
        ↓
允许更多 exploration
        ↓
可能找到更复杂 solution
```

于是可以把 reasoning 能力粗略理解为：

$$
Capability
=
f(
ModelSize,
TrainingCompute,
RLCompute,
ContextLength
)
$$

而不再仅仅：

$$
Capability=f(ModelSize)
$$

---

# 二十九、一个很有意思的实验：小模型 + 长思考

论文做了一个很有意思的 Ablation。

比较：

```text
Large Model
```

和：

```text
Smaller Model
```

一开始：

```text
Large Model > Small Model
```

但经过 RL 之后，如果小模型获得更多：

```text
CoT tokens
```

那么：

```text
Small Model
+
Longer Reasoning
```

可以逐渐追近较大的模型。

也就是说：

```text
Parameter Scaling
```

和：

```text
Test-time Compute Scaling
```

之间存在一定替代关系。

不过论文同时指出：

> 大模型通常仍然拥有更好的 token efficiency，而且如果追求最高性能，大模型 + 更长 Context 的上限通常更高。([arXiv][1])

---

# 三十、这和 OpenAI o1 的 Test-Time Compute 思路有什么关系

可以把传统模型理解为：

```text
Prompt
↓
一次前向推理
↓
Answer
```

而 reasoning model：

```text
Prompt
↓
大量 reasoning tokens
↓
探索
↓
反思
↓
纠错
↓
Answer
```

于是：

```text
Inference Compute ↑
        ↓
Reasoning Capability ↑
```

这就是：

# Test-Time Compute Scaling

也就是说智能能力不再完全在：

```text
模型参数
```

里。

还可以来自：

```text
模型愿意花多少计算量去思考。
```

---

# 三十一、为什么不能无限增加 CoT

问题来了：

```text
Context ↑
CoT ↑
Performance ↑
```

是不是直接：

```text
1M Token CoT
10M Token CoT
```

就完事了？

当然不是。

因为会出现：

```text
Overthinking
```

包括：

```text
重复分析
无效搜索
循环推理
重复验证
过度回溯
Token Waste
```

而论文的实验也显示：

```text
RL Training
        ↓
Accuracy ↑
同时
Length ↑
```

因此后期必须开始优化：

```text
Reasoning Efficiency
```

这就引出了：

# Long2Short

---

# 三十二、Long2Short 是什么

Long-CoT：

```text
Question
↓
10000 Tokens Thinking
↓
Correct Answer
```

虽然很强，但在线服务成本很高。

于是目标变成：

```text
Question
↓
3000 Tokens Thinking
↓
Same Correct Answer
```

也就是说：

> **把长推理模型学到的 reasoning prior 压缩进短推理模型。**

论文把它称为：

```text
Context Compression
```

([arXiv][1])

---

# 三十三、Long2Short 的四种方法

Kimi k1.5 实验了：

```text
Model Merging

Shortest Rejection Sampling

DPO

Long2Short RL
```

([arXiv][1])

---

# 三十四、方法一：Model Merging

准备：

```text
Long-CoT Model
+
Short-CoT Model
```

然后简单进行：

```text
Parameter Average
```

得到：

```text
Merged Model
```

直觉是：

```text
Long Model
提供 reasoning ability

Short Model
提供 token efficiency
```

两者融合。([arXiv][1])

---

# 三十五、方法二：Shortest Rejection Sampling

对于一个 Question：

```text
采样 8 个 Response
```

例如：

| Response | 是否正确 |  长度 |
| -------- | ---- | --: |
| A        | ✓    | 10K |
| B        | ×    |  5K |
| C        | ✓    |  4K |
| D        | ✓    |  7K |
| E        | ×    |  3K |

选择：

```text
C
```

因为：

```text
Correct
+
Shortest
```

然后：

```text
Shortest Correct Response
        ↓
SFT
```

这样模型逐渐学会：

```text
更短的正确推理路径
```

论文实验中：

```text
n = 8
```

([arXiv][1])

---

# 三十六、方法三：DPO

继续使用 Long-CoT Model 生成多个答案。

选择：

```text
Positive
=
Shortest Correct Response
```

而：

```text
Negative
=
Long Incorrect Response
+
明显更长的 Correct Response
```

例如：

```text
Chosen:
3K token + 正确

Rejected:
10K token + 正确

或者：

8K token + 错误
```

然后进行：

```text
DPO
```

于是模型学习：

```text
正确且短
>
正确但啰嗦
>
错误
```

([arXiv][1])

---

# 三十七、方法四：Long2Short RL

这是论文效果最好的方法之一。

第一阶段：

```text
正常 Long-CoT RL
```

得到：

```text
Long Reasoning Model
```

然后选择：

```text
Performance
+
Token Efficiency
```

平衡比较好的 checkpoint。

第二阶段再进行：

```text
RL
+
Strong Length Penalty
+
Smaller Max Rollout Length
```

从而逼迫模型：

```text
保留能力
但是减少 reasoning tokens
```

([arXiv][1])

---

# 三十八、Long2Short 的实验效果

Kimi k1.5 的 Long2Short RL 在 AIME 2024 上：

```text
Pass@1 = 60.8
```

平均只使用：

```text
3272 Tokens
```

论文认为，相比 DPO、Model Merge 和 Shortest Rejection Sampling，Long2Short RL 在性能/Token 效率上表现最好。([arXiv][1])

---

# 三十九、多模态 RL

Kimi k1.5 还有一个非常重要的特点：

> 它并不是纯文本 reasoning model，而是多模态 reasoning model。

训练数据包含：

```text
Text
+
Vision
```

视觉数据包括：

```text
Chart Understanding
OCR
Visual Reasoning
Visual Coding
Math / Science Images
Image-grounded QA
```

论文还使用一种很有意思的数据：

```text
Text-rendered Data
```

即：

```text
普通文本
代码
结构化数据
     ↓
渲染为图片
     ↓
让模型通过 Vision 理解
```

这样可以增强：

```text
Text → Vision
```

之间的一致性。([arXiv][1])

---

# 四十、SFT 数据规模

论文公开的 Vanilla SFT 数据大约包括：

```text
1M Text Examples
+
1M Text-Vision Examples
```

文本部分大致为：

| 类型               |   数量 |
| ---------------- | ---: |
| General QA       | 500K |
| Coding           | 200K |
| Math / Science   | 200K |
| Creative Writing |   5K |
| Long Context     |  20K |

多模态部分约：

```text
1M
```

覆盖：

```text
OCR
Chart
Visual Coding
Visual Reasoning
Image QA
Math/Science Vision
```

([arXiv][1])

---

# 四十一、Context Training

模型的 SFT 采用两阶段 Context Length：

```text
Stage 1
32K Tokens
1 Epoch

↓

Stage 2
128K Tokens
1 Epoch
```

也就是说模型不是突然从短上下文切换到：

```text
128K
```

而是先：

```text
32K
```

再：

```text
128K
```

完成长上下文激活。([arXiv][1])

---

# 四十二、Coding RL

Coding 是非常适合 RL 的领域。

因为 Reward 非常明确：

```text
Code
 ↓
Compile
 ↓
Execute
 ↓
Test Cases
 ↓
Pass / Fail
```

Kimi 构建自己的：

```text
Code Sandbox
```

支持：

```text
MultiPL-E
DMOJ
Lean
Jupyter
```

等环境。

训练流程：

```text
Question
   ↓
LLM
   ↓
Generated Code
   ↓
Sandbox
   ↓
Test Cases
   ↓
Reward
   ↓
Policy Update
```

([arXiv][1])

这也是为什么：

```text
Math
Coding
```

一直是 reasoning RL 最容易产生明显收益的领域：

> **Reward 足够客观。**

---

# 四十三、核心实验结果

Kimi k1.5 Long-CoT 的论文结果包括：

| Benchmark     |           Kimi k1.5 |
| ------------- | ------------------: |
| MATH-500      |            **96.2** |
| AIME 2024     |            **77.5** |
| Codeforces    | **94th percentile** |
| LiveCodeBench |            **62.5** |
| MathVista     |            **74.9** |
| MMMU          |                70.0 |

论文当时将它与 o1、o1-mini、QwQ、QVQ 等模型进行了比较。([arXiv][1])

Short-CoT 版本：

| Benchmark     | Kimi k1.5 Short |
| ------------- | --------------: |
| MATH-500      |        **94.6** |
| AIME 2024     |        **60.8** |
| LiveCodeBench |        **47.3** |
| IF-Eval       |            87.2 |
| C-Eval        |            88.3 |

([arXiv][1])

需要注意，这些都是论文所处时间点的 benchmark 对比，不应该直接理解为 2026 年当前所有模型的实时排行榜。

---

# 四十四、Kimi k1.5 最核心的技术贡献

如果把整篇论文压缩成 5 点，就是：

## 1. RL Context Scaling

```text
Context Length ↑
       ↓
Reasoning Steps ↑
       ↓
Search Space ↑
       ↓
Reasoning Capability ↑
```

---

## 2. Long-CoT RL

不是直接教模型一个固定思维链，而是让模型通过 Reward：

```text
探索自己的 reasoning strategy
```

逐渐出现：

```text
Planning
Reflection
Backtracking
Self-correction
```

---

## 3. 不依赖 Value Model / MCTS

整体框架相对简单：

```text
Policy
+
Reward
+
Long Rollout
+
Policy Optimization
```

而不是：

```text
Policy
+
Critic
+
Process Reward
+
MCTS
+
Search Tree
```

---

## 4. Partial Rollout

解决：

```text
128K Long-CoT
```

带来的计算瓶颈。

```text
Long trajectory
↓
切成多个 segment
↓
Replay Buffer
↓
跨 iteration 继续生成
```

---

## 5. Long2Short

把：

```text
“会想很久”
```

进一步训练成：

```text
“会快速想对”
```

---

# 四十五、整套系统最重要的闭环

可以把 Kimi k1.5 抽象成：

```text
                  RL Prompt Set
                       │
                       ▼
                 Long-CoT SFT
                       │
                       ▼
                  Policy Model
                       │
                       ▼
                   Rollout
                       │
                 Long Context
                       │
                       ▼
         Planning / Reflection / Search
                       │
                       ▼
                  Final Answer
                       │
                       ▼
                    Reward
                       │
                       ▼
              Policy Optimization
                       │
                       ▼
                 Better Policy
                       │
                       └─────────┐
                                 │
                                 ▼
                            再次 Rollout
```

不断循环：

```text
Explore
↓
Reward
↓
Learn
↓
Explore Better
```

这就是论文所谓：

# Scaling RL

的核心。

---

# 四十六、Kimi k1.5 与 DeepSeek-R1 的区别

结合你前面正在看的 DeepSeek-R1，这里特别值得对比。

| 维度                   | Kimi k1.5                     | DeepSeek-R1               |
| -------------------- | ----------------------------- | ------------------------- |
| 核心主题                 | Long-context RL Scaling       | RL 驱动 Reasoning Emergence |
| RL 算法                | Online Mirror Descent Variant | GRPO                      |
| Value Model          | 不需要                           | GRPO 也不需要                 |
| Long-CoT             | 核心                            | 核心                        |
| Context Scaling      | **重点强调 128K**                 | 不是论文主线                    |
| MCTS                 | 不需要                           | 不需要                       |
| Process Reward Model | 不需要                           | 不依赖                       |
| Partial Rollout      | **核心工程创新**                    | 非核心                       |
| Long2Short           | **系统研究**                      | 主要使用蒸馏                    |
| Multimodal RL        | 是                             | R1 论文主要是文本 reasoning      |

所以二者虽然都属于：

```text
Reasoning RL
```

但研究重点其实不同。

---

# 四十七、两篇论文最根本的思想差异

DeepSeek-R1 更像是在回答：

> **强化学习能不能让 Reasoning 行为自己涌现？**

即：

```text
Base Model
↓
RL
↓
Self-reflection
Verification
Backtracking
Long Reasoning
```

Kimi k1.5 更像是在回答：

> **既然 RL 可以产生 Reasoning，那么这套 RL 怎么继续 Scale？**

于是重点放在：

```text
128K Context
Partial Rollout
Policy Optimization
Sampling
Infrastructure
Long2Short
```

所以如果按研究路线来看：

```text
DeepSeek-R1
     ↓
“Reasoning 可以通过 RL 学出来”

Kimi k1.5
     ↓
“那 Reasoning RL 怎么继续扩大规模？”
```

这两个视角放在一起看，非常有意思。

---

# 四十八、工程上的最大难点

Kimi k1.5 真正困难的其实不只是 RL 算法。

从工程上看：

```text
Reasoning RL
```

需要同时解决：

```text
Rollout Throughput

Long Sequence Training

Reward Evaluation

Replay Buffer

GPU Scheduling

Training / Inference Switching

Distributed Checkpoint

Code Sandbox

Multimodal Data

Failure Recovery
```

所以 reasoning model 并不是：

```text
换一个 RL loss
```

就能训练出来。

实际上它是：

> **算法 + 数据 + 推理系统 + 分布式训练 + Reward Infrastructure**

组成的一整套系统工程。

---

# 四十九、论文的局限

Kimi k1.5 也暴露出一些很明显的问题。

### 1. Long-CoT 成本非常高

虽然 Partial Rollout 优化训练成本，但：

```text
Long Reasoning
```

本身仍然意味着：

```text
Inference latency ↑

KV Cache ↑

GPU Memory ↑

Token Cost ↑
```

---

### 2. Reward Verification 仍然困难

数学、代码比较容易：

```text
答案
Test Case
```

都可以自动检查。

但是：

```text
开放式写作
商业决策
复杂分析
事实推理
主观问题
```

很难得到绝对可靠的：

```text
Reward
```

---

### 3. Length ≠ Intelligence

论文观察到：

```text
Longer CoT
```

往往有助于性能。

但不能反过来说：

```text
越长越聪明
```

模型可能只是：

```text
重复
犹豫
低效搜索
```

这也是 Long2Short 和 Length Penalty 存在的原因。

---

### 4. Credit Assignment 仍未彻底解决

只使用最终 Reward：

```text
10000-token CoT
            ↓
          Reward = 1
```

很难知道：

```text
到底哪一步是关键的？
```

Kimi 选择弱化传统 Value Function，并鼓励完整 trajectory exploration，但论文最后仍然把更好的 credit assignment 列为后续值得研究的方向。([arXiv][1])

---

# 五十、这篇论文最值得记住的一个公式

与其记论文复杂的 Policy Optimization 公式，不如记住这个概念公式：

$$
\boxed{
Reasoning\ Capability
\approx
Model\ Capability
+
Test\ Time\ Compute
+
RL\ Exploration
}
$$

进一步：

$$
Test\ Time\ Compute
\approx
Reasoning\ Tokens
$$

所以：

```text
模型参数
```

不再是唯一的 Scaling 维度。

还可以：

```text
让同一个模型
多思考一段时间。
```

---

# 五十一、一句话理解 Kimi k1.5

如果只记一句：

> **Kimi k1.5 的核心不是发明一个更复杂的搜索算法，而是把“长上下文中的自由探索”本身变成搜索，让 LLM 通过 RL 学会在长 CoT 中规划、试错、回溯和自我修正。**

所以整套思想是：

```text
不要显式构造：

MCTS
+
Value Tree
+
Process Reward

而是：

给模型更大的 Context
        +
让模型大量探索
        +
用最终 Reward 训练
        ↓
让搜索能力内化进模型
```

---

# 五十二、全文技术脉络总结

最后把整篇论文压缩成这一张逻辑图：

```text
            Pretraining
                 │
                 ▼
             Vanilla SFT
                 │
                 ▼
          Long-CoT Warm-up
                 │
                 ▼
        ┌─────────────────┐
        │   RL Prompt Set │
        │                 │
        │ Diversity       │
        │ Difficulty      │
        │ Verifiability   │
        └────────┬────────┘
                 │
                 ▼
            Long-CoT RL
                 │
        ┌────────┼─────────┐
        │        │         │
        ▼        ▼         ▼
   Long Context Reward   Sampling
      128K               Strategy
        │        │         │
        └────────┼─────────┘
                 ▼
      Policy Optimization
                 │
                 ▼
       Planning / Reflection
       Backtracking / Search
                 │
                 ▼
         Better Reasoning
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
   Long-CoT Model     Long2Short
                           │
          ┌────────────────┼──────────────┐
          │                │              │
       Merge              DPO             RL
          │                │              │
          └────────────────┼──────────────┘
                           ▼
                    Short-CoT Model
                           │
                           ▼
                  高性能 + 低 Token
```

而支撑这一切的底层系统是：

```text
Rollout Workers
        │
        ▼
Partial Rollout
        │
        ▼
Replay Buffer
        │
        ▼
Reward / Code Sandbox
        │
        ▼
Trainer Workers
        │
        ▼
Policy Update
```

Kimi k1.5 真正值得关注的地方也就在这里：

> **它把“Reasoning Model”从单纯的算法问题，进一步变成了一个可以 Scaling 的 RL 系统工程问题。** ([arXiv][1])

[1]: https://arxiv.org/pdf/2501.12599 "Kimi k1.5: Scaling Reinforcement Learning with LLMs"
[2]: https://arxiv.org/abs/2501.12599?utm_source=chatgpt.com "Kimi k1.5: Scaling Reinforcement Learning with LLMs"
[3]: https://github.com/MoonshotAI/Kimi-k1.5 "GitHub - MoonshotAI/Kimi-k1.5 · GitHub"
