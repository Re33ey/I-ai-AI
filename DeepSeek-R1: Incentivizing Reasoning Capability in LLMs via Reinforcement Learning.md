# DeepSeek-R1：通过强化学习激励大语言模型推理能力

> **论文：** DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
> **机构：** DeepSeek-AI
> **核心关键词：** Reasoning Model、Reinforcement Learning、GRPO、Long CoT、Test-Time Scaling、Distillation、Reward Model
> **核心模型：** DeepSeek-R1-Zero、DeepSeek-R1

---

# 一、背景

## 1.1 大模型为什么需要“推理训练”

传统 LLM 的核心训练方式可以粗略理解为：

```text
海量文本
   ↓
Pre-training
   ↓
Base Model
   ↓
SFT / RLHF
   ↓
Chat Model
```

预训练解决的主要问题是：

> **让模型获得语言、知识和一定程度的推理能力。**

但当问题变复杂，例如：

```text
数学竞赛
程序设计竞赛
复杂逻辑推理
科学问题
多步骤规划
```

仅仅依赖“下一个 Token 预测”往往不够。

一个模型可能：

```text
Question
   ↓
Model
   ↓
Answer
```

而强推理模型更像：

```text
Question
   ↓
理解问题
   ↓
构造方案
   ↓
推导
   ↓
发现问题
   ↓
回退
   ↓
尝试另一种方案
   ↓
验证
   ↓
Answer
```

也就是引入更长的 **Chain of Thought（CoT）**。

OpenAI o1 系列的重要方向之一，就是让模型在推理过程中消耗更多计算资源，即所谓的 **Inference/Test-Time Scaling**。DeepSeek-R1 研究的核心问题则进一步变成：

> **一定要由人类告诉模型“应该怎样推理”吗？能不能只告诉它结果对不对，让模型自己探索出推理方法？**

论文认为，可以。

这也是 DeepSeek-R1 最有价值的地方。([arXiv][1])

---

# 二、目标

论文主要研究四个问题。

第一：

> **不使用推理 SFT 数据，直接对 Base Model 做强化学习，能不能产生推理能力？**

于是产生了：

```text
DeepSeek-V3-Base
       ↓
      RL
       ↓
DeepSeek-R1-Zero
```

第二：

> 如果纯 RL 已经能够产生推理能力，如何解决可读性差、中英文混杂等问题？

于是形成完整的：

```text
DeepSeek-R1
```

第三：

> 大模型通过 RL 学出来的推理模式，能否迁移给小模型？

因此进行了：

```text
DeepSeek-R1
    ↓
生成推理数据
    ↓
Distillation
    ↓
1.5B / 7B / 8B / 14B / 32B / 70B
```

第四：

> PRM、MCTS 等看起来很合理的方法，为什么在大规模 Reasoning Model 训练中并没有成为最终方案？

这部分论文也专门给出了失败经验。([arXiv][1])

---

# 三、范围

整个论文可以拆成三个主要研究对象：

| 模型                  | 核心目的                    | 训练方法                       |
| ------------------- | ----------------------- | -------------------------- |
| DeepSeek-R1-Zero    | 验证“纯 RL 能否产生推理能力”       | Base Model → RL            |
| DeepSeek-R1         | 构建真正可用的 Reasoning Model | Cold Start → RL → SFT → RL |
| DeepSeek-R1-Distill | 将推理能力迁移到小模型             | R1 生成数据 → SFT              |

其中最重要的不是某一个 benchmark 数字，而是三条技术结论：

```text
结论 1
Reasoning 不完全依赖人工 CoT。

结论 2
RL 可以通过结果奖励，让模型自己搜索推理策略。

结论 3
大模型探索出的 Reasoning Pattern
可以通过蒸馏传递给小模型。
```

([arXiv][1])

---

# 四、现状分析

## 4.1 传统 Reasoning Model 的基本训练路线

DeepSeek-R1 之前，一个典型方案通常是：

```text
Base Model
     ↓
人工编写高质量 CoT
     ↓
SFT
     ↓
模型模仿人工推理过程
     ↓
RL / Reward Model
```

这种方法的问题在于：

### 人类 CoT 成本很高

复杂数学和代码推理的标注，需要专家。

例如一道奥赛题可能需要：

```text
问题
↓
人工推导 20 步
↓
检查
↓
修改
↓
形成高质量 CoT
```

大量生产成本非常高。

### 人类推理方式未必是最优方式

这是论文非常重要的一层观点。

如果 SFT 数据告诉模型：

```text
A → B → C → D
```

模型容易学习：

> 遇到这种问题就模仿 A → B → C → D。

但真正最优的路径可能是：

```text
A
↓
E
↓
发现错误
↓
回退
↓
F
↓
验证
↓
D
```

如果一开始就用大量人工 CoT 强约束模型，反而可能限制探索空间。

论文因此提出：

> 不先告诉模型“怎样推理”，只通过最终结果给予奖励。

让模型自己搜索策略。([arXiv][2])

---

# 五、总体技术方案

DeepSeek-R1 最重要的一张图，就是完整训练流水线：

```text
                    ┌─────────────────┐
                    │ DeepSeek-V3-Base│
                    └────────┬────────┘
                             │
                        Cold Start
                        Long CoT SFT
                             │
                             ▼
                       R1 Dev-1
                             │
                             │ Reasoning RL
                             │
               Accuracy + Language Reward
                             │
                             ▼
                       R1 Dev-2
                             │
                        Rejection
                         Sampling
                             │
              ┌──────────────┴─────────────┐
              │                            │
        Reasoning Data              Non-Reasoning Data
              │                            │
              └──────────────┬─────────────┘
                             │
                            SFT
                             │
                             ▼
                       R1 Dev-3
                             │
                 Diverse Prompts + RL
                             │
             Rule Reward + Preference RM
                             │
                             ▼
                      DeepSeek-R1
```

论文中的原始 Pipeline 图明确展示了 **两次 SFT + 两次 RL** 的多阶段设计。([arXiv][2])

这套架构实际上体现了 DeepSeek 对 Reasoning Model 的一个关键判断：

> **RL 负责“能力探索”，SFT 负责“行为塑形”。**

不是：

```text
SFT vs RL
```

而是：

```text
SFT + RL
```

两者解决完全不同的问题。

---

# 六、详细设计

# 6.1 DeepSeek-R1-Zero：最关键的实验

R1-Zero 是整篇论文最值得研究的部分。

因为它进行了一个非常干净的实验：

```text
DeepSeek-V3-Base
        ↓
       GRPO
        ↓
DeepSeek-R1-Zero
```

中间：

```text
没有 Reasoning SFT
没有人工 CoT 示范
```

也就是说，人没有告诉模型：

```text
你应该检查答案
你应该反思
你应该回退
你应该多尝试几种方案
```

这些行为是 RL 过程中自己出现的。([arXiv][2])

---

# 6.2 为什么使用 GRPO

DeepSeek 并没有使用传统 PPO，而采用了：

> **Group Relative Policy Optimization，GRPO**

传统 PPO 大致需要：

```text
Policy Model
Reward Model
Value Model / Critic
Reference Model
```

问题是 Value Model 通常和 Policy Model 规模接近。

假设主模型已经是：

```text
数百 Billion 参数
```

再维护一个相近规模 Critic，训练成本极高。

GRPO 的关键改进：

```text
PPO：

Response
   ↓
Reward
   ↓
Value Model
   ↓
Advantage


GRPO：

一个 Question
   ↓
生成 G 个 Response
   ↓
获得 G 个 Reward
   ↓
组内比较
   ↓
计算 Relative Advantage
```

因此：

> **GRPO 不再需要独立 Value Model。**

论文也明确指出，这是它相对于 PPO 在计算和显存成本上的核心优势。([arXiv][2])

---

# 6.3 GRPO 核心数学原理

假设一个问题：

$$
q
$$

模型一次生成 \(G\) 个答案：

$$
o_1,o_2,\dots,o_G
$$

这些答案对应奖励：

$$
r_1,r_2,\dots,r_G
$$

GRPO 定义第 \(i\) 个答案的 Advantage：

$$
A_i =
\frac{
r_i-\operatorname{mean}(r_1,\dots,r_G)
}{
\operatorname{std}(r_1,\dots,r_G)
}
$$

([arXiv][2])

举个非常直观的例子。

某一道数学题生成 4 个回答：

| Answer | Reward |
| ------ | -----: |
| A      |      1 |
| B      |      0 |
| C      |      1 |
| D      |      0 |

模型并不需要一个 Critic 告诉它：

```text
这个状态 Value = 0.73
```

而是直接通过组内比较：

```text
A、C > 平均水平
B、D < 平均水平
```

因此：

```text
增加 A、C 这类生成路径概率
降低 B、D 这类生成路径概率
```

这其实非常像：

> **同一道题让模型参加多次考试，然后让“同组学生”互相比成绩。**

---

# 6.4 GRPO Loss

论文使用类似 PPO 的 clipped objective：

$$
J_{GRPO}(\theta)
=
E
\left[
\frac{1}{G}
\sum_i
\left(
\min
\left(
r_i(\theta)A_i,
\operatorname{clip}
(r_i(\theta),1-\epsilon,1+\epsilon)A_i
\right)
-
\beta D_{KL}
\right)
\right]
$$

其中：

$$
r_i(\theta)
=
\frac{
\pi_\theta(o_i|q)
}{
\pi_{\theta_{old}}(o_i|q)
}
$$

同时加入 KL：

$$
D_{KL}(\pi_\theta || \pi_{ref})
$$

目的非常关键：

```text
Reward
 ↓
鼓励模型变强

KL
 ↓
防止模型一下跑得太远
```

所以本质上：

```text
最大化任务奖励
      +
限制 Policy 更新幅度
```

([arXiv][2])

---

# 6.5 R1-Zero 的 Reward 设计

R1-Zero 的 Reward 极其克制。

主要只有两类：

```text
Reward
├── Accuracy Reward
└── Format Reward
```

([arXiv][2])

## Accuracy Reward

数学题：

```text
Model Answer
     ↓
提取最终答案
     ↓
和 Ground Truth 比较
     ↓
Correct / Incorrect
```

代码题：

```text
Model Code
     ↓
Compiler
     ↓
Test Cases
     ↓
Pass / Fail
```

这是一类非常关键的任务：

> **Verifiable Tasks，可验证任务。**

因为 Reward 足够客观。

---

# 6.6 为什么不直接训练“推理过程奖励”

这可能是论文里最容易被忽略、但工程价值很高的地方。

DeepSeek 没有在 R1-Zero 的 Reasoning RL 里依赖 Neural PRM。

也就是说没有：

```text
Step 1 正确 +0.2
Step 2 正确 +0.2
Step 3 错误 -0.3
Step 4 正确 +0.1
```

主要原因是：

> Reward Model 本身可能被 Policy Model “骗”。

即：

```text
Reward Hacking
```

模型最终学到的可能不是：

```text
如何正确解决问题
```

而是：

```text
如何让 Reward Model 觉得我解决了问题
```

当 RL 训练非常长时，这个问题尤其严重。([arXiv][2])

因此 DeepSeek 的设计思想是：

> **能用确定性 Verifier，就尽量不要让另一个 LLM 判断。**

这是整篇论文非常重要的工程原则。

---

# 七、Reasoning 为什么会“涌现”

这是 R1-Zero 最惊艳的实验结果。

训练过程中，研究团队观察到：

```text
RL Step ↑
     ↓
平均输出 Token ↑
     ↓
推理时间 ↑
     ↓
AIME Accuracy ↑
```

也就是说：

> 没有人奖励模型“写得更长”。

但为了拿到更高的 correctness reward，模型自己逐渐学会：

```text
困难问题
     ↓
需要更多 Token
     ↓
更多推导
     ↓
更多验证
     ↓
更高正确率
```

论文训练曲线中，随着 RL 进行，R1-Zero 的平均输出长度明显增长，同时 AIME 成绩持续提升。

这其实就是一种：

# Learned Test-Time Scaling

模型自己学会：

```text
简单问题
→ 少想

困难问题
→ 多想
```

v2 后续分析进一步发现，在一组 2024 年数学竞赛问题上，R1 对较简单问题可能使用不足 7,000 个 thinking tokens，而最困难的问题可以超过 18,000 个；模型表现出明显的自适应计算分配能力。([arXiv][2])

这件事的意义非常大。

过去 Scaling Law 更多关注：

```text
参数量 ↑
训练 Token ↑
训练 FLOPs ↑
```

Reasoning Model 又增加了一个维度：

```text
Inference Compute ↑
```

于是能力扩展开始变成：

```text
Pre-training Scaling
        +
Post-training Scaling
        +
Test-time Scaling
```

---

# 八、Aha Moment 到底是什么

论文观察到 R1-Zero 在训练过程中出现一个很有意思的现象。

模型会突然开始出现类似：

```text
Wait...
```

然后重新检查前面的推导。

也就是：

```text
开始推理
   ↓
走了一条路线
   ↓
Wait...
   ↓
重新分析
   ↓
发现错误
   ↓
改变策略
```

这种：

```text
Reflection
Self-verification
Backtracking
Alternative Exploration
```

并没有通过 SFT 显式教给模型。

论文把这个阶段称为：

> **Aha Moment**

并认为它体现了 RL self-evolution。([arXiv][2])

但这里要特别避免一个误解：

> 这不意味着模型突然产生了类似人类的“意识”。

DeepSeek 在 v2 中甚至专门强调，很多看起来非常“人类化”的第一人称思考样式包含工程设计和数据塑形因素，不应该由此推断模型获得了人类式智能或自主意识。([arXiv][2])

---

# 九、R1-Zero 的问题

纯 RL 很强，但不好直接拿去做产品。

主要有三个问题。

## 9.1 可读性差

模型优化目标主要是：

```text
答案正确
```

而不是：

```text
人类容易阅读
```

所以可能出现：

```text
冗长
反复
结构混乱
```

---

## 9.2 Language Mixing

例如：

```text
首先我们设 x = ...
Wait, this assumption may be incorrect.
重新考虑这个 condition...
Therefore...
```

中文问题里突然英语，或者反过来。

从 Reward 的角度：

```text
答案是对的
→ 得分
```

因此模型没有动力保持语言一致。

---

## 9.3 通用能力不足

R1-Zero 的 RL 数据集中于：

```text
Math
Code
Logic
```

因此：

```text
Reasoning 很强
```

不代表：

```text
写文章
聊天
开放问答
翻译
角色扮演
```

也同样优秀。

论文因此从 R1-Zero 继续构建 DeepSeek-R1。([arXiv][2])

---

# 十、DeepSeek-R1：Cold Start

完整 R1 不再坚持：

```text
100% Pure RL
```

而是在 RL 前加入少量：

> **Cold Start Long-CoT Data**

论文描述为数千条量级。([arXiv][2])

训练变成：

```text
DeepSeek-V3-Base
      ↓
少量高质量 Long CoT
      ↓
SFT
      ↓
RL
```

Cold Start 的核心目的并不是：

> 教会模型所有推理能力。

而更多是：

```text
约束输出习惯
提高可读性
提高语言一致性
提供更稳定的 RL 初始状态
```

v2 进一步解释，Cold Start 数据包括对 R1-Zero 推理轨迹的人类整理：先由人工将原始推理转换成更自然的对话式风格，再利用 LLM 扩写，并再次由人工审核。([arXiv][2])

因此可以把它理解成：

```text
R1-Zero：
让模型自由探索

R1：
保留探索能力
+
人为规定“更适合产品使用的表达方式”
```

---

# 十一、第一阶段 Reasoning RL

Cold Start SFT 后，再进行大规模 Reasoning RL。

主要奖励：

```text
Reward =
    Accuracy Reward
  + Language Consistency Reward
```

语言一致性 Reward 大致是：

$$
Reward_{language}
=
\frac{
N(\text{目标语言单词})
}{
N(\text{全部单词})
}
$$

([arXiv][2])

例如中文问题：

```text
1000 个词
900 中文
100 英文
```

Language Reward 大约：

$$
0.9
$$

这样训练后：

```text
中英文乱跳
↓
Reward 降低
```

模型自然学习：

```text
中文问题 → 中文推理
英文问题 → 英文推理
```

值得注意的是，论文消融实验指出，这种语言一致性约束会造成轻微的能力损失，但换来了更好的用户体验和可读性。([arXiv][2])

这很好地体现了一条现实规律：

> **模型能力最优 ≠ 产品体验最优。**

---

# 十二、Rejection Sampling

完成第一阶段 RL 后，DeepSeek 有了一个很强的 Reasoning Model。

接下来用它制造训练数据：

```text
Prompt
   ↓
R1 Intermediate Model
   ↓
生成多个 Responses
   ↓
Verifier
   ↓
保留正确 Response
   ↓
清洗
   ↓
SFT Dataset
```

这就是：

# Rejection Sampling

不是每个模型答案都拿去训练。

而是：

```text
Generate N
↓
Verify
↓
Reject Incorrect
↓
Keep Correct
```

在 Reasoning 数据中，还会过滤：

```text
语言混杂
长而混乱的段落
异常 Code Block
低质量 CoT
```

([arXiv][2])

---

# 十三、800K SFT 数据

这一阶段最终得到约：

$$
804,745
$$

条 SFT 样本。

具体数据如下：

| Domain    |     Samples | Avg Tokens |
| --------- | ----------: | ---------: |
| Math      |     395,285 |     6094.2 |
| Code      |     211,129 |     7435.7 |
| STEM      |      10,124 |     4928.8 |
| Logic     |      10,395 |     2739.0 |
| General   |     177,812 |     1419.8 |
| **Total** | **804,745** | **5355.3** |

([arXiv][2])

从逻辑上大约可以理解为：

```text
800K
│
├── ~600K Reasoning
│
│   ├ Math
│   ├ Code
│   ├ STEM
│   └ Logic
│
└── ~200K Non-Reasoning
    ├ Writing
    ├ Factual QA
    ├ Translation
    ├ Self Cognition
    ├ Program Repair
    └ Front-end / Engineering
```

论文明确给出的数字约为 **600K reasoning samples + 200K non-reasoning samples**。([arXiv][2])

这一步非常值得注意。

DeepSeek 实际上做了一次：

# Model → Data → Model

```text
强模型
 ↓
产生高质量 Reasoning Data
 ↓
筛选
 ↓
重新训练 Base Model
```

这个闭环已经比简单的：

```text
人工 Data → Model
```

复杂很多。

---

# 十四、第二阶段 SFT

数据准备好以后：

```text
DeepSeek-V3-Base
      ↓
~800K Data
      ↓
SFT
      ↓
R1 Dev-3
```

为什么又重新从 Base Model 做 SFT？

因为此时的目标已经不只是：

```text
数学推理
```

还希望模型同时具备：

```text
Reasoning
Writing
QA
Translation
Instruction Following
Coding
General Chat
```

也就是说这一阶段完成：

> **Reasoning Capability + General Capability 的融合。**

---

# 十五、第二阶段 RL

最后还要进行一次 RL。

这时候训练 Prompt 已不再局限于 Reasoning。

而是：

```text
Diverse Prompts
```

Reward 则变成：

$$
Reward =
Reward_{reasoning}
+
Reward_{general}
+
Reward_{language}
$$

其中：

$$
Reward_{reasoning}=Reward_{rule}
$$

对于数学、代码、逻辑：

```text
Rule-based Verifier
```

对于通用问题：

```text
Preference Reward Model
```

([arXiv][2])

所以完整 Reward 架构可以理解为：

```text
                    Reward
                       │
          ┌────────────┼────────────┐
          │            │            │
       Reasoning     General     Language
          │            │            │
     Rule-based    Preference     Language
      Verifier         RM        Consistency
```

这比传统统一一个 Reward Model 更合理：

> **不同任务使用不同的评价体系。**

---

# 十六、Helpfulness 与 Safety Reward

通用任务没有标准答案。

例如：

```text
帮我设计一个 Redis 缓存方案
```

你很难写：

```text
if answer == ground_truth:
    reward = 1
```

因此 DeepSeek 使用 Reward Model。

论文 v2 给出的数据规模为：

```text
Helpfulness RM：
66,000 preference pairs

Safety RM：
106,000 prompts
```

([arXiv][2])

而且有一个很有意思的设计：

### Helpfulness

主要评价：

```text
Final Answer
```

尽量不要干扰内部 reasoning。

### Harmlessness

评价：

```text
Reasoning
+
Final Answer
```

因为安全风险可能在整个输出过程中出现。

---

# 十七、DeepSeek-R1 最终训练流程总结

把整篇论文压缩成一张图，就是：

```text
                           DeepSeek-V3-Base
                                  │
                                  │
                         Cold Start Long CoT
                                  │
                                 SFT
                                  ↓
                              R1 Dev-1
                                  │
                   ┌──────────────┴──────────────┐
                   │ First Reasoning RL           │
                   │                             │
                   │ Accuracy Reward             │
                   │ Language Consistency        │
                   └──────────────┬──────────────┘
                                  ↓
                              R1 Dev-2
                                  │
                         Rejection Sampling
                                  │
                     ┌────────────┴───────────┐
                     │                        │
                 600K Reasoning           200K General
                     │                        │
                     └────────────┬───────────┘
                                  │
                               ~800K
                                  │
                                 SFT
                                  ↓
                              R1 Dev-3
                                  │
                    ┌─────────────┴──────────────┐
                    │ Second RL                  │
                    │                            │
                    │ Rule Reward                │
                    │ Preference Reward          │
                    │ Language Reward            │
                    └─────────────┬──────────────┘
                                  ↓
                           DeepSeek-R1
```

这就是整篇论文最核心的工程架构。([arXiv][2])

---

# 十八、实验结果

挑几个最具代表性的指标。

| Benchmark                 | DeepSeek-R1 |
| ------------------------- | ----------: |
| AIME 2024 Pass@1          |   **79.8%** |
| MATH-500 Pass@1           |   **97.3%** |
| GPQA Diamond              |   **71.5%** |
| LiveCodeBench             |   **65.9%** |
| Codeforces Rating         |    **2029** |
| Codeforces Percentile     |   **96.3%** |
| SWE Verified              |   **49.2%** |
| ArenaHard                 |   **92.3%** |
| AlpacaEval 2.0 LC-WinRate |   **87.6%** |

([arXiv][2])

其中 AIME 2024：

```text
DeepSeek-R1    79.8%
OpenAI o1      79.2%
```

论文报告的测试设置下二者基本处于同一水平。([arXiv][2])

不过不要简单得出：

```text
R1 > o1
```

这种结论。

因为不同 Benchmark：

```text
R1 有些更强
o1 有些更强
```

比如工程编码任务 Aider 上，论文自己承认 o1-1217 表现更好。([arXiv][2])

---

# 十九、训练每一阶段到底贡献了什么

论文的 stage-by-stage 数据尤其有意思。

以 LiveCodeBench 为例：

| 难度     | R1-Zero |  Dev1 |  Dev2 |  Dev3 |    R1 |
| ------ | ------: | ----: | ----: | ----: | ----: |
| Easy   |   98.07 | 99.52 |   100 |   100 |   100 |
| Medium |   58.78 | 73.31 | 81.76 | 81.42 | 83.45 |
| Hard   |   17.09 | 23.21 | 30.36 | 33.16 | 34.44 |

([arXiv][2])

非常明显：

```text
Easy
几乎没有提升空间

Medium
明显提升

Hard
持续提升
```

说明 Post-training 的真正价值主要集中在：

> **复杂问题，而不是简单题。**

这也是 Reasoning Model 的核心价值所在。

---

# 二十、蒸馏：为什么小模型突然也会“推理”

这是整篇论文的第二个重要发现。

DeepSeek 用 R1 生成大约：

```text
800K samples
```

然后直接 SFT：

```text
Qwen / Llama Base
        ↓
R1 Reasoning Data
        ↓
SFT
        ↓
R1-Distill
```

这里甚至：

> **没有额外执行 RL。**

([arXiv][2])

结果非常强：

| 模型                   | AIME 2024 | MATH-500 |     GPQA | LiveCodeBench |
| -------------------- | --------: | -------: | -------: | ------------: |
| R1-Distill-Qwen-1.5B |      28.9 |     83.9 |     33.8 |          16.9 |
| R1-Distill-Qwen-7B   |      55.5 |     92.8 |     49.1 |          37.6 |
| R1-Distill-Qwen-14B  |      69.7 |     93.9 |     59.1 |          53.1 |
| R1-Distill-Qwen-32B  |  **72.6** | **94.3** | **62.1** |      **57.2** |
| R1-Distill-Llama-70B |      70.0 |     94.5 |     65.2 |          57.5 |

([arXiv][2])

---

# 二十一、蒸馏真正说明了什么

这说明 Reasoning Capability 至少包含两个部分：

```text
Reasoning Capability
│
├── 搜索 / 探索能力
│
└── 已经被发现的 Reasoning Pattern
```

大模型通过 RL：

```text
探索大量 Solution Space
↓
找到有效模式
```

例如：

```text
Reflection
Verification
Backtracking
Decomposition
Alternative Strategy
Long CoT
```

然后小模型不需要重新花同等代价探索。

直接：

```text
Teacher
   ↓
Reasoning Trajectories
   ↓
Student
```

即可学习这些模式。

因此可以类比：

```text
大模型 RL
≈ 科学家做研究、发现方法

Distillation
≈ 把研究成果写成教材

小模型 SFT
≈ 学生学习教材
```

这是理解 R1 Distillation 最好的方式之一。

论文实验也发现，对较小模型而言：

> 直接从大型 Reasoning Model 蒸馏，效果往往比让小模型自己从头 RL 探索更好。([arXiv][2])

---

# 二十二、为什么“大模型先探索，小模型再蒸馏”很重要

这意味着未来 Reasoning Model 的训练路线未必是：

```text
每个 Model
都独立 RL
```

而可能是：

```text
超大 Teacher
     ↓
昂贵 RL
     ↓
发现 Reasoning Patterns
     ↓
生成 Data
     ↓
───────────────
↓       ↓      ↓
7B     14B    32B
```

也就是：

> **Reasoning Search Cost 只需要主要支付一次。**

随后通过 Distillation 摊薄。

这对企业非常重要，因为大规模 RL 最昂贵的部分正是 Rollout + Verification + Training。

---

# 二十三、训练成本

v2 还给出了非常难得的训练成本数据。

按照论文假设：

```text
H800 租赁价格：
$2 / GPU Hour
```

估算为：

| 阶段                | H800 GPU Hours |      估算成本 |
| ----------------- | -------------: | --------: |
| R1-Zero           |           101K |     $202K |
| SFT Data Creation |             5K |      $10K |
| R1                |            41K |      $82K |
| **Total**         |       **147K** | **$294K** |

([arXiv][2])

这里需要特别注意：

> **$294K 不是 DeepSeek-R1 从零训练的全部成本。**

它描述的是论文列出的相关 **后训练阶段计算成本估算**，而不是 DeepSeek-V3 Base 的完整预训练成本。

把 "$294K" 理解成：

```text
DeepSeek-R1 整个模型只花了 29.4 万美元
```

是不准确的。

---

# 二十四、为什么没有采用 PRM

论文专门写了失败案例。

PRM：

> Process Reward Model

思路听起来非常合理：

```text
完整 CoT：

Step 1 → Reward
Step 2 → Reward
Step 3 → Reward
Step 4 → Reward
```

相比只看 Final Answer，理论上学习信号更密集。

但实际遇到三个问题。

第一：

```text
什么叫一个“Step”？
```

自然语言推理没有严格的步骤边界。

第二：

```text
如何判断中间 Step 是否正确？
```

很多推理中：

```text
当前看起来错误
↓
后面可能修正
```

或者：

```text
当前看起来正确
↓
后面发现前提错误
```

第三，也是最严重的：

```text
PRM
↓
可以被 Policy Hack
```

而重新训练 PRM 又会增加：

```text
Data
Compute
Pipeline Complexity
```

最终 DeepSeek 认为其收益不足以覆盖复杂度。([arXiv][2])

---

# 二十五、为什么 MCTS 也没有成为核心方案

DeepSeek 也测试了：

> Monte Carlo Tree Search。

思想是：

```text
Problem
  ↓
Step 1
├── Step 2A
│   ├── Step 3A
│   └── Step 3B
│
└── Step 2B
    ├── Step 3C
    └── Step 3D
```

类似 AlphaGo 搜索。

问题在于围棋和自然语言差别巨大。

围棋：

```text
合法动作数量有限
State 清晰
Reward 清晰
```

自然语言：

```text
下一个 Token 数万种
↓
一句话几十 Token
↓
Reasoning 几千至几万 Token
```

搜索空间几乎爆炸。

而且 MCTS 又高度依赖：

```text
Value Model
```

如果 Value 判断不准：

```text
Search Direction 错
↓
整个 Tree 都被带偏
```

论文因此认为，MCTS 在当前大规模语言模型 reasoning 训练中仍面临明显的 scalability 问题。([arXiv][2])

---

# 二十六、R1 给出的三个核心训练原则

把整篇论文抽象掉具体参数，我认为最值得记住的是下面三个原则。

## 原则一：能验证，就可以 RL

关键不是：

```text
这个问题对人类来说难不难
```

而是：

```text
答案能不能可靠验证
```

比如：

```text
奥数题
人类很难
但答案可以自动验证

Code
人类很难
但 Test Case 可以自动验证
```

这类问题非常适合：

```text
大量 Rollout
+
Automatic Verifier
+
RL
```

论文甚至认为，未来只要存在高质量 verifier，很多极难任务都有潜力通过这种方式持续优化。([arXiv][2])

---

## 原则二：不要过早限制 Reasoning Space

R1-Zero 的思想可以概括为：

```text
不要：
“按照这种方式思考。”

而是：
“最后答案正确即可。”
```

然后：

```text
Trial
↓
Error
↓
Reward
↓
Policy Update
↓
新 Strategy
```

模型自然发现：

```text
Reflection
Verification
Backtracking
Long Thinking
```

这可能比大量人工规定 Reasoning Pattern 更具扩展性。

---

## 原则三：探索和对齐应该分开

RL 擅长：

```text
Search
Explore
Optimize
```

SFT 擅长：

```text
Readable
Stable
Human-aligned
General behavior
```

所以最佳 Pipeline 并不是：

```text
纯 RL
```

也不是：

```text
纯 SFT
```

而是：

```text
SFT
 ↓
RL
 ↓
SFT
 ↓
RL
```

论文最终也明确指出，两者都不可缺少：只做 RL 容易在难以定义 reward 的任务上产生 reward hacking 或异常行为；只做 SFT 又限制了模型通过探索发现更优 reasoning trajectory 的能力。([arXiv][2])

---

# 二十七、工程角度理解 R1

如果我们自己要实现一个迷你 R1 Pipeline，可以抽象成下面这些系统：

```text
                    ┌───────────────┐
                    │ Prompt Dataset│
                    └───────┬───────┘
                            ↓
                     ┌─────────────┐
                     │Policy Model │
                     └──────┬──────┘
                            │
                      N × Rollout
                            │
                 ┌──────────▼──────────┐
                 │    Verifier Layer   │
                 │                     │
                 │ Math Checker        │
                 │ Code Sandbox        │
                 │ Format Checker      │
                 │ Reward Model        │
                 └──────────┬──────────┘
                            ↓
                         Rewards
                            ↓
                     ┌────────────┐
                     │    GRPO    │
                     └─────┬──────┘
                           ↓
                      Policy Update
                           │
                           └──────────────┐
                                          ↓
                                        Loop
```

因此一个 Reasoning Model Training Platform 的核心模块至少包括：

```text
Dataset System
Rollout Engine
Inference Cluster
Verifier
Reward Service
GRPO Trainer
Reference Model
Checkpoint Manager
Evaluation System
Rejection Sampling
SFT Pipeline
Observability
```

这已经不再只是：

```text
“训练一个 Transformer”
```

而是一个复杂的：

> **LLM + RL + Distributed Inference + Evaluation + Data Flywheel 系统。**

---

# 二十八、非功能设计与工程挑战

## 28.1 Rollout 成本

普通 SFT：

```text
Prompt + Target
↓
Forward
↓
Backward
```

RL：

```text
Prompt
↓
生成 16 个甚至更多 Response
↓
每个可能几万 Token
↓
Verifier
↓
GRPO
```

因此最大成本之一不是 Backpropagation，而是：

> **Rollout。**

论文中 R1-Zero 每个问题采样 16 个输出；最大生成长度最初为 32,768 tokens，8.2K step 之后提升到 65,536 tokens。总训练 10,400 steps。([arXiv][2])

这解释了为什么 Reasoning RL 特别依赖：

```text
高吞吐推理集群
KV Cache
Continuous Batching
高效通信
异步 Rollout
```

---

## 28.2 Reward Reliability

整个系统存在一个核心约束：

$$
RL\ Quality \approx Reward\ Quality
$$

如果 Reward 错：

```text
Policy 会非常努力地优化错误目标。
```

而且模型越强：

```text
越容易找到 Reward 漏洞
```

所以一个非常重要的工程优先级应该是：

```text
Verifier Quality
>
Reward Complexity
```

与其建设一个非常复杂但不可靠的 Reward Model，不如建设一个简单、稳定、难 Hack 的 deterministic verifier。

---

# 二十九、论文的局限性

R1 不是“Reasoning 已经解决”。

论文自己暴露了不少问题。

### 软件工程能力

因为真实软件工程任务 evaluation 时间很长，很难大规模放进同步 RL Loop。

论文因此承认：

```text
Software Engineering RL 数据不足
```

这也是为什么 SWE / Aider 等工程 Benchmark 没有出现数学那种巨大飞跃。([arXiv][2])

### Open-ended Reward

例如：

```text
写一篇优秀小说
设计一个优秀系统
写一个有创意的广告
```

没有明确：

```text
Ground Truth
```

因此无法简单 Rule-Based RL。

这仍然是 Pure RL Scaling 的核心难点。([arXiv][2])

### 长 CoT 并不意味着永远正确

模型可能：

```text
想了 20,000 Token
```

但其实从第 2,000 Token 就走错了。

论文也观察到 R1 仍可能陷入错误 reasoning path；增加多条独立 reasoning chain 后，Pass@64 高于 Pass@1，说明单条长 CoT 并没有消除搜索失败。([arXiv][2])

---

# 三十、这篇论文真正改变了什么

如果只记一句：

> **DeepSeek-R1 的贡献不是“发明了 CoT”，而是证明了强推理模式可以通过 Outcome-based Reinforcement Learning 在大型 Base Model 上自主涌现。**

更完整一点，可以总结成：

```text
以前：

Human
 ↓
Reasoning Data
 ↓
Model


R1：

Environment / Verifier
 ↓
Reward
 ↓
Model Exploration
 ↓
Reasoning Strategy


然后：

Large Reasoning Model
 ↓
Reasoning Data
 ↓
Smaller Models
```

这意味着 Reasoning Model 的开发范式正在从：

```text
教模型“正确答案怎么写”
```

逐渐转向：

```text
构建一个可靠环境，
让模型自己寻找解决问题的方法。
```

---

# 三十一、最终技术结论

DeepSeek-R1 的整体技术思想，可以最终压缩成这张图：

```text
                 ┌────────────────────┐
                 │ Strong Base Model  │
                 └─────────┬──────────┘
                           │
                           ↓
                  ┌─────────────────┐
                  │ Verifiable Task │
                  └────────┬────────┘
                           │
                           ↓
                    Multiple Rollouts
                           │
                           ↓
                    Rule Verifier
                           │
                           ↓
                        Reward
                           │
                           ↓
                         GRPO
                           │
                           ↓
             ┌─────────────────────────┐
             │ Emerging Reasoning      │
             │                         │
             │ • Long CoT              │
             │ • Reflection            │
             │ • Verification          │
             │ • Backtracking          │
             │ • Strategy Switching    │
             └────────────┬────────────┘
                          │
                    Data Generation
                          │
                          ↓
                 Rejection Sampling
                          │
                          ↓
                     High-quality
                   Reasoning Dataset
                          │
                 ┌────────┴────────┐
                 ↓                 ↓
               SFT              Distill
                 ↓                 ↓
           DeepSeek-R1       Small Models
```

技术启示，

**第一，Reasoning 可以被 Incentivize，而不一定必须被 Demonstrate。**

**第二，Verifier 是 Reasoning RL 最重要的基础设施之一。**

**第三，大模型负责探索 Reasoning Space，小模型可以通过蒸馏学习已经发现的 Reasoning Pattern。**

因此从技术演进角度来看，DeepSeek-R1 真正值得关注的并不是某个 Benchmark 超过了谁，而是它展示了一条相对完整的路线：

$$
\boxed{
Base\ Model
\rightarrow
RL\ Exploration
\rightarrow
Reasoning\ Emergence
\rightarrow
Data\ Flywheel
\rightarrow
SFT/RL
\rightarrow
Distillation
}
$$

这套范式后来已经成为理解 **Reasoning Model / Thinking Model / Test-Time Scaling** 非常重要的一条主线。([arXiv][1])

**论文原文：** [arXiv: DeepSeek-R1](https://arxiv.org/abs/2501.12948?utm_source=chatgpt.com)
**官方代码与模型：** [DeepSeek-R1 GitHub](https://github.com/deepseek-ai/DeepSeek-R1?utm_source=chatgpt.com)

[1]: https://arxiv.org/abs/2501.12948 "[2501.12948] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
[2]: https://arxiv.org/pdf/2501.12948 "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
