# The Illusion of Thinking：大模型推理能力的边界与复杂度陷阱

## 1. 背景

随着 OpenAI o 系列、DeepSeek-R1、Claude Thinking、Gemini Thinking 等推理模型出现，大语言模型逐渐从传统的：

```text
Question
   ↓
直接生成 Answer
```

演变为：

```text
Question
   ↓
Reasoning / Thinking
   ↓
多步分析
   ↓
自我检查
   ↓
Answer
```

这类模型通常被称为：

> Large Reasoning Model，LRM，大型推理模型。

它们与普通 LLM 最大的区别之一，是在输出最终答案之前，会消耗额外的推理计算资源执行较长的 Chain-of-Thought 或类似的内部推理过程。

传统观点通常认为：

```text
更多 Thinking
      ↓
更多 Test-Time Compute
      ↓
更充分的搜索和自我纠错
      ↓
更好的复杂问题解决能力
```

例如在数学、编程、逻辑推理任务上，推理模型确实展现出了明显提升。

但这里存在一个非常重要的问题：

> 模型到底是真的获得了可扩展、可泛化的推理能力，还是只是在特定难度范围内，通过更长的搜索过程获得了更好的结果？

传统 Benchmark 很难回答这个问题。

因为常见数学 Benchmark，如：

```text
MATH-500
AIME 2024
AIME 2025
GSM8K
```

存在几个问题：

1. 不同题目本身难度不连续；
2. 无法精确控制问题复杂度；
3. 训练数据可能包含相似题目；
4. 通常只检查最终答案；
5. 很难分析模型中间到底如何推理。

因此 Apple 的研究团队提出了一个不同的研究思路：

> 不再单纯观察 Benchmark Accuracy，而是研究模型能力随“问题复杂度”增长时如何变化。

这就是论文：

**The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity**

的核心研究目标。论文认为，只有能够连续改变问题复杂度，才能真正观察推理模型的 Scaling Behavior。

---

# 2. 目标

论文主要试图回答以下几个问题。

## 2.1 Reasoning Model 是否真正具备可扩展推理能力

假设问题复杂度：

```text
Complexity ↑
```

理论上，如果模型真的掌握了某种通用算法，则即使准确率有所下降：

```text
Reasoning Effort
```

至少应该随着问题复杂度继续增加。

例如：

```text
简单题
Thinking: 1K tokens

中等题
Thinking: 5K tokens

困难题
Thinking: 15K tokens

更困难
Thinking: 30K tokens
```

这才比较符合：

> 问题越困难 → 搜索越多 → 推理越长

这一基本逻辑。

论文想验证现实中的 LRM 是否如此。

---

## 2.2 Thinking 是否始终优于 Non-Thinking

论文对比：

```text
Reasoning Model

vs

Standard LLM
```

例如：

```text
Claude 3.7 Sonnet Thinking
          vs
Claude 3.7 Sonnet

DeepSeek-R1
          vs
DeepSeek-V3
```

并尽可能比较相似的 inference compute。

核心问题是：

> Thinking 本身到底贡献了多少能力？

---

## 2.3 Test-Time Scaling 是否可以无限扩展

很多推理模型的发展路线隐含着一个假设：

```text
增加推理 Token
        ↓
增加 inference compute
        ↓
允许模型进行更多搜索
        ↓
提升 Accuracy
```

即所谓：

# Test-Time Scaling

论文试图研究：

```text
Inference Compute
```

是否存在明显收益边界。

---

# 3. 范围

论文重点研究的是：

```text
Algorithmic Reasoning
Planning
Compositional Reasoning
Search
Exact Computation
```

而不是知识问答能力。

其主要测试模型包括：

```text
Claude 3.7 Sonnet Thinking
Claude 3.7 Sonnet

DeepSeek-R1
DeepSeek-V3

OpenAI o-series
```

其中重点分析 Claude 和 DeepSeek，是因为研究者可以获得更丰富的 reasoning trace 信息；OpenAI o 系列主要用于最终准确率实验。论文实验通常给 Claude 和 DeepSeek 最大约 64K token 的生成预算，每个 puzzle instance 生成 25 个样本。

---

# 4. 现状分析

## 4.1 当前 Reasoning Model 的基本机制

今天的 Reasoning Model 可以粗略理解成：

```text
Transformer
    ↓
Next Token Prediction
    ↓
生成 reasoning token
    ↓
reasoning token 成为新的上下文
    ↓
继续预测
    ↓
形成长推理链
```

与普通模型：

```text
Question
 ↓
Answer
```

相比，推理模型更像：

```text
Question

 ↓

尝试方案 A

 ↓

发现问题

 ↓

重新分析

 ↓

尝试方案 B

 ↓

验证

 ↓

修正

 ↓

Answer
```

这实际上为模型增加了一种：

> inference-time search

能力。

---

# 5. 总体研究方案

论文没有直接依赖传统数学 Benchmark，而设计了一套：

# Controllable Puzzle Environment

核心思想：

```text
保持规则不变
      ↓
只改变 Problem Size
      ↓
精确增加 Problem Complexity
```

这样就可以观察：

```text
Complexity
   ↓
Accuracy
Thinking Tokens
Reasoning Trace
Failure Pattern
```

之间的关系。

论文使用了四类 Puzzle。

---

# 6. 四类实验环境

## 6.1 Tower of Hanoi

即经典的：

# 汉诺塔

三个柱子：

```text
A    B    C
```

初始：

```text
A:

1
2
3
4
```

目标：

```text
C:

1
2
3
4
```

规则：

* 每次只能移动一个盘子；
* 大盘不能放在小盘上；
* 只能移动柱子顶部盘子。

如果有：

```text
n
```

个盘子，最少移动次数：

```text
2^n - 1
```

因此复杂度增长非常快。

例如：

```text
n = 3

7 moves
```

```text
n = 5

31 moves
```

```text
n = 10

1023 moves
```

```text
n = 20

1,048,575 moves
```

因此汉诺塔非常适合测试：

> 模型能否真正执行递归算法。

论文并不只评价最优性，而会验证动作是否合法以及最终状态是否正确。

---

# 6.2 Checker Jumping

可以理解成：

```text
R R R _ B B B
```

目标：

```text
B B B _ R R R
```

棋子只能：

```text
移动一格
```

或者：

```text
跨过一个对方棋子
```

其最优步数随规模增长约为：

```text
(n + 1)^2 - 1
```

因此复杂度呈多项式增长。

这个任务主要测试：

```text
Sequential Planning
+
State Tracking
```

---

# 6.3 River Crossing

经典：

# 过河问题

存在：

```text
Actor
Agent
Boat
```

并具有一组状态约束。

模型必须找到一串合法动作：

```text
State0
 ↓
State1
 ↓
State2
 ↓
...
 ↓
Goal
```

每一步都不能违反约束。

本质属于：

```text
Constraint Satisfaction
+
Planning
```

问题。

论文通过改变 actor/agent pair 的数量增加问题复杂度。

---

# 6.4 Blocks World

例如：

```text
A
B
C
```

需要重排为某种目标：

```text
C
A
B
```

模型必须：

```text
识别当前状态
    ↓
规划动作
    ↓
移动 Block
    ↓
更新状态
    ↓
继续规划
```

这是 AI Planning 领域非常经典的环境。

---

# 7. 核心实验结果

论文最重要的发现可以浓缩为一张图：

```text
Accuracy
   ▲
   │
   │                 Reasoning Model
   │                /\
   │               /  \
   │              /    \
   │             /      \
   │ Standard   /        \
   │  Model    /          \
   │──────\───/            \
   │       \                \
   │        \                \
   │         \                \
   │          \________________\____
   │
   └───────────────────────────────►
             Complexity

       ①        ②          ③

      Low     Medium      High
```

论文将模型行为分成：

# 三个 Complexity Regime

这也是整篇论文最重要的结论之一。

---

# 8. 第一阶段：Low Complexity

低复杂度：

```text
Standard LLM
>
Reasoning Model
```

这是一个比较反直觉的发现。

简单问题，例如：

```text
2~3 步就可以解决
```

普通模型往往：

```text
直接识别模式
    ↓
快速生成
    ↓
正确答案
```

而 Reasoning Model 可能：

```text
理解问题
 ↓
尝试方案 A
 ↓
验证 A
 ↓
重新思考
 ↓
尝试 B
 ↓
又回到 A
 ↓
Answer
```

这种现象通常被称为：

# Overthinking

即：

> 本来已经得到正确答案，却继续进行不必要的推理。

论文在 reasoning trace 中观察到：

```text
Correct Solution
```

有时很早就已经出现。

但模型继续：

```text
Exploration
Reflection
Verification
Alternative Solution
```

最终：

```text
更多 Tokens
```

并不一定：

```text
更高 Accuracy
```

因此在简单任务上：

```text
Thinking ≠ Benefit
```

甚至可能：

```text
Thinking → Cost ↑
Accuracy ↓
```

论文把这视为推理模型效率上的一个重要问题。

---

# 9. 第二阶段：Medium Complexity

进入中等复杂度：

```text
Reasoning Model
>
Standard Model
```

这是当前 LRM 最有优势的区域。

为什么？

因为 Standard Model 很容易：

```text
直觉预测
 ↓
错误
 ↓
直接输出
```

而 LRM 可以：

```text
尝试
 ↓
失败
 ↓
Reflection
 ↓
修改方案
 ↓
重新搜索
 ↓
找到答案
```

这说明：

# Self-Correction

确实有价值。

这个阶段：

```text
Complexity ↑

Thinking Tokens ↑

Accuracy Advantage ↑
```

因此 Test-Time Compute 在这里非常有效。

可以理解成：

```text
更多 inference compute
       ↓
更大的 search space
       ↓
更高找到正确路径的概率
```

这正是 Reasoning Model 当前真正强大的地方。

---

# 10. 第三阶段：High Complexity

当复杂度继续增加，论文观察到一个非常明显的现象：

```text
Standard Model → Fail

Reasoning Model → Fail
```

最终：

```text
Accuracy ≈ 0
```

论文称之为：

# Complete Accuracy Collapse

即：

> 在超过某个 Complexity Threshold 后，模型性能并不是平滑下降，而可能快速接近完全失败。

论文认为，这表明现有 LRM 的推理能力并不能简单通过增加 inference compute 无限扩展。

---

# 11. 最关键发现：Thinking Token 反而下降

这是论文最有意思的实验之一。

正常情况下我们会认为：

```text
Complexity ↑

Thinking ↑
```

实验初期确实如此：

```text
Thinking Tokens
      ▲
      │
      │       /
      │      /
      │     /
      │    /
      │   /
      │  /
      │ /
      └────────────────► Complexity
```

但达到某个 Critical Complexity 后：

```text
Thinking Tokens
      ▲
      │
      │        /\
      │       /  \
      │      /    \
      │     /      \
      │    /        \
      │___/          \____
      └────────────────────► Complexity
```

也就是：

```text
Complexity ↑

Thinking Tokens ↓
```

论文尤其强调：

> 这种下降发生时，模型并不一定已经耗尽最大 token budget。

也就是说，并不是单纯：

```text
达到 Max Tokens
```

才停止。

而更像：

```text
Problem Too Complex
       ↓
模型无法形成有效搜索策略
       ↓
Reasoning Search Collapse
       ↓
Thinking 减少
       ↓
输出错误答案
```

作者将其解释为当前 LRM 的一种：

# Inference-Time Scaling Limit

即：

```text
增加 Test-Time Compute
```

并不能无限弥补：

```text
Reasoning Capability
```

本身的不足。

---

# 12. 为什么会发生 Reasoning Collapse

这里可以从 Transformer 的计算方式理解。

LLM 的推理过程本质还是：

```text
P(token_t | token_1 ... token_t-1)
```

也就是：

> 根据前面的 token 预测下一个 token。

即使 Reasoning Model 看起来在：

```text
Planning
Reflection
Search
Backtracking
```

底层仍然是：

```text
Autoregressive Token Generation
```

因此它并不是一个真正意义上的：

```text
Symbolic Search Engine
```

也不是：

```text
Program Interpreter
```

更不是天然具备显式：

```text
Stack
Graph Search
Dynamic Programming
Constraint Solver
```

的软件系统。

---

# 13. 汉诺塔揭示出的关键问题

汉诺塔其实有非常清晰的递归算法：

```text
Hanoi(n, A, B, C):

if n == 1:
    move A → C
else:
    Hanoi(n-1, A, C, B)
    move A → C
    Hanoi(n-1, B, A, C)
```

理论上，一个真正掌握算法的系统：

```text
知道 Hanoi Algorithm
```

之后：

```text
n = 5
n = 10
n = 15
```

只是执行次数不同。

也就是说应该体现：

# Algorithmic Scaling

但论文认为实验显示：

```text
模型知道算法
```

并不等于：

```text
模型可以稳定执行算法
```

这是：

# Algorithm Knowledge

与：

# Algorithm Execution

之间的重要区别。

LLM 很可能能够解释：

```text
Tower of Hanoi uses recursion.
Minimum moves = 2^n - 1.
```

但当要求真正执行大量状态转移时：

```text
State 1
State 2
State 3
...
State 100
...
```

模型可能逐渐：

```text
丢失状态
产生非法动作
重复动作
跳过步骤
形成错误分支
```

论文因此认为当前 LRM 在 exact computation 上仍然存在明显限制。

---

# 14. Reasoning 与 Search 的关系

可以把 LRM 的 reasoning process 粗略理解为：

```text
                    Question
                       │
              ┌────────┴────────┐
              ↓                 ↓
           Path A             Path B
              │                 │
         ┌────┴────┐        ┌───┴───┐
         ↓         ↓        ↓       ↓
        A1        A2       B1      B2
```

传统算法：

```text
DFS
BFS
A*
Dynamic Programming
```

有非常明确的：

```text
State Representation
Visited Set
Search Queue
Backtracking
```

而 LRM 的 Search 更接近：

```text
Implicit Search
```

状态主要存储于：

```text
Token Context
```

因此容易发生：

```text
重复探索
忘记旧状态
错误状态传播
错误路径锁定
```

---

# 15. Reasoning Trace 中的三个典型行为

论文对中间推理过程进行了分析，而不仅仅看最终 Accuracy。

这也是这篇论文的重要贡献。

## 15.1 简单问题

模型通常：

```text
很早找到正确答案
```

但继续：

```text
Thinking
```

表现为：

```text
Correct Solution
      ↓
继续探索
      ↓
Alternative
      ↓
Reflection
      ↓
又回到 Correct Solution
```

即：

# Overthinking

---

# 15.2 中等问题

正确答案出现时间更晚：

```text
Wrong
 ↓
Wrong
 ↓
Wrong
 ↓
Correction
 ↓
Correct
```

此时：

```text
Self-Correction
```

是真正有价值的。

---

# 15.3 高复杂度

进入高复杂度之后：

```text
Wrong
 ↓
Wrong
 ↓
Wrong
 ↓
Wrong
 ↓
End
```

模型甚至：

```text
无法进入 Correct Path
```

论文将这种情况理解为：

```text
Search Space
```

超过当前模型可有效处理的范围。

论文还观察到失败案例中，模型有时会较早锁定一个错误方向，然后把剩余大量推理资源用于围绕这个方向继续推演，而不是成功纠正。

---

# 16. Thinking Token 不等于有效计算

这篇论文实际上指出：

```text
Thinking Tokens
```

不是一个非常可靠的：

```text
Reasoning Compute
```

指标。

例如：

```text
10K Thinking Tokens
```

可能包含：

```text
2K Effective Reasoning

+
3K Repetition

+
2K Wrong Search

+
2K Self Reflection

+
1K Redundant Explanation
```

所以：

```text
Token ↑
```

不代表：

```text
Effective Search ↑
```

更不代表：

```text
Algorithmic Compute ↑
```

这也是未来 Reasoning Model 的核心优化方向。

---

# 17. 一个重要概念：Effective Reasoning Compute

真正应该关注的可能不是：

```text
Raw Thinking Tokens
```

而是：

```text
Effective Reasoning Compute
```

可以粗略定义：

```text
Effective Reasoning Compute
=
Useful Search
+
Correct State Updates
+
Successful Verification
+
Useful Backtracking
```

而不是：

```text
Total Generated Tokens
```

未来 Reasoning Model 的目标可能不是：

```text
Think Longer
```

而是：

# Think Better

---

# 18. 为什么传统 Benchmark 可能掩盖问题

假设：

```text
MATH Benchmark
```

包含：

```text
Problem A
Problem B
Problem C
Problem D
```

我们并不知道：

```text
Complexity(A)
Complexity(B)
Complexity(C)
Complexity(D)
```

之间的精确关系。

模型得到：

```text
Accuracy = 85%
```

也无法知道：

```text
复杂度增加
```

时：

```text
Accuracy
```

到底如何变化。

而论文的方法是：

```text
n = 1
n = 2
n = 3
...
n = 20
```

然后观察：

```text
Accuracy(n)
```

因此可以得到：

# Reasoning Scaling Curve

这就是：

> Lens of Problem Complexity

真正的含义。

论文也指出，在 MATH-500 上 thinking / non-thinking 模型的 pass@k 差距相对有限，而在 AIME24、AIME25 上差距发生变化；作者认为传统 Benchmark 中任务复杂度和潜在数据污染等因素很难被分离，因此才转向可控 Puzzle。

---

# 19. 论文的核心结论模型

整篇论文可以概括为：

```text
                  Problem Complexity
                          ↑
                          │
           ┌──────────────┼──────────────┐
           │              │              │
           │              │              │
         LOW           MEDIUM           HIGH
           │              │              │
           ↓              ↓              ↓

      Standard LLM      LRM优势        Both Fail

           │              │              │
           ↓              ↓              ↓

      不需要Thinking    Thinking有效    Reasoning Collapse
```

因此：

```text
Reasoning Benefit
```

并不是：

```text
monotonically increasing
```

而更像：

```text
          Benefit
             ▲
             │
             │       /\
             │      /  \
             │     /    \
             │____/      \____
             └─────────────────► Complexity
```

即存在一个：

# Reasoning Sweet Spot

当前 LRM 最适合：

```text
Medium Complexity
```

问题。

---

# 20. 对 Test-Time Scaling 的影响

过去一种常见 Scaling 思路：

```text
Pretraining Scaling

Parameters ↑
Data ↑
Compute ↑
```

近年来出现：

```text
Test-Time Scaling

Reasoning Tokens ↑
Samples ↑
Search ↑
Verification ↑
```

论文给出的提醒是：

```text
Test-Time Compute
```

并不是万能解。

可能存在：

```text
C < C1

Thinking 没必要
```

```text
C1 < C < C2

Thinking 非常有效
```

```text
C > C2

继续增加 Thinking
收益快速降低
```

所以未来真正的问题不是：

> 能不能让模型 Thinking 100K tokens？

而是：

> 这些计算是否真正转化成了有效搜索、状态管理与验证？

---

# 21. 对 Agent 系统的启示

这篇论文其实对 AI Agent 很重要。

Agent 通常需要：

```text
Goal
 ↓
Planning
 ↓
Tool Call
 ↓
Observation
 ↓
Replanning
 ↓
Tool Call
```

如果完全依赖 LRM 内部：

```text
Planning
```

那么复杂任务很可能遇到：

```text
Reasoning Collapse
```

因此更可靠的 Agent 架构应该：

```text
                 LLM
                  │
          ┌───────┼────────┐
          ↓       ↓        ↓
       Planner  Memory   Verifier
          │                │
          ↓                ↓
       Executor        Constraint
          │
          ↓
        Tools
```

而不是：

```text
一个 LLM
    ↓
Thinking 50K tokens
    ↓
希望它自己解决所有事情
```

---

# 22. External Tool 为什么重要

假设任务是：

```text
392847 × 293847
```

与其让模型：

```text
Thinking
Thinking
Thinking
```

更好的方式通常是：

```text
LLM
 ↓
Calculator
 ↓
Result
```

同理：

```text
Planning
```

可以使用：

```text
Search Algorithm
```

```text
SQL
```

使用：

```text
Database
```

```text
Code
```

使用：

```text
Compiler
Test Runner
```

```text
数学
```

使用：

```text
Python
CAS
Calculator
```

未来 LLM 更可能作为：

# Reasoning Orchestrator

而不是承担所有底层精确计算。

---

# 23. 更合理的下一代 Reasoning Architecture

可以从论文推导出一种更可靠的架构：

```text
                    User Problem
                         │
                         ↓
                     LLM Router
                         │
             ┌───────────┼───────────┐
             │           │           │
             ↓           ↓           ↓
          Simple       Reasoning   Algorithmic
          Problem       Problem      Problem
             │           │           │
             ↓           ↓           ↓
        Direct Answer    LRM       External Solver
                             │
                     ┌───────┴────────┐
                     ↓                ↓
                  Search           Verifier
                     │                │
                     └────────┬───────┘
                              ↓
                          Final Answer
```

即：

```text
简单问题
→ 不 Thinking
```

```text
中等问题
→ LRM Thinking
```

```text
高复杂度精确问题
→ LLM + Solver / Tools
```

这样比：

```text
Everything → Long CoT
```

更合理。

---

# 24. 对模型训练的启示

目前很多 Reasoning Training：

```text
RL
 ↓
Reward Correct Answer
 ↓
鼓励 Long Reasoning
```

但真正需要优化的可能是：

```text
Reasoning Efficiency
```

包括：

```text
1. 是否重复搜索

2. 是否识别错误路径

3. 是否及时 Backtracking

4. 是否维护稳定状态

5. 是否调用外部算法

6. 是否知道什么时候停止 Thinking
```

也就是说未来 Reward 不应该只关注：

```text
Correct Answer
```

还可以考虑：

```text
Correct Answer
+
Reasoning Cost
+
Reasoning Efficiency
+
Verification Quality
```

---

# 25. 对推理预算分配的启示

当前很多系统使用固定：

```text
reasoning_effort = high
```

但论文揭示：

```text
High Reasoning
```

不一定总是最好。

更好的方式可能是：

# Adaptive Reasoning Budget

例如：

```text
Problem
 ↓
Estimate Complexity
 ↓

Low
→ 500 tokens

Medium
→ 5000 tokens

High
→ 10000 tokens + tools
```

这比：

```text
所有问题

→ 20000 thinking tokens
```

更节省成本。

---

# 26. 与 DeepSeek-R1 的关系

DeepSeek-R1 的一个重要思想是：

```text
RL
 ↓
模型逐渐学习
Reflection
Verification
Long CoT
```

而本论文并不是否定这种方法。

更准确地说：

```text
RL Reasoning
```

确实扩展了模型能够处理的问题区间：

```text
普通 LLM
     ↓
较低 Complexity Limit
```

变成：

```text
Reasoning Model
        ↓
更高 Complexity Limit
```

但论文认为：

```text
Complexity Limit
```

并没有消失。

只是：

```text
Boundary → 向右移动
```

可以表示为：

```text
Accuracy
 ▲
 │
 │ Standard LLM
 │ ────────\
 │          \
 │           \
 │
 │ Reasoning Model
 │ ─────────────────\
 │                   \
 │                    \
 └────────────────────────► Complexity
```

因此：

> Reasoning Model 扩展了推理边界，但不意味着已经获得无限可扩展的通用算法推理能力。

---

# 27. 论文值得注意的争议

这篇论文影响很大，但也不能简单把：

```text
LRM 无法推理
```

当成已经被证明的事实。

论文发表后出现了针对实验设计的公开批评。

其中一个评论工作指出几个主要问题：

```text
Tower of Hanoi
```

在部分高复杂度设置中，要求模型显式输出完整动作序列，这本身可能产生极大的输出长度需求；

此外，该评论还质疑论文部分 River Crossing 实例的可解性，以及自动评价是否充分区分了：

```text
Reasoning Failure
```

和：

```text
Output / Experimental Constraint
```

的问题。

因此必须区分两个命题。

论文较强的表述是：

```text
LRMs possess fundamental reasoning limits.
```

而从实验中更稳妥能够得出的结论是：

```text
当前 LRM 在这些可控规划任务和所采用的评价方式下，
随着复杂度提升表现出明显的性能边界和不稳定性。
```

后一个结论证据更直接。

前一个结论：

```text
这些失败是否代表“真正的底层推理能力极限”
```

仍然需要更多实验验证。

---

# 28. 论文的局限

## 28.1 Puzzle ≠ 所有真实世界 Reasoning

Puzzle 主要测试：

```text
Algorithmic
Planning
Exact State Tracking
```

而真实世界问题还包含：

```text
Semantic Reasoning
Knowledge
Common Sense
Creativity
Approximate Reasoning
```

因此不能简单推出：

```text
Puzzle Failure
=
所有 Reasoning Failure
```

---

## 28.2 显式输出与内部能力可能混在一起

例如汉诺塔：

```text
20 disks
```

理论最优步骤：

```text
2^20 - 1
=
1,048,575
```

即使系统“知道怎么做”，要求其：

```text
逐步输出全部动作
```

本身也是一个极大的生成任务。

因此：

```text
不能完整输出
```

与：

```text
不知道算法
```

并非完全等价。

这也是后续批评的重点之一。

---

# 29. 这篇论文真正重要的地方

这篇论文真正值得关注的并不是标题中的：

# Illusion

而是它提出了一种非常重要的研究范式：

```text
不要只问：

Model Accuracy是多少？
```

而应该问：

```text
Accuracy 如何随 Complexity 变化？
```

以及：

```text
Thinking Cost
如何随 Complexity 变化？
```

最终研究：

```text
Accuracy = f(Complexity)

Compute = g(Complexity)

Reasoning Behavior = h(Complexity)
```

这比单独一个：

```text
Benchmark Score = 92%
```

更能够揭示模型真实能力结构。

---

# 30. 工程结论

对于实际开发来说，这篇论文可以总结成六个非常重要的原则。

## 30.1 不要默认 Thinking 越长越好

```text
Long CoT
≠
Better Reasoning
```

---

## 30.2 推理模型最有价值的是中等复杂度任务

```text
Simple
→ 普通模型足够

Medium
→ Reasoning Model

Very High
→ LRM + Tools
```

---

## 30.3 精确计算不要只依赖语言模型

例如：

```text
Math
SQL
Planning
Constraint Solving
```

尽量：

```text
LLM
+
Deterministic Tool
```

---

## 30.4 Agent 必须具备外部状态

不要让几十步任务状态全部依赖：

```text
Context Window
```

而应该：

```text
Memory
Database
State Machine
Workflow Engine
```

保存状态。

---

## 30.5 Reasoning 必须配合 Verification

推荐：

```text
Generator
   ↓
Verifier
   ↓
Correction
```

而不仅是：

```text
Generator
   ↓
Longer Thinking
```

---

## 30.6 应该动态分配推理预算

未来模型系统应从：

```text
Fixed Reasoning Budget
```

发展为：

```text
Adaptive Test-Time Compute
```

---

# 31. 最终总结

这篇论文试图挑战一个正在逐渐形成的直觉：

```text
Thinking More
=
Reasoning Better
```

论文实验提出：

```text
Thinking More
```

只有在一定复杂度范围内才能稳定带来收益。

整个现象可以概括成：

```text
                Problem Complexity
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ↓               ↓               ↓

       Low            Medium           High
        │               │               │
        ↓               ↓               ↓

Standard LLM       Reasoning LLM       Both Fail
更高效             优势最大             能力下降

        │               │               │
        ↓               ↓               ↓

Overthinking      Self-Correction     Collapse
```

因此，论文真正传递的核心思想可以浓缩为一句话：

> **推理模型的关键不是能不能“想得更久”，而是它能否随着问题复杂度增长，把额外计算稳定转化为有效的搜索、状态维护、纠错和算法执行。**

目前的 LRM 已经证明：

```text
Test-Time Compute
```

可以显著提高推理能力。

但这篇论文认为，同样重要的一件事是：

```text
Test-Time Scaling
≠
Unlimited Reasoning Scaling
```

当前 Reasoning Model 更像是：

```text
一个拥有更强搜索能力的语言模型
```

而不是：

```text
一个已经掌握任意复杂算法执行能力的通用计算系统。
```

这也是为什么下一阶段的大模型发展，很可能不会只是：

```text
More Tokens
+
Longer CoT
```

而会越来越走向：

```text
LLM
+
Adaptive Reasoning
+
Search
+
Memory
+
Verifier
+
Code Execution
+
External Tools
```

也就是从：

# “让模型想得更久”

逐渐转向：

# “让模型知道如何计算、如何验证，以及什么时候应该调用工具”。

---

# 32. 参考论文

**Shojaee, P., Mirzadeh, I., Alizadeh, K., Horton, M., Bengio, S., & Farajtabar, M.**

**The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity**

NeurIPS 2025 / arXiv:2506.06941。论文最初于 2025 年 6 月提交，目前 arXiv 页面显示最新为 v3，2025 年 11 月 20 日修订。

Apple 官方研究页面将其核心结果总结为：LRM 在复杂度超过一定阈值后出现准确率崩溃；reasoning effort 会先随复杂度增长、随后下降；在低、中、高三个复杂度区间，普通 LLM 和 LRM 表现出不同的相对优势。
