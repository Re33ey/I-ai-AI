# s1: Simple Test-Time Scaling 技术解析

## 一、背景

近年来，大语言模型的能力提升主要依赖两条路线：

1. **Training-time Scaling**

   * 增加模型参数量
   * 增加训练数据
   * 增加训练 FLOPs
   * 使用 SFT、RLHF、RL 等后训练方法

2. **Test-time Scaling**

   * 模型参数不发生变化
   * 在推理阶段投入更多计算资源
   * 让模型进行更长、更深入或更多次的推理

传统 Scaling Law 主要关注：

```text
更多参数
   +
更多数据
   +
更多训练计算
   ↓
更强模型
```

而 Test-time Scaling 提出了另一种思路：

```text
模型已经训练完成

        ↓

同一个模型
        ↓
推理阶段投入更多计算
        ↓
更多思考 / 搜索 / 验证
        ↓
更高正确率
```

这意味着：

> 模型能力不仅取决于模型有多大，还取决于回答问题时愿意使用多少计算资源。

s1 论文的核心问题正是：

> 能不能用一种非常简单的方法，让普通开源模型获得明显的 Test-time Scaling 能力？

论文 **s1: Simple test-time scaling** 最初于 2025 年 1 月发布，后发表于 EMNLP 2025。作者来自 Stanford、University of Washington、Allen Institute for AI 等机构。([arXiv][1])

---

# 二、目标

s1 并没有试图构建非常复杂的强化学习系统。

它提出了一个非常激进的问题：

> 实现 Reasoning Model，真的需要非常复杂的大规模 RL 吗？

作者希望寻找一个尽可能简单的方案。

最终给出的 Recipe 只有两个核心部分：

```text
s1
│
├── 1. s1K
│      1000 条高质量推理训练数据
│
└── 2. Budget Forcing
       推理阶段控制模型思考长度
```

也就是说：

```text
少量高质量 Reasoning Data
           +
简单 SFT
           +
Test-time Compute Control
           ↓
       Reasoning Model
```

论文使用 **Qwen2.5-32B-Instruct** 作为基础模型，并仅使用 1000 个经过筛选的问题及推理轨迹进行监督微调。([arXiv][1])

---

# 三、什么是 Test-Time Scaling

## 3.1 基本概念

Test-Time Scaling，也可以理解为：

> 推理时计算扩展。

传统模型通常是：

```text
Question
   ↓
Model
   ↓
Answer
```

整个推理过程基本是一次生成。

而 Test-Time Scaling 允许：

```text
Question
   ↓
Thinking
   ↓
Check
   ↓
Re-think
   ↓
Verify
   ↓
Alternative Solution
   ↓
Final Answer
```

因此模型可能使用：

```text
500 tokens
1000 tokens
3000 tokens
10000 tokens
```

完成同一个问题。

---

## 3.2 为什么更多计算可能提高正确率

例如一道数学题：

```text
x² - 5x + 6 = 0
```

模型第一次推理：

```text
x = 1, 6
```

如果直接输出：

```text
Answer = 1, 6
```

就是错误答案。

但如果继续思考：

```text
Wait.

检查：

(x - 2)(x - 3)
= x² - 5x + 6

所以：

x = 2, 3
```

答案就被修正了。

这就是 Test-Time Scaling 的核心直觉：

```text
更多推理
   ↓
更多检查
   ↓
更多纠错机会
   ↓
更高正确率
```

当然：

> 更多 token 并不必然意味着更聪明。

如果模型本身不会推理，让它重复思考 100 次也可能只是重复错误。

因此真正的问题是：

```text
模型是否学习到了
「继续思考 → 检查 → 修正」
这种行为模式？
```

s1 的答案是：

**可以通过高质量 Reasoning SFT 教会模型这种模式。**

---

# 四、s1 整体架构

s1 可以理解为两个阶段。

```text
                 s1
                  │
        ┌─────────┴─────────┐
        │                   │
     Training            Inference
        │                   │
      s1K              Budget Forcing
        │                   │
        ↓                   ↓
Qwen2.5-32B-Instruct    控制 Thinking Tokens
        │                   │
        └─────────┬─────────┘
                  ↓
               s1-32B
```

训练阶段解决：

> 怎么让模型学习复杂推理模式？

推理阶段解决：

> 怎么控制模型使用多少计算资源？

---

# 五、s1K：只有 1000 条训练数据

s1 一个非常值得关注的地方是：

> Reasoning SFT 数据只有约 1000 条。

数据集被称为：

```text
s1K
```

这里：

```text
K ≈ 1000
```

作者从更大的候选数据集中进行筛选。

s1K 的选择重点是三个指标：

```text
Difficulty
Diversity
Quality
```

即：

```text
难度
多样性
质量
```

论文通过消融实验强调，这三个因素对于少量数据获得良好推理能力非常重要。([arXiv][1])

---

# 六、Difficulty：为什么数据必须够难

如果 SFT 数据都是：

```text
1 + 1 = ?
2 + 3 = ?
```

模型学不到真正复杂的 reasoning pattern。

作者更倾向选择：

```text
复杂数学
科学问题
逻辑推理
竞赛问题
```

其核心原则是：

```text
Easy Example

Question
   ↓
直接得到 Answer


Hard Example

Question
   ↓
分析
   ↓
假设
   ↓
推导
   ↓
发现问题
   ↓
修正
   ↓
验证
   ↓
Answer
```

后者包含更多有价值的：

```text
Reasoning Trajectory
```

也就是：

> 推理轨迹。

---

# 七、Diversity：为什么不能全是数学题

假设 1000 条数据全都是：

```text
二次方程
```

模型很容易学习成：

```text
Quadratic Equation Solver
```

而不是通用 Reasoning Model。

因此需要覆盖：

```text
Math

Science

Logic

Physics

Chemistry

Reasoning

Puzzle
```

从而让模型学习更加通用的：

```text
Problem Solving Pattern
```

而不是单一题型模板。

---

# 八、Quality：推理轨迹比答案更重要

Reasoning Model 的训练数据通常类似：

```text
<Question>

...

<Reasoning>

Step 1
Step 2
Step 3
...
Step N

<Answer>
```

真正重要的不是：

```text
Answer
```

而是：

```text
Reasoning Trace
```

例如：

```text
Question
   ↓
理解问题
   ↓
拆解问题
   ↓
选择方法
   ↓
计算
   ↓
验证
   ↓
纠错
   ↓
Answer
```

SFT 实际上是在学习：

```text
P(reasoning, answer | question)
```

而不是单纯：

```text
P(answer | question)
```

因此数据质量非常重要。

---

# 九、核心创新：Budget Forcing

整个 s1 最有意思的技术其实不是 SFT。

而是：

# Budget Forcing

它解决的问题是：

> 如何控制模型到底思考多久？

---

# 十、正常 Reasoning Model 的问题

正常生成：

```text
Question
   ↓
Reasoning
   ↓
Model decides to stop
   ↓
Answer
```

什么时候停止由模型决定。

例如：

```text
Thinking tokens = 1200

EOS
```

模型输出：

```text
EOS
```

代表：

```text
我推理完了。
```

但问题在于：

> 模型可能结束得太早。

例如：

```text
Question

↓

Reasoning

↓

得到答案 A

↓

EOS
```

但如果继续：

```text
Wait...
```

可能会：

```text
重新检查
    ↓
发现错误
    ↓
答案 A → 答案 B
```

---

# 十一、Budget Forcing 的核心思想

s1 做了一件非常简单甚至有点“粗暴”的事情。

当模型准备结束推理时：

```text
Model:

...
Therefore answer = 42

EOS
```

系统不允许它结束。

而是在上下文里加入类似：

```text
Wait
```

让模型继续生成。

于是：

```text
Model:

Therefore answer = 42.

Wait.

Let me verify this calculation...

...
```

模型可能开始：

```text
验证

重新计算

检查边界

重新审视假设
```

论文将这种通过追加 **“Wait”** 继续延长推理、或者在预算耗尽时强制终止推理的方法称为 **Budget Forcing**。([arXiv][1])

---

# 十二、Budget Forcing 的完整流程

可以表示为：

```text
Question
   ↓
Model Thinking
   ↓
Reasoning
   ↓
模型准备结束
   ↓
是否达到 Thinking Budget？
   │
   ├── Yes
   │     ↓
   │   输出答案
   │
   └── No
         ↓
      append "Wait"
         ↓
      继续 Thinking
         ↓
      再次检查
```

循环：

```text
while thinking_tokens < budget:

    reasoning = model.generate()

    if model wants to stop:
        append("Wait")
```

最终：

```text
达到预算
   ↓
停止 Thinking
   ↓
要求 Final Answer
```

---

# 十三、Budget 到底是什么

这里的 Budget 通常指：

```text
Thinking Token Budget
```

例如：

```text
Budget = 512 tokens
Budget = 1024 tokens
Budget = 2048 tokens
Budget = 4096 tokens
```

可以理解成：

> 给模型多少“思考经费”。

例如：

```text
Question

Thinking Budget = 512

Question
   ↓
Thinking 512 tokens
   ↓
Answer
```

如果提高：

```text
Thinking Budget = 4096
```

模型就有更多机会：

```text
推导
检查
回溯
纠错
```

---

# 十四、为什么一个 “Wait” 有用

这是 s1 非常有意思的一点。

模型在大量 Reasoning 数据中会学到类似语言模式：

```text
Wait

But...

Let's reconsider...

Actually...

Let me verify...

Another way...
```

这些 token 往往意味着：

```text
重新思考
```

因此：

```text
Wait
```

类似于给模型一个：

```text
Reasoning Trigger
```

即：

```text
你先别回答。

继续检查一下。
```

在很多情况下，模型随后会主动进行二次验证。论文明确指出，这种延长推理的方式有时能够让模型修正此前错误的推理步骤。([arXiv][1])

---

# 十五、一个完整示例

假设：

```text
Question:

一个商品原价 100，
先涨价 20%，
再降价 20%，
最终多少钱？
```

第一次 reasoning：

```text
涨20%
再降20%

20% - 20% = 0%

所以还是100。
```

模型准备结束。

系统：

```text
Wait
```

模型继续：

```text
Wait.

涨价以后价格：

100 × 1.2 = 120

然后下降20%：

120 × 0.8 = 96

所以最终是96。
```

于是：

```text
第一次答案：
100 ❌

继续推理：
96 ✅
```

这正体现：

```text
Test-Time Compute
       ↓
Self Correction
       ↓
Higher Accuracy
```

---

# 十六、Test-Time Scaling Curve

理想情况下：

```text
Accuracy
  ↑
  │
  │              ●
  │           ●
  │        ●
  │     ●
  │  ●
  │
  └──────────────────→
          Compute
```

即：

```text
Thinking Tokens ↑

Accuracy ↑
```

论文报告，在 AIME24 上，通过 Budget Forcing 增加测试时计算后，s1-32B 的准确率可以从约 **50% 提升到 57%**。([arXiv][1])

这就是所谓：

```text
Test-Time Scaling
```

---

# 十七、s1 与普通 Chain-of-Thought 的区别

普通 CoT：

```text
Question
   ↓
Think step by step
   ↓
Reasoning
   ↓
Answer
```

模型自己决定：

```text
思考多久
```

而 s1：

```text
Question
   ↓
Reasoning
   ↓
Budget Controller
   ↓
继续思考
   ↓
Reasoning
   ↓
Answer
```

主要区别：

| 技术                | 是否控制推理计算量 |
| ----------------- | --------- |
| 普通 CoT            | 否         |
| s1 Budget Forcing | 是         |

所以：

```text
CoT

解决：
怎么让模型展开推理


Budget Forcing

解决：
让模型展开多少推理
```

---

# 十八、与 Self-Consistency 的区别

Self-Consistency 通常是：

```text
Question
   │
   ├── Reasoning 1 → Answer A
   ├── Reasoning 2 → Answer B
   ├── Reasoning 3 → Answer A
   ├── Reasoning 4 → Answer A
   └── Reasoning 5 → Answer C

               ↓

          Majority Vote

               ↓

             A
```

这属于：

```text
Parallel Scaling
```

即：

> 同时生成很多条推理路径。

而 s1 的 Budget Forcing 更偏：

```text
Sequential Scaling
```

也就是：

```text
一条 Reasoning

不断继续
    ↓
继续
    ↓
继续
    ↓
修正
```

因此：

```text
Self-Consistency

横向增加推理路径


s1

纵向增加单条推理深度
```

---

# 十九、与 Best-of-N 的区别

Best-of-N：

```text
Question
   ↓
生成 N 个 Answer
   ↓
Reward Model
   ↓
选最好一个
```

例如：

```text
Response 1 → score 0.3

Response 2 → score 0.9

Response 3 → score 0.6

↓

选择 Response 2
```

计算主要花在：

```text
更多 Sampling
```

而 Budget Forcing：

```text
Single Response
     ↓
Longer Reasoning
```

所以：

```text
Best-of-N
= Width Scaling


Budget Forcing
= Depth Scaling
```

---

# 二十、Width Scaling 与 Depth Scaling

Test-time Scaling 可以粗略分为：

```text
                 Test-Time Scaling

                       │

         ┌─────────────┴─────────────┐

         │                           │

   Depth Scaling                Width Scaling

         │                           │

 Longer Reasoning              More Samples

 Budget Forcing                Best-of-N

 Sequential Search             Majority Voting
```

这是理解 Reasoning Model 非常重要的一个框架。

---

# 二十一、进一步组合

单纯不断增加 Thinking Token 存在问题：

```text
Question
   ↓
500 tokens
   ↓
1000 tokens
   ↓
5000 tokens
   ↓
10000 tokens
   ↓
Context Window
   ↓
到达极限
```

而且可能出现：

```text
Overthinking
```

例如：

```text
正确答案
   ↓
继续推理
   ↓
开始怀疑
   ↓
错误修改
```

因此论文还讨论：

```text
Sequential Scaling
        +
Parallel Scaling
```

例如：

```text
Question
   │
   ├── Long Reasoning 1
   ├── Long Reasoning 2
   ├── Long Reasoning 3
   └── Long Reasoning 4
             ↓
          Voting
             ↓
          Answer
```

s1 项目页明确指出，当单条 sequential reasoning 太长、受到上下文窗口限制时，可以把 Budget Forcing 和 parallel scaling 结合起来继续扩展计算。([Simple Scaling][2])

---

# 二十二、s1 与 PPO / GRPO 的根本区别

这里非常容易混淆。

DeepSeek-R1 一类方法主要关注：

```text
Training-Time Reasoning Scaling
```

比如：

```text
Question
   ↓
Policy Model
   ↓
Response
   ↓
Reward
   ↓
RL
   ↓
更新参数
```

GRPO：

```text
一个 Question
   ↓
生成多个 Response
   ↓
Reward
   ↓
Group Relative Advantage
   ↓
更新 Policy
```

核心是：

```text
训练模型
```

而 s1 的核心方案非常不同：

```text
SFT
+
Budget Forcing
```

Budget Forcing 发生于：

```text
Inference Time
```

不需要因为每一道测试题再次更新模型参数。

所以：

```text
GRPO

优化的是：
Model Parameters


Budget Forcing

优化的是：
Inference Compute
```

---

# 二十三、训练阶段 vs 推理阶段

可以非常直观地理解。

## RL

```text
训练阶段：

Question
   ↓
Reasoning
   ↓
Reward
   ↓
Gradient
   ↓
Update Parameters
```

结果：

```text
模型本身变强
```

---

## Test-Time Scaling

```text
推理阶段：

Question
   ↓
Model
   ↓
Thinking
   ↓
更多 Compute
   ↓
Better Answer
```

结果：

```text
参数没变

但这一次回答更加认真
```

---

# 二十四、s1 的工程实现

假设使用类似 Chat Template：

```text
<|im_start|>user

question

<|im_end|>

<|im_start|>assistant

<|think|>

reasoning

</think>

answer
```

普通推理：

```python
output = model.generate(
    prompt,
    max_new_tokens=4096
)
```

Budget Forcing 的伪代码则可以理解为：

```python
reasoning = ""

while token_count(reasoning) < thinking_budget:

    output = model.generate(
        prompt + reasoning
    )

    if output.wants_to_stop():

        reasoning += output

        reasoning += "\nWait\n"

    else:

        reasoning += output
```

达到：

```text
thinking_budget
```

之后结束 reasoning。

然后生成：

```text
Final Answer
```

真实实现会涉及 stop token、token budget 和推理框架等细节，但核心思想就是如此。

官方 s1 GitHub 仓库同时提供了普通 vLLM 推理以及带 Budget Forcing 的 vLLM 推理实现。([GitHub][3])

---

# 二十五、为什么 s1 很重要

s1 真正重要的地方并不是：

```text
Wait
```

这个 token 有多神奇。

真正重要的是它证明：

```text
Reasoning Capability
```

不一定完全依赖：

```text
超大规模 RL
```

一种极简路线也可以获得很强的 reasoning：

```text
High-quality Data
        +
SFT
        +
Test-Time Scaling
```

也就是：

```text
训练阶段
学会怎么思考

+

推理阶段
决定思考多久
```

---

# 二十六、一个更深层的理解

传统模型架构可以理解：

```text
Model

Question
   ↓
Fixed Compute
   ↓
Answer
```

而 Reasoning Model 正逐渐变成：

```text
Question
   ↓
Difficulty Estimation
   ↓
Compute Allocation
   ↓
Reasoning
   ↓
Verification
   ↓
Answer
```

简单问题：

```text
1 + 1 = ?

↓

几十 tokens
```

复杂问题：

```text
证明一道数学定理

↓

几千甚至几万 tokens
```

未来模型可能动态决定：

```text
这个问题值得花多少 Compute？
```

这就是：

```text
Adaptive Test-Time Compute
```

---

# 二十七、为什么 Test-Time Scaling 是 LLM 的重要方向

传统 Scaling：

```text
GPT-3
↓
更大模型
↓
GPT-4
↓
更大训练 Compute
```

成本发生在：

```text
Pretraining
```

而 Test-Time Scaling：

```text
同一个 Model

Easy Question
↓
少量 Compute

Hard Question
↓
大量 Compute
```

意味着计算资源可以：

```text
按需分配
```

理论上更加灵活。

---

# 二十八、s1 的局限性

s1 虽然简单，但并不代表：

```text
thinking tokens 越多越好
```

它存在几个明显问题。

### 1. Overthinking

模型可能：

```text
第一次已经正确
    ↓
继续思考
    ↓
开始怀疑
    ↓
修改为错误答案
```

---

### 2. Token Cost

如果：

```text
普通回答 = 500 tokens

Reasoning = 10000 tokens
```

推理成本可能提高：

```text
20 倍
```

---

### 3. Latency

更多 reasoning：

```text
TTFT
+
Generation Time
```

都会增加。

例如：

```text
500 tokens

可能数秒

10000 tokens

可能几十秒甚至更久
```

---

### 4. Context Window

长 reasoning 可能达到：

```text
Context Limit
```

例如：

```text
32K
64K
128K
```

---

### 5. 不一定真正“思考”

一个很重要的问题：

```text
Longer Reasoning
≠
Better Reasoning
```

模型可能只是：

```text
重复
啰嗦
绕圈
```

因此真正高效的 Test-Time Scaling 需要解决：

```text
Compute Allocation
```

即：

> 每增加一个 token，到底有没有增加有效推理？

---

# 二十九、s1 的实验意义

论文在 MATH500、AIME24 和 GPQA Diamond 等推理任务上评估模型。官方项目页将 s1 描述为基于 1000 条数据和 Budget Forcing 获得强推理能力的极简方案。([Simple Scaling][2])

论文报告 s1-32B 在部分竞赛数学设置中达到或超过当时的 o1-preview 表现，并称在 MATH/AIME24 上最高存在约 27% 的优势。这个结论需要结合论文具体 benchmark、评分设置和模型版本理解，而不能泛化成“s1 全面强于 o1-preview”。([arXiv][1])

---

# 三十、s1.1

官方项目后来又发布了：

```text
s1.1
```

其思路并没有改变：

```text
仍然使用同一批 s1K 问题
```

但更换了 reasoning trace 来源。

官方仓库说明，s1.1 使用 DeepSeek-R1 生成的 reasoning traces 替代了早期版本中的 Gemini reasoning traces，并基于这些数据训练新的 s1.1-32B。([GitHub][3])

这实际上进一步说明：

```text
Question Quality
       +
Reasoning Trace Quality
```

对 Reasoning SFT 的效果非常关键。

---

# 三十一、与现代 Reasoning Model 的统一理解

可以把很多推理模型放入下面的框架：

```text
                  Reasoning Model

                       │
          ┌────────────┴────────────┐
          │                         │
    Training Scaling          Test-Time Scaling
          │                         │
     SFT / RL / GRPO          Longer Reasoning
          │                         │
     学会如何推理                使用更多推理
          │                         │
          └────────────┬────────────┘
                       ↓
                  Better Answer
```

例如：

```text
SFT
→ 教模型推理格式和模式

RL / GRPO
→ 强化有效推理行为

Budget Forcing
→ 控制推理长度

Best-of-N
→ 增加候选答案

Self-Consistency
→ 多路径投票

Verifier
→ 验证候选答案

Search
→ 探索推理空间
```

这些技术本质上并不冲突。

未来系统完全可能组合成：

```text
Reasoning SFT
      ↓
Reasoning RL
      ↓
Dynamic Thinking Budget
      ↓
Parallel Sampling
      ↓
Verifier
      ↓
Search
      ↓
Final Answer
```

---

# 三十二、核心总结

如果只记住 s1 的三个关键词：

```text
s1
│
├── 1000
│
├── SFT
│
└── Budget Forcing
```

第一：

> **1000 条精选推理数据。**

强调：

```text
Difficulty
Diversity
Quality
```

第二：

> **用 SFT 让普通模型学习 Reasoning Trace。**

基础模型：

```text
Qwen2.5-32B-Instruct
```

经过：

```text
s1K SFT
```

得到：

```text
s1-32B
```

第三，也是最核心的：

> **Budget Forcing 控制 Test-Time Compute。**

流程：

```text
Question
   ↓
Reasoning
   ↓
准备结束
   ↓
"Wait"
   ↓
继续 Reasoning
   ↓
检查 / 修正
   ↓
Answer
```

最终形成：

```text
Thinking Budget ↑
        ↓
Test-Time Compute ↑
        ↓
Reasoning Depth ↑
        ↓
部分任务 Accuracy ↑
```

---

# 三十三、一句话理解 s1

可以把 s1 理解成：

> **先用少量高质量数据教会模型“怎么思考”，再在推理阶段通过 Budget Forcing 控制模型“思考多久”。**

它揭示出的核心思想甚至比具体方法本身更重要：

```text
LLM Scaling
```

已经不仅仅是：

```text
Train Bigger Models
```

还可以是：

```text
Let Models Compute Longer
```

即：

> **Scaling Training Compute → Scaling Inference Compute。**

而这正是现代 Reasoning Model 技术路线中最值得关注的变化之一。


[1]: https://arxiv.org/abs/2501.19393?utm_source=chatgpt.com "s1: Simple test-time scaling"
[2]: https://simplescaling.github.io/?utm_source=chatgpt.com "s1: Simple test-time scaling"
[3]: https://github.com/simplescaling/s1/blob/main/README.md?utm_source=chatgpt.com "s1/README.md at main · simplescaling/s1 · GitHub"
