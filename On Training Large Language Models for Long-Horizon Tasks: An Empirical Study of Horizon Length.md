## 一、核心思想

假设让一个 AI Agent 完成下面的任务：

> 从业务需求出发，找到相关代码、修改接口、调整数据库、编写单元测试、执行测试、修复错误，最后提交代码。

Agent 可能已经知道每一步怎么做，但只要中间一步出错，就可能导致最终任务失败。

论文研究的不是模型能否理解这个需求，而是：

为什么模型明明会执行每个步骤，却很难稳定地连续执行几十个步骤？

传统原子动作

1

2

3

4

5

6

7

8

8 次连续决策

Horizon Reduction：宏动作抽象

宏动作 A

封装步骤 1–4

宏动作 B

封装步骤 5–8

示意：8 次决策压缩成 2 次高层决策

这里最关键的区别是：任务并没有消失，底层操作也没有必然减少，但模型需要亲自做出的高层决策次数减少了。

这正是论文提出的 Horizon Reduction（任务视野长度缩减）思想。

## 二、研究背景：什么是 Long-Horizon Task？

Long-Horizon Task 通常翻译为“长时序任务”或“长链路任务”。

它不是指模型一次生成特别长的回答，也不等于上下文窗口很大，而是指模型需要经过多轮决策和环境交互，才能完成最终目标。

例如：

| 任务        | 典型交互特点        |
| --------- | ------------- |
| 普通问答      | 一次回答即可完成      |
| 数学推理      | 可能需要多步内部推理    |
| 代码 Agent  | 多次搜索、修改、编译和测试 |
| 浏览器 Agent | 多轮导航、点击、输入和验证 |
| 自动化业务流程   | 跨接口、跨系统进行连续决策 |

这里必须区分三个概念。

### 2.1 三种 Horizon 的严格定义

论文定义了三种不同的 Horizon。

1\. Goal Distance：目标距离

用 \\(d(s_0,g)\\) 表示。

指在初始状态 \\(s_0\\) 下，通过最优策略到达目标 \\(g\\) 所需要的最少原子动作数量。

例如，某个数独有 20 个需要填写的空格，每次只允许填写一个格子，那么目标距离就是 20。

2\. Interaction Budget：交互预算

用 \\(H\_{\max}\\) 表示。

指环境允许 Agent 执行的最大动作次数。

例如，一个 Agent 最多允许调用工具 50 次，这属于交互预算，而不是任务自身的目标距离。

3\. Effective Horizon：有效执行长度

用 \\(h\_\pi(s_0,g)\\) 表示。

指模型按照策略 \\(\pi\\) 实际完成任务所使用的决策步数。

对成功轨迹而言：

\\[ d(s_0,g)\leq h\_\pi(s_0,g)\leq H\_{\max} \\]

需要注意，这个不等式使用的是相同的原子动作定义。如果引入宏动作，就要区分底层原子动作数量与高层决策次数。

### 2.2 三者的区别

假设一个任务理论上只需要执行 10 步，但模型执行过程中出现了反复尝试。

| 指标                 | 示例值 | 说明              |
| ------------------ | --- | --------------- |
| Goal Distance      | 10  | 理论最少需要 10 个原子动作 |
| Interaction Budget | 30  | 最多允许执行 30 步     |
| Effective Horizon  | 18  | 模型实际执行了 18 步才成功 |

传统研究经常通过修改 \\(H\_{\max}\\) 来研究长任务，但这篇论文重点关注的是任务自身所需的最少动作长度，以及如何降低模型实际需要面对的决策长度。

这是一个非常重要的研究视角转变。

## 三、为什么长任务容易失败？

论文将核心困难归纳为两个方面：探索困难（Exploration Difficulty）与信用分配困难（Credit Assignment）。

### 3.1 探索困难：成功路径的概率快速下降

假设模型每一步做出正确选择的概率都是 \\(p\\)，且为简化分析，暂时假设各步成功概率独立。

那么连续 \\(T\\) 步都正确的概率为：

\\[ P(\text{Success})=p^T \\]

交互模拟：单步准确率与任务成功率

95%

80%

90%

100%

5 步任务

# 77.4%

理论成功率

20 步任务

# 35.8%

理论成功率

50 步任务

# 7.7%

理论成功率

任务成功率

0%25%50%75%100%5101520253035404550

说明：该公式用于直观展示误差累积，并非论文训练实验的实际成功率。

以上是独立同分布假设下的理论模型；真实 Agent 的动作误差通常相互依赖。

这个简单模型说明：

即使单步准确率很高，随着任务链路越来越长，端到端成功率也可能迅速下降。

在真实任务中，困难还不止于此。一个错误动作可能修改环境状态，让后面的决策更难，从而进一步影响成功概率。

这也意味着，单纯提升模型某一道题的正确率，并不能保证 Agent 长时间执行任务的可靠性。

### 3.2 信用分配困难：最终失败，到底是哪一步错了？

假设某个 Agent 执行了 20 个动作，其中前 19 个正确，最后一个错误。

如果采用只有最终成功或失败的稀疏奖励机制：

\\[ R(\tau)= \begin{cases} 1,&\text{任务成功}\\\ 0,&\text{任务失败} \end{cases} \\]

那么整条轨迹可能只获得失败反馈。

这会带来一个问题：

训练算法很难知道究竟应该惩罚哪一步，而不应该惩罚哪一步。

论文使用的基础策略梯度思想可以写成：

\\[ \nabla\_\theta J(\theta)= \mathbb E\_{\tau\sim\pi\_\theta} \left[ \sum\_{t=0}^{T-1} A_t\nabla\_\theta\log\pi\_\theta(a_t\mid s_t) \right] \\]

其中：

- \\(\pi\_\theta\\)：模型策略
- \\(a_t\\)：第 \\(t\\) 步动作
- \\(s_t\\)：当前环境状态
- \\(A_t\\)：第 \\(t\\) 步优势估计（Advantage）
- \\(J(\theta)\\)：期望收益目标

如果一条失败轨迹中的多个步骤都被分配负优势，即使某些中间动作完全正确，它们也可能受到错误的负向更新。

作者进一步分析了 Token 级梯度：当某个已采样 Token 获得负优势时，优化会降低它的概率，并将概率质量重新分配给其他 Token，其中可能包含大量无关选项。

随着长任务中错误负反馈不断积累，策略可能逐渐偏离原本有效的行为。

这里需要区分论文的实测与解释： 长 Horizon 下的训练性能崩溃以及输出长度异常增长是实验观察；负优势更新累积导致策略失稳，是作者结合梯度机制提出的解释，并非已经排除其他因素的唯一因果结论。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

+1



## 四、论文如何设计实验？

这篇论文较有价值的部分，是它没有直接拿一个复杂的代码 Agent 测成功率，而是专门构造了能够控制任务长度的实验环境。

因为如果直接比较两个不同难度的任务：

- 任务 A：执行 5 步
- 任务 B：执行 50 步

任务 B 失败，可能是因为执行步数多，也可能是因为题目本身更难。

作者希望排除后一种干扰。

### 4.1 实验环境一：Sudoku（数独）

[Program to Check (or Solve) Sudoku Puzzles | Science Project](https://images.openai.com/static-rsc-4/PWFbkq-3UgA7U778fl9oS4qRLSu2keoWG6TCYXRoK7-CZAaiWlzBvutYGGmlwaRJhdFKG416NIKLITWjlWruvd5AE524lJX2Y_kMjNaYIgDafp-xHBasoA6Fe3qJo2poHE7Lj4xQOTR6ky7JUa-3Za8avfMktW7nXel57iVz8Uc?purpose=inline)

[THINKFUN Rush Hour | Puzzle-puzzle.cz](https://images.openai.com/static-rsc-4/dakZzFP6d5o_K4v-GlrktIOHTEmS89X-f65iCCH-MYUxR1B8_skzrS9OzyRZGykpaEvQzELopAXjmUXP3PLbd-8Od_GfN-Ie_mCOmmaqIn8LQM88g0nJE_1f7Ok2gno1AjFb8IBiYd3p-quNtU7Qwzvi6TKIArJ4-Yq0U1EuoF0?purpose=inline)

[Screenshot from Rush Hour (1996). | Download Scientific Diagram](https://images.openai.com/static-rsc-4/ZUWygno7q_amE7tm3mTGn8fxdLf7Yd3KRiJYwuWxcWLhFAZAMh9Hsulemq8X8ORjvt_EsKpgkpAtbg_hwCI6TUfMiiD5OXL5IwSvQAQG_HNRkYdStfP4OBqd0xhW_KDpbMLVnzfhBGoRZhBRNYPTu6Oq1ks3XFYTn3XxizNqyAE?purpose=inline)

5

数独是论文的主要实验环境。

作者让 Agent 每次填写一个格子，通过改变待填写的空格数量控制目标距离。

同时，使用 HoDoKu 求解器筛选只需要基础技巧即可解决的数独，并通过单步完整求解形式验证模型的解题能力。

这样可以减少“推理难度随任务步数增加”造成的混淆。

数独数据集按照目标距离划分为七个等级：

| 难度等级 | Goal Distance | 用途       |
| ---- | ------------- | -------- |
| L1   | 11–15         | 训练与测试    |
| L2   | 16–20         | 训练与测试    |
| L3   | 21–25         | 训练与测试    |
| L4   | 26–30         | 训练与测试    |
| L5   | 31–35         | 未见长度泛化测试 |
| L6   | 36–40         | 未见长度泛化测试 |
| L7   | 41–45         | 未见长度泛化测试 |

每个训练等级包含 640 个样本；L1–L6 各有 100 个测试样本，L7 有 50 个。

这里的 L1–L7 不是通用的数学难度分级，而是作者为本研究构造的动作长度分级。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



### 4.2 实验环境二：Rush Hour

Rush Hour 是一种滑块停车解谜游戏。

与数独不同，它主要考察空间操作和规划能力。作者以最优解所需的最少移动次数衡量目标距离。

加入这个实验的意义是：验证训练现象并非仅由数独的任务规则导致。

### 4.3 实验环境三：WebShop

WebShop 模拟基于自然语言购物需求的网页交互。

与两个游戏环境相比，它更加接近真实的 Web Agent，包括页面观察、商品搜索与多步决策。

论文还在这一环境中测试了 Horizon Reduction 的有效性，但这不意味着已经完整覆盖真实企业网站中的登录、权限、动态页面和各种外部系统异常。

### 4.4 训练配置

论文的主要实验采用以下配置：

| 项目      | 配置                              |
| ------- | ------------------------------- |
| 基础模型    | Qwen3-1.7B                      |
| 初始训练    | 专家轨迹 SFT                        |
| 专家来源    | 包括 GPT-5-mini 等较强模型             |
| 强化学习    | 基于 REINFORCE 的优化实现              |
| RL 训练轮数 | 4 Epoch                         |
| 训练及推理温度 | 0.8                             |
| 评测      | 每道题采样 4 条轨迹，统计 pass\@K / avg\@K |
| 扩展验证    | 4B 模型、GRPO-style 优化器、WebShop    |

这些设置让研究能够在相对可控的条件下分析任务长度，而不是将模型参数量、算法复杂度、任务领域等全部混在一起。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

+1



## 五、核心实验结果：长链路会使强化学习训练崩溃

论文 Figure 3 是理解结论的重要图表。

[DAIR.AI (@dair_ai) on X](https://images.openai.com/static-rsc-4/EPPOrkiGmxbyleyrZe4QXivLLJqRDp4vcpztj3rqAz6JAUnY9tHgHwaWSD42mQSWxZGFZiZy-nu3w43v0IpIrFiXh9T-AYvPgvrsVV-zq6ifq2rhSWWqG5oJ8Q5xKpkmvO74RTiQH3Zqi7ZVSXav2DsUUUaS7sM5tnJILdaiIeU?purpose=inline)

[x.com](https://x.com/dair_ai/status/2053495683835973684)

Figure 3：原子动作与宏动作在数独、Rush Hour 上的训练和测试表现。建议结合原论文第 5 页查看完整曲线。

原始实验展示出一种值得注意的现象：

在较短的 L1–L2 任务中，使用原子动作进行强化学习能够持续改善表现。

但当训练任务增长到 L3–L4 时，原子动作策略出现严重不稳定，部分训练曲线先上升，随后发生明显崩溃。

相比之下，使用宏动作的训练能够维持更稳定的表现，并得到更好的最终成功率。

这说明：RL 优化器可能没有变，任务推理规则也没有发生实质变化，单纯的交互长度增长就足以使训练行为显著恶化。&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



论文还通过更进一步的消融实验排除了一个可能的解释：宏动作表现好，只是因为模型本身更擅长这种表达方式。

作者使用同一个宏动作策略，但限制每轮只能执行一个原子操作，人为重新拉长交互链路。结果训练又出现性能崩溃。

这为“有效执行长度本身是关键因素”的判断提供了更直接的证据。

## 六、核心方法：Horizon Reduction

论文并没有提出一个复杂的新型强化学习优化器，而是从任务表示与交互结构入手，提出两条主要路径。

### 6.1 方法一：Macro Actions（宏动作）

宏动作就是把多个底层原子动作封装成一个更高层的动作。

例如，原始数独需要逐格操作：

经过宏动作抽象，可以允许 Agent 在一个决策回合里提交多个填写操作。

对应到软件工程场景：

原子动作设计：

```
1. 打开文件 A
2. 搜索方法 B
3. 编辑方法 B
4. 打开文件 C
5. 修改测试用例
6. 执行测试
```

宏动作设计：

```
1. update_code_and_tests(change_spec)
2. run_targeted_tests()
```

宏动作不是让系统神奇地省掉编辑和测试，而是把一部分确定性的操作序列交给高级工具执行，减少模型逐步控制的次数。

#### 宏动作长度并不是越大越好

论文还比较了三种策略：

| 动作方式           | 行为特点           | 实验发现      |
| -------------- | -------------- | --------- |
| Atomic Action  | 每轮只执行一个原子动作    | 长任务中易失稳   |
| Fixed Macro    | 每轮必须执行固定数量的动作  | 僵硬，容易过度执行 |
| Flexible Macro | 模型可以决定一轮执行多少动作 | 整体效果更好    |

作者在 GPT-5-mini、Gemini-3-Flash 的数独实验中发现，灵活宏动作优于固定长度宏动作。

因此，真正有效的设计不是无条件要求模型一次执行更多步骤，而是允许模型根据任务状态动态决定动作粒度。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



### 6.2 方法二：Subgoal Decomposition（子目标分解）

宏动作主要通过减少高层决策次数缩短链路。

子目标分解则通过把一条长任务拆为多个可验证的短阶段，缩短需要进行信用分配的范围。

例如：

最终目标：完成软件功能开发

01

设计接口

独立验证与反馈

02

实现逻辑

独立验证与反馈

03

完成测试

独立验证与反馈

04

部署验证

独立验证与反馈

工程化示意，并非论文实际采用的软件开发测试流程。

对于一个完整目标 \\(g\\)，可以将它拆成：

\\[ g\rightarrow(g_1,g_2,\ldots,g_k) \\]

每个子目标都能够单独提供反馈。

论文具体在数独中采用宫格完成作为可验证子目标，给完成正确宫格的行为提供中间奖励，并将轨迹按子目标完成节点切分，分别计算回报。

由此产生的效果是：

原先一条长轨迹上的稀疏反馈，转变为多个相对较短、拥有局部反馈的训练片段。

这有助于缓解信用分配问题。

不过需要明确：子目标分解并不必然减少端到端的原子操作总数，真正缩短的是训练中需要归因的决策跨度。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



## 七、Horizon Generalization：短任务训练，为什么能泛化到长任务？

这是我认为整篇论文最值得关注的发现之一。

传统直觉是：想让模型完成 50 步任务，就应该尽量用 50 步甚至更长的任务训练它。

论文却发现，经过合理训练的较短任务策略，可以泛化到训练阶段未见过的更长任务。

作者称之为 Horizon Generalization（跨任务长度泛化）。

### 7.1 背后的机制

作者提出两个重要的解释。

第一，经过稳定训练后，模型可以获得更高的单步决策准确率。

第二，宏动作能够减少决策节点数量，即使部分情况下宏动作策略的单步准确率更低，也可能凭借更短的有效执行链路取得更高的最终成功率。

例如，考虑下面这个纯理论示意：

| 策略   | 每步成功率 | 决策次数 | 最终成功率   |
| ---- | ----- | ---- | ------- |
| 原子动作 | 98%   | 40   | 约 44.6% |
| 宏动作  | 95%   | 10   | 约 59.9% |

即使宏动作单次决策的可靠性略低，因为决策次数大幅减少，最终整体成功率仍然可能更高。

但必须注意，这个例子假设错误独立且各步准确率固定，并不代表论文实测结果。

### 7.2 Curriculum Learning（课程学习）

论文进一步研究了训练顺序。

对于 Rush Hour 任务，作者比较了三种策略：

- Short-only：只训练目标距离为 4–9 的任务。
- Long-only：直接训练目标距离为 10–12 的任务。
- Curriculum：先训练短任务，再使用所得策略继续训练长任务。

实验发现，直接进行较长任务训练时模型改善有限，而短任务预训练再迁移到较长任务能够获得明显提升。

这表明：先建立稳定的短链路能力，再逐步增加任务长度，可能比直接把模型投入复杂长任务更有效。

这里的泛化仍然需要任务具有相似的决策规则和推理结构，不能理解为学会短数独之后，就能自动完成任意复杂的长任务。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



## 八、论文对 Agent 系统设计有什么启示？

接下来是基于论文研究结果进行的工程化分析，不是作者已经在企业生产环境中验证过的结论。

### 8.1 Agent 工具设计：优先提供高层语义操作

例如在一个客服 Agent 系统中，假设业务目标是查询客户近期风险信息并给出建议。

一种设计是让模型逐步调用多个原子接口：

```
query_customer()
query_sessions()
query_risk_signals()
query_customer_profile()
aggregate_evidence()
generate_suggestion()
```

另一种设计是提供一个高级业务工具：

```
analyze_customer_risk(customer_id)
```

工具内部执行确定性的查询、聚合和部分规则计算，最后把结构化结果交给 Agent。

第二种方式降低了 Agent 的决策次数，也减少了中间调用状态需要由模型自行维护的机会。

但是它也带来新的要求：工具内部必须具备可观测性、超时控制、部分失败处理、权限校验以及必要的幂等机制。

不能只是把十个不可靠的接口简单包装起来，就认为系统整体一定更加可靠。

### 8.2 Agent 工作流：让长任务拥有清晰的检查点

在复杂业务中，推荐将任务拆成可验证的子目标，并对每个阶段记录状态。

一个适用于工程系统的设计是：

\#chatgpt-mermaid-\_r_11m\_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(13, 13, 13);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-\_r_11m\_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-\_r_11m\_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-\_r_11m\_ .error-icon{fill:rgb(243, 243, 243);}#chatgpt-mermaid-\_r_11m\_ .error-text{fill:rgb(13, 13, 13);stroke:rgb(13, 13, 13);}#chatgpt-mermaid-\_r_11m\_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-\_r_11m\_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-\_r_11m\_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-\_r_11m\_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-\_r_11m\_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-\_r_11m\_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-\_r_11m\_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-\_r_11m\_ .marker.cross{stroke:rgb(143, 143, 143);}#chatgpt-mermaid-\_r_11m\_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-\_r_11m\_ p{margin:0;}#chatgpt-mermaid-\_r_11m\_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(13, 13, 13);}#chatgpt-mermaid-\_r_11m\_ .cluster-label text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-\_r_11m\_ .cluster-label span{color:rgb(13, 13, 13);}#chatgpt-mermaid-\_r_11m\_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-\_r_11m\_ .label text,#chatgpt-mermaid-\_r_11m\_ span{fill:rgb(13, 13, 13);color:rgb(13, 13, 13);}#chatgpt-mermaid-\_r_11m\_ .node rect,#chatgpt-mermaid-\_r_11m\_ .node circle,#chatgpt-mermaid-\_r_11m\_ .node ellipse,#chatgpt-mermaid-\_r_11m\_ .node polygon,#chatgpt-mermaid-\_r_11m\_ .node path{fill:rgb(226, 242, 230);stroke:rgb(107, 198, 127);stroke-width:1px;}#chatgpt-mermaid-\_r_11m\_ .rough-node .label text,#chatgpt-mermaid-\_r_11m\_ .node .label text,#chatgpt-mermaid-\_r_11m\_ .image-shape .label,#chatgpt-mermaid-\_r_11m\_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-\_r_11m\_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-\_r_11m\_ .rough-node .label,#chatgpt-mermaid-\_r_11m\_ .node .label,#chatgpt-mermaid-\_r_11m\_ .image-shape .label,#chatgpt-mermaid-\_r_11m\_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-\_r_11m\_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-\_r_11m\_ .root .anchor path{fill:rgb(143, 143, 143)!important;stroke-width:0;stroke:rgb(143, 143, 143);}#chatgpt-mermaid-\_r_11m\_ .arrowheadPath{fill:rgb(143, 143, 143);}#chatgpt-mermaid-\_r_11m\_ .edgePath .path{stroke:rgb(143, 143, 143);stroke-width:1px;}#chatgpt-mermaid-\_r_11m\_ .flowchart-link{stroke:rgb(143, 143, 143);fill:none;}#chatgpt-mermaid-\_r_11m\_ .edgeLabel{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-\_r_11m\_ .edgeLabel p{background-color:rgb(252, 252, 252);}#chatgpt-mermaid-\_r_11m\_ .edgeLabel rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-\_r_11m\_ .labelBkg{background-color:rgba(252, 252, 252, 0.5);}#chatgpt-mermaid-\_r_11m\_ .cluster rect{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-\_r_11m\_ .cluster text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-\_r_11m\_ .cluster span{color:rgb(13, 13, 13);}#chatgpt-mermaid-\_r_11m\_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(243, 243, 243);border:1px solid rgba(0, 0, 0, 0.1);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-\_r_11m\_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(13, 13, 13);}#chatgpt-mermaid-\_r_11m\_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-\_r_11m\_ .icon-shape,#chatgpt-mermaid-\_r_11m\_ .image-shape{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-\_r_11m\_ .icon-shape p,#chatgpt-mermaid-\_r_11m\_ .image-shape p{background-color:rgb(252, 252, 252);padding:2px;}#chatgpt-mermaid-\_r_11m\_ .icon-shape .label rect,#chatgpt-mermaid-\_r_11m\_ .image-shape .label rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-\_r_11m\_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-\_r_11m\_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-\_r_11m\_ .node .neo-node{stroke:rgb(107, 198, 127);}#chatgpt-mermaid-\_r_11m\_ [data-look="neo"].node rect,#chatgpt-mermaid-\_r_11m\_ [data-look="neo"].cluster rect,#chatgpt-mermaid-\_r_11m\_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-\_r_11m\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_11m\_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-\_r_11m\_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-\_r_11m\_-gradient);stroke-width:1px;}#chatgpt-mermaid-\_r_11m\_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_11m\_ [data-look="neo"].node .neo-line path{stroke:rgb(107, 198, 127);filter:none;}#chatgpt-mermaid-\_r_11m\_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-\_r_11m\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_11m\_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-\_r_11m\_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-\_r_11m\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_11m\_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-\_r_11m\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_11m\_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(0, 105, 42);}#chatgpt-mermaid-\_r_11m\_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(0, 105, 42);}#chatgpt-mermaid-\_r_11m\_ .node tspan[font-weight="normal"],#chatgpt-mermaid-\_r_11m\_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-\_r_11m\_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(237, 250, 242);stroke:rgb(166, 211, 184);stroke-width:1px;}#chatgpt-mermaid-\_r_11m\_ .node rect,#chatgpt-mermaid-\_r_11m\_ .node circle,#chatgpt-mermaid-\_r_11m\_ .node ellipse,#chatgpt-mermaid-\_r_11m\_ .node polygon,#chatgpt-mermaid-\_r_11m\_ .node path{fill:rgb(217, 244, 228);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-\_r_11m\_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-\_r_11m\_ .node.mermaid-decision .label-container{fill:rgb(237, 250, 242);stroke:rgb(195, 220, 205);stroke-dasharray:2,2;}#chatgpt-mermaid-\_r_11m\_ .edgePaths .flowchart-link{stroke:rgb(143, 143, 143);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-\_r_11m\_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-\_r_11m\_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}接收用户任务规划子目标执行当前子目标结果是否通过验证?保存检查点局部重试或重新规划全部子目标完成?全局验证并输出结果通过未通过否是

这样可以避免某个中间阶段出错后，必须重新执行整个长流程。

对于无法回滚的外部动作，例如资金操作、消息发送、生产环境变更等，还需要增加操作确认、审计记录与补偿逻辑。

### 8.3 强化学习训练：奖励应该尽量可验证

论文表明，只有任务最终成功或失败的稀疏奖励，会让长轨迹的信用分配更加困难。

可以考虑设计以下奖励信号：

\\[ R = \lambda_1 R\_{\text{success}} +\lambda_2 R\_{\text{subgoal}} +\lambda_3 R\_{\text{valid}} -\lambda_4 C\_{\text{failure}} \\]

其中：

| 奖励项                       | 作用          |
| ------------------------- | ----------- |
| \\(R\_{\text{success}}\\) | 最终任务完成奖励    |
| \\(R\_{\text{subgoal}}\\) | 可验证子目标完成奖励  |
| \\(R\_{\text{valid}}\\)   | 动作格式、约束满足奖励 |
| \\(C\_{\text{failure}}\\) | 无效操作或违规行为代价 |

这是便于工程理解的建议性设计，并非论文的原始奖励公式。

论文实际采用的 REINFORCE 实现会分别处理轨迹级与步骤级反馈，归一化后组合优势项：

\\[ A_t=\hat r_t^{\text{traj}} +\alpha\hat r_t^{\text{step}} \\]

其中 \\(\alpha=0.2\\)。

还通过重要性采样权重缓解轨迹复用和采样策略滞后造成的分布偏移。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



对于具体业务，更重要的是确保子目标奖励真正反映目标进展，避免模型为了获得局部奖励而牺牲整体结果。

## 九、如何建立一套面向 Long-Horizon 的 Agent 评测体系？

如果要把论文思想用于 Agent 评测平台建设，我建议不要只关注最终任务成功率，还应该建立任务长度维度的评测能力。

### 9.1 核心评测指标

| 评测指标                      | 衡量内容          |
| ------------------------- | ------------- |
| Task Success Rate         | 最终任务成功率       |
| Goal Distance             | 任务理论所需最少步骤    |
| Effective Horizon         | 成功轨迹的实际决策步数   |
| Step Accuracy             | 每一步决策的准确率     |
| Macro Action Success Rate | 宏动作成功执行比例     |
| Subgoal Completion Rate   | 子目标完成率        |
| Invalid Action Rate       | 不合法动作比例       |
| Recovery Rate             | 发生错误之后的恢复能力   |
| Tool Calls                | 平均工具调用次数      |
| Token / Latency Cost      | Token 消耗和执行时延 |

### 9.2 设计 Horizon 分层评测

可以按照任务动作长度，建立例如以下评测分层。

| 层级 | 建议决策步数 | 评测侧重点      |
| -- | ------ | ---------- |
| H1 | 1–5    | 基础任务能力     |
| H2 | 6–10   | 短链路可靠性     |
| H3 | 11–20  | 多步骤协调能力    |
| H4 | 21–40  | 长任务稳定性     |
| H5 | 41+    | 长链路泛化与恢复能力 |

以上是建议性的工程分层，不是论文采用的数独 L1–L7 标准。

对各个层级分别统计成功率，有助于识别模型在哪个任务长度范围开始明显退化。

### 9.3 不要忽略成本与效率

宏动作能够减少模型与环境之间的交互轮数，但一个宏动作内部可能包含很多实际工具执行操作。

因此建议同时记录：

\\[ \text{Decision Compression Ratio} = \frac{\text{Atomic Decision Count}} {\text{Macro Decision Count}} \\]

以及实际的总成本：

\\[ C\_{\text{total}} = C\_{\text{model}}+ C\_{\text{tools}}+ C\_{\text{verification}}+ C\_{\text{recovery}} \\]

这两项可以帮助判断某种 Horizon Reduction 方案是否真正值得应用于生产环境。

## 十、论文的局限性

虽然论文结论有启发性，但阅读时也需要关注其适用边界。

第一，受控实验并不等于所有真实任务。

Sudoku 与 Rush Hour 有明确规则、相对清晰的状态和可自动判断的成功条件。真实企业 Agent 则可能涉及动态数据、不确定环境、含糊目标与难以自动验证的中间结果。

第二，宏动作存在粒度权衡。

过大的宏动作可能隐藏内部失败，降低灵活性，甚至增加出错后的补偿成本。论文也通过对比固定长度和灵活长度宏动作，展示了动作粒度的重要性。

第三，较大的模型也并非天然解决 Horizon 瓶颈。

论文在 4B 模型上观察到类似的长 Horizon 训练失稳现象，但这并不能直接推出任意规模模型都会以完全相同的方式崩溃。

第四，跨长度泛化不等于跨能力泛化。

从短任务泛化到较长任务，需要任务之间具有相似的基础规则。对于引入全新工具、知识或推理能力的任务，仅依赖 Horizon Generalization 可能不足。

第五，论文重点研究训练问题。

生产 Agent 的最终效果同时受到模型训练、工具设计、上下文管理、任务规划和系统可靠性的影响。Horizon Reduction 是其中一个重要方向，而不是所有长任务失败问题的唯一解法。

## 十一、总结：这篇论文真正告诉了我们什么？

我认为可以将这篇论文的贡献归纳成三个层次。

第一层：发现问题

在控制推理复杂度的条件下，单独增长任务执行链路，仍然能够导致强化学习训练明显不稳定。

第二层：提出解决原则

通过宏动作减少高层决策次数，或者通过子目标分解缩短奖励归因跨度，可以有效缓解长任务训练瓶颈。

第三层：验证泛化潜力

先在较短任务上形成稳定能力，再迁移或继续训练到更长任务，是一条值得关注的训练路径。

这篇论文对当前 LLM Agent 研究最有价值的提醒是：

> 提升 Agent 的长任务能力，不仅要让模型更聪明，还要让任务的决策结构更容易学习、更容易执行，并且更容易获得正确反馈。

对于算法研究者，它提示我们要关注 Exploration、Credit Assignment 和 Horizon-aware Training。

对于 Agent 平台工程师，它提示我们重新考虑工具抽象、任务分解、阶段验证与评测指标。

最终的目标不是让 Agent 永远执行更少的底层操作，而是让模型在完成同样复杂任务时，承担更少的不必要决策，以及更短、更清晰的学习与反馈链路。
