# Transformer 架构详解：从 Attention 到大语言模型

## 一、背景

Transformer 是当前大语言模型最核心的基础架构之一。

我们现在常见的模型，例如：

* GPT 系列
* Claude
* Gemini
* Llama
* Qwen
* DeepSeek
* BERT

虽然具体实现已经存在大量差异，但其核心思想大多来源于 Transformer。

Transformer 最早在 2017 年论文《Attention Is All You Need》中提出。

在 Transformer 出现之前，NLP 领域主要使用：

* RNN
* LSTM
* GRU

来处理文本序列。

Transformer 最大的变化是：

> 不再依赖 RNN 按顺序逐个处理 Token，而是通过 Attention 机制直接计算不同 Token 之间的关系。

因此 Transformer 具备两个非常重要的优势：

1. 能够捕获长距离依赖关系。
2. 能够进行大规模并行计算。

这两个特点，为后续大语言模型的发展奠定了基础。

---

# 二、先理解大模型到底在做什么

一个大语言模型，本质上是在解决一个问题：

> 根据前面的 Token，预测下一个 Token。

例如输入：

```text
中国的首都是
```

模型实际上计算：

```text
北京      0.92
上海      0.03
广州      0.01
深圳      0.01
其他      0.03
```

于是模型选择：

```text
北京
```

然后输入变成：

```text
中国的首都是北京
```

再继续预测下一个 Token。

因此整个生成过程可以理解成：

```text
输入文本
   ↓
Tokenizer
   ↓
Token ID
   ↓
Embedding
   ↓
Transformer
   ↓
得到所有 Token 的上下文表示
   ↓
LM Head
   ↓
下一个 Token 的概率
   ↓
选择 Token
   ↓
继续生成
```

所以 Transformer 最核心的任务并不是直接“输出文字”。

它负责的是：

> 对输入 Token 进行上下文建模。

---

# 三、Transformer 整体架构

原始 Transformer 使用：

```text
Encoder + Decoder
```

整体结构：

```text
                  Transformer
                       │
            ┌──────────┴──────────┐
            │                     │
         Encoder               Decoder
            │                     │
       理解输入信息          根据输入生成内容
```

经典结构：

```text
Input
  │
  ↓
Embedding
  │
  ↓
Positional Encoding
  │
  ↓
┌─────────────────┐
│ Encoder Layer 1 │
├─────────────────┤
│ Encoder Layer 2 │
├─────────────────┤
│       ...       │
├─────────────────┤
│ Encoder Layer N │
└─────────────────┘
        │
        ↓
   Encoder Output
        │
        ↓
┌─────────────────┐
│ Decoder Layer 1 │
├─────────────────┤
│ Decoder Layer 2 │
├─────────────────┤
│       ...       │
├─────────────────┤
│ Decoder Layer N │
└─────────────────┘
        │
        ↓
     Linear
        │
        ↓
     Softmax
        │
        ↓
   Output Token
```

但是现在的大模型不一定完整使用 Encoder + Decoder。

通常可以分成三类：

| 类型              | 架构                | 典型模型           |
| --------------- | ----------------- | -------------- |
| Encoder-only    | 只有 Encoder        | BERT           |
| Decoder-only    | 只有 Decoder        | GPT、Llama、Qwen |
| Encoder-Decoder | Encoder + Decoder | T5             |

当前主流生成式大模型，大多数采用：

```text
Decoder-only Transformer
```

所以理解现代 LLM 时，重点理解 Decoder Transformer 即可。

---

# 四、Transformer 的完整数据流

假设输入：

```text
我喜欢人工智能
```

整个处理过程可以简化成：

```text
文本
 │
 ↓
Tokenizer
 │
 ↓
Token ID
 │
 ↓
Embedding
 │
 ↓
位置编码
 │
 ↓
Transformer Block
 │
 ├── Self-Attention
 │
 ├── Residual
 │
 ├── LayerNorm
 │
 ├── FFN
 │
 ├── Residual
 │
 └── LayerNorm
 │
 ↓
Transformer Block
 │
 ↓
...
 │
 ↓
Transformer Block
 │
 ↓
LM Head
 │
 ↓
Softmax
 │
 ↓
下一个 Token
```

一个现代大模型，本质上就是：

```text
Embedding

+

几十层甚至上百层 Transformer Block

+

LM Head
```

---

# 五、第一步：Tokenizer

Transformer 并不能直接处理：

```text
我喜欢人工智能
```

模型只能处理数字。

所以首先需要 Tokenizer。

例如：

```text
我喜欢人工智能
```

可能被切分为：

```text
我
喜欢
人工
智能
```

每个 Token 对应一个 ID：

```text
我       → 1024
喜欢     → 5832
人工     → 9281
智能     → 7356
```

最终模型真正收到的是：

```text
[1024, 5832, 9281, 7356]
```

注意：

> Token 不一定等于一个汉字，也不一定等于一个单词。

例如：

```text
Transformer
```

有可能被切成：

```text
Trans
former
```

也可能整体是一个 Token。

具体取决于 Tokenizer 的词表。

---

# 六、第二步：Embedding

Token ID 本身没有语义。

例如：

```text
苹果 = 1001
香蕉 = 1002
汽车 = 1003
```

1001 和 1002 数字接近，并不代表：

```text
苹果和香蕉语义相近。
```

所以需要把 Token ID 转换成向量。

这就是：

```text
Embedding
```

例如：

```text
苹果
 ↓
[0.21, 0.87, -0.13, ..., 0.42]
```

假设模型隐藏维度：

```text
hidden_size = 4096
```

那么：

```text
一个 Token
       ↓
4096 维向量
```

如果输入：

```text
我 喜欢 人工 智能
```

有 4 个 Token：

```text
Token 1 → 4096维向量
Token 2 → 4096维向量
Token 3 → 4096维向量
Token 4 → 4096维向量
```

最终得到一个矩阵：

```text
4 × 4096
```

一般表示：

```text
sequence_length × hidden_size
```

---

# 七、为什么需要位置编码

Attention 本身并不知道 Token 的顺序。

例如：

```text
我喜欢你
```

和：

```text
你喜欢我
```

如果只是看 Token 集合：

```text
我
喜欢
你
```

实际上完全一样。

但语义明显不同。

因此模型必须知道：

```text
谁在第1个位置
谁在第2个位置
谁在第3个位置
```

这就是：

```text
Position Encoding
```

也就是：

> 给 Token 增加位置信息。

早期 Transformer 使用：

```text
Sin / Cos 位置编码
```

现代 LLM 常见：

```text
RoPE
```

即：

```text
Rotary Position Embedding
旋转位置编码
```

例如：

```text
Token Embedding
       +
Position Information
       ↓
最终输入向量
```

可以简单理解为：

```text
“我”的语义
+
“我出现在第1个位置”
```

共同组成模型真正处理的信息。

---

# 八、Transformer 最核心部分：Self-Attention

理解 Transformer，最关键的就是理解：

```text
Self-Attention
```

可以翻译为：

> 自注意力机制。

它解决的问题是：

> 当前 Token 应该关注其他哪些 Token？

例如：

```text
小明把书送给了小红，因为她很喜欢阅读。
```

当模型处理：

```text
她
```

的时候，需要判断：

```text
她指的是谁？
```

模型应该重点关注：

```text
小红
```

而不是：

```text
书
```

Attention 做的事情就是计算：

```text
“她”
```

和其他 Token 的相关程度。

例如：

```text
                Attention Score

小明               0.08
书                 0.02
送给               0.05
小红               0.76
因为               0.03
喜欢               0.04
阅读               0.02
```

于是模型知道：

```text
她 ↔ 小红
```

关系比较强。

---

# 九、为什么会出现 Q、K、V

这是 Transformer 最容易把人绕晕的地方。

Self-Attention 会把每个 Token 的向量转换成三个向量：

```text
Q：Query
K：Key
V：Value
```

即：

```text
Token Embedding
       │
       ├── Wq → Q
       │
       ├── Wk → K
       │
       └── Wv → V
```

其中：

```text
Q = XWq

K = XWk

V = XWv
```

Wq、Wk、Wv 都是模型训练出来的参数。

---

# 十、怎么理解 Q / K / V

可以用“搜索系统”理解。

假设你在搜索：

```text
北京天气
```

那么：

```text
Query = 你想找什么
```

数据库中的每条内容有：

```text
Key = 它描述什么
Value = 它真正包含的信息
```

于是：

```text
Query
  ↓
和 Key 比较
  ↓
找到相关内容
  ↓
读取 Value
```

Attention 完全类似。

因此可以记住：

```text
Q：我想找什么？

K：我有什么特征？

V：我真正携带什么信息？
```

---

# 十一、通过一个例子彻底理解 Q/K/V

假设：

```text
小明喜欢吃苹果
```

模型处理：

```text
喜欢
```

的时候。

“喜欢”的 Query 可以理解为：

```text
我正在寻找：

谁喜欢？
喜欢什么？
```

然后与所有 Token 的 Key 比较：

```text
喜欢的 Q
   │
   ├── 小明的 K
   │
   ├── 喜欢的 K
   │
   ├── 吃的 K
   │
   └── 苹果的 K
```

得到：

```text
小明     0.40
喜欢     0.10
吃       0.15
苹果     0.35
```

说明：

```text
喜欢
```

和：

```text
小明
苹果
```

关系较强。

然后：

```text
0.40 × 小明的 V
+
0.10 × 喜欢的 V
+
0.15 × 吃的 V
+
0.35 × 苹果的 V
```

得到新的：

```text
“喜欢”的表示
```

此时这个向量已经不再只是：

```text
喜欢
```

本身的含义。

而是融合了：

```text
谁喜欢
喜欢什么
上下文是什么
```

因此：

```text
原始 Token 向量
        ↓
Self-Attention
        ↓
带上下文信息的 Token 向量
```

这就是 Transformer 最核心的思想。

---

# 十二、Attention 的数学公式

Transformer 最经典的公式：

```text
Attention(Q, K, V)
=
softmax(QKᵀ / √dk)V
```

完整来看：

```text
Q
 │
 │
 × Kᵀ
 │
 ↓
QKᵀ
 │
 ↓
除以 √dk
 │
 ↓
Softmax
 │
 ↓
Attention Weight
 │
 ×
 V
 │
 ↓
Output
```

分四步理解。

---

## 12.1 第一步：Q × Kᵀ

```text
QKᵀ
```

用于计算：

> Token 与 Token 之间的相关程度。

例如：

```text
            我   喜欢   AI

我         1.2   0.3   0.1
喜欢       0.2   1.1   0.8
AI         0.1   0.7   1.3
```

这就是：

```text
Attention Score Matrix
```

---

# 十三、为什么除以 √dk

公式：

```text
QKᵀ / √dk
```

为什么不直接：

```text
QKᵀ
```

因为向量维度越大，点积结果越容易变得非常大。

例如：

```text
dk = 128
```

如果直接计算：

```text
Q · K
```

数值可能很大。

经过 Softmax 后可能变成：

```text
[0.000001, 0.999998, 0.000001]
```

Softmax 太极端，会导致梯度非常小，训练变困难。

因此除以：

```text
√dk
```

让数值保持相对稳定。

这也是为什么叫：

```text
Scaled Dot-Product Attention
```

即：

> 缩放点积注意力。

---

# 十四、Softmax 在做什么

假设相关性分数：

```text
[2.0, 1.0, 0.5]
```

经过：

```text
Softmax
```

可能变成：

```text
[0.63, 0.23, 0.14]
```

满足：

```text
0.63 + 0.23 + 0.14 = 1
```

这就代表：

```text
63% 注意第一个 Token
23% 注意第二个 Token
14% 注意第三个 Token
```

所以 Softmax 本质上就是：

> 把相关性分数转换成注意力权重。

---

# 十五、最后乘 V

得到 Attention Weight 后：

```text
Attention Weight × V
```

例如：

```text
0.63 × V1
+
0.23 × V2
+
0.14 × V3
```

最终形成一个新的向量。

因此 Self-Attention 可以概括成：

```text
Q × K
   ↓
计算“我应该关注谁”
   ↓
Softmax
   ↓
得到关注比例
   ↓
加权 V
   ↓
得到新的上下文表示
```

最重要的一句话：

> Q 和 K 决定关注谁，V 决定真正获取什么信息。

---

# 十六、什么是 Multi-Head Attention

Transformer 不只计算一次 Attention。

而是同时计算多组 Attention：

```text
Multi-Head Attention
```

例如有：

```text
32 Heads
```

相当于：

```text
Head 1
Head 2
Head 3
...
Head 32
```

每个 Head 都有自己的：

```text
Wq
Wk
Wv
```

因此不同 Head 可以学习不同关系。

例如：

```text
“小明昨天去北京旅游，因为那里天气很好。”
```

可能出现：

```text
Head 1 → 学习主谓关系

小明 → 去
```

```text
Head 2 → 学习地点关系

去 → 北京
```

```text
Head 3 → 学习指代关系

那里 → 北京
```

```text
Head 4 → 学习时间关系

昨天 → 去
```

所以：

```text
Single Head
```

类似从一个角度理解句子。

而：

```text
Multi-Head Attention
```

类似：

> 从多个不同维度同时理解句子。

---

# 十七、Multi-Head Attention 的完整结构

假设：

```text
hidden_size = 4096
head_num = 32
```

则：

```text
head_dim
=
4096 / 32
=
128
```

也就是说：

```text
4096 维
     ↓
拆成 32 组
     ↓
每组 128 维
```

结构：

```text
Input
  │
  ├───────────┬───────────┬───────────┐
  ↓           ↓           ↓           ↓
Head1       Head2       Head3       ...
  │           │           │
Attention  Attention  Attention
  │           │           │
  └───────────┴───────────┘
             ↓
          Concat
             ↓
           Linear
             ↓
           Output
```

即：

```text
MultiHead(Q,K,V)
=
Concat(head1, head2, ..., headN)Wo
```

---

# 十八、为什么还需要 FFN

Attention 主要解决：

```text
Token 和 Token 之间的信息交换。
```

但模型不仅需要：

```text
找到谁和谁有关系。
```

还需要：

```text
对得到的信息进一步计算、抽象和转换。
```

所以 Transformer Block 里还有：

```text
FFN
```

即：

```text
Feed Forward Network
前馈神经网络
```

经典形式：

```text
FFN(x)
=
Activation(xW1 + b1)W2 + b2
```

结构：

```text
hidden_size
    │
    ↓
扩大维度
    │
    ↓
Activation
    │
    ↓
降低维度
    │
    ↓
hidden_size
```

例如：

```text
4096
 ↓
16384
 ↓
激活函数
 ↓
4096
```

可以粗略理解：

```text
Attention
=
让 Token 之间交流

FFN
=
每个 Token 自己思考、加工信息
```

这是理解 Transformer Block 很好用的一种方式。

---

# 十九、现代 LLM 中的 FFN

现代模型往往不再简单使用：

```text
ReLU
```

而是：

```text
GELU
SwiGLU
GeGLU
```

例如很多现代 LLM 使用：

```text
SwiGLU
```

相比传统 FFN，可以进一步提升模型表达能力。

因此一个现代 Transformer Block 可以粗略表示：

```text
Attention
   ↓
信息交流

FFN / SwiGLU
   ↓
信息加工
```

---

# 二十、Residual Connection

Transformer 还有一个非常重要的结构：

```text
Residual Connection
```

即：

```text
残差连接
```

普通网络：

```text
x
 ↓
Layer
 ↓
y
```

残差网络：

```text
       ┌─────────────┐
       │             │
x ─────┤             +
       │             ↑
       ↓             │
     Layer ──────────┘
```

也就是：

```text
Output = x + Layer(x)
```

为什么需要？

因为 Transformer 可能非常深：

```text
32 层
64 层
80 层
100+ 层
```

如果每层都完全重新转换信息，训练容易出现：

```text
梯度消失
训练不稳定
原始信息丢失
```

Residual Connection 给信息提供了一条：

```text
高速公路
```

使原始信息可以直接向后传播。

---

# 二十一、LayerNorm

Transformer 中还大量使用：

```text
Layer Normalization
```

即：

```text
LayerNorm
```

其主要作用是：

> 稳定每一层输入数据的数值分布。

例如一层输出：

```text
[-100, 20, 300, -50]
```

另一层输出：

```text
[0.001, 0.01, 0.1]
```

数值范围差异过大，会让网络训练不稳定。

Normalization 会把这些数据调整到更稳定的分布范围。

因此 LayerNorm 的主要目标可以理解为：

```text
降低训练难度
+
提高数值稳定性
+
让深层网络更容易训练
```

---

# 二十二、一个完整 Transformer Block

到这里就可以看完整结构了。

经典 Transformer Block：

```text
Input
  │
  ├────────────────────┐
  │                    │
  ↓                    │
Multi-Head Attention   │
  │                    │
  └────────── + ───────┘
               │
               ↓
           LayerNorm
               │
               ├────────────────────┐
               │                    │
               ↓                    │
              FFN                   │
               │                    │
               └──────── + ─────────┘
                          │
                          ↓
                      LayerNorm
                          │
                          ↓
                        Output
```

现代 LLM 中更常见：

```text
Pre-Norm
```

结构大致是：

```text
x
│
├────────────────────────────┐
│                            │
↓                            │
LayerNorm                    │
│                            │
↓                            │
Attention                    │
│                            │
└──────────── + ─────────────┘
               │
               ↓
               x'
               │
               ├──────────────────────┐
               │                      │
               ↓                      │
           LayerNorm                  │
               │                      │
               ↓                      │
              FFN                     │
               │                      │
               └──────── + ───────────┘
                          │
                          ↓
                        Output
```

可以记成：

```text
x = x + Attention(Norm(x))

x = x + FFN(Norm(x))
```

---

# 二十三、为什么 Transformer 要堆几十层

单层 Transformer 只能进行有限的信息加工。

因此需要不断重复：

```text
Attention
+
FFN
```

例如：

```text
Embedding
     ↓
Transformer Layer 1
     ↓
Transformer Layer 2
     ↓
Transformer Layer 3
     ↓
...
     ↓
Transformer Layer 32
```

可以粗略理解：

```text
低层
↓
学习词法、局部关系

中层
↓
学习句法、语义关系

高层
↓
形成更复杂的抽象表示
```

当然真实模型内部并没有如此严格的层级边界，但这种理解方式有助于建立直觉。

---

# 二十四、GPT 为什么只需要 Decoder

原始 Transformer 是：

```text
Encoder + Decoder
```

但 GPT 的目标是：

```text
预测下一个 Token。
```

例如：

```text
今天北京天气
```

预测：

```text
很好
```

然后：

```text
今天北京天气很好
```

再预测：

```text
，
```

继续：

```text
今天北京天气很好，
```

再预测下一个 Token。

因此 GPT 只需要：

```text
Decoder
```

所以称为：

```text
Decoder-only Transformer
```

结构：

```text
Input Token
    ↓
Embedding
    ↓
Transformer Decoder Block
    ↓
Transformer Decoder Block
    ↓
...
    ↓
Transformer Decoder Block
    ↓
LM Head
    ↓
Softmax
    ↓
Next Token
```

---

# 二十五、GPT 为什么不能看到未来 Token

假设训练数据：

```text
我喜欢人工智能
```

模型正在预测：

```text
人工
```

如果它能直接看到后面的：

```text
人工智能
```

那么预测就失去意义了。

所以 Decoder Attention 使用：

```text
Causal Mask
```

也叫：

```text
Masked Self-Attention
```

例如：

```text
         我   喜欢   人工   智能

我       ✓    ×      ×      ×

喜欢     ✓    ✓      ×      ×

人工     ✓    ✓      ✓      ×

智能     ✓    ✓      ✓      ✓
```

意思是：

```text
Token 1
只能看到 Token 1

Token 2
可以看到 Token 1~2

Token 3
可以看到 Token 1~3
```

不能看到未来 Token。

因此：

> GPT 是一个自回归语言模型。

---

# 二十六、什么叫 Autoregressive

Autoregressive：

```text
自回归
```

就是：

> 使用已经生成出来的内容，继续预测下一步。

例如：

```text
输入：

中国
```

预测：

```text
的
```

变成：

```text
中国的
```

继续预测：

```text
首都
```

变成：

```text
中国的首都
```

继续：

```text
是
```

继续：

```text
北京
```

所以：

```text
Token1
 ↓
Token2
 ↓
Token3
 ↓
Token4
...
```

每一次预测都依赖之前所有 Token。

---

# 二十七、LM Head 是什么

经过几十层 Transformer 后：

```text
Token
```

已经变成一个上下文向量。

例如：

```text
hidden_size = 4096
```

输出：

```text
[0.13, -0.72, ..., 0.56]
```

但是我们真正需要预测的是：

```text
词表中的哪个 Token？
```

假设词表：

```text
vocab_size = 100000
```

所以需要：

```text
4096
 ↓
Linear
 ↓
100000
```

这个 Linear 层通常叫：

```text
LM Head
```

输出：

```text
100000 个分数
```

对应词表中的：

```text
Token 1
Token 2
Token 3
...
Token 100000
```

再经过采样算法选择下一个 Token。

---

# 二十八、完整的大语言模型推理流程

把前面的内容串起来：

```text
用户：

Transformer是什么？
        │
        ↓
Tokenizer
        │
        ↓
Token IDs
        │
        ↓
Embedding
        │
        ↓
Position Information
        │
        ↓
┌───────────────────────┐
│ Transformer Block 1   │
│                       │
│ Attention             │
│ +                     │
│ FFN                   │
└───────────────────────┘
        │
        ↓
┌───────────────────────┐
│ Transformer Block 2   │
└───────────────────────┘
        │
        ↓
       ...
        │
        ↓
┌───────────────────────┐
│ Transformer Block N   │
└───────────────────────┘
        │
        ↓
     LM Head
        │
        ↓
 Token Probability
        │
        ↓
 Sampling
        │
        ↓
      “Transformer”
```

然后把：

```text
Transformer
```

放回输入：

```text
Transformer是什么？Transformer
```

重新执行一遍。

继续生成：

```text
是一种
```

再继续：

```text
Transformer是什么？Transformer是一种
```

如此循环。

直到：

```text
EOS
```

或者达到：

```text
max_tokens
```

---

# 二十九、Transformer 最大的问题：Attention 计算复杂度

标准 Attention：

```text
Attention(Q,K,V)
=
softmax(QKᵀ / √d)V
```

假设：

```text
Sequence Length = n
```

Q：

```text
n × d
```

K：

```text
n × d
```

计算：

```text
QKᵀ
```

得到：

```text
n × n
```

所以 Attention 复杂度大致是：

```text
O(n²)
```

这就是长上下文成本非常高的重要原因。

例如：

```text
1K Token
→ 约 1M Attention关系

10K Token
→ 约 100M

100K Token
→ 约 10B
```

因此现代大模型一直在优化：

```text
长上下文
Attention 内存
KV Cache
计算复杂度
```

---

# 三十、KV Cache 是什么

大模型生成时有一个非常典型的问题。

假设已经生成：

```text
我喜欢人工智能，因为
```

然后预测下一个 Token。

之前：

```text
我
喜欢
人工
智能
因为
```

对应的 K 和 V 已经计算过了。

如果每生成一个 Token 都重新计算：

```text
前面所有 Token 的 K / V
```

会非常浪费。

因此会缓存：

```text
K
V
```

这就是：

```text
KV Cache
```

下一次生成：

```text
只计算新 Token 的 Q/K/V
```

历史 Token：

```text
K/V
```

直接读取 Cache。

结构：

```text
历史 Token
    │
    └── K/V ──→ KV Cache
                    ↑
                    │
新 Token ──→ Q/K/V │
          │
          ↓
       Attention
```

因此 KV Cache 可以显著提升：

```text
大模型 Decode 阶段
```

的推理速度。

---

# 三十一、为什么上下文越长越吃显存

因为 KV Cache 会随着：

```text
Token 数量
```

持续增长。

大致关系：

```text
KV Cache
∝
Layer Number
×
Sequence Length
×
KV Head Number
×
Head Dimension
```

所以：

```text
上下文越长
      ↓
KV Cache 越大
      ↓
GPU 显存占用越高
```

这也是：

```text
128K Context
1M Context
```

并不是简单扩大一个参数就可以实现。

---

# 三十二、MHA、MQA、GQA

为了减少 KV Cache，现代大模型对 Attention 做了很多优化。

传统：

```text
MHA
Multi-Head Attention
```

每个 Head 都有自己的：

```text
Q
K
V
```

例如：

```text
32 Q Heads
32 K Heads
32 V Heads
```

KV Cache 很大。

于是出现：

```text
MQA
Multi-Query Attention
```

变成：

```text
32 Q Heads

1 K Head
1 V Head
```

显著减少 KV Cache。

但是可能损失一定模型能力。

因此很多现代 LLM 使用：

```text
GQA
Grouped Query Attention
```

例如：

```text
32 Q Heads
8 K Heads
8 V Heads
```

即：

```text
多个 Q Head
共享一组 K/V
```

取得：

```text
性能
与
推理效率
```

之间的平衡。

可以简单记：

```text
MHA
能力强
KV Cache 大

MQA
KV Cache 最小
共享程度最高

GQA
折中方案
```

---

# 三十三、MoE 和 Transformer 什么关系

现在很多模型还使用：

```text
MoE
Mixture of Experts
```

例如：

```text
DeepSeek
Mixtral
```

MoE 通常不是把 Transformer 推翻。

而是修改 Transformer 中的：

```text
FFN
```

传统：

```text
Attention
   ↓
FFN
```

MoE：

```text
Attention
   ↓
Router
   │
   ├── Expert 1
   ├── Expert 2
   ├── Expert 3
   ├── ...
   └── Expert N
```

每个 Token 只激活其中少数 Expert。

例如：

```text
总参数：600B

实际激活：
30B
```

这样就可以做到：

```text
模型容量很大
+
每次计算量相对较小
```

所以很多 MoE 模型依然属于：

```text
Transformer Architecture
```

只不过：

```text
Dense FFN
```

被替换成：

```text
Sparse MoE FFN
```

---

# 三十四、现代 LLM 的典型 Transformer Block

今天一个现代 LLM 的 Transformer Block，更接近：

```text
              Input
                │
                ├─────────────────┐
                │                 │
                ↓                 │
             RMSNorm              │
                │                 │
                ↓                 │
               RoPE               │
                │                 │
                ↓                 │
        GQA / Attention            │
                │                 │
                └────── + ────────┘
                        │
                        ↓
                      Hidden
                        │
                        ├─────────────────┐
                        │                 │
                        ↓                 │
                     RMSNorm              │
                        │                 │
                        ↓                 │
                  SwiGLU / MoE            │
                        │                 │
                        └────── + ────────┘
                                │
                                ↓
                              Output
```

所以现代大模型虽然仍然叫：

```text
Transformer
```

但是与 2017 年原始 Transformer 已经有不少变化。

常见技术包括：

```text
RoPE
RMSNorm
SwiGLU
GQA
KV Cache
FlashAttention
MoE
```

---

# 三十五、从工程视角理解 Transformer

如果从 Java / 后端工程师比较熟悉的角度理解，可以把 Transformer 想象成一个数据加工 Pipeline。

例如：

```text
Request
   ↓
参数解析
   ↓
业务处理
   ↓
RPC
   ↓
DB
   ↓
Response
```

Transformer 类似：

```text
Token
  ↓
Embedding
  ↓
Attention
  ↓
FFN
  ↓
Attention
  ↓
FFN
  ↓
...
  ↓
预测结果
```

其中：

```text
Attention
```

更像：

> 全局信息查询 / 信息聚合。

而：

```text
FFN
```

更像：

> 本地业务逻辑处理。

可以把一层 Transformer 理解成：

```text
先查上下文
     ↓
得到相关信息
     ↓
自己加工
     ↓
把结果传给下一层
```

几十层 Transformer 就是几十轮：

```text
信息查询
+
信息加工
```

---

# 三十六、一句话理解 Transformer 的每个组件

最后可以用下面这张表建立整体认知：

| 组件                       | 作用                       |
| ------------------------ | ------------------------ |
| Tokenizer                | 把文本变成 Token              |
| Token ID                 | Token 在词表中的编号            |
| Embedding                | 把 Token 转换成向量            |
| Position Encoding / RoPE | 告诉模型 Token 的位置           |
| Q                        | 当前 Token 想找什么            |
| K                        | 当前 Token 有什么特征           |
| V                        | 当前 Token 真正携带的信息         |
| QKᵀ                      | 计算 Token 之间相关性           |
| Softmax                  | 转换成注意力权重                 |
| Attention                | 聚合其他 Token 的信息           |
| Multi-Head Attention     | 从多个角度学习关系                |
| FFN                      | 对 Token 信息进一步加工          |
| SwiGLU                   | 现代 FFN 常用结构              |
| Residual                 | 保留原始信息，帮助训练深层网络          |
| LayerNorm / RMSNorm      | 稳定网络训练                   |
| Causal Mask              | 防止 GPT 偷看未来 Token        |
| Transformer Block        | Attention + FFN 等组件组成的一层 |
| LM Head                  | 把隐藏状态转换成词表分数             |
| Softmax / Sampling       | 选择下一个 Token              |
| KV Cache                 | 缓存历史 K/V，加速生成            |
| GQA                      | 减少 KV Cache              |
| MoE                      | 用多个 Expert 扩大模型容量        |

---

# 三十七、Transformer 最核心的思想

如果整场技术分享只需要记住三件事，可以记：

## 第一：Embedding

```text
文字
↓
数字
↓
向量
```

让模型能够处理语言。

## 第二：Attention

```text
当前 Token
↓
寻找相关 Token
↓
聚合上下文信息
```

让模型理解：

```text
Token 与 Token 的关系。
```

## 第三：FFN

```text
上下文信息
↓
非线性加工
↓
更高级的表示
```

不断堆叠：

```text
Attention
+
FFN
```

最终产生越来越复杂的语言表示。

因此可以把 Transformer 最核心的运行过程概括成：

```text
       Token
         │
         ↓
     Embedding
         │
         ↓
┌───────────────────┐
│                   │
│    Attention      │
│       ↓           │
│       FFN         │
│                   │
└───────────────────┘
         │
        × N
         │
         ↓
     LM Head
         │
         ↓
   Next Token
```

---

# 三十八、总结

Transformer 并不神秘。

它最核心的思想其实可以压缩成一句话：

> 让每一个 Token 根据当前上下文找到自己应该关注的信息，再对这些信息进行加工。

其中：

```text
Q/K
```

负责：

```text
寻找信息
```

```text
V
```

负责：

```text
提供信息
```

```text
Attention
```

负责：

```text
信息交换
```

```text
FFN
```

负责：

```text
信息加工
```

```text
Residual
```

负责：

```text
信息保留
```

```text
Norm
```

负责：

```text
保证整个系统稳定运行
```

几十层 Transformer Block 不断重复这个过程：

```text
查信息
 ↓
融合信息
 ↓
加工信息
 ↓
继续查信息
 ↓
继续加工
```

最终：

```text
输入 Token
      ↓
Transformer
      ↓
上下文语义表示
      ↓
预测下一个 Token
```

这就是 GPT、Llama、Qwen、DeepSeek 等现代大语言模型最核心的工作方式。

---

# 架构图

```text
                           Large Language Model
                                    │
                                    ▼
                               用户输入文本
                                    │
                                    ▼
                               Tokenizer
                                    │
                                    ▼
                                Token IDs
                                    │
                                    ▼
                               Embedding
                                    │
                                    ▼
                                  RoPE
                                    │
                                    ▼
                    ┌──────────────────────────┐
                    │    Transformer Block     │
                    │                          │
                    │  RMSNorm                 │
                    │     ↓                    │
                    │  Self-Attention          │
                    │     ↓                    │
                    │  Residual                │
                    │     ↓                    │
                    │  RMSNorm                 │
                    │     ↓                    │
                    │  FFN / MoE               │
                    │     ↓                    │
                    │  Residual                │
                    └──────────────────────────┘
                                    │
                                    │ × N Layers
                                    ▼
                                 LM Head
                                    │
                                    ▼
                              Token Scores
                                    │
                                    ▼
                               Sampling
                                    │
                                    ▼
                              Next Token
                                    │
                                    └──────────┐
                                               │
                                               ▼
                                      放回输入继续生成
```

最终可以记成一个非常简单的公式：

```text
LLM
≈
Tokenizer
+
Embedding
+
N × Transformer Block
+
LM Head
+
Sampling
```

而：

```text
Transformer Block
≈
Attention
+
FFN
+
Residual
+
Normalization
```

再继续往下拆：

```text
Attention
≈
Q
+
K
+
V
+
Softmax
```

因此最终知识树就是：

```text
LLM
│
├── Tokenizer
│
├── Embedding
│
├── Transformer
│   │
│   ├── Attention
│   │   ├── Q
│   │   ├── K
│   │   ├── V
│   │   ├── Mask
│   │   └── Softmax
│   │
│   ├── Multi-Head / GQA
│   ├── FFN / SwiGLU / MoE
│   ├── Residual
│   ├── RMSNorm
│   └── RoPE
│
├── LM Head
│
├── Sampling
│
└── KV Cache
```

掌握这张知识树，基本就建立了理解现代大语言模型架构的核心骨架。
