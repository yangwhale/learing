# Section 3: Transformer 语言模型

> 对应原始 PDF Section 3.1-3.5 (pages 13-28) | 共 11 个 Problem，27 分

本章从零构建完整的 Transformer 语言模型：从基础模块（Linear、Embedding）开始，逐步实现 RMSNorm、SwiGLU FFN、RoPE、Attention，最终组装为完整模型。

---

## 3.1 Transformer 语言模型概览

参见 **Figure 1**（`images/figure1_transformer_lm.png`）和 **Figure 2**（`images/figure2_transformer_block.png`）。

Transformer 语言模型的核心组件：
- Token Embedding 层：将 token ID 映射为向量
- 多个 Transformer Block（堆叠）
- 最终的 RMSNorm + 线性投影层（LM Head）

每个 Transformer Block 包含：
- RMSNorm → Multi-Head Self-Attention → 残差连接
- RMSNorm → Feed-Forward Network → 残差连接

---

## 3.2 批量处理与 Einsum

### 批量处理 (Batching)

训练时我们并行处理多个序列。对于 batch 中的每个序列，模型的计算是独立的。

### Einsum 表示法

`torch.einsum` 提供了一种简洁的方式来表示张量运算。

**示例 1**：Batched 矩阵乘法

```python
# A: (batch, m, n), B: (batch, n, p) → C: (batch, m, p)
C = torch.einsum('bmn, bnp -> bmp', A, B)
```

**示例 2**：带 Broadcast 的运算

```python
# A: (batch, seq_len, d), B: (d,) → C: (batch, seq_len, d)
C = torch.einsum('bsd, d -> bsd', A, B)  # element-wise with broadcast
```

**示例 3**：Pixel Mixing / Token Mixing

```python
# weights: (seq_len, seq_len), x: (batch, seq_len, d) → y: (batch, seq_len, d)
y = torch.einsum('ts, bsd -> btd', weights, x)
```

### 3.2.1 数学符号约定

| 符号 | 含义 |
|------|------|
| $x$ | 标量 (scalar)，粗体 $\mathbf{x}$ 表示向量 |
| $W$ | 大写表示矩阵 |
| $\mathbf{x}$ | **行向量** (row vector)，shape $(1, d)$ |
| $W \in \mathbb{R}^{d_1 \times d_2}$ | 矩阵，$d_1$ 行 $d_2$ 列 |

> **重要约定**：本作业中的向量默认为**行向量**（不同于许多教材使用列向量的习惯）。

---

## 3.3 基础构建模块

### 3.3.1 参数初始化

所有参数使用 **truncated normal** 初始化：

| 模块 | 参数 | 初始化 |
|------|------|--------|
| Linear | weight | `trunc_normal_(mean=0, std=0.02)` |
| Embedding | weight | `trunc_normal_(mean=0, std=0.02)` |
| RMSNorm | weight (gain) | 全 1 (`ones`) |

> `torch.nn.init.trunc_normal_(tensor, mean=0, std=0.02)`

### 3.3.2 Linear 模块

**不使用 bias** (no bias term)。

$$\mathbf{y} = \mathbf{x} W^T$$

其中 $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$，$\mathbf{x} \in \mathbb{R}^{d_{\text{in}}}$，$\mathbf{y} \in \mathbb{R}^{d_{\text{out}}}$。

> 为什么？当数据是行向量时，PyTorch 的 `nn.Linear` 内部计算 $y = x W^T + b$。权重形状是 `(d_out, d_in)`。

### Problem (linear): Linear Module (1 point)

实现不带 bias 的 Linear 模块。

**测试**：`adapters.get_Linear` → `uv run pytest -k test_linear`

### 3.3.3 Embedding 模块

Token Embedding 将 token ID 映射为 $d_{\text{model}}$ 维向量。

$$\mathbf{e}_i = E[x_i, :] \in \mathbb{R}^{d_{\text{model}}}$$

其中 $E \in \mathbb{R}^{V \times d_{\text{model}}}$，$V$ 是词汇表大小。

### Problem (embedding): Embedding Module (1 point)

实现 Embedding 模块。

**测试**：`adapters.get_Embedding` → `uv run pytest -k test_embedding`

---

## 3.4 Pre-Norm Transformer Block

Pre-norm 结构（区别于原始 Transformer 的 Post-norm）：

$$\mathbf{h} = \mathbf{x} + f(\text{Norm}(\mathbf{x}))$$

即先归一化再进行 self-attention 或 FFN 运算，然后通过残差连接。

### 3.4.1 RMSNorm

Root Mean Square Layer Normalization (Zhang & Sennrich [11])：

$$\text{RMSNorm}(\mathbf{x})_i = \frac{x_i}{\sqrt{\frac{1}{d} \sum_{j=1}^{d} x_j^2 + \epsilon}} \cdot g_i$$

其中 $\mathbf{g} \in \mathbb{R}^d$ 是可学习的 gain 参数（初始化为全 1），$\epsilon$ 是防止除零的小常数。

> **实现提示**：归一化计算时将输入 **upcast 到 float32** 以保证数值稳定性，然后将结果 cast 回输入的原始 dtype。

### Problem (rmsnorm): RMS Normalization (1 point)

**构造参数**：`d_model`（特征维度）

**测试**：`adapters.get_RMSNorm` → `uv run pytest -k test_rmsnorm`

---

### 3.4.2 Position-wise Feed-Forward Network (SwiGLU)

参见 **Figure 3**（`images/figure3_silu_relu.png`）。

**SiLU**（Sigmoid Linear Unit，也叫 Swish）（公式 3）：

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

**GLU**（Gated Linear Unit）（公式 4）：

$$\text{GLU}(\mathbf{x}, \mathbf{g}) = \mathbf{x} \odot \sigma(\mathbf{g})$$

**SwiGLU**（SiLU + GLU，Shazeer [13]）（公式 5）：

$$\text{FFN}(\mathbf{x}) = W_2 (\text{SiLU}(W_1 \mathbf{x}) \odot W_3 \mathbf{x})$$

其中：
- $W_1, W_3 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$
- $W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$
- $\odot$ 表示逐元素乘法

**d_ff 计算**：

$$d_{\text{ff}} = \text{round\_to\_multiple\_of\_64}\left(\left\lfloor \frac{8}{3} d_{\text{model}} \right\rfloor\right)$$

> **引自 Shazeer [13]**: "...making SiLU a 'smooth' version of ReLU, where the zero-slope region is replaced with a smooth, non-zero one."

### Problem (positionwise_feedforward): FFN with SwiGLU (2 points)

**构造参数**：`d_model`（模型维度），`d_ff`（中间层维度）

三个权重矩阵 $W_1, W_2, W_3$，均无 bias。

**测试**：`adapters.get_FFN` → `uv run pytest -k test_positionwise_feedforward`

---

### 3.4.3 旋转位置编码 (Rotary Positional Embeddings, RoPE)

RoPE (Su et al. [14]) 通过旋转矩阵将位置信息编码到 query 和 key 中。

**旋转矩阵**（公式 6-8）：

$$R_{\Theta, m} = \begin{pmatrix} \cos(m\theta_1) & -\sin(m\theta_1) & & \\ \sin(m\theta_1) & \cos(m\theta_1) & & \\ & & \ddots & \\ & & & \cos(m\theta_{d/2}) & -\sin(m\theta_{d/2}) \\ & & & \sin(m\theta_{d/2}) & \cos(m\theta_{d/2}) \end{pmatrix}$$

其中 $\theta_i = \theta_{\text{base}}^{-2i/d}$，$\theta_{\text{base}}$ 通常取 10,000。

**关键特性**：

1. RoPE 矩阵是分块对角矩阵，每个 $2 \times 2$ 块执行一个旋转操作
2. **没有可学习参数** — 完全基于位置的确定性函数
3. 只应用于 query 和 key 向量，**不应用于 value 向量**

**实现技巧**：

- 预计算 cos 和 sin 值
- 跨 batch 和 attention head 复用 cos/sin（仅依赖位置，不依赖具体 token）
- 使用 `register_buffer` 存储预计算值（不参与梯度计算）

### Problem (rope): Rotary Positional Embeddings (2 points)

**构造参数**：`theta` 值（默认 10000），`d_model`（向量维度），`max_seq_len`（最大序列长度）

**接口**：接受 shape `(batch, seq_len, num_heads, d_model)` 的 query 和 key 向量。

**测试**：`adapters.get_RoPE` → `uv run pytest -k test_rope`

---

### 3.4.4 缩放点积注意力 (Scaled Dot-Product Attention)

**Softmax**（公式 9）：

$$\text{softmax}(\mathbf{x})_i = \frac{e^{x_i}}{\sum_{j=1}^{d} e^{x_j}}$$

> **数值稳定性技巧**：计算 softmax 时，先减去最大值 $\max(\mathbf{x})$。

**Attention 公式**（公式 10）：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Masking**：

- `mask` 是一个 boolean 张量
- `True` = 允许注意 (attend)
- `False` = 不允许注意，将对应位置设为 $-\infty$

$$\text{Attention}(Q, K, V, M) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \tilde{M}\right)V$$

其中 $\tilde{M}_{ij} = 0$ 当 $M_{ij} = \text{True}$，$\tilde{M}_{ij} = -\infty$ 当 $M_{ij} = \text{False}$。

### Problem (softmax): Softmax (1 point)

实现数值稳定的 softmax 函数。

**测试**：`adapters.get_softmax` → `uv run pytest -k test_softmax`

### Problem (scaled_dot_product_attention): Attention (5 points)

实现缩放点积注意力。

**输入**：
- $Q \in \mathbb{R}^{n \times d}$：query 矩阵
- $K \in \mathbb{R}^{m \times d}$：key 矩阵
- $V \in \mathbb{R}^{m \times v}$：value 矩阵
- $M \in \{0, 1\}^{n \times m}$：mask 矩阵（可选）

**输出**：$\text{Attention}(Q, K, V, M) \in \mathbb{R}^{n \times v}$

**测试**：`adapters.run_scaled_dot_product_attention` → `uv run pytest -k test_scaled_dot_product_attention`

---

### 3.4.5 因果多头自注意力 (Causal Multi-Head Self-Attention)

每个 attention head 独立计算注意力：

$$\text{head}_i = \text{Attention}(X W_i^Q, X W_i^K, X W_i^V, M)$$

（公式 12）

$$\text{MultiHead}(X, M) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

（公式 13-14）

其中：
- $X \in \mathbb{R}^{s \times d_{\text{model}}}$
- $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$，$d_k = d_{\text{model}} / h$
- $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$

**因果遮罩** (Causal Mask)：

$$M_{ij} = \begin{cases} \text{True} & \text{if } j \leq i \\ \text{False} & \text{if } j > i \end{cases}$$

保证每个位置只能注意到之前的位置（和自身），不能看到未来的 token。

**RoPE 应用**：RoPE 只应用于 Q 和 K，不应用于 V。

### Problem (multihead_self_attention): Causal Multi-Head Self-Attention (5 points)

**构造参数**：`d_model`，`num_heads`

**权重矩阵**：$W^Q, W^K, W^V, W^O$（均无 bias）。

> 可将 Q、K、V 的投影矩阵合并为一个大矩阵以提高效率。

**测试**：`adapters.get_MultiHeadSelfAttention` → `uv run pytest -k test_multihead_self_attention`

---

## 3.5 完整 Transformer 语言模型

### Transformer Block

将 Multi-Head Self-Attention 和 FFN 组合为一个 Transformer Block：

$$\mathbf{h}_1 = \mathbf{x} + \text{MHA}(\text{RMSNorm}(\mathbf{x}))$$

$$\mathbf{h}_2 = \mathbf{h}_1 + \text{FFN}(\text{RMSNorm}(\mathbf{h}_1))$$

### Problem (transformer_block): Transformer Block (3 points)

**构造参数**：`d_model`，`num_heads`，`d_ff`

**测试**：`adapters.get_TransformerBlock` → `uv run pytest -k test_transformer_block`

### Transformer LM

完整的 Transformer 语言模型由以下部分组成：

1. Token Embedding 层
2. $N$ 个 Transformer Block
3. 最终的 RMSNorm
4. LM Head 线性层（输出 vocab_size 维度）

### Problem (transformer_lm): Transformer LM (3 points)

**构造参数**：`vocab_size`，`context_length`，`d_model`，`num_layers`，`num_heads`，`d_ff`，`attn_pdrop`，`residual_pdrop`

**测试**：`adapters.get_TransformerLM` → `uv run pytest -k test_transformer_lm`

---

## 3.6 资源核算 (Resource Accounting)

理解模型的计算和内存需求是训练大型语言模型的关键技能。

### FLOPs 计算

一个矩阵乘法 $A \in \mathbb{R}^{m \times n}$ 与 $B \in \mathbb{R}^{n \times p}$ 的浮点运算数为：

$$\text{FLOPs}(A \times B) = 2mnp$$

（$mn$ 个 dot product，每个 $2p$ 次运算：$p$ 次乘法 + $p$ 次加法）

### 峰值内存分解

| 组件 | 说明 |
|------|------|
| **Parameters** | 模型所有可学习参数 |
| **Gradients** | 每个参数对应一个梯度（与参数同大小） |
| **Optimizer state** | AdamW: 一阶矩 $m$ + 二阶矩 $v$（各与参数同大小） |
| **Activations** | 中间计算结果，用于反向传播 |

### GPT-2 XL 参数

| Hyperparameter | 值 |
|------|------|
| $d_{\text{model}}$ | 1600 |
| $d_{\text{ff}}$ | 6400 |
| `num_heads` | 25 |
| `num_layers` | 48 |
| `context_length` | 1024 |
| `vocab_size` | 50257 |

### Problem (transformer_accounting): Resource Accounting (5 points)

使用 GPT-2 XL 配置回答以下问题（展示计算步骤）：

**(a)** 精确计算模型中的可学习参数总数（不能用 `model.parameters()`）。

**(b)** 以 BF16 精度存储模型权重需要多少字节内存？

**(c)** 使用 AdamW 优化器以 BF16 精度训练模型，峰值训练内存（不含 activations）是多少字节？

**(d)** 模型单次前向传播的 FLOPs 是多少？

**(e)** 在 B200 GPU（2,250 BF16 TFLOPS）上，给定 batch_size=16，MFU (Model FLOP Utilization) 为 0.35 时，每次前向传播需要多少 wall-clock 时间？
