# Section 7: 实验 (Experiments)

> 对应原始 PDF Section 7.1-7.5 (pages 38-47) | 共 10 个 Problem，22 分

运行实验来验证你的实现，并在排行榜上与同学竞争。

---

## 7.1 如何运行实验 (How to Run Experiments)

### Experiment Logging

训练过程中应记录以下信息：

| 记录项 | 说明 |
|--------|------|
| 训练 loss | 每 N 步记录一次 |
| 验证 loss | 每 N 步在验证集上评估 |
| Wall-clock time | 训练的实际耗时 |

可以使用 [Weights & Biases](https://docs.wandb.ai/quickstart) 进行实验跟踪和可视化。

### Problem (experiment_log): 实验日志 (3 points)

截图展示你的实验日志系统。

---

## 7.2 TinyStories 实验

### TinyStories 数据集

TinyStories (Eldan and Li, 2023) 是一个由 GPT-3.5/GPT-4 生成的简单英文故事数据集，适合小规模模型训练和快速迭代。

**数据示例**：

> Once upon a time, there was a shy girl named Lily. She liked to play with her toys at home and didn't like to go outside. One day, her mom asked her to go to the store to buy some food. Lily was scared, but she wanted to help her mom. [...]

### 默认超参数

| 超参数 | 值 |
|--------|------|
| `vocab_size` | 10,000 |
| `context_length` | 256 |
| `d_model` | 512 |
| `d_ff` | ≈ 1,344 (SwiGLU: round(8/3 × 512) to multiple of 64) |
| `num_layers` | 4 |
| `num_heads` | 16 |
| `batch_size` | 256 sequences |
| `total_tokens` | 327,680,000 (= 256 × 256 × 5,000 steps) |
| `max_lr` | 1e-3 |
| `min_lr` | 1e-4 |
| `warmup_iters` | 200 |
| `lr_decay_iters` | 5,000 |
| `weight_decay` | 0.1 |
| `gradient_clipping` | 1.0 |

> 这些超参数预期在约 5,000 次迭代时达到约 **1.0** 的训练 loss。

### 调试建议

- 从小数据子集开始
- 在开始阶段用 SGD 验证 loss 是否下降
- 在开始阶段打印梯度范数，确认没有梯度消失/爆炸
- 如果 loss 不下降，逐步替换为 PyTorch 标准实现来定位问题

> **Low-Resource Tips**:
> - **CPU 训练**：使用 `device = 'cpu'`，可在笔记本上运行
> - **Apple Silicon**：使用 `device = 'mps'`，MPS 后端注意事项：
>   - 不支持 `torch.compile()`
>   - 不支持 TF32 数学模式

### Problem (learning_rate): 学习率调优 (3 points)

在 TinyStories 上训练模型。使用默认超参数（或自行调整），**验证集 loss ≤ 1.45**。

**Deliverable**: 训练/验证损失曲线。描述使用的超参数以及调优过程。

**资源限制**：3 B200-hours

---

### Problem (batch_size_experiment): Batch Size 实验 (1 point)

尝试不同的 batch size 来训练模型。选择至少 3 个不同的 batch size，保持总 token 数恒定。

**Deliverable**: 学习曲线 (loss vs wall-clock time)。对比不同 batch size 的收敛速度和最终质量。

---

### Problem (generate): 文本生成 (1 point)

使用训练好的 TinyStories 模型，在 decoding 任务中生成几段故事文本。

**Deliverable**: 展示生成的文本。定性评估：文本是否连贯？是否像 TinyStories 风格？

---

## 7.3 消融实验 (Ablations)

以下 4 个实验各修改模型的一个组件，观察对训练效果的影响。

### Problem (layer_norm_ablation): 移除 RMSNorm (1 point)

移除 Transformer Block 中的所有 RMSNorm 层，观察训练效果。

**Deliverable**: 训练曲线对比（有 vs 无 RMSNorm）。

---

### Problem (pre_norm_ablation): Post-Norm vs Pre-Norm (1 point)

将 Pre-norm 结构改为 Post-norm 结构：

**Pre-norm（当前使用）**：

$$\mathbf{h} = \mathbf{x} + f(\text{Norm}(\mathbf{x}))$$

**Post-norm（公式 25-28）**：

$$\mathbf{h} = \text{Norm}(\mathbf{x} + f(\mathbf{x}))$$

对 Attention 和 FFN 子层分别应用。注意：Post-norm 结构中，最后一个 Transformer Block 之后的那个额外 RMSNorm 应**移除**（因为每个子层输出已经归一化了）。

**Deliverable**: 训练曲线对比（Pre-norm vs Post-norm）。

---

### Problem (no_pos_emb): NoPE vs RoPE (1 point)

移除 RoPE（即不使用任何位置编码），观察模型是否仍能学习。

**Deliverable**: 训练曲线对比（RoPE vs NoPE）。分析无位置编码的模型能否正常工作。

---

### Problem (swiglu_ablation): SwiGLU vs SiLU-only FFN (1 point)

将 SwiGLU FFN 替换为简单的 SiLU FFN（没有 gating mechanism）：

$$\text{FFN}_{\text{SiLU}}(\mathbf{x}) = W_2 \cdot \text{SiLU}(W_1 \mathbf{x})$$

其中 $W_1 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$，$W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$。

**公平对比**：将 SiLU FFN 的 $d_{\text{ff}}$ 设为 $4 \times d_{\text{model}}$（不使用 8/3 公式），使两种 FFN 的参数量大致相当。

**Deliverable**: 训练曲线对比（SwiGLU vs SiLU-only）。

---

## 7.4 在 OpenWebText 上运行 (Running on OpenWebText)

### OpenWebText 数据集

OpenWebText (OWT) 是从 web crawl 创建的数据集，其文本比 TinyStories 更加真实、复杂和多样化。

**OWT 数据示例**：

> Baseball Prospectus director of technology Harry Pavlidis took a risk when he hired Jonathan Judge.
>
> Pavlidis knew that, as Alan Schwarz wrote in The Numbers Game, "no corner of American culture is more precisely counted, more passionately quantified, than performances of baseball players." [...]
>
> "He freaks us out." Harry Pavlidis [...]

### 注意事项

从 TinyStories 切换到 OWT 后，可能需要**重新调优超参数**，特别是学习率和 batch size。

### Problem (main_experiment): OWT 实验 (2 points)

使用与 TinyStories 相同的模型架构和总训练迭代数，在 OpenWebText 上训练模型。

**Deliverable**:

1. OWT 上的学习曲线。描述 TinyStories 和 OWT 的 loss 差异——如何解释这种差异？
2. OWT 语言模型生成的文本。文本的流畅度如何？为什么在相同模型架构和计算预算下，OWT 模型的输出质量比 TinyStories 模型更差？

**资源限制**：2 B200-hours

---

## 7.5 自定义修改 + 排行榜 (Your Own Modification + Leaderboard)

恭喜你走到这里！在这个部分，你将尝试改进 Transformer 架构，看看你的超参数和架构选择如何与班上其他同学比较。

### 排行榜规则

| 限制项 | 要求 |
|--------|------|
| **Runtime** | 提交最多在 **1 块 B200 GPU** 上运行 **45 分钟** |
| **Data** | 只能使用提供的 **OpenWebText 训练数据集** |
| **其他** | 随意 |

### 改进思路

**1. SOTA 开源 LLM 系列**

- **Llama 3** (Grattafiori et al., 2024)
- **Qwen 2.5** (Yang et al., 2024)

研究这些模型的架构选择和训练技巧。

**2. NanoGPT Speedrun 仓库**

仓库地址：`github.com/KellerJordan/modded-nanogpt`

社区成员发布了很多有趣的 speedrun 预训练修改。

**经典修改示例 — Weight Tying（权重绑定）**：

将输入嵌入层和输出 LM head 的权重绑定在一起（Vaswani et al. Section 3.4, Chowdhery et al. Section 2）。

> **提示**：如果使用 weight tying，可能需要降低 embedding/LM head 初始化的标准差。

**3. 建议的测试流程**

先在 OWT 的小子集或 TinyStories 上测试修改，验证有效后再进行完整的 45 分钟训练。

**4. 关于可推广性**

排行榜中表现好的修改**不一定能推广到更大规模的预训练**。后续课程中的 scaling laws 单元会进一步探讨。

### Problem (leaderboard): 排行榜 (6 points)

在排行榜规则下训练模型，目标是在 **0.75 B200-hours** 内最小化 validation loss。

**Deliverable**:
- 最终 validation loss
- 学习曲线（x 轴为 wall-clock time，≤ 45 分钟）
- 做了什么修改的描述
- 排行榜至少要达到 naive baseline 的 **5.0 loss**

**提交方式**：提交到排行榜仓库 `github.com/stanford-cs336/assignment1-basics-leaderboard`
