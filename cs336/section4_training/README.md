# Section 4: 训练 (Training)

> 对应原始 PDF Section 4.1-4.5 (pages 28-34) | 共 6 个 Problem，8 分

有了模型，接下来需要训练它。本章涵盖：损失函数、优化器 (SGD/AdamW)、学习率调度、梯度裁剪。

---

## 4.1 交叉熵损失 (Cross-Entropy Loss)

语言建模的目标是最大化模型生成训练数据的概率。等价于最小化负对数似然，即**交叉熵损失**。

### 损失公式（公式 16-17）

给定输入 token 序列 $x_{1:m}$，模型对第 $t$ 个位置的预测是一个概率分布 $\hat{p}_t$：

$$\hat{p}_t(w) = P_\theta(x_t = w \mid x_{1:t-1})$$

交叉熵损失为：

$$\mathcal{L} = -\frac{1}{m} \sum_{t=1}^{m} \log \hat{p}_t(x_t)$$

### Perplexity（公式 18）

$$\text{PPL} = \exp(\mathcal{L})$$

Perplexity 可以直观理解为"模型在每个 token 上的等效选择数"。

### 数值稳定性

为了避免 log-softmax 的数值问题：
- 先对 logits 减去最大值
- 在 log-space 中计算

**重要**：交叉熵是在 **logits 上**计算的（不是在 softmax 之后的概率上），以确保数值稳定性。

### Problem (cross_entropy): Cross-Entropy Loss (1 point)

实现交叉熵损失函数。

**输入**：logits $(B, s, V)$，targets $(B, s)$

**输出**：标量损失值

**测试**：`adapters.run_cross_entropy` → `uv run pytest -k test_cross_entropy`

---

## 4.2 SGD 优化器

### 更新规则（公式 19-20）

$$g_t = \nabla_\theta \mathcal{L}(\theta_{t-1})$$

$$\theta_t = \theta_{t-1} - \eta \cdot g_t$$

### SGD 完整实现参考

```python
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        super().__init__(params, defaults=dict(lr=lr))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is not None:
                    p -= lr * p.grad
```

### 训练循环示例

```python
optimizer = SGD(model.parameters(), lr=0.01)
for batch in dataloader:
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Problem (learning_rate_tuning): SGD 学习率调优 (1 point)

使用 SGD 优化器训练模型，找到训练 100 步后能达到最低 loss 的学习率。

**Deliverable**: 用不同学习率训练的损失曲线。

---

## 4.3 AdamW 优化器

### Algorithm 1: AdamW（完整伪代码）

参见 **Algorithm 1**（`images/algorithm1_adamw.png`）。

**AdamW 算法步骤**：

1. **输入**：初始参数 $\theta_0$，学习率 $\alpha$，衰减率 $\beta_1, \beta_2$，正则化常数 $\lambda$，稳定性常数 $\epsilon$
2. **初始化**：$m_0 = 0$，$v_0 = 0$，$t = 0$
3. **循环** 直到收敛：
   - $t = t + 1$
   - $g_t = \nabla_\theta \mathcal{L}(\theta_{t-1})$
   - $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ （一阶矩估计）
   - $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$ （二阶矩估计）
   - $\hat{m}_t = m_t / (1 - \beta_1^t)$ （偏差校正）
   - $\hat{v}_t = v_t / (1 - \beta_2^t)$ （偏差校正）
   - $\theta_t = \theta_{t-1} - \alpha (\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) + \lambda \theta_{t-1})$ （参数更新 + 权重衰减）

### AdamW 默认超参数

| 超参数 | 默认值 |
|--------|--------|
| $\alpha$ (learning rate) | 0.001 |
| $\beta_1$ | 0.9 |
| $\beta_2$ | 0.999 |
| $\epsilon$ | 1e-8 |
| $\lambda$ (weight decay) | 0.0 |

### Problem (adamw): AdamW Optimizer (2 points)

实现 AdamW 优化器。

**注意**：$\theta_{t-1}$ 是 weight decay **之前**的值。

**测试**：`adapters.get_AdamW` → `uv run pytest -k test_adamw`

### Problem (adamw_accounting): AdamW 资源核算 (2 points)

使用 GPT-2 XL 配置、batch size 16、context length 1024：

**(a)** 使用 AdamW 训练一步（前向 + 反向 + 更新）需要多少 FLOPs？

> **提示**：前向传播 FLOPs 已在 `transformer_accounting` 中计算。反向传播的 FLOPs 约为前向的**两倍**。

**(b)** MFU 为 0.35 时，在 B200 GPU 上训练一步需要多少 wall-clock time？

---

## 4.4 学习率调度 (Learning Rate Schedule)

### Cosine Annealing with Warmup（公式 21-22）

学习率调度分为三个阶段：

**Phase 1: Linear Warmup**（$t < T_w$）

$$\eta(t) = \eta_{\max} \cdot \frac{t}{T_w}$$

**Phase 2: Cosine Annealing**（$T_w \leq t < T_c$）

$$\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\pi \cdot \frac{t - T_w}{T_c - T_w}\right)\right)$$

**Phase 3: Post-annealing**（$t \geq T_c$）

$$\eta(t) = \eta_{\min}$$

其中：
- $T_w$：warmup 步数
- $T_c$：cosine annealing 结束步数
- $\eta_{\max}$：最大学习率
- $\eta_{\min}$：最小学习率（通常为 $\eta_{\max}$ 的 10%）

### Problem (learning_rate_schedule): Cosine LR Schedule (1 point)

实现 cosine learning rate schedule，包含线性 warmup 和 post-annealing 阶段。

**测试**：`adapters.get_lr_cosine_schedule` → `uv run pytest -k test_lr_schedule`

---

## 4.5 梯度裁剪 (Gradient Clipping)

当梯度的 L2 范数超过阈值 $c$ 时，将所有梯度等比缩小：

$$\mathbf{g} \leftarrow \begin{cases} \mathbf{g} & \text{if } \|\mathbf{g}\|_2 \leq c \\ c \cdot \frac{\mathbf{g}}{\|\mathbf{g}\|_2 + \epsilon} & \text{if } \|\mathbf{g}\|_2 > c \end{cases}$$

其中 $\|\mathbf{g}\|_2$ 是所有参数梯度拼接后的全局 L2 范数，$\epsilon$ 用于防止除零。

### Problem (gradient_clipping): Gradient Clipping (1 point)

实现基于 L2 范数的梯度裁剪。

**测试**：`adapters.run_gradient_clipping` → `uv run pytest -k test_gradient_clipping`
