# Section 2: Byte-Pair Encoding (BPE) 分词器

> 对应原始 PDF Section 2.1-2.7 (pages 3-12) | 共 7 个 Problem，42 分

本章实现一个完整的字节级 BPE 分词器：从 Unicode 基础知识出发，经过 BPE 训练算法，到分词器的编码/解码，最后在真实数据集上进行实验。

---

## 2.1 Unicode 标准

Unicode 是一种文本编码标准，将字符映射为整数**码点** (code points)。截至 Unicode 17.0（2025 年 9 月发布），标准定义了 159,801 个字符，涵盖 172 种文字。例如，字符 `"s"` 的码点是 115（通常表示为 `U+0073`，其中 `U+` 是常规前缀，`0073` 是 115 的十六进制），字符 `"牛"` 的码点是 29275。在 Python 中，可以用 `ord()` 函数将单个 Unicode 字符转为整数表示，用 `chr()` 函数将整数码点转为对应字符的字符串。

```python
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

### Problem (unicode1): Understanding Unicode (1 point)

**(a)** `chr(0)` 返回什么 Unicode 字符？

**Deliverable**: 一句话回答。

**(b)** 这个字符的字符串表示 (`__repr__()`) 与它的打印表示有何不同？

**Deliverable**: 一句话回答。

**(c)** 当这个字符出现在文本中时会发生什么？用以下代码在 Python 解释器中实验：

```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

**Deliverable**: 一句话回答。

---

## 2.2 Unicode 编码

虽然 Unicode 标准定义了从字符到码点（整数）的映射，但直接在 Unicode 码点上训练分词器是不切实际的，因为词汇表太大（约 15 万项），且许多字符非常罕见。我们将使用 Unicode **编码** (encoding)，将 Unicode 字符转换为字节序列。Unicode 标准定义了三种编码：UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网上的主导编码（超过 98% 的网页使用）。

在 Python 中用 `encode()` 函数将 Unicode 字符串编码为 UTF-8。要访问 Python `bytes` 对象的底层字节值，可以迭代它（如调用 `list()`）。用 `decode()` 函数将 UTF-8 字节串解码为 Unicode 字符串。

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # 获取编码字符串的字节值（0 到 255 的整数）
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 129, 130, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
>>> # 一个字节不一定对应一个 Unicode 字符！
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

通过将 Unicode 码点转换为字节序列（如通过 UTF-8 编码），我们实质上是将一个码点序列（21 位整数，有 159,801 个有效值）转换为一个字节值序列（0 到 255 的整数）。256 长度的字节词汇表要可管理*得多*。使用字节级分词时，我们不需要担心词汇外 (out-of-vocabulary) token，因为**任何**输入文本都可以表示为 0 到 255 的整数序列。

### Problem (unicode2): Unicode Encodings (3 points)

**(a)** 为什么我们更倾向于在 UTF-8 编码的字节上训练分词器，而不是 UTF-16 或 UTF-32？

**Deliverable**: 一到两句话回答。

**(b)** 考虑以下（不正确的）函数，它试图将 UTF-8 字节串解码为 Unicode 字符串。为什么这个函数不正确？请给出一个会产生错误结果的输入字节串示例。

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

**Deliverable**: 一个使 `decode_utf8_bytes_to_str_wrong` 产生错误输出的输入字节串示例，并用一句话解释为什么该函数不正确。

**(c)** 给出一个不能解码为任何 Unicode 字符的两字节序列。

**Deliverable**: 一个例子，附一句话解释。

---

## 2.3 子词分词 (Subword Tokenization)

在字节级表示的基础上，我们希望找到一种分词方案，在**纯字节级分词**和**词级分词**之间取得平衡：

- **字节级分词**的问题：序列过长。每个 Unicode 字符可能需要多个字节表示，导致序列长度大幅增加。更长的序列意味着更高的计算成本（Transformer 的 self-attention 复杂度与序列长度的平方成正比），也使模型更难学习长距离依赖关系。
- **词级分词**的问题：词汇外 (OOV) 问题。对于未在训练数据中出现的词，模型无法处理。此外，词级分词无法捕捉词内的共享子结构（如 "learning" 和 "learned" 共享词根 "learn"）。

**子词分词** (subword tokenization) 是两者的折中方案：将常见的字符序列合并为单个 token，同时保留对罕见序列的字节级表示能力。

我们将实现 **Byte Pair Encoding (BPE)** 分词器。BPE 最初是一种数据压缩算法 (P. Gage [5])，后被 Sennrich et al. [3] 引入到 NLP 中用于子词分词。BPE 的核心思想是迭代地将训练语料中最常见的相邻 token 对合并为新的 token，从而构建子词词汇表。

---

## 2.4 BPE 分词器训练

BPE 分词器的训练过程包含三个主要步骤：

### 步骤 1：词汇表初始化

词汇表 (vocabulary) 是从 bytestring token 到 integer ID 的一一映射。初始词汇表包含 256 个条目，对应 0 到 255 的所有单字节值。

### 步骤 2：预分词 (Pre-tokenization)

预分词将输入文本分割成较小的块，BPE 合并只在每个块内部进行，不会跨块边界。

**原始 BPE** 简单地按空格分割文本。

**现代分词器**（如 GPT-2, Radford et al. [6]）使用基于正则表达式的预分词器：

```python
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

测试预分词器：

```python
>>> # requires `regex` package
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

> **实现提示**：使用 `re.finditer` 而不是 `re.findall`，可以避免将所有预分词的词存储在内存中，改为惰性迭代。

### 步骤 3：计算 BPE 合并

在完成预分词后，迭代执行以下操作直到达到目标词汇表大小：

1. 统计训练语料中所有相邻 token 对的出现频率
2. 选择出现频率最高的 token 对
3. 将该 token 对合并为一个新 token，添加到词汇表中
4. 在训练语料中将所有该 token 对的出现替换为新 token

**平局处理**：当多对 token 具有相同频率时，选择字典序 (lexicographic order) 更大的对：

```python
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```

### 特殊 Token

某些特殊 token（如 `<|endoftext|>`）必须在分词过程中保留为单个 token，不能被拆分或合并。这些特殊 token 被添加到词汇表中，每个有固定的 token ID。

---

### Example (bpe_example): BPE 训练示例

以下是一个 Sennrich et al. [3] 风格的 BPE 训练过程示例。

**训练语料库**：

| 词 | 频率 |
|---|---|
| `l o w` | 5 |
| `l o w e r` | 2 |
| `w i d e s t` | 3 |
| `n e w e s t` | 6 |

词汇表含特殊 token `<|endoftext|>` 和 256 个字节值。

**合并过程**：

**第 1 轮**：`(e, s)` 和 `(s, t)` 均出现 9 次。按字典序 `(s, t)` > `(e, s)`，合并 `(s, t) → st`。

**第 2 轮**：`(e, st)` 出现 9 次，合并为 `est`。

**第 3 轮**：`(l, o)` 出现 7 次，合并为 `lo`。

**第 4 轮**：`(lo, w)` 出现 7 次，合并为 `low`。

**第 5 轮**：`(n, e)` 出现 6 次（`(e, w)` 也是 6 次，但 `(n, e)` 字典序更大），合并为 `ne`。

**第 6 轮**：`(ne, w)` 出现 6 次，合并为 `new`。

取 6 次合并后，词汇表包括：`[<|endoftext|>, ...256 BYTE CHARS, st, est, ow, low, west, ne]`

此时 `newest` 会被编码为 `[ne, west]`。

---

## 2.5 BPE 分词器训练实验

### 实现优化建议

**并行预分词**：使用 Python 的 `multiprocessing` 模块并行处理预分词。将输入文本分成多个 chunk，每个 chunk 由一个进程处理。注意 chunk 边界应在 special token 处分割。

**预分词前移除 special tokens**：在对文本进行正则表达式预分词之前，先用 `re.split` 和 `re.escape` 将 special tokens 从文本中分离出来。

**优化合并步骤**：通过增量更新计数来优化，只更新受合并影响的 pair 的计数，而不是重新统计所有 pair。注意 BPE 训练的合并部分在 Python 中**不可并行化**。

### Low-Resource Tips

**Profiling**：使用 `cProfile` 或 `py-spy` 找出代码中的性能瓶颈。

**Downscaling**：在调试阶段，使用小数据集（如 TinyStories validation set，约 22,000 个文档）而不是完整数据集。

---

### Problem (train_bpe): BPE Tokenizer Training (15 points)

实现 BPE 分词器训练函数。

**函数接口**：

```python
def train_bpe(
    input_path: str,
    vocab_size: int = 256,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练 BPE 分词器。

    Args:
        input_path: str — 训练语料文本文件的路径。
        vocab_size: int — 目标词汇表大小（必须 >= 256）。
        special_tokens: list[str] | None — 特殊 token 列表。

    Returns:
        vocab: dict[int, bytes] — 词汇表映射。
        merges: list[tuple[bytes, bytes]] — BPE 合并列表（按训练顺序）。
    """
```

**测试**：`adapters.run_train_bpe` → `uv run pytest tests/test_train_bpe.py`

**正则引擎说明**：GPT-2 预分词正则使用 Unicode 属性（`\p{L}`、`\p{N}`），需要使用 `regex` 包或 Oniguruma 引擎。

---

### Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)

**(a)** 在 TinyStories 数据集上训练 BPE 分词器（vocab_size=10,000，special token: `<|endoftext|>`）。序列化词汇表和合并列表，报告训练时间、峰值内存、最长 token。

**资源限制**：≤ 30 分钟（无 GPU），≤ 30 GB RAM

**(b)** Profile 你的代码，哪部分最耗时？

---

### Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)

**(a)** 在 OpenWebText 上训练 BPE 分词器（vocab_size=32,000）。

**资源限制**：≤ 12 小时，≤ 100 GB RAM

**(b)** 对比 TinyStories 分词器和 OWT 分词器的差异。

---

## 2.6 BPE 分词器：编码和解码

训练完 BPE 分词器后，我们需要实现两个核心功能：
- **编码 (Encode)**：将文本字符串转换为 token ID 序列
- **解码 (Decode)**：将 token ID 序列转换回文本字符串

### 2.6.1 编码文本

编码过程分为两步：

**Step 1: 预分词 + UTF-8 字节表示**

首先对输入文本进行预分词（使用与训练时相同的正则表达式），然后将每个预分词 token 转换为 UTF-8 字节序列。

**Step 2: 按训练时的合并顺序应用合并**

对每个预分词 token 的字节序列，按照训练时记录的合并顺序依次应用合并操作。

### Example (bpe_encoding): BPE 编码示例

**词汇表**：`{0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b'at', 8: b' c', 9: b'the', 10: b' at'}`

**合并列表**：`[(b't', b'h'), (b'a', b't'), (b'th', b'e'), (b' ', b'c')]`

**编码 `'the cat ate'`**：

预分词 → `['the', ' cat', ' ate']`

- `'the'` → `[b't', b'h', b'e']` → 合并 `(t,h)→th` → `[b'th', b'e']` → 合并 `(th,e)→the` → `[b'the']` → `[9]`
- `' cat'` → `[b' ', b'c', b'a', b't']` → 合并 `(a,t)→at` → `[b' ', b'c', b'at']` → 合并 `(' ',c)→' c'` → `[b' c', b'at']` → `[8, 7]`
- `' ate'` → `[b' ', b'a', b't', b'e']` → 合并 `(a,t)→at` → `[b' ', b'at', b'e']` → 无更多合并 → `[0, 7, 3]`

**最终编码结果**：`[9, 8, 7, 0, 7, 3]`

### Special Tokens 处理

编码时在正则预分词**之前**，先将输入文本按 special tokens 分割。Special tokens 直接映射为对应的 token ID。

### Memory Considerations

对于大文件，应分块处理。需要确保 token 不会跨 chunk 边界被拆分。

### 2.6.2 解码文本

1. 对每个 token ID，查找词汇表中对应的字节序列
2. 拼接所有字节序列
3. 将字节串解码为 Unicode 字符串

对于无效 Unicode 字节序列，使用 `errors='replace'` 替换为 `U+FFFD`：

```python
>>> b'\x80\x81'.decode("utf-8", errors="replace")
'��'
```

---

### Problem (tokenizer): Implementing the Tokenizer (15 points)

实现 `Tokenizer` 类：

```python
class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None): ...

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str,
                   special_tokens: list[str] | None = None): ...

    def encode(self, text: str) -> list[int]: ...

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]: ...

    def decode(self, ids: list[int]) -> str: ...
```

**测试**：`adapters.get_tokenizer` → `uv run pytest tests/test_tokenizer.py`

---

## 2.7 Experiments

### Problem (tokenizer_experiments): Experiments with Tokenizers (4 points)

**(a)** 从 TinyStories 和 OpenWebText 各随机抽取 10 个文档，分别用 10K 和 32K 词汇表分词器编码。计算压缩率 (bytes/token)。

**(b)** 用 TinyStories 分词器对 OWT 样本分词的结果如何？为什么？

**(c)** 估算分词器吞吐量 (bytes/s)，估算对 The Pile (825 GB) 完整分词需要多久。

**(d)** 将训练/验证集编码为 token ID 序列，序列化为 `uint16` NumPy 数组。为什么 `uint16` 合适？
