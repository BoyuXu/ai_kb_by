# 序列建模演进：从 RNN 到 Mamba

> **一句话总结**：用户的行为是一个时间序列，怎么从这个序列里提取"兴趣"是推荐/搜索/LLM的核心问题。从 RNN 的逐步遗忘，到 Transformer 的全局注意力，到 Mamba 的线性复杂度——每一代都在解决上一代的瓶颈。
>
> **为什么要学**：序列建模是推荐系统区别于传统机器学习的核心能力，也是理解 LLM 的基础。

**相关概念页**：[Attention in RecSys](attention_in_recsys.md) | [Embedding全景](embedding_everywhere.md) | [生成式推荐](generative_recsys.md)

---

## 1. RNN / GRU / LSTM：逐步处理序列

### 核心思想
按时间顺序一步一步处理，每步更新一个"记忆状态"：

$$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t)$$

### GRU（推荐系统常用）

$$\mathbf{z}_t = \sigma(W_z [\mathbf{h}_{t-1}, \mathbf{x}_t])  \quad \text{（更新门：保留多少旧信息）}$$
$$\mathbf{r}_t = \sigma(W_r [\mathbf{h}_{t-1}, \mathbf{x}_t])  \quad \text{（重置门：遗忘多少旧信息）}$$
$$\tilde{\mathbf{h}}_t = \tanh(W [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])$$
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

### 在推荐中的应用
- **GRU4Rec**（2015）：第一个用 RNN 做 session-based 推荐
- **DIEN**（2019）：GRU + Attention 控制更新门 → AUGRU

### 瓶颈
| 问题 | 原因 | 后果 |
|------|------|------|
| 长距离遗忘 | 梯度消失/爆炸 | 序列>100就丢信息 |
| 无法并行 | 必须按时间顺序计算 | 训练慢 |
| 固定容量 | 隐状态维度固定 | 信息瓶颈 |

---

## 2. Transformer / Self-Attention：全局并行

### 核心思想
不再逐步处理，而是让序列中**任意两个位置直接交互**：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

### 关键改进
| 特性 | RNN | Transformer |
|------|-----|-------------|
| 距离建模 | 靠传递（越远越弱） | 直接交互（任意距离一样） |
| 并行性 | 串行 | 完全并行 |
| 记忆容量 | 固定隐状态 | 随序列长度线性增长（KV Cache） |

### 在推荐中的应用

| 模型 | 方式 | Mask | 适用场景 |
|------|------|------|---------|
| SASRec | 自回归 | Causal（只看左边） | Next-item prediction |
| BERT4Rec | 双向 | Random mask | 序列理解/补全 |
| BST | 编码器 | 无 mask | 特征提取 |
| HSTU | ReLU 替代 softmax | Causal | 万亿参数推荐模型 |

### 瓶颈
- **$O(N^2)$ 复杂度**：序列长度 N 的二次方 → 长序列（>2K）很慢
- **KV Cache 内存**：推理时缓存所有历史 KV → 内存爆炸

📄 详见 [Attention in RecSys](attention_in_recsys.md)

---

## 3. 长序列优化方案

面对 Transformer 的 $O(N^2)$ 瓶颈，推荐和 NLP 分别探索了不同的路径：

### 推荐系统的方案：先剪枝再 Attention

| 方法 | 思路 | 复杂度 | 序列上限 |
|------|------|--------|---------|
| SIM | 先检索 top-K，再 Attention | $O(K)$ | 10K+ |
| ETA | SimHash 近似检索 | $O(1)$ lookup | 10K+ |
| SDIM | 多头 hash 采样 | $O(K)$ | 10K+ |
| SparseCTR | 个性化分块+三分支稀疏注意力 | $O(N \cdot w)$ | 10K+ |

**直觉**：推荐场景下，万级行为中只有少数跟候选相关 → 不需要全部做 Attention。

### NLP/LLM 的方案：修改 Attention 本身

| 方法 | 思路 | 复杂度 |
|------|------|--------|
| Sparse Attention | 只看固定窗口+间隔位置 | $O(N\sqrt{N})$ |
| Linear Attention | kernel 近似替代 softmax | $O(N)$ |
| FlashAttention | 不改数学，改访存 | $O(N^2)$ 但实际快 2x |
| Sliding Window | 固定窗口（Mistral） | $O(Nw)$ |

📄 FlashAttention 详见 [Attention in RecSys](attention_in_recsys.md) §6

---

## 4. Mamba / SSM：线性复杂度的新范式

### 核心思想
**状态空间模型（State Space Model）**：把连续动力系统离散化

$$\mathbf{h}_t = \bar{A} \mathbf{h}_{t-1} + \bar{B} \mathbf{x}_t$$
$$\mathbf{y}_t = C \mathbf{h}_t$$

### Mamba 的关键创新：选择性机制

普通 SSM 的 $A, B$ 是固定参数 → 对所有输入一视同仁。

Mamba 让 $B, C, \Delta$（步长）**依赖于输入**：
- 重要的输入 → 大步长 → 多写入状态
- 不重要的输入 → 小步长 → 快速跳过

**直觉**：类似 GRU 的门控，但在连续空间中操作。

### 为什么比 Transformer 快？

| 维度 | Transformer | Mamba |
|------|-------------|-------|
| 训练复杂度 | $O(N^2)$ | $O(N)$ |
| 推理复杂度 | $O(N)$（有 KV Cache） | $O(1)$（固定状态） |
| 推理内存 | $O(N)$（KV Cache 线性增长） | $O(1)$（固定大小状态） |
| 并行训练 | ✅ | ✅（scan 并行） |
| 长距离建模 | ✅（直接） | 通过状态压缩（有损） |

### 在推荐中的潜力
- 万级用户行为序列 → Mamba 天然适合
- 推理时固定状态大小 → 延迟可控
- 选择性遗忘 → 天然过滤无关行为

**但**：截至 2026，Mamba 在推荐中的工业应用还不多，Transformer + 剪枝（SIM/ETA）仍是主流。

---

## 5. 各场景序列建模对比

| 场景 | 序列内容 | 典型长度 | 主流方案 |
|------|---------|---------|---------|
| 推荐排序 | 用户点击/购买历史 | 50-200 | DIN/DIEN → Transformer |
| 推荐长序列 | 用户全量行为 | 1K-100K | SIM/ETA（检索+Attention） |
| 搜索排序 | query-doc 交互 | 128-512 tokens | Cross-Encoder Transformer |
| LLM | 自然语言 | 4K-128K tokens | Transformer + FlashAttention |
| LLM 长上下文 | 超长文本 | 128K-1M | Transformer + Rope + Mamba 混合 |

---

## 6. 演进总结

```
RNN/GRU (2015-2018) ──── 顺序处理，长距离遗忘
    │                    推荐：GRU4Rec, DIEN
    ▼
Transformer (2017-now) ── 全局注意力，O(N²)
    │                    推荐：SASRec, BST, HSTU
    │
    ├─ 推荐路线 ──────── SIM/ETA: 先检索再 Attention
    │                    解决万级序列，保持精度
    │
    ├─ NLP路线 ──────── FlashAttention: 不改数学改硬件
    │                    + Sparse/Sliding Window
    │
    └─ SSM路线 ──────── Mamba: 选择性状态空间模型
                         O(N) 训练，O(1) 推理
                         推荐领域刚起步
```

## 面试高频问题

1. **GRU 和 LSTM 的区别？推荐为什么更常用 GRU？** → GRU 2 个门 vs LSTM 3 个门，GRU 参数少推理快，推荐场景序列较短，GRU 够用。
2. **Transformer 在推荐中和 NLP 中用法有什么区别？** → 推荐通常用 causal mask 做 next-item，序列较短（<200），embedding 是 ID 不是 word；NLP 序列长，token 是词。
3. **为什么推荐的长序列方案是 SIM 而不是 Linear Attention？** → 推荐场景下相关行为是稀疏的（万级行为中只有几十个相关），先检索再精算更高效；Linear Attention 适合全局信息的场景。
4. **Mamba 能替代 Transformer 吗？** → 某些场景可以（长序列推理效率高），但 Transformer 在需要精确远距离依赖的任务上仍有优势。混合架构（Transformer + Mamba）可能是方向。
