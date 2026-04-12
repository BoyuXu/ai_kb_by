# 序列建模演进：DIN 到 Mamba

> **创建日期**: 2026-04-13 | **合并来源**: 用户行为序列建模, 长序列用户行为建模技术演进, 序列推荐高效注意力与状态空间模型_20260411, 20260411_sequential_and_generative_rec
>
> **核心命题**: 用户行为序列建模从 Pooling 到 Target Attention 到 Transformer 到 SSM/Mamba，核心挑战是长序列效率与信息保留的平衡

---

## 一、技术演进脉络

```
Pooling (Sum/Mean)
  → DIN (Target Attention, 2018)
    → DIEN (兴趣演化 GRU, 2019)
      → BST (Transformer, 2019)
        → SIM (超长序列两阶段检索, 2020)
          → ETA (LSH 加速, 2021)
            → SparseCTR (稀疏注意力, 2022)
              → HSTU (去 softmax Transformer, 2024)
                → Mamba4Rec/SIGMA (SSM, 2025)
                  → FuXi-Linear / M2Rec (线性注意力/多尺度Mamba, 2026)

效率演进: O(n²) → O(n·k) → O(n) → O(1) 推理
```

---

## 二、核心模型详解

### 2.1 DIN — Target Attention 的起点

$$
\text{weight}_i = \text{softmax}(f(e_{item}, e_{history_i}))
$$
$$
v_U = \sum_i \text{weight}_i \cdot e_{history_i}
$$

**核心洞察**：用户兴趣是多样的，不同候选物品应该激活不同的历史行为。DIN 用候选 item 作为 query 对历史行为加权，而非简单 mean pooling。

**DIN vs 标准 Self-Attention**：DIN 是 target-aware（候选 item 作 query），标准 attention 是自注意力。

### 2.2 DIEN — 兴趣演化建模

$$
h_t = \text{GRU}(e_t, h_{t-1})
$$
$$
h'_t = \text{AUGRU}(h_t, h'_{t-1}, \alpha_t)
$$

两层 GRU 结构：
1. **兴趣提取层**（GRU）：从行为序列提取兴趣状态
2. **兴趣演化层**（AUGRU）：用 target attention 引导兴趣的演化方向

辅助损失：用 next-item prediction 监督兴趣提取层。

### 2.3 SIM — 超长序列建模 (阿里)

解决万级行为序列的计算问题，两阶段架构：

**Stage 1: General Search Unit (GSU)** — 从数万行为中快速检索相关子序列
- Hard Search：按 Category/Brand 属性匹配
- Soft Search：向量 embedding + FAISS/ScaNN Top-K 检索

**Stage 2: Exact Search Unit (ESU)** — 对子序列精确建模
- Multi-Head Attention / Transformer
- 复杂度从 O(n²) n=10000+ 降到 O(m²) m=50-200

### 2.4 ETA — LSH 加速的端到端长序列

$$
\text{SimHash}(x) = \text{sign}(Rx), \quad R \sim \mathcal{N}(0, 1)
$$

用 Locality Sensitive Hashing 替代 SIM 的 GSU 阶段，端到端可训练，避免硬搜索的信息损失。

### 2.5 HSTU — 生成式序列建模 (Meta)

$$
\mathbf{O} = \text{silu}(Q) \cdot \text{silu}(K)^\top \cdot V \cdot M_{\text{causal}}
$$

- 去掉 softmax 和 LayerNorm，用 silu 替代
- 推理加速 3-5 倍，适合千级 token 序列
- 统一召回与排序为单一自回归模型
- 验证推荐领域 Scaling Law 存在性

### 2.6 SORT — 工业级排序 Transformer (电商)

四步全栈优化：
1. **Request-centric sample**：同请求样本聚合在同 batch
2. **Local Attention**：请求内注意力替代 full attention
3. **Query Pruning**：剪掉低价值查询
4. **Generative Pre-Training**：特征序列自回归预训练

$$
\mathcal{L}_{\text{GPT}} = -\sum_{i} \log P(f_i | f_1, ..., f_{i-1})
$$

实验：orders +6.35%, GMV +5.47%, latency -44.67%, throughput +121.33%。

---

## 三、O(n²) 的替代路径

### 3.1 路径对比

| 方法 | 架构类型 | 复杂度 | 核心创新 | 关键指标 |
|------|---------|--------|---------|---------|
| FuXi-Linear | 线性注意力 | O(n) | 时序保持+线性位置双通道 | prefill 10x加速, decode 21x加速 |
| BlossomRec | 稀疏注意力 | O(n·k), k<<n | 长短期双稀疏+门控融合 | 内存显著减少 |
| SIGMA | Mamba (SSM) | O(n) | PF-Mamba双向 + DS Gate + FE-GRU | 5数据集 SOTA |
| M2Rec | Mamba+FFT+LLM | O(n) | 多尺度时序/频域/语义 | HR@10 +3.2% |

### 3.2 FuXi-Linear — 线性注意力

**核心问题**：直接将线性注意力用于序列推荐效果不佳，因为时序衰减和语义相关性耦合。

**Temporal Retention Channel**（时序保持通道）：
$$
\text{TRC}(Q, K, V, \Delta t) = \sum_{j \leq i} \phi(Q_i) \cdot \phi(K_j)^T \cdot V_j \cdot \gamma(\Delta t_{ij})
$$

$\phi(\cdot)$ 实现线性化，$\gamma(\Delta t_{ij})$ 是周期性时间衰减函数，独立于语义通道。

**Linear Positional Channel**（线性位置通道）：
$$
\text{LPC}(x) = \sum_{j \leq i} k_{\text{pos}}(i, j) \cdot V_j
$$

在 O(n) 复杂度内编码位置关系。在千级 token 长度下观察到 power-law scaling。

### 3.3 SIGMA — 双向 Mamba 序列推荐

**问题**：原始 Mamba 是单向的，但推荐中双向上下文重要。

**PF-Mamba（Partially Flipped Mamba）**：
$$
h_{\text{bi}} = \text{DS-Gate}(h_{\text{forward}}, h_{\text{backward}})
$$

DS Gate（Dense Selective Gate）是 input-sensitive 的，根据输入动态调整前向/后向权重。

**FE-GRU（Feature Extract GRU）**：
$$
h_{\text{short}} = \text{GRU}(x_{t-w:t})
$$

专门捕捉短期依赖，弥补 Mamba 状态估计不稳定导致的短期模式遗漏。

### 3.4 BlossomRec — 块级稀疏注意力

$$
\text{Output} = \sigma(g) \odot \text{Attn}_{\text{long-term}}(x) + (1 - \sigma(g)) \odot \text{Attn}_{\text{short-term}}(x)
$$

长期：全局采样稀疏 pattern；短期：局部窗口稀疏 pattern；$g$ 为可学习门控。

### 3.5 M2Rec — 多尺度 Mamba

$$
h_{\text{final}} = \alpha \cdot h_{\text{Mamba}} + \beta \cdot h_{\text{FFT}} + \gamma \cdot h_{\text{LLM}}
$$
$$
[\alpha, \beta, \gamma] = \text{softmax}(\text{MLP}([h_{\text{Mamba}}; h_{\text{FFT}}; h_{\text{LLM}}]))
$$

FFT 通道：$h_{\text{FFT}} = \text{IFFT}(\text{Filter}(\text{FFT}(x)))$，频域分离周期性趋势和噪声。

---

## 四、方法选择指南

| 场景 | 推荐方法 | 理由 |
|------|---------|------|
| 序列 < 200 | SASRec (标准 Attention) | O(n²) 可接受，效果最好 |
| 序列 200-500 | DIN/BST | 工业验证充分 |
| 序列 500-2000 | BlossomRec 或 FuXi-Linear | 稀疏/线性注意力平衡效率效果 |
| 序列 > 2000 | FuXi-Linear | 线性复杂度 + power-law scaling |
| 超长序列（万级）| SIM | 两阶段检索，工业验证 |
| 需要双向上下文 | SIGMA | PF-Mamba 双向建模 |
| 数据稀疏 + 频域 | M2Rec | 多尺度融合补充信息 |
| 统一召回+排序 | HSTU | 生成式统一，最前沿 |

---

## 五、工业实践经验

### 5.1 长序列处理工程要点
- **序列截断**：基于时间衰减采样限制长度（通常 200-500）
- **KV-cache 增量推理**：新 action 只增量计算，复杂度 O(1)，比全序列快 10 倍
- **INT8 量化 + 蒸馏**：模型体积压缩 1/4
- **批处理调度**：多用户请求打包 batch，利用 GPU 并行

### 5.2 LinkedIn Feed-SR 工业经验
- Transformer 序列排序在工业场景验证：time spent +2.10%
- 工业约束：低延迟、可解释性、稳定性优先

### 5.3 时序-语义解耦是关键（FuXi-Linear）
- 直接线性注意力效果不佳，需要独立处理时间信号
- Temporal Retention Channel 避免串扰

---

## 六、面试高频考点

**Q1: DIN 的 target attention 和标准 attention 区别？**
A: DIN 是 target-aware：用候选 item 作 query 对历史行为加权。标准 attention 是自注意力（序列内部元素互相关注）。DIN 的核心公式：$\text{weight}_i = \text{softmax}(f(e_{item}, e_{history_i}))$。

**Q2: 为什么标准 Transformer 不适合超长序列推荐？**
A: Self-attention O(n²) 复杂度，n>1000 时计算和内存不可接受。推荐场景用户行为序列动辄上千甚至上万，需要亚二次复杂度方案。

**Q3: 线性注意力的核心原理？**
A: 将 softmax(QK^T)V 分解为 φ(Q)(φ(K)^T V)，利用结合律将复杂度从 O(n²d) 降到 O(nd²)。当 d << n 时效率提升巨大。FuXi-Linear 进一步引入时序保持通道避免信号丢失。

**Q4: Mamba 相比 Transformer 的优劣？**
A: 优势：O(n) 复杂度；隐式压缩历史，推理 O(1) 内存。劣势：(1) 天然单向，需 PF-Mamba 改造；(2) 状态估计不稳定，短期依赖差；(3) 缺乏显式注意力权重，可解释性弱。

**Q5: SIM 的两阶段设计为什么必要？**
A: 万级行为序列全部 attention 不可行。GSU 快速筛选（O(n) 或 O(1) 检索），ESU 精确建模（O(m²) m远小于n）。Hard Search 简单但有信息损失，Soft Search（向量检索）端到端更优但需要维护 ANN 索引。

**Q6: SIGMA 的 DS Gate 和普通门控区别？**
A: 普通门控权重固定或仅依赖位置，DS Gate 是 input-sensitive——根据当前输入动态调整前向/后向 Mamba 融合权重。某些 position 前向信息更重要（购买序列因果性），某些需要后向（浏览序列上下文补充）。

**Q7: M2Rec 为什么引入 FFT？**
A: 用户行为有周期性（每周末购物、每天通勤听播客），纯时域模型难以显式捕捉。FFT 转换到频域，可学习滤波器分离周期趋势和噪声，再 IFFT 转回。

**Q8: 生成式序列模型（HSTU）的在线延迟如何控制在 10-20ms？**
A: (1) KV-cache 增量推理（O(1)）；(2) Pointwise Aggregated Attention 去 softmax；(3) INT8 量化+蒸馏压缩 1/4；(4) 批处理调度利用 GPU 并行；(5) 序列长度截断 200-500。

---

## 相关概念

- [[concepts/sequence_modeling_evolution|序列建模演进]]
- [[concepts/attention_in_recsys|Attention 在搜广推中的演进]]
- [[concepts/generative_recsys|生成式推荐统一视角]]
