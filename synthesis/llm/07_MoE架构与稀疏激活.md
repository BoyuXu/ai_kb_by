# MoE 架构与稀疏激活：从路由机制到分布式推理

> 综合自 MoE架构设计与推理优化 | 更新：2026-04-13 | 领域：MoE/稀疏激活
> 关联：[[concepts/attention_in_recsys]] | [[01_LLM推理优化全景]]

---

## 创新点 vs 之前方案

| 维度 | 之前方案 | MoE 创新 |
|------|---------|---------|
| 模型结构 | Dense FFN（所有参数参与每次计算） | 稀疏激活：N 个 Expert 只激活 Top-K |
| 路由粒度 | Switch: Top-1, Mixtral: 8 大 Expert | DeepSeek-V3: 256 小 Expert + 共享 Expert |
| KV Cache | 完整 Multi-Head KV 缓存 | MLA 低秩压缩 KV Cache 10x+ |
| 推理部署 | TP 切 head，Attention+FFN 共存 | MegaScale-Infer：解耦部署 + 乒乓流水线 |
| 通信 | NCCL All-Reduce（密集集合通信） | M2N 零拷贝 RDMA（稀疏点对点 token dispatch） |

---

## 一、MoE 核心公式

### Gate（路由机制）

$$
G(x) = \text{TopK}(\text{softmax}(W_g \cdot x))
$$

$$
y = \sum_{i \in \text{TopK}} G_i(x) \cdot E_i(x)
$$

- $x \in \mathbb{R}^d$：输入 token 隐藏表示
- $W_g \in \mathbb{R}^{N \times d}$：门控权重，$N$ 为 Expert 数
- $G_i(x)$：token 分配给 Expert $i$ 的门控权重
- $E_i(x)$：第 $i$ 个 Expert（小型 FFN）输出
- TopK：只保留得分最高的 K 个 Expert（$K=1$ 最高效，$K=2$ 效果更好）

**直觉**：Router 是调度员，看到 token 后决定交给哪几个专家处理，输出是各专家的加权和。

### 负载均衡 Auxiliary Loss

$$
\mathcal{L}_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i
$$

- $f_i$：实际分配到 Expert $i$ 的 token 比例
- $P_i$：Router 给 Expert $i$ 的平均概率
- $\alpha$：平衡系数（0.01-0.1）

**直觉**：某 Expert 同时被大量 token 选中（$f_i$ 大）且 Router 给高概率（$P_i$ 大）时，loss 惩罚不均衡。最小化时趋向 $f_i = P_i = 1/N$。

---

## 二、技术演进时间线

| 年份 | 模型 | Expert 数 | 激活数 | 核心设计 |
|------|------|----------|--------|---------|
| 2021 | Switch Transformer (Google) | 大 Expert | Top-1 | 首次大规模验证 MoE |
| 2023 | Mixtral 8x7B (Mistral) | 8 大 Expert | Top-2 | 首个好用的开源 MoE |
| 2024 | DeepSeek-V2 | MoE + MLA | Top-K | KV Cache 压缩 10x |
| 2025-Q1 | DeepSeek-V3 (671B) | 256 小 Expert | Top-8 + 共享 | 训练成本仅 Dense 的 1/3 |
| 2025-Q2 | MegaScale-Infer (字节) | - | - | Attention/FFN 解耦部署 |
| 2025 | Qwen3-235B | 128 Expert | 8 激活 | 大规模稀疏 |

---

## 三、DeepSeek-V3 精细路由设计

- **256 小 Expert**（vs Mixtral 8 大），每 token 选 top-8 激活
- **1 个共享 Expert** 永远激活处理通用知识
- Router 只是 linear(hidden_dim → 256)，开销可忽略
- 总参 671B 但每 token 只激活 37B

### MLA（Multi-head Latent Attention）

KV Cache 压缩到低秩 latent 向量，推理时只缓存 latent 而非完整 K/V。KV Cache 减少 10x+。

与 GQA 互补：GQA 减少 head 数，MLA 压缩维度。

---

## 四、MegaScale-Infer 解耦架构

### 核心思想

Attention 层（计算密集）和 FFN/Expert 层（内存密集）特性完全不同，分开部署到不同 GPU 组。

```
Attention GPU Group → [Token Dispatch M2N] → Expert GPU Group
Expert GPU Group   → [Results Return]     → Attention GPU Group
```

### 乒乓流水线（核心 trick）

```
Attn GPU:   [处理 batch-A] [处理 batch-B] [处理 batch-C]
Expert GPU:      [处理 batch-A] [处理 batch-B]
通信:        A→E   B→E   C→E   A←E   B←E

关键：通信与计算重叠 → 通信延迟几乎被隐藏
```

### M2N 通信库 vs NCCL

| 特性 | NCCL | M2N |
|------|------|-----|
| 通信模式 | 集合通信（All-Reduce） | 稀疏点对点（token dispatch） |
| 初始化 | 秒级 | 毫秒级 |
| 适用场景 | Dense 模型 TP | MoE Expert Parallelism |
| 拷贝 | GPU→CPU→GPU | 零拷贝 RDMA |

---

## 五、MoE vs Dense 选型

| 场景 | 推荐 | 原因 |
|------|------|------|
| 算力受限、追求能力 | MoE | 同推理成本能力更强 |
| 简单部署 | Dense | 无需 Expert Parallelism |
| 边端设备 | Dense | MoE 内存 = 总参数量 |
| 超大规模训练 | MoE | 训练成本仅 Dense 的 1/3 |

---

## 六、核心洞察

1. **MoE = "大模型小计算"**：DeepSeek-V3 总参 671B 但每 token 只激活 37B，效果接近同参数 Dense 但推理成本 1/18
2. **Router 是灵魂**：Top-K routing 主流，$K=1$ 最高效，$K=2$ 效果更好
3. **负载均衡是训练核心挑战**：Expert Collapse 用 Auxiliary Loss 解决
4. **MoE 推理需专用架构**：token-to-expert 的 AllToAll 通信是瓶颈
5. **共享 Expert = 全科医生**：保证基础能力，专科 Expert 负责差异化

---

## 面试高频 Q&A

### Q1: MoE 模型的基本结构？
**30秒**：每个 Transformer 层的 FFN 替换为 N 个 Expert + 1 个 Router。Router 对每个 token 选 Top-K Expert，输出 = $\sum(G_i \times E_i(x))$。$K=1$ 最高效，$K=2$ 效果更好。

### Q2: 专家坍缩怎么解决？
**30秒**：加 Auxiliary Loss $\mathcal{L}_{aux} = \alpha N \sum f_i P_i$，强制均匀分配。$\alpha$ 太大影响主 loss，太小不均衡，通常 0.01-0.1。

### Q3: MoE 推理的通信挑战？
**30秒**：EP 将不同 Expert 放不同 GPU，每 token 需 AllToAll 通信。减少：Expert 分组本地化、增大 batch 均摊、MegaScale-Infer 解耦架构。

### Q4: DeepSeek-V3 精细路由设计？
**30秒**：256 小 Expert（vs Mixtral 8 大），每 token 选 top-8 激活。1 个共享 Expert 永远激活处理通用知识。Router 只是 linear(hidden_dim → 256)，开销可忽略。

### Q5: MLA 是什么？
**30秒**：DeepSeek-V2 将 KV Cache 压缩到低秩 latent 向量，推理时只缓存 latent 而非完整 K/V。减少 10x+。与 GQA 互补：GQA 减少 head 数，MLA 压缩维度。

### Q6: MoE vs Dense 怎么选？
**30秒**：算力受限选 MoE（同推理成本能力更强）；简单部署选 Dense（无需 EP）；边端选 Dense（MoE 内存 = 总参数量）。

### Q7: FFN 层为什么是"内存密集"？
**30秒**：N 个 Expert 常驻 GPU 内存但每 token 只激活 Top-K，compute/memory ratio 极低，GPU 计算单元等内存加载。

### Q8: 乒乓流水线为什么能隐藏通信？
**30秒**：batch 分成 micro-batches。Expert GPU 处理 A 时，Attention GPU 处理 B，同时 A 结果在通信返回。三动作重叠，通信被计算覆盖。

---

## 记忆助手

- **MoE = 专家会诊**：每个 token 不找所有医生（Dense FFN），而是挂号分诊（Router）到 1-2 个专科医生
- **Router = 分诊台**：看到症状（token）后决定挂哪个科，softmax 给出各科权重
- **负载均衡 = 不让某科室排长队**：辅助损失惩罚 Expert 过度集中
- **MegaScale-Infer = 门诊和检验分开**：Attention（门诊）和 FFN（检验）不同机器，乒乓传菜
- **共享 Expert = 全科医生**：每次都参与，保证基础能力

---

## 相关概念

- [[concepts/attention_in_recsys|Attention 在搜广推中的演进]]
- [[01_LLM推理优化全景|推理优化与 MoE 的协同]]
