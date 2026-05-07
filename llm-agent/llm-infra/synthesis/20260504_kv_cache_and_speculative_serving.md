# KV Cache 恢复、压缩与投机式服务前沿 (2025-2026)

> 覆盖论文：CacheFlow (2604.25080), RocketKV (2502.14051), FreeKV (2505.13109), PPD (2603.13358), HierSpec (2505.22179)
> 交叉引用：[[20260503_kv_cache_frontier_2026.md]]、[[20260419_KV_cache_quantization_adaptive_methods.md]]

---

## 1. 技术演进总览

KV Cache 优化已从单一维度扩展为三大并行方向：

```
KV Cache 优化前沿
|
|-- 恢复（Restoration）: 如何高效从外部存储恢复 KV Cache
|   |-- CacheFlow: 3D 并行恢复（token x layer x GPU）
|   |-- PPD: 多轮场景下 prefill-decode 解耦 + 本地缓存复用
|
|-- 压缩（Compression）: 如何减少 KV Cache 内存占用
|   |-- RocketKV: 两阶段（粗粒度淘汰 + 细粒度稀疏注意力）
|   |-- FreeKV: 投机式检索 + 双缓冲流式召回
|
|-- 投机式加速（Speculative Serving）: 如何与量化协同
|   |-- HierSpec: 投机解码 + 4-bit 量化的兼容性与分层框架
```

---

## 2. 逐篇精读

### 2.1 CacheFlow: 3D-Parallel KV Cache Restoration (2604.25080)

**核心问题**：长上下文 LLM serving 中，KV Cache 恢复成为主要瓶颈。现有方法将恢复视为单个请求级别的「重算 vs I/O 传输」权衡。

**方法：3D 并行抽象**

将 KV Cache 恢复分解为三维并行：

| 维度 | 并行策略 | 描述 |
|------|----------|------|
| Token-level | 重算 token_1..k 与 I/O 传输 token_{k+1..n} 重叠 | 前面的 token 重算快，后面的走 I/O |
| Layer-level | Pipeline：底层重算与高层 KV 传输重叠 | 利用 Transformer 层间依赖 |
| Multi-GPU | 多 GPU 并行恢复 | 水平扩展 |

**核心组件：Batch-Aware Two-Pointer Scheduler**

调度器联合优化多请求的 compute 和 I/O 分配：

$$\text{Priority}(op) = \frac{\Delta C_{\text{recompute}}(op)}{t_{\text{IO}}(op)}$$

优先执行「每单位 I/O 时间能最大减少重算开销」的操作。两个指针分别追踪 compute frontier 和 I/O frontier，动态分配资源。

**关键结果**：
- TTFT（Time-To-First-Token）降低 **10%-62%**
- 在多种模型、工作负载、硬件上一致有效

**工程意义**：打破了「重算 or I/O」的二选一困境，通过细粒度重叠实现 Pareto 最优。

---

### 2.2 RocketKV: Two-Stage KV Cache Compression (2502.14051)

**核心问题**：KV Cache 随输入长度线性增长，制约 decode 阶段的内存带宽和容量。

**方法：两阶段压缩**

**Stage 1 — 粗粒度永久淘汰**：
- 采用 SnapKV 方法，根据注意力分数统计永久淘汰低重要性 token 的 KV
- 压缩后的 KV Cache 用于后续所有 decode step

**Stage 2 — 细粒度动态选择（Hybrid Sparse Attention, HSA）**：
- 对 Stage 1 保留的 KV 做 top-k 稀疏注意力
- 通过**双维度降维**近似注意力分数：
  - Head dimension reduction：压缩 head 维度估计注意力
  - Sequence dimension reduction：压缩序列维度加速选择

**自适应压缩分解**：

给定目标压缩率 $R$，自动分配两阶段的压缩比：

$$R = R_1 \times R_2$$

其中 $R_1$ 是 Stage 1 淘汰比，$R_2$ 是 Stage 2 稀疏比。机制自动搜索最优 $(R_1, R_2)$ 组合。

**关键结果**：
- 压缩比高达 **400x**
- 端到端加速 **3.7x**（A100）
- 峰值内存降低 **32.6%**
- 精度损失可忽略

**多轮变体 RocketKV-MT**：Stage 1 不淘汰而是标记，保留完整 KV 供后续轮次使用，decode 时仍只用 Stage 1 筛选的子集。

---

### 2.3 FreeKV: Speculative KV Cache Retrieval (2505.13109)

**核心问题**：KV Cache 卸载到 CPU 后，选择 + 召回操作阻塞 decode 关键路径。

**方法：算法-系统协同优化**

**算法层 — 投机式检索（Speculative Retrieval）**：

核心观察：**相邻 decode step 的 query 向量高度相似**。

$$\text{sim}(q_t, q_{t+1}) \approx 1 - \epsilon, \quad \epsilon \ll 1$$

因此可以用 step $t$ 的选择结果预测 step $t+1$ 需要的 KV：
1. 在 step $t$ 执行 attention 的同时，用 $q_t$ 预测 step $t+1$ 的 KV 选择
2. 提前从 CPU 召回预测的 KV 到 GPU
3. Step $t+1$ 开始时，KV 已经在 GPU 上了

**Fine-grained Correction**：预测不准时做局部修正，保证精度。

**系统层**：
- **Hybrid KV Layout**：CPU-GPU 混合存储，消除碎片化传输
- **Double-Buffered Streamed Recall**：双缓冲流式召回，计算和 I/O 完全重叠

**关键结果**：
- 比 SOTA KV retrieval 方法**快 13x**
- 近无损精度
- 训练-free

**与 RocketKV 的互补**：RocketKV 做压缩减少存储量，FreeKV 做高效检索减少访问延迟，两者可组合使用。

---

### 2.4 PPD: Prefill-Decode Disaggregation for Multi-turn (2603.13358)

**核心问题**：Prefill-Decode 分离架构在多轮对话中有两个低效：
1. 每轮都需要 prefill 新 prompt + 上轮 response
2. Prefill-Decode 节点间反复传输 KV Cache，带宽饱和

**核心洞察：Not All Prefills Are Equal**

两种 prefill 操作代价差异巨大：
- **Full Prefill**：从头计算所有 token 的 KV（首轮或 cache miss）
- **Append Prefill**：仅处理新 token，复用已缓存 KV（第 2+ 轮）

Append prefill 对 decode 的干扰远小于 full prefill。

**PPD (Prefill Prefill-capable Decode) 架构**：

```
Turn 1: Query → Prefill Node (full prefill) → KV transfer → Decode Node
Turn 2+: Query → Decode Node (append prefill, 本地KV复用, 无需传输)
```

动态路由决策：

$$\text{Route}(req) = \begin{cases} \text{Decode Node (local)} & \text{if append-prefill AND cache hit} \\ \text{Prefill Node} & \text{otherwise} \end{cases}$$

通过可配置 SLO 权重调节 TTFT vs TPOT 的平衡。

**关键结果**：
- Turn 2+ TTFT 降低 **68%**
- TPOT 保持竞争力
- 有效缓解高负载下的 KV 传输拥塞
- 与传统 PD 部署无缝集成

---

### 2.5 HierSpec: Speculative Decoding Meets Quantization (2505.22179)

**核心问题**：Speculative decoding 和量化都能加速 LLM 推理，但二者结合时效果如何？

**关键发现（兼容性评估）**：

对 EAGLE-2（先进的 speculative decoding 方法）应用于 4-bit 量化模型：
- 4-bit 量化减少了内存带宽瓶颈，但 speculative decoding 的 tree-style 验证引入大量计算
- Tree verification 在 4-bit 模型上的时间开销远超单 token forward pass
- 结果：**4-bit 量化的内存收益被 speculative decoding 的计算开销部分抵消**

**HierSpec 分层框架**：

将 speculative decoding 分为 drafting 和 verification 两个阶段分别优化：

| 阶段 | 策略 | 目标 |
|------|------|------|
| Drafting | 接近 EAGLE-2 的速度 | 快速生成候选 token |
| Verification | 匹配 baseline（无 spec-dec）的验证时间 | 用量化模型高效验证 |

**核心思路**：drafting 用高精度快速模型，verification 用量化模型降低内存开销。

**工程意义**：不能简单地「量化 + speculative decoding」叠加，需要分层设计才能实现加速叠加而非相互抵消。

---

## 3. 技术对比矩阵

| 维度 | CacheFlow | RocketKV | FreeKV | PPD | HierSpec |
|------|-----------|----------|--------|-----|----------|
| 优化目标 | TTFT | 内存+带宽 | 检索延迟 | 多轮TTFT | 解码吞吐 |
| 方法类型 | 并行调度 | 两阶段压缩 | 投机检索 | 架构解耦 | 分层框架 |
| 训练需求 | 无 | 无 | 无 | 无 | 需训练 draft |
| 适用场景 | 长上下文/多轮 | 长上下文 | KV卸载场景 | 多轮对话 | 量化部署 |
| 加速倍数 | 1.1-1.6x TTFT | 3.7x E2E | 13x 检索 | 68% TTFT降低 | -- |
| 压缩比 | N/A | 400x | N/A | N/A | N/A |
| 精度影响 | 无 | 可忽略 | 近无损 | 无 | 近无损 |

---

## 4. 系统组合视角

这 5 篇论文覆盖了 LLM serving 的不同瓶颈点，可以组合使用：

```
请求到达
  |
  v
[PPD 路由] → Turn 1: Prefill Node / Turn 2+: Decode Node (本地 append)
  |
  v
[RocketKV] 压缩 KV Cache（400x），减少存储
  |
  v
[FreeKV] 投机式检索被卸载到 CPU 的 KV，13x 加速
  |
  v
[CacheFlow] 3D 并行恢复 cache miss 的 KV
  |
  v
[HierSpec] 分层 speculative decoding + 量化加速 decode
```

---

## 5. 面试高频 Q&A

### Q1: KV Cache 恢复（restoration）的瓶颈在哪？CacheFlow 如何解决？

**A**: 瓶颈在于长上下文下，KV Cache 要么从头重算（计算密集），要么从 CPU/远端加载（I/O 密集）。传统方法二选一。

CacheFlow 引入 3D 并行：token 维度（前面重算 + 后面 I/O）、layer 维度（底层重算 + 高层 I/O pipeline）、GPU 维度（多卡并行恢复）。核心是 two-pointer scheduler 贪心分配 compute 和 I/O 资源，TTFT 降低 10%-62%。

### Q2: RocketKV 的两阶段分别做什么？为什么不直接用一阶段？

**A**:
- Stage 1（粗粒度淘汰）：用 SnapKV 永久删除不重要的 KV token，大幅减少候选集
- Stage 2（细粒度稀疏）：对剩余 KV 做 top-k 稀疏注意力，进一步降低计算量

单阶段的问题：如果只做粗粒度，可能误删后续重要的 token；如果只做细粒度，候选集太大导致选择本身开销大。两阶段配合，粗筛 + 精选，实现 400x 压缩且精度几乎无损。

### Q3: FreeKV 的投机式检索为什么可行？

**A**: 基于观察：相邻 decode step 的 query 向量高度相似（$\text{sim}(q_t, q_{t+1}) \approx 1$）。因此 step $t$ 选出的 KV token 大概率也是 step $t+1$ 需要的。FreeKV 利用这个特性，在 step $t$ 计算 attention 的同时，异步从 CPU 召回 step $t+1$ 预测需要的 KV。偶尔预测错误时做 fine-grained correction。

### Q4: 为什么 speculative decoding + 量化不能简单叠加？

**A**: HierSpec 发现，4-bit 量化减少了内存带宽瓶颈，使得 decode 不再是纯 memory-bound。此时 speculative decoding 的 tree-style verification 变成 compute-bound 操作，额外计算开销抵消了量化带来的内存收益。解决方案是分层设计：drafting 用高精度模型保持速度，verification 用量化模型节省内存。

### Q5: PPD 在多轮场景下比标准 PD 分离好在哪？

**A**: 标准 PD 每轮都要：(1) prefill 节点处理新 prompt + 上轮 response，(2) 传输 KV 到 decode 节点。PPD 发现 Turn 2+ 是 append-prefill（只处理新 token），代价远小于 full prefill，可以直接在 decode 节点本地完成，避免 KV 传输。Turn 2+ TTFT 降低 68%。

---

## 6. 开放问题与趋势

1. **KV Cache 压缩 + 检索的统一框架**：RocketKV 压缩 + FreeKV 检索能否在一个系统中协同？
2. **跨请求 KV 共享**：多个请求共享相同前缀的 KV Cache（如 system prompt），CacheFlow 的 batch-aware 调度可扩展
3. **量化与 speculative decoding 的最优配比**：HierSpec 揭示了兼容性问题，但最优的精度-速度 Pareto 前沿仍需探索
4. **Disaggregation 的极限**：PPD 扩展了 PD 分离到 3 类节点，未来可能进一步细分（如 retrieval node、reranking node）

---

*Updated: 2026-05-04 | 概念页关联：[[embedding_everywhere.md]]、[[sequence_modeling_evolution.md]]*
