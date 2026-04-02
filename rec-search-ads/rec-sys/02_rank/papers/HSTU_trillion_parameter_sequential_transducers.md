# HSTU: Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations

> 来源：arXiv 2402.17152, Meta | 年份：2024 | 领域：rec-sys/02_rank（序列推荐/大规模模型）

## 问题定义

工业推荐系统面临超长用户行为序列建模挑战：
- 用户历史行为可达**数万条**（Meta 用户平均日交互数百次）
- 传统 Transformer 注意力复杂度 $O(n^2)$ 无法处理如此长序列
- 随着模型规模扩展到**万亿参数**，需要高效架构和分布式训练方案
- 推荐系统是否也遵循类似 LLM 的 **Scaling Law**？

**核心问题**：如何在工业级推荐系统中高效建模极长用户序列并扩展到超大规模模型？

## 模型结构图

```
┌─────────────────────────────────────────────────────────────┐
│                    HSTU Architecture                         │
│                                                             │
│  用户行为序列: [a₁, a₂, ..., a₁₀₀₀₀] (click/view/like/...)│
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Level 1: Item-level Attention (短时间窗口内)        │    │
│  │                                                     │    │
│  │  Session 1:     Session 2:      Session K:          │    │
│  │  [a₁...aₘ]     [aₘ₊₁...a₂ₘ]   [...aₙ]            │    │
│  │    ↓               ↓              ↓                  │    │
│  │  Linear Attn    Linear Attn    Linear Attn           │    │
│  │  O(m·d²)       O(m·d²)        O(m·d²)              │    │
│  │    ↓               ↓              ↓                  │    │
│  │  [s₁]           [s₂]           [sₖ]  (session repr) │    │
│  └──────┬────────────┬──────────────┬──────────────────┘    │
│         └────────────┼──────────────┘                        │
│                      ↓                                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Level 2: Segment-level Attention (跨session)        │    │
│  │                                                     │    │
│  │  [s₁, s₂, ..., sₖ] → Segment Attention → u         │    │
│  │  (粗粒度，K << N)                                    │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         ↓                                    │
│  ┌──────────────────────┴──────────────────────────────┐    │
│  │  Transducer Output: Next-action prediction           │    │
│  │  P(a_{n+1} | u, context)                            │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 核心方法与完整公式

### 公式1：线性注意力机制

$$\text{LinearAttn}(Q, K, V) = \phi(Q) \cdot (\phi(K)^T V)$$

**解释：**
- $\phi$：特征映射函数（如 ELU+1 或 ReLU）
- 标准注意力：先计算 $QK^T$（$n \times n$ 矩阵），复杂度 $O(n^2d)$
- 线性注意力：先计算 $\phi(K)^T V$（$d \times d$ 矩阵），复杂度 $O(nd^2)$
- 当 $n \gg d$ 时（推荐场景中用户序列极长），效率提升巨大

### 公式2：两级注意力

**Item-level（session 内）：**
$$h_i^{item} = \text{LinearAttn}(q_i, K_{session}, V_{session})$$

**Segment-level（跨 session）：**
$$s_k = \text{Pool}(h_1^{item}, \ldots, h_m^{item}) \quad \text{（session k 的压缩表示）}$$
$$u = \text{Attn}(q_{target}, [s_1, \ldots, s_K], [s_1, \ldots, s_K])$$

**解释：**
- Item-level 关注 session 内的精细交互模式
- Segment-level 跨 session 建模长期兴趣演变
- 类似人类记忆：近期详细，远期模糊

### 公式3：Scaling Law（推荐系统版）

$$L(N) = L_\infty + \frac{C}{N^\alpha}$$

**解释：**
- $L(N)$：参数量为 $N$ 时的 loss
- $L_\infty$：不可约损失（数据噪声下界）
- $\alpha$：幂律指数（推荐系统中 $\alpha \approx 0.05-0.1$，比 NLP 的 $\alpha \approx 0.07$ 类似）
- 从 10B → 1T 参数，性能持续提升

### 公式4：Transducer 流式推理

$$P(a_{n+1} | a_1, \ldots, a_n) = \text{Softmax}(W_{out} \cdot u + b)$$

$$u_{n+1} = \text{Update}(u_n, a_n) \quad \text{（增量更新，不重算历史）}$$

**解释：** 借鉴语音识别 Transducer，支持流式在线推理，每个新行为只需增量计算。

## 与基线方法对比

| 方法 | 序列长度 | 注意力复杂度 | 参数规模 | Scaling Law |
|------|---------|-------------|---------|------------|
| **DLRM** | 无序列 | N/A | ~10B | 未验证 |
| **SASRec** | ~200 | $O(n^2)$ | ~100M | 不适用 |
| **DIN/DIEN** | ~50-200 | $O(n)$ | ~1B | 未验证 |
| **SIM** | ~10000 | $O(K)$ 检索 | ~1B | 未验证 |
| **HSTU** | ~10000+ | $O(nd^2)$ | **~1T** | ✅ 验证 |

## 实验结论

- **离线指标**：NE（Normalized Entropy）相比 DLRM 系列基线显著降低
- **在线 A/B**：Facebook 短视频推荐 CTR 和互动率均有显著提升
- **效率**：支持处理 10,000+ token 的用户序列，训练速度比标准 Transformer 快 10x
- **扩展性**：万亿参数模型训练成功，验证推荐系统的 scaling law
- **消融**：两级注意力 vs 单级：+2.3% NE 改善；线性注意力 vs 标准注意力：10x 速度提升，NE 仅下降 0.1%

## 工程落地要点

1. **序列截断策略**：按时间滑动窗口截断，session 切分点根据间隔时间（>30min = 新 session）
2. **线性注意力数值稳定性**：使用 ELU+1 而非 ReLU，避免零值导致的梯度消失
3. **KV Cache 增量推理**：缓存历史 session 的 segment 表示，新行为只增量计算当前 session
4. **分布式训练**：模型并行 + 数据并行 + 流水线并行三维并行，专为 Meta 基础设施优化
5. **特征工程**：行为序列需要细粒度时间戳、行为类型（click/view/like）、停留时长等
6. **冷启动**：新用户序列短，需用内容特征（item embeddings）补充

## 面试考点

**Q1：HSTU 如何解决 Transformer 在长序列推荐中的效率问题？**
> 使用线性注意力：$\phi(Q)(\phi(K)^TV)$，利用矩阵乘法结合律先计算 $K^TV$（$d \times d$ 矩阵），将复杂度从 $O(n^2d)$ 降至 $O(nd^2)$。当 $n=10000, d=128$ 时，加速约 78 倍。

**Q2：两级注意力设计的意义？**
> Item-level 关注 session 内精细交互模式，Segment-level 跨 session 建模长期兴趣。两级结构在保持长期记忆的同时聚焦近期行为细节，类似"近期详细、远期模糊"的记忆机制。

**Q3：推荐系统的 Scaling Law 与 LLM 有何异同？**
> 同：都验证幂律关系 $L \propto N^{-\alpha}$。异：推荐系统的 "tokens" 是用户行为（点击/观看），非文字；数据分布随时间变化更剧烈（需持续更新）；特征工程更复杂（ID + 统计 + 上下文特征）。

**Q4：HSTU 的 Transducer 结构对在线推理的优势？**
> Transducer 支持流式增量推理：每个新用户行为只需增量更新当前 session 的表示，无需重新计算完整历史。这对实时推荐（<10ms 响应）至关重要。

**Q5：HSTU 与 SIM（Search-based Interest Model）处理长序列的区别？**
> SIM 用检索+精排两阶段：先从长序列检索 TopK 相关行为，再精排。HSTU 用层级注意力端到端建模：session 内精细注意力 + 跨 session 粗粒度注意力。SIM 依赖检索质量，HSTU 端到端优化但计算量更大。

**Q6：万亿参数推荐模型的参数主要在哪里？**
> 99%+ 的参数在 Embedding 层（数亿用户 ID + 数亿 item ID + 各种特征 ID），MLP/Attention 层参数相对很少。这与 LLM（主要参数在 Transformer 层）根本不同。
