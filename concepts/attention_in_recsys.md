# Attention 机制在搜广推中的演进

> **一句话总结**：Attention 的本质是"按需分配权重"——从 DIN 的候选物品问历史行为"你跟我有多相关"，到 Transformer 的任意 token 问任意 token"我们之间什么关系"，再到 FlashAttention 的"怎么算得又快又省"。
>
> **为什么要学**：Attention 是搜广推和 LLM 的共同底层语言。理解它的演进，等于理解过去 7 年推荐系统最核心的建模思路变化。

**相关概念页**：[序列建模演进](sequence_modeling_evolution.md) | [Embedding全景](embedding_everywhere.md) | [生成式推荐](generative_recsys.md)

---

## 1. 起点：Avg Pooling 的问题

推荐排序需要把用户的历史行为序列压缩成一个向量。最简单的方法是 **平均池化**：

$$\mathbf{v}_u = \frac{1}{T} \sum_{t=1}^{T} \mathbf{e}_t$$

问题很明显：用户点过 100 个商品，你给每个商品同等权重，跟当前候选物品无关的行为全混进来了。买过手机壳的人，在看手机时，手机壳的信号应该放大，零食的信号应该缩小。

---

## 2. DIN: Target Attention（2018，阿里）

**核心思想**：让候选物品去"问"历史行为——"你跟我有多相关？"

$$\alpha_t = \text{MLP}\left([\mathbf{e}_t;\, \mathbf{e}_a;\, \mathbf{e}_t \odot \mathbf{e}_a;\, \mathbf{e}_t - \mathbf{e}_a]\right)$$

$$\mathbf{v}_u = \sum_{t=1}^{T} \alpha_t \cdot \mathbf{e}_t$$

其中 $\mathbf{e}_a$ 是候选物品 embedding，$\mathbf{e}_t$ 是第 $t$ 个历史行为 embedding。

**和 Self-Attention 的区别**：
- Self-Attention: 序列内部互相看 → $Q=K=V=$ 行为序列
- Target Attention: 候选物品作为 Query 去看行为序列 → $Q=$ 候选, $K=V=$ 行为

**解决了什么**：同一个用户面对不同候选物品，表示不同（动态兴趣）。

**工程细节**：DICE 激活函数替代 PReLU，mini-batch aware 正则化。

📄 详见 [rec-sys/02_rank/synthesis/CTR模型深度解析.md](../rec-search-ads/rec-sys/02_rank/synthesis/CTR模型深度解析.md)

---

## 3. DIEN: Attention + 时序演化（2019，阿里）

**DIN 的问题**：把历史行为当集合（无序），忽略了兴趣随时间变化。

**DIEN 方案**：两层结构
1. **兴趣抽取层**：GRU 建模序列时序 → 得到隐状态序列 $\mathbf{h}_1, ..., \mathbf{h}_T$
2. **兴趣演化层**：AUGRU（Attention-based GRU）→ 候选物品的 attention 控制 GRU 的更新门

$$\tilde{u}_t = \alpha_t \cdot u_t$$

$$\mathbf{h}_t' = (1 - \tilde{u}_t) \odot \mathbf{h}_{t-1}' + \tilde{u}_t \odot \tilde{\mathbf{h}}_t$$

**直觉**：attention 权重低的时刻（跟候选无关的行为），GRU 更新门被缩小，几乎不更新状态 → 自动跳过无关行为。

**辅助损失**：用下一个点击行为监督 GRU 隐状态，解决兴趣抽取层训练不稳定的问题。

**和 DIN 的核心区别**：DIN 是无序加权求和，DIEN 是有序演化后取最终状态。

📄 详见 [rec-sys/02_rank/synthesis/用户行为序列建模.md](../rec-search-ads/rec-sys/02_rank/synthesis/用户行为序列建模.md)

---

## 4. SIM: 两阶段检索式 Attention（2020，阿里）

**DIEN 的问题**：GRU 是 $O(T)$ 顺序计算，序列长度限制在 ~100。但用户真实行为可能上万。

**SIM 方案**：先检索，再精排
1. **GSU（General Search Unit）**：从万级行为中快速检索 top-K 相关行为
   - Hard Search: 按类目匹配（最快，但粗糙）
   - Soft Search: embedding 近似最近邻（更准）
2. **ESU（Exact Search Unit）**：对 top-K 做精确 Target Attention（同 DIN）

**本质**：把 Attention 从 $O(T)$ 降到 $O(K)$，$K \ll T$。

**后续变体**：
- **ETA**：用 SimHash 做近似检索，$O(1)$ 查找
- **SDIM**：用多头 hash 采样，无需 ANN 索引

📄 详见 [rec-sys/long-sequence/synthesis/长序列用户行为建模技术演进.md](../rec-search-ads/rec-sys/long-sequence/synthesis/长序列用户行为建模技术演进.md)

---

## 4-B. SparseCTR：推荐专用稀疏注意力（WWW 2026，美团）

SIM/ETA 用"检索+精排"两阶段解决长序列，但检索阶段不参与端到端训练。SparseCTR 提出**推荐场景定制的稀疏自注意力**，一步到位。

**三分支设计**：
- **Global Branch**：chunk 代表向量之间的注意力 → 长期兴趣
- **Transition Branch**：相邻 chunk 之间 → 兴趣转移
- **Local Branch**：chunk 内部 → 短期兴趣

$$\text{Attn}_{\text{sparse}} = \alpha_g \cdot \text{Attn}_{\text{global}} + \alpha_t \cdot \text{Attn}_{\text{transition}} + \alpha_l \cdot \text{Attn}_{\text{local}}$$

**个性化分块**：按用户行为的时间连续性自适应切分（不同用户不同阈值），保证连续购物 session 不被切断。

**复合时间编码**：每个 attention head 有独立的时间偏置系数，同时编码顺序关系和周期关系（如周末购物模式）。

**关键结果**：
- 在线 CTR +1.72%，CPM +1.41%（美团 A/B 测试）
- 展现 Scaling Law：三个数量级 FLOPs 上效果持续提升
- 端到端训练，不需要 SIM 的两阶段分离

**和 SIM 的核心区别**：SIM 是"先用便宜方法找相关行为，再精算"；SparseCTR 是"用结构化稀疏注意力直接建模全序列"。SIM 只捕捉与候选相关的行为，SparseCTR 同时捕捉全局兴趣、兴趣转移和短期兴趣。

📄 详见 [[长序列用户行为建模技术演进|rec-sys/long-sequence/synthesis/长序列用户行为建模技术演进.md]]

---

## 5. Transformer Attention 进入推荐（2019+）

### 5.1 自回归推荐（SASRec / BERT4Rec）

把用户行为序列当"语言"，用 Transformer 做 next-item prediction：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

- **SASRec**: Causal Mask（只看左边），自回归生成
- **BERT4Rec**: 双向 Mask（随机遮掩），理解能力更强但不能直接做生成

### 5.2 Cross-Attention 在搜索重排

搜索 Reranker 中，query 和 document 之间用 Cross-Attention：
- $Q =$ query tokens, $K = V =$ document tokens
- 全交互打分（Cross-Encoder），精度最高但最慢

### 5.3 HSTU: 推荐的 Scaling 时代（Meta 2024）

Meta 用 **1.5 万亿参数** 的 Transformer 做推荐，证明推荐系统也有 Scaling Law：
- 用 ReLU 替代 softmax（稀疏激活 + 推理更快）
- 去掉 LayerNorm（推荐场景稳定性够）
- 行为序列直接当 token 输入

📄 详见 [rec-sys/02_rank/synthesis/推荐系统ScalingLaw_Wukong.md](../rec-search-ads/rec-sys/02_rank/synthesis/推荐系统ScalingLaw_Wukong.md)

---

## 6. FlashAttention: 算得又快又省（2022-2024）

不是改 Attention 的数学，而是改计算方式：

**问题**：标准 Attention 需要 $O(N^2)$ 的中间矩阵 $S = QK^\top$ 存在 HBM 中，内存瓶颈。

**FlashAttention 方案**：
1. **Tiling**: 把 Q/K/V 分块加载到 SRAM（快但小）
2. **Online Softmax**: 分块计算 softmax，不需要完整 $S$ 矩阵
3. **不存中间结果**: 反向传播时重新计算（用计算换内存）

**对推荐的影响**：使长序列 Transformer 在推荐中变得可行（如 HSTU）。

📄 详见 [llm-infra/synthesis/FlashAttention3与LLM推理基础设施.md](../llm-agent/llm-infra/synthesis/FlashAttention3与LLM推理基础设施.md)

---

## 7. 演进总结：一张图看清

```
Avg Pooling (无权重)
    │
    ▼
DIN Target Attention (2018) ─── 候选问历史"你相关吗"
    │
    ▼
DIEN AUGRU (2019) ──────────── + 时序演化，Attention 控制 GRU 门
    │
    ▼
SIM 两阶段 (2020) ─────────── 先检索 top-K 再 Attention（万级序列）
    │                           └─ ETA / SDIM（hash 近似加速）
    │
    ├─ SparseCTR (2026) ──────── 三分支稀疏 Attention + 个性化分块（端到端）
    ▼
Transformer Self-Attention ──── 序列内部全交互（SASRec/BERT4Rec）
    │
    ├─ Cross-Attention ────────── query-document 跨序列交互（搜索重排）
    │
    ├─ HSTU (2024) ────────────── 1.5T 参数推荐 Transformer（Scaling Law）
    │
    └─ FlashAttention ─────────── 不改数学改计算（Tiling + Online Softmax）
```

## 面试高频问题

1. **DIN 和 Self-Attention 的区别是什么？** → Target Attention 用候选做 Query，Self-Attention 序列内部互相看。
2. **为什么 SIM 不直接用 Transformer？** → $O(N^2)$ 在万级序列不可行，SIM 先剪枝到 top-K 再精算。
3. **HSTU 为什么用 ReLU 替代 softmax？** → 推荐场景不需要概率归一化，ReLU 更稀疏且推理更快。
4. **FlashAttention 改变了 Attention 的计算结果吗？** → 没有，数学等价，只改了访存模式。
5. **SparseCTR 的三分支注意力各自解决什么问题？** → Global 捕捉长期偏好，Transition 捕捉兴趣漂移，Local 捕捉短期活跃行为。个性化分块保证 session 完整性。比 SIM 的 top-K 检索更全面。
