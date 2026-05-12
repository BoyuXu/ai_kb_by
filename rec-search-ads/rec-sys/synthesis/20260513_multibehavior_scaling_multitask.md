# Multi-Behavior + Scaling Law + Multi-Task Learning: 10 篇前沿论文综合精读

> **日期**：2026-05-13
> **覆盖论文**：MBGen / Multi-Behavior Survey / EST / Deep Mutual Learning / ApEn Scaling Law / SeqRec Scaling Law / VL-JEPA / RMTL / Multi-Task Survey / EMPRA
> **三大主题**：(1) Multi-Behavior Recommendation (2) Scaling Laws in CTR/SeqRec (3) Multi-Task Learning for Rec-Sys
> **附加**：VL-JEPA (多模态 JEPA) + EMPRA (对抗排序攻击)

**相关概念页**：[[多目标优化]] | [[序列建模演进]] | [[生成式推荐]] | [[Embedding无处不在]] | [[推荐中的注意力机制]]
**相关 synthesis**：[[20260503_scaling_sequence_multitask_frontier]] | [[20260504_scaling_coldstart_ctr_frontier]] | [[20260420_feature_interaction_ctr_advances]]

---

## Section 1: Multi-Behavior Recommendation

### 1.1 MBGen: Multi-Behavior Generative Recommendation (CIKM 2024, arxiv 2405.16871)

| 维度 | 内容 |
|------|------|
| **Problem** | 多行为序列推荐 (MBSR) 需要同时建模不同行为类型 (click/cart/purchase)，传统方法将行为类型作为辅助信号，未能充分利用行为序列的生成式建模潜力 |
| **Method** | 将 MBSR 分解为两步生成：(1) 给定 item 序列，预测下一个行为类型；(2) 给定 item 序列 + 目标行为类型，预测下一个 item。将 behavior 和 item 都 tokenize 后交替排列成统一序列，用自回归生成 |
| **Key Innovation** | Position-Routed Sparse Architecture：利用 token 序列的异构性 (behavior token vs item token)，按位置路由到不同 Expert，高效扩展模型 |
| **Results** | 在公开数据集上显著优于现有 MBSR 模型，多任务能力天然涌现 (行为预测 + 物品推荐同时完成) |
| **Keywords** | Multi-Behavior, Generative Recommendation, Sparse MoE, Token Interleaving |

**核心公式**：

$$P(\text{next}) = P(b_{t+1} | s_{1:t}) \times P(i_{t+1} | s_{1:t}, b_{t+1})$$

其中 $s_{1:t} = [i_1, b_1, i_2, b_2, \ldots, i_t, b_t]$ 是 item-behavior 交替序列。

**架构设计要点**：
```
Input:  [item_1, click, item_2, cart, item_3, purchase, ...]
         ↓ Tokenize (behavior + item 各自独立 codebook)
Backbone: Transformer with Position-Routed Sparse Architecture
         ↓ 偶数位 → Item Expert, 奇数位 → Behavior Expert
Output: next_behavior_token → next_item_token (两步自回归)
```

### 1.2 Multi-Behavior Recommender Systems: A Survey (PAKDD 2025, arxiv 2503.06963)

| 维度 | 内容 |
|------|------|
| **Problem** | 传统推荐只用单一交互类型 (如 purchase)，忽略了 click/add-to-cart/favorite 等丰富行为信号 |
| **Method** | 系统综述，从三个维度分类：(1) Data Modeling: 如何在输入层表示多行为 (2) Encoding: 如何将多行为编码为向量 (3) Training: 如何优化多行为模型 |
| **Key Taxonomy** | Data Modeling: graph-based vs sequence-based; Encoding: GNN / Attention / Contrastive; Training: multi-task / auxiliary / cascade |
| **Keywords** | Survey, Multi-Behavior, GNN, Contrastive Learning, Behavior Cascade |

**多行为建模的三种范式**：

| 范式 | 代表方法 | 优势 | 局限 |
|------|---------|------|------|
| 图建模 | MBGCN, KMCLR | 捕获行为间拓扑关系 | 难以建模时序 |
| 序列建模 | MBGen, CMBF | 天然保留时序 | 行为类型信息融合方式受限 |
| 对比学习 | CML, MBSSL | 增强跨行为表示对齐 | 负样本构造困难 |

---

## Section 2: Scaling Laws in CTR / Sequential Recommendation

### 2.1 EST: Efficiently Scalable Transformer for CTR (Taobao, arxiv 2602.10811)

| 维度 | 内容 |
|------|------|
| **Problem** | 现有 CTR 模型在 scale up 时遇到信息瓶颈：early aggregation 将用户行为压缩后再建模，丢失 token-level 细粒度信号，无法解锁 scaling 收益 |
| **Method** | Efficiently Scalable Transformer (EST)：将所有原始输入（用户行为序列、候选 item 特征、context 特征）拼接为单一序列，实现 fully unified modeling，无损聚合 |
| **Key Innovation** | 两个核心模块：(1) Lightweight Cross-Attention (LCA)：裁剪冗余自交互，聚焦高影响力的跨特征依赖；(2) Content Sparse Attention (CSA)：基于内容相似度动态选择高信号行为 token |
| **Results** | 展现稳定的 power-law scaling relationship。部署淘宝展示广告：RPM +3.27%, CTR +1.22% |
| **Keywords** | Scaling Law, Unified Modeling, Sparse Attention, Cross-Attention, CTR |

**Scaling Law 公式**：

$$\text{Loss}(N) = \alpha \cdot N^{-\beta} + \gamma$$

其中 $N$ 为模型参数量。EST 的关键发现：只有 fully unified modeling（不做 early aggregation）才能展现稳定的 power-law scaling。

**LCA vs 标准 Self-Attention**：
```
Standard Self-Attention: 所有 token 两两交互 → O(n^2)
LCA: 只保留 cross-feature 交互 (user-item, context-item)
     裁剪同类 token 间冗余交互 → 效率提升 + 质量不降
CSA: 在 user behavior 序列中，按内容相似度 top-k 选择
     → 稀疏化长序列注意力
```

### 2.2 Scaling Law of Large Sequential Recommendation Models (RecSys 2024, arxiv 2311.11351)

| 维度 | 内容 |
|------|------|
| **Problem** | Scaling Law 在 NLP/CV 已证实，但推荐场景有独特挑战：数据稀疏性、item vocabulary 规模有限、用户行为模式异构 |
| **Method** | 在纯 ID-based sequential recommendation 上系统研究 scaling 效应。使用 Transformer backbone，逐步增大模型参数和训练数据量 |
| **Key Innovation** | 首次在推荐系统中验证了 power-law scaling relationship (纯 ID, 无 side info)，并分析了与 LLM scaling law 的异同 |
| **Results** | 模型性能随参数量/数据量增长呈 power-law 下降，但推荐系统的 scaling 效率低于 NLP（数据稀疏性是主要瓶颈） |
| **Keywords** | Scaling Law, Sequential Recommendation, ID-based, Power Law |

### 2.3 Optimizing Sequential Recommendation with Scaling Laws and Approximate Entropy (arxiv 2412.00430)

| 维度 | 内容 |
|------|------|
| **Problem** | 推荐系统 Scaling Law 只看数据量不够——数据质量同样影响模型性能，但缺乏量化指标 |
| **Method** | 提出 Performance Law for SR：拟合 HR/NDCG 指标与模型配置的关系。引入 Approximate Entropy (ApEn) 衡量数据质量，比传统数据量指标更精细 |
| **Key Innovation** | 将数据质量纳入推荐 Scaling Law：$\text{Perf}(N, D, Q) = f(N^{-\alpha}, D^{-\beta}, Q^{\gamma})$，其中 $Q$ 用 ApEn 度量 |
| **Results** | 在多个数据集和模型规模上验证了预测准确性，能指导最优模型配置选择 |
| **Keywords** | Scaling Law, Data Quality, Approximate Entropy, Sequential Recommendation |

**Scaling Law 三要素对比**：

| 要素 | LLM Scaling Law | 推荐 Scaling Law (SeqRec) | EST (CTR) |
|------|----------------|--------------------------|-----------|
| 模型规模 $N$ | $L \propto N^{-0.076}$ | 验证 power-law 但指数更小 | power-law 需 unified modeling |
| 数据量 $D$ | $L \propto D^{-0.095}$ | 推荐数据稀疏 → 更陡 | 行为序列长度是关键 |
| 数据质量 $Q$ | 未显式建模 | ApEn 度量 → 低熵数据更有效 | CSA 隐式选择高质量行为 |
| 计算效率 $C$ | Chinchilla 最优 | 稀疏性是瓶颈 | LCA + CSA 实现高效 scaling |

---

## Section 3: Multi-Task Learning for Recommender Systems

### 3.1 Deep Mutual Learning across Task Towers (arxiv 2309.10357)

| 维度 | 内容 |
|------|------|
| **Problem** | 多任务推荐中，各 task tower 独立训练，缺乏正向知识共享；底层参数共享无法充分促进上层 tower 间的互学习 |
| **Method** | 在 task tower 之间引入 Deep Mutual Learning (DML) 机制：每个 tower 不仅优化自身 task loss，还通过 KL 散度互相学习对方的预测分布 |
| **Key Innovation** | 兼容多种 backbone (Shared-Bottom, MMoE, PLE, CGC)，在不改变底层架构的前提下增强 tower 间知识传递 |
| **Results** | 在多个多任务推荐基准上一致提升各任务性能，尤其对稀疏任务 (CVR) 提升更大 |
| **Keywords** | Multi-Task Learning, Mutual Learning, KL Divergence, Task Tower |

**DML Loss**：

$$\mathcal{L}_{\text{total}} = \sum_{t=1}^{T} \mathcal{L}_t^{\text{task}} + \lambda \sum_{t \neq t'} D_{\text{KL}}(p_t \| p_{t'})$$

其中 $\mathcal{L}_t^{\text{task}}$ 是第 $t$ 个任务的原始 loss，$D_{\text{KL}}$ 是 tower $t$ 和 $t'$ 输出分布的 KL 散度。$\lambda$ 控制互学习强度。

### 3.2 RMTL: Multi-Task Recommendations with Reinforcement Learning (WWW 2023, arxiv 2302.03328)

| 维度 | 内容 |
|------|------|
| **Problem** | (1) 现有 MTL 推荐模型基于 item-wise 构建，忽略 session-level 的行为模式；(2) 多目标权重 balancing 通常靠线性加权，不够动态 |
| **Method** | RL-enhanced MTL framework (RMTL)：用 RL agent 动态调整多任务 loss 权重，同时建模 session-wise interaction pattern |
| **Key Innovation** | 将多任务 loss weighting 转化为 RL 问题：state = 当前各 task 的 loss/梯度统计，action = 各 task 权重，reward = 综合指标提升 |
| **Results** | 在 Kuaishou 等工业数据集上优于固定权重和 Uncertainty Weighting 方法 |
| **Keywords** | Multi-Task Learning, Reinforcement Learning, Dynamic Weighting, Session-aware |

**RL 动态加权框架**：
```
State:  s_t = [L_1(t), L_2(t), ..., L_T(t), grad_norm_1, ..., grad_norm_T]
Action: a_t = [w_1(t), w_2(t), ..., w_T(t)]  (各任务权重)
Reward: r_t = Delta(综合在线指标)
Policy: π(a|s) — 用 PPO 或 DQN 优化
```

### 3.3 Multi-Task Deep Recommender Systems: A Survey (arxiv 2302.03525)

| 维度 | 内容 |
|------|------|
| **Problem** | 多任务学习在推荐系统中广泛应用，但缺乏系统性分类和对比 |
| **Method** | 从 task relation 和 methodology 两个维度分类：Task Relation: parallel / cascaded / auxiliary-with-main; Methodology: parameter sharing / optimization / training mechanism |
| **Key Taxonomy** | Parameter Sharing: Shared-Bottom → MMoE → PLE → CGC; Optimization: gradient manipulation, loss balancing; Training: cascade training, curriculum learning |
| **Keywords** | Survey, Multi-Task Learning, Parameter Sharing, Gradient Manipulation |

**MTL for RecSys 演进路线**：

```
Shared-Bottom (2017)
  ↓ 任务差异大 → Expert 分离
MMoE (Google 2018) — 多 Expert + Gate
  ↓ Expert 坍缩 → 私有 Expert
PLE / CGC (Tencent 2020) — 共享 + 私有 Expert
  ↓ Tower 间缺乏交互
Deep Mutual Learning (2023) — Tower 间 KL 互学习
  ↓ 静态权重 → 动态权重
RMTL (2023) — RL 动态调整 task weights
  ↓ 行为类型融入
MBGen (2024) — 多行为生成式统一
  ↓ Scaling Law 驱动
EST (2026) — Unified Modeling + Efficient Scaling
```

---

## Section 4: Cross-Domain Papers

### 4.1 VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language (arxiv 2512.10942)

| 维度 | 内容 |
|------|------|
| **Problem** | 传统 VLM 用自回归 token 生成，计算成本高且 50% 以上参数用于 text decoder |
| **Method** | 在 JEPA 框架下做 vision-language 联合建模：预测 target text 的连续 embedding 而非离散 token，聚焦 task-relevant semantics |
| **Key Innovation** | (1) 50% 更少可训练参数 vs 标准 VLM；(2) 推理时选择性解码 (selective decoding)：仅在需要文本输出时调用 text decoder，减少 2.85x 解码操作；(3) 原生支持 classification / retrieval / VQA 无需架构修改 |
| **Results** | 1.6B 参数，在 8 个视频分类 + 8 个视频检索数据集上超越 CLIP/SigLIP2，VQA 上与 InstructBLIP/QwenVL 相当 |
| **Keywords** | JEPA, Vision-Language, Embedding Prediction, Selective Decoding |

**与推荐系统的关联**：VL-JEPA 的 "预测 embedding 而非 token" 思路与推荐系统中的 representation learning 高度相关：
- 推荐中可借鉴：对用户行为序列预测下一个 item 的 embedding 而非 item ID，避免 vocabulary 规模限制
- Selective Decoding 思路可用于推荐系统的动态推理：简单请求走 embedding matching，复杂请求走 full decoding

### 4.2 EMPRA: Embedding Perturbation Rank Attack (TOIS, arxiv 2412.16382)

| 维度 | 内容 |
|------|------|
| **Problem** | 黑盒 Neural Ranking Models 面临对抗攻击风险，现有攻击方法依赖特定 surrogate model，泛化性差 |
| **Method** | 在 sentence-level embedding 空间做扰动，引导文档 embedding 向 query 相关上下文方向偏移，同时保持语义完整性 |
| **Key Innovation** | 不依赖任何 surrogate NRM，在 In-Distribution 和 Out-of-Distribution 场景下均有效 |
| **Results** | 96% 原排 51-100 的文档被攻击进 top 10；65% 原排 996-1000 的文档进 top 10 |
| **Keywords** | Adversarial Attack, Neural Ranking, Embedding Perturbation, Black-box |

**对推荐系统的启示**：
- 推荐系统同样依赖 embedding-based retrieval，类似攻击可能影响双塔召回模型
- 防御思路：embedding 空间正则化、对抗训练、异常 embedding 检测

---

## 技术演进全景图

```
Multi-Task Learning 演进:
  Shared-Bottom → MMoE → PLE → DML (tower互学习) → RMTL (RL动态权重)
                                                        ↓
Multi-Behavior Recommendation:                    多行为融合
  单行为 → 图建模(MBGCN) → 序列建模(CMBF) → 生成式统一(MBGen)
                                                        ↓
Scaling Law in RecSys:                            规模化驱动
  LLM Scaling Law → 纯ID验证(SeqRec) → 数据质量(ApEn) → 统一建模(EST)
                                                        ↓
Cross-Domain Insights:
  VL-JEPA (embedding prediction) ←→ 推荐 representation learning
  EMPRA (embedding attack) ←→ 推荐系统鲁棒性
```

---

## 核心公式速查

### Scaling Law 家族

| 名称 | 公式 | 来源 |
|------|------|------|
| LLM Scaling | $L(N,D) = \alpha N^{-a} + \beta D^{-b} + \gamma$ | Kaplan 2020 |
| SeqRec Scaling | $L(N) \propto N^{-\alpha}$ (纯 ID, power-law verified) | arxiv 2311.11351 |
| ApEn Performance Law | $\text{Perf}(N, D, Q) = f(N, D, \text{ApEn})$ | arxiv 2412.00430 |
| EST Power-Law | $\text{Loss}(N) = \alpha N^{-\beta} + \gamma$ (需 unified modeling) | arxiv 2602.10811 |

### Multi-Task Loss 家族

| 名称 | 公式 | 来源 |
|------|------|------|
| 固定权重 | $\mathcal{L} = \sum_k w_k \mathcal{L}_k$ | Baseline |
| 不确定性加权 | $\mathcal{L} = \sum_k \frac{1}{2\sigma_k^2} \mathcal{L}_k + \log \sigma_k$ | Kendall 2018 |
| DML 互学习 | $\mathcal{L} = \sum_t \mathcal{L}_t + \lambda \sum_{t \neq t'} D_{\text{KL}}(p_t \| p_{t'})$ | arxiv 2309.10357 |
| RMTL RL 加权 | $w_t = \pi_\theta(s_t)$, $s_t$ = loss/gradient 统计 | arxiv 2302.03328 |

### Multi-Behavior 生成式

| 名称 | 公式 | 来源 |
|------|------|------|
| MBGen 两步生成 | $P(\text{next}) = P(b_{t+1}|s) \cdot P(i_{t+1}|s, b_{t+1})$ | arxiv 2405.16871 |

---

## 工业实践总结

| 论文 | 公司/场景 | 核心指标提升 |
|------|----------|-------------|
| EST | 淘宝展示广告 | RPM +3.27%, CTR +1.22% |
| MBGen | 学术验证 (CIKM 2024) | 多数据集 SOTA |
| RMTL | 快手 (WWW 2023) | 多任务综合指标显著提升 |
| SeqRec Scaling | 学术验证 (RecSys 2024) | 确认推荐 power-law scaling |
| VL-JEPA | Meta Research | 1.6B 参数，8 视频 benchmark SOTA |
| EMPRA | 学术/安全 (TOIS) | 96% 攻击成功率 |

---

## 面试考点 Q&A

### Q1: 多行为推荐 (Multi-Behavior) 和多任务学习 (Multi-Task) 有什么区别和联系？

**A**: 区别在于粒度和目标：
- **Multi-Behavior**：关注用户交互行为类型的多样性 (click/cart/buy)，本质是**输入端**的多样性
- **Multi-Task**：关注预测目标的多样性 (CTR/CVR/完播率)，本质是**输出端**的多样性

联系：多行为信息可以增强多任务学习。MBGen 展示了一种优雅的统一：将行为类型和物品 ID 都作为生成目标，behavior prediction 和 item prediction 自然变成两个子任务。

传统方法把行为类型作为辅助特征输入 Multi-Task 模型（如 ESMM 用 click → conversion cascade），MBGen 则直接将行为类型变成生成目标。

### Q2: EST 为什么强调 "unified modeling" 才能展现 scaling law？

**A**: 核心在于信息瓶颈：
- **Early aggregation** (传统方法)：先把用户行为序列用 pooling/attention 压缩为一个向量，再与其他特征交互。信息在压缩时已丢失，增大模型只是在更大的 capacity 里拟合一个低信息量的信号。
- **Unified modeling** (EST)：所有 raw tokens (行为序列 + item 特征 + context) 拼成一个序列，模型直接在 token 级别做交互。信息无损，增大模型时可以捕获更细粒度的 pattern。

这与 LLM 的 scaling 道理一致：如果你先把输入做有损压缩再喂给大模型，模型再大也无法从被压缩的信息里学到更多。

### Q3: 推荐系统的 Scaling Law 和 LLM 的 Scaling Law 有什么异同？

**A**:

| 维度 | LLM | 推荐系统 |
|------|-----|---------|
| 数据特征 | 文本连续、稠密 | 行为稀疏、异构 |
| Scaling 效率 | 高 ($\alpha \approx 0.076$) | 低 (数据稀疏是瓶颈) |
| 模型架构 | 统一 Transformer | 异构 (Embedding + MLP + Transformer) |
| 数据质量 | 隐含在数据筛选中 | ApEn 显式衡量 |
| 工程约束 | 可以大模型 + 慢推理 | P99 < 50ms，严格延迟限制 |

关键发现：推荐系统的 scaling 不仅取决于 $N$ 和 $D$，还需要数据质量 $Q$（ApEn 度量）和建模方式（unified vs aggregated）。

### Q4: Deep Mutual Learning (DML) 和 Knowledge Distillation (KD) 有什么区别？

**A**:
- **KD**：单向蒸馏，teacher → student，teacher 固定
- **DML**：双向互学习，所有 tower 同时是 teacher 和 student，端到端训练

DML 优势：(1) 不需要预训练好的 teacher；(2) 各 task tower 能互相补充信息——CTR tower 可能学到了某些 pattern 对 CVR tower 有用，反之亦然；(3) 实现简单，只需加 KL loss。

适用场景：当多个任务有互补性但又有差异时（如 CTR 和 CVR），DML 比简单共享参数更灵活。

### Q5: RMTL 用 RL 做多任务权重调整相比传统方法有什么优势？

**A**:
- **静态权重** ($w_k$ 固定)：无法适应训练动态，某些任务后期已收敛但仍占权重
- **不确定性加权** (Kendall)：自动学习但仍是标量权重，无法感知任务间交互
- **RMTL (RL)**：state 包含各任务的 loss 值、梯度范数等信息，action 是动态权重，reward 是综合指标

RL 的优势在于：(1) 能感知 session-level 行为模式；(2) 权重随训练过程动态调整；(3) 可以优化长期指标而非贪心优化当前 loss。

但也有挑战：RL 训练不稳定、超参多、难以在线调试。实际工业中 Pareto-MTL 或 GradNorm 可能更实用。

### Q6: MBGen 的 Position-Routed Sparse Architecture 如何工作？

**A**: MBGen 利用 token 序列的异构性进行路由：
- 偶数位 (item tokens) → Item Expert 处理
- 奇数位 (behavior tokens) → Behavior Expert 处理
- 这是一种比 MoE 更轻量的稀疏架构：不需要 Gate 网络，位置本身就是路由信号

优势：(1) 避免不同类型 token 互相干扰；(2) 各 Expert 可以针对性设计（item Expert 更大、behavior Expert 更小）；(3) 计算效率高（每个 token 只过一个 Expert）。

### Q7: VL-JEPA 的 "预测 embedding 而非 token" 对推荐有什么启发？

**A**: 推荐系统中存在类似选择：
- **生成 item ID** (TIGER/MTGR)：需要 vocabulary，受 item 规模限制
- **生成 item embedding** (VL-JEPA 启发)：在连续空间预测，再用 ANN 检索

JEPA 范式在推荐中的潜力：
1. 训练时预测 embedding 可以 bypass vocabulary 瓶颈
2. 推理时用 ANN 检索比 beam search 生成更快
3. 支持 selective decoding：简单用户走 embedding matching，复杂用户走完整生成

### Q8: EMPRA 攻击对推荐系统安全有什么启示？

**A**: EMPRA 的攻击思路可迁移到推荐场景：
- **攻击面**：推荐系统的双塔召回模型也是 embedding-based，恶意 item 可以通过操控 item embedding 提升排名
- **具体威胁**：刷单、虚假评论可被视为 "embedding perturbation"，将低质量 item 推到前排
- **防御**：(1) embedding 空间正则化防止异常偏移；(2) 对抗训练增强鲁棒性；(3) 异常检测 (embedding 变化速度监控)

---

## 延伸阅读

- [[20260503_scaling_sequence_multitask_frontier]] — ULTRA-HSTU/SparseCTR/LONGER 等 Scaling + 序列前沿
- [[20260504_scaling_coldstart_ctr_frontier]] — LUM/MixFormer 等 Scaling + CTR 前沿
- [[07_多任务学习与MoE]] — MMoE/PLE/SMES/DHEN 基础
- [[生成式推荐范式统一_20260403]] — 生成式推荐全景
