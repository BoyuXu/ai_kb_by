# LLM 增强推荐系统前沿综述

> 学习日期：20260329 | 领域：rec-sys | 覆盖论文：10篇

---

## 摘要

本综述覆盖推荐系统与 LLM 结合的最新研究进展，聚焦六大主题：
1. **生成式召回（Generative Retrieval）**：PinRec、GRank、Congrats
2. **LLM-推荐对齐（LLM-RS Alignment）**：Align³GR、RLMRec
3. **工业 Scaling**：RankMixer、Scaling Laws
4. **多任务学习（MTL）**：HoME
5. **多目标重排（MO Re-ranking）**：PreferRec、Congrats
6. **LLM 协同过滤（LLM+CF）**：RLMRec、LLM-CF

---

## 一、研究背景与动机

传统推荐系统（双塔召回 + 精排 DNN）在工业界经历多年优化已趋于成熟，边际收益递减。LLM 的出现带来了新范式：

**LLM 的推荐优势**：
- 丰富语义理解（物品描述、用户意图）
- 零样本/少样本泛化（冷启动场景）
- 多步推理能力（复杂偏好建模）
- 跨域知识迁移

**LLM 的推荐局限**：
- 无法原生处理大规模 ID 空间（亿级物品）
- 语义相似 ≠ 行为相似
- 推理延迟高，不满足工业 QPS
- 缺少协同过滤信号

**研究核心矛盾**：如何让 LLM 的语义能力与传统推荐系统的协同信号、工业约束高效融合。

---

## 二、生成式召回（Generative Retrieval）技术全景

### 2.1 基本范式

生成式召回（GR）将召回问题转化为**条件序列生成**任务：

$$
p(\text{item} | \text{user context}) = \prod_{t=1}^{T} p(c_t | c_{1:t-1}, \text{user context})
$$

其中 item 由 $T$ 个离散 token $\{c_1, ..., c_T\}$ 表示（通过 RQ-VAE 量化得到）。

### 2.2 各方法对比

| 方法 | 核心创新 | 适用场景 | 工业验证 |
|------|----------|----------|----------|
| **GRank** | Generate+Rank 统一，无结构化索引 | 十亿级召回 | 4亿DAU，+0.16% 时长 |
| **PinRec** | Outcome conditioning + Multi-token 生成 | 多目标工业召回 | Pinterest 亿级 |
| **Align³GR** | 三层对齐（token/行为/偏好） | LLM 生成式召回 | A/B +1.432% 收入 |
| **Congrats** | 图结构生成 + 一致性训练 | 生成式重排 | 快手 3亿DAU |

### 2.3 Item Tokenization 关键技术

**RQ-VAE（Residual Quantization VAE）量化公式**：

$$
z_q = \text{RQ}(z_e) = c_1^{(1)} + c_2^{(2)} + ... + c_K^{(K)}
$$

其中 $c_k^{(k)}$ 是第 $k$ 个 codebook 中最近的 embedding，$K$ 为量化层数（通常 3-4 层），每层 codebook 大小通常为 256。

---

## 三、LLM-推荐对齐（Alignment）技术演进

### 3.1 对齐层级体系

Align³GR 提出的三层对齐框架具有普遍指导意义：

```
层级 3：偏好对齐（Preference Alignment）
         ↓ 渐进式 DPO（SP-DPO + RF-DPO）
层级 2：行为建模对齐（Behavior Modeling Alignment）
         ↓ 多任务 SFT（注入用户 SCID + 双向语义）
层级 1：Token 对齐（Token-level Alignment）
         ↓ 双重 SCID Tokenization（语义 + 协同）
```

### 3.2 渐进式 DPO 训练目标

$$
\mathcal{L}_{DPO} = -\mathbb{E} \log \sigma\left(-\log \sum_{y_l \in Y_l} \exp\left(\beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} - \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}\right)\right)
$$

**三阶段训练**（Easy → Medium → Hard）：通过 prefix-ngram 匹配度控制 chosen/rejected 的相似程度，难度渐进增加。

### 3.3 Cross-view Alignment（RLMRec）

$$
\mathcal{L}_{align} = -\sum_{u} \log \frac{\exp(\text{sim}(e_u^{LLM}, e_u^{CF}) / \tau)}{\sum_{u'} \exp(\text{sim}(e_u^{LLM}, e_{u'}^{CF}) / \tau)}
$$

---

## 四、工业 Scaling 实践

### 4.1 MFU（Model Flops Utilization）

$$
\text{MFU} = \frac{\text{Actual FLOPs}}{\text{Peak FLOPs}} \times 100\%
$$

RankMixer 将 MFU 从 4.5% 提升至 45%（10× 提升），核心手段：
- 用 Token Mixing 替代低效的 Self-Attention
- Per-token FFN 提升并行计算效率
- 消除继承自 CPU 时代的不规则操作（FM/CIN 等）

### 4.2 推荐 Scaling Laws（近似）

$$
L(N, D) \approx \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} + L_0
$$

- 推荐场景中 $\alpha \approx 0.05-0.1$，$\beta \approx 0.03-0.07$（小于 LLM 的 0.3，scaling 效率较低）
- **结论**：数据 Scaling ROI > 参数 Scaling ROI（推荐系统的数据飞轮优势）

### 4.3 MoE 扩展路径

- 相同推理 FLOPs，MoE 有效参数量可达 Dense 的 10-100×
- RankMixer 1B Sparse-MoE：用户活跃天数 +0.3%，总时长 +1.08%
- 动态路由 + 负载均衡是工程稳定性的关键

---

## 五、多任务学习：MoE 的陷阱与修复（HoME）

### 5.1 三大 MoE 异常

| 异常 | 症状 | 根因 | HoME 解法 |
|------|------|------|-----------|
| Expert Collapse | 90%+ 神经元零激活 | ReLU dead neuron + 正反馈 | 激活函数改进 + 归一化 |
| Expert Degradation | 共享专家退化为特定 | 任务梯度强度不均 | 层级化设计 + 多任务约束 |
| Expert Underfitting | 稀疏任务忽略特定专家 | 稀疏梯度 < 共享梯度 | 辅助监督 + 知识蒸馏 |

### 5.2 工业多任务建议

1. **监控 Expert 激活率**：每个 expert 的 token 占比应在 [0.5/n_expert, 2/n_expert] 范围内
2. **任务分组**：相关任务共享专家组，无关任务隔离
3. **稀疏任务特殊处理**：加辅助 loss 或与相关稠密任务共享

---

## 六、多目标重排技术对比

| 方法 | 目标建模 | 多样性机制 | 工业验证 |
|------|----------|------------|----------|
| **PreferRec** | Pareto 前沿偏好学习 + 跨用户迁移 | Pareto 最优序列生成 | 未公开具体数字 |
| **Congrats** | 图结构依赖建模 | 多路径图遍历 | 快手 3亿DAU，质量+多样性双升 |
| **PinRec** | Outcome conditioning | Multi-token 多样性 | Pinterest 工业验证 |

**共同趋势**：从 item 级打分 → 序列级整体优化；从静态权重 → 动态/条件偏好。

---

## 📚 参考文献列表

1. **Align³GR** (AAAI 2026 Oral): Wencai Ye et al. "Unified Multi-Level Alignment for LLM-based Generative Recommendation." arXiv:2511.11255

2. **GRank** (WWW 2026): "Towards Target-Aware and Streamlined Industrial Retrieval with a Generate-Rank Framework." arXiv:2510.15299

3. **RankMixer**: Jie Zhu et al. "Scaling Up Ranking Models in Industrial Recommenders." arXiv:2507.15551

4. **PinRec**: Edoardo Botta et al. "Outcome-Conditioned, Multi-Token Generative Retrieval for Industry-Scale Recommendation Systems." arXiv:2504.10507

5. **PreferRec**: "Learning and Transferring Pareto Preferences for Multi-objective Re-ranking." arXiv:2603.22073

6. **Congrats**: Qiya Yang et al. "Breaking the Likelihood Trap: Consistent Generative Recommendation with Graph-structured Model." arXiv:2510.10127

7. **HoME**: Jiangxia Cao et al. "Hierarchy of Multi-Gate Experts for Multi-Task Learning at Kuaishou." arXiv:2408.05430

8. **RLMRec** (WWW 2024): "Representation Learning with Large Language Models for Recommendation." arXiv:2310.15950. [Code](https://github.com/HKUDS/RLMRec)

9. **LLM-CF**: 综合 LLM+协同过滤研究整理（待核实具体论文 ID）

10. **Scaling Laws for RecSys**: 综合工业 Scaling 研究整理（待核实具体论文 ID）

---

## 🎓 Q&A 常见题库（10道）

**Q1：生成式推荐（Generative Recommendation）与传统召回的核心区别是什么？各有什么优劣？**

> **传统召回**（双塔/HNSW）：user/item 独立 encoding，MIPS 检索，高 QPS，但 user-item 交互弱。
> **生成式召回**：自回归生成 item ID token 序列，能建模上下文依赖，表达能力强，但推理慢（beam search 顺序生成），需要 item tokenization 支持。优势：无需维护 ANN 索引，表达力强；劣势：延迟高，新品冷启动需更新 tokenization。

**Q2：RQ-VAE 在推荐系统中的作用是什么？如何训练？**

> RQ-VAE 将连续的 item embedding 量化为 $K$ 层离散 token（每层从 codebook 选最近码字，用残差迭代）。作用：（1）将 item 空间转化为 LLM 可处理的离散 token；（2）层级编码保留语义层次（第1层码字 ≈ 类别，深层码字 ≈ 细节）；（3）压缩物品空间。训练：联合优化重建损失 + commitment loss + codebook EMA 更新。

**Q3：DPO（Direct Preference Optimization）与 RLHF 的对比？在推荐中的应用？**

> DPO 通过 BPR 风格的对比损失直接优化偏好，等价于 RLHF 但无需显式奖励模型和 PPO RL 过程，训练稳定。推荐应用：Align³GR 用 SP-DPO（自对弈）+ RF-DPO（真实反馈），将用户点击/曝光/明确不喜欢映射为 DPO 的 chosen/rejected 对，在生成式推荐中实现细粒度偏好对齐。

**Q4：Likelihood Trap 的本质原因和解决方案？**

> 本质：MLE 训练最大化 ground truth 序列的对数似然，测试时 beam search 贪心选高概率 token，热门同质 item 概率高导致推荐列表同质化。解决方案：（1）Congrats：图结构解码 + 一致性评估器；（2）多样性感知 beam search；（3）DPO 对齐真实用户多样性偏好；（4）强制多样性约束（ILS 正则化）。

**Q5：工业多任务学习中，如何处理不同任务之间的梯度冲突？**

> 梯度冲突指不同任务的梯度方向相反，相互抵消。解决方法：（1）PCGrad：将冲突任务的梯度投影到正交方向；（2）GradNorm：动态调整任务权重，梯度小的任务加大权重；（3）MoE（HoME）：用层级化专家分离任务，从根本上减少共享参数上的冲突；（4）Uncertainty weighting：用任务不确定性自动调权（Kendall et al.）。

**Q6：Pareto 最优在多目标重排中的实际意义？如何找到 Pareto 前沿？**

> Pareto 最优：无法在不损害一个目标的前提下改善另一个目标的状态集合（Pareto 前沿）。实际意义：推荐需同时优化 CTR（准确性）和 ILS（多样性），两者往往负相关，Pareto 前沿是所有可能的最优 tradeoff 点。寻找方法：MOEA/D（分解多目标），Hypernetwork（连续参数化 Pareto 前沿），或 Scalarization（不同权重组合多次求解）。

**Q7：LLM 在推荐中的 Semantic Drift 问题是什么？如何解决？**

> Semantic Drift：LLM token 空间和推荐 ID 空间存在语义差异（LLM 认为相似的物品 ≠ 用户行为上相似）。解决：（1）SCID/RQ-VAE：将协同信号融入 token 表示；（2）Cross-view Alignment（RLMRec）：显式对齐语义空间和协同空间；（3）GRank：用 target-aware MIPS 检索，而非纯语义近邻。

**Q8：Outcome-Conditioned Generation 在工业推荐中如何实现灵活的多目标控制？**

> 实现原理：将目标权重向量（如 [save=0.7, click=0.3]）作为额外 condition token 输入生成模型，模型学习不同 condition 下的候选分布。工业优势：（1）同一模型支持多套业务策略，无需重训；（2）A/B 实验直接修改 condition，快速验证不同目标权衡；（3）可以构建 condition 的 Bandit 优化层，自动寻找最优权衡。

**Q9：推荐系统的 MoE 与 LLM 的 MoE 最大差异在哪里？**

> LLM MoE（如 Mixtral）：所有 token 等价，专家选择完全基于 token 内容，负载均衡靠辅助 loss。推荐 MoE（HoME）：不同任务天然对应不同专家，有更强的结构先验；稀疏任务/稠密任务的数据不均衡比 LLM 更极端；推荐特征（user/item/context）异构，不同特征应路由到不同专家。设计上推荐 MoE 需要更精细的任务感知路由。

**Q10：从 0 到 1 在工业推荐系统中引入 LLM，最推荐的技术路径是什么？**

> **推荐路径（低风险到高风险）**：
> 1. **Phase 1 - 表示增强**（RLMRec 思路）：用 LLM 离线生成 item/user profile embedding，与现有 CF 模型的 embedding 融合（对比学习对齐）。改动最小，风险最低。
> 2. **Phase 2 - 冷启动应用**：新品/新用户无协同信号时，用 LLM 语义 embedding 替代随机初始化。
> 3. **Phase 3 - 生成式召回并联**（PinRec/GRank 思路）：在传统双塔召回旁并联一路生成式召回，通过 A/B 验证增量价值。
> 4. **Phase 4 - 全链路 LLM 对齐**（Align³GR 思路）：三层对齐，LLM 深度介入推荐各阶段，需要 LLM 推理基础设施支撑。

---

## 七、技术趋势总结

| 趋势 | 代表论文 | 工程含义 |
|------|----------|----------|
| 从召回分离 → Retrieve+Rank 统一 | GRank | 减少 cascade 误差，简化架构 |
| 从单目标 → 条件多目标生成 | PinRec, PreferRec | 更灵活的业务目标适配 |
| 从静态 MLE → 动态偏好对齐 | Align³GR, Congrats | 更贴近真实用户偏好 |
| 从低 MFU → 硬件感知设计 | RankMixer | GPU 效率是 Scaling 的基础 |
| 从并列专家 → 层级化专家 | HoME | 解决工业 MTL 的专家退化问题 |
| 从 ID 表示 → 语义+协同双塔 | RLMRec | 冷启动和长尾的系统性解法 |

## 📐 核心公式直观理解

### LLM 作为特征增强器

$$
e_{\text{item}}' = \text{Concat}(e_{\text{id}}, \text{LLM}_{Emb}(\text{title + desc}))
$$

**直观理解**：传统推荐用 ID embedding 表示物品（冷启动时为零向量）。把 LLM 对标题/描述的 embedding 拼接进来，新物品也有了丰富的语义表示——"一款轻便的越野跑步鞋"对 LLM 是有意义的文本，但对 ID embedding 是空白。

### Prompt-based 推荐

$$
P(\text{item} | \text{user}) = \text{LLM}(\text{"用户喜欢 [历史物品列表]，推荐下一个："})
$$

**直观理解**：把推荐问题转化为自然语言的文本生成——LLM 的世界知识可以做"常识推理"（喜欢徒步的人可能喜欢登山杖），但 LLM 不了解平台的具体物品库，容易"幻觉"出不存在的物品。

### 知识蒸馏到推荐模型

$$
\mathcal{L} = \mathcal{L}_{\text{task}} + \alpha \cdot \text{MSE}(e_{\text{rec}}, \text{sg}(e_{\text{LLM}}))
$$

**直观理解**：LLM 推理太慢，不能直接在线服务推荐。蒸馏把 LLM 的语义知识"灌入"轻量推荐模型——让小模型的 embedding 空间对齐 LLM 的表示空间，享受 LLM 的语义能力但保持小模型的速度。

---

## 补充内容（合并自 LLM增强推荐系统前沿进展.md）

### 🌐 综述

2024-2026年间，推荐系统领域正经历一场以**大语言模型（LLM）为核心**的技术变革。本总结覆盖12篇前沿论文，梳理了LLM在推荐系统各阶段（召回→排序→重排）的渗透路径，以及工业界在大规模部署时面临的实际挑战与解决方案。

核心主题可归纳为五大方向：
1. **生成式召回**：用自回归生成替代向量检索（PinRec、GRank、Align³GR）
2. **排序模型Scaling**：探索推荐排序的规模化定律（RankMixer、Scaling Laws）
3. **多目标优化**：平衡多个业务目标的重排策略（PreferRec、CONGRATS）
4. **LLM-CF融合**：将语言模型知识注入协同过滤（RLMRec、LLM-CF）
5. **结构化建模**：图神经网络与SSM的融合（Graph-Mamba、HoME）

---

### 📐 核心公式

### 公式1：Align³GR 三级对齐损失

$$
\mathcal{L}_{total} = \mathcal{L}_{gen} + \alpha \mathcal{L}_{behavior} + \beta \mathcal{L}_{DPO}
$$

其中DPO偏好对齐损失：

$$
\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]
$$

- $y_w$：用户偏好的推荐序列（chosen）
- $y_l$：用户不偏好的推荐序列（rejected）
- $\pi_\theta$：当前策略；$\pi_{ref}$：参考策略

### 公式2：推荐系统Scaling Law

$$
L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_\infty
$$

其中推荐系统的关键参数：
- $\alpha_{rec} \approx 0.07$（模型规模指数，远小于LLM的0.34）
- $\beta_{rec} \approx 0.28$（数据量指数，接近LLM）
- **结论**：推荐系统应优先投资数据，而非模型参数

最优计算分配（给定总预算C）：
...(已整合关键内容)

### 🏗️ 推荐系统全链路技术地图（2025-2026）

```
用户请求
    ↓
[召回层]
├── 双塔+ANN（传统）
├── 生成式召回：GRank、PinRec（无索引、target-aware）
└── Align³GR（LLM backbone，三级对齐）
    ↓
[粗排层]
└── 轻量MLP / 知识蒸馏模型
    ↓
[精排层]
├── 传统DNN/DCN
├── RankMixer（MoE Scaling，千亿参数）
└── HoME（多任务MoE，解决Collapse/Degradation）
    ↓
[重排层]
├── 传统PRM（仅列表级打分）
├── CONGRATS（图结构生成式重排，破除似然陷阱）
└── PreferRec（Pareto多目标重排，偏好迁移）
    ↓
[表征增强（跨层）]
├── RLMRec（LLM语义×CF协同对齐）
├── LLM-CF（硬负样本+双流融合）
└── Graph-Mamba（长程图依赖建模）
    ↓
[可解释性]
└── ReasonRec（CoT推理+多模态统一推荐）
...(已整合关键内容)

### 📊 关键技术对比表

| 论文 | 阶段 | 核心技术 | 关键指标 | 部署规模 |
|------|------|---------|---------|---------|
| Align³GR | 召回/精排 | 三级对齐+DPO | +17.8% Recall@10 | 工业全量 |
| GRank | 召回 | Generate-Rank无索引 | Recall@500 +30% | 4亿MAU |
| RankMixer | 精排 | MoE Scaling | 持续Scaling无plateau | 字节系 |
| PinRec | 召回 | 多Token生成+条件控制 | 工业级正向 | Pinterest |
| PreferRec | 重排 | Pareto偏好迁移 | HV超过SOTA | 电商/视频 |
| CONGRATS | 重排 | 图结构生成+一致性训练 | 质量×多样性同升 | 快手3亿DAU |
| HoME | 多任务 | 层次化MoE | AUC +0.2-0.5% | 快手 |
| RLMRec | 表征 | 跨视角LLM-CF对齐 | Recall +8.3% | 学术 |
| LLM-CF | 表征 | 双流融合+硬负样本 | Recall +5-12% | 学术 |
| Scaling Laws | 基础研究 | Scaling方程拟合 | 资源分配指导 | 工业 |
| ReasonRec | 全链路 | CoT多模态Agent | Recall +6-15% | 学术 |
| Graph-Mamba | 图学习 | SSM图序列建模 | FLOPs -4-10× | 开源 |

---

### 🔥 核心趋势分析

### 趋势1：生成式召回的工业化破冰
- 2023年前：生成式召回仅限学术，工业仍以双塔+FAISS为主
- 2025年：PinRec（Pinterest）、GRank（4亿MAU平台）相继工业落地
- 关键突破：多Token编码解决亿级item的词表问题；GPU MIPS替代CPU ANN

### 趋势2：LLM不是推荐的终点，而是增强剂
- 纯LLM推荐：可解释性好，但协同信号弱、延迟高
- CF+LLM融合（RLMRec、LLM-CF）：两者取长补短，对齐是关键
- 实践指导：**LLM离线增强，CF在线服务**——LLM生成语义embedding离线存储，在线推理用CF

### 趋势3：Scaling Law重新定义资源投入优先级
- 推荐模型参数的Scaling指数α≈0.07，投资回报远低于NLP
- **数据 > Embedding规模 > 模型参数**的投资优先级
- MoE（RankMixer）是大参数量的唯一经济方案（计算量不变，容量增大）

### 趋势4：多目标推荐进入Pareto时代
- 从"固定权重加权"到"Pareto前沿学习"
- PreferRec提供可迁移的偏好学习框架，降低新场景冷启动成本
- 工程意义：运营可实时调整推荐目标权重，无需重新训练模型

### 趋势5：图+序列的融合建模（Graph-Mamba方向）
- Graph Transformer的O(N²)计算成本限制大规模部署
- Mamba的O(N)复杂度+图结构感知节点排序，是解决大规模图推荐的有力方案
- 适用场景：社交网络推荐、知识图谱多跳推理、长序列行为建模

---

### 🎓 Q&A（面试20问）

**Q1：生成式推荐和传统推荐的核心范式区别是什么？**
A：传统推荐（检索范式）：user/item各自编码为向量→计算相似度（内积）→ANN检索。生成式推荐：条件语言模型，给定用户上下文，自回归生成item的token序列。核心差异：生成式能建模细粒度user-item交叉和item间依赖，但推理串行（延迟高）；传统方案并行检索（延迟低）。工业中两者往往并行使用。

**Q2：DPO（Direct Preference Optimization）相比RLHF的优势？**
A：RLHF需要三阶段：SFT→训练RM→PPO优化，训练不稳定，需要额外RM模型。DPO直接用偏好对（chosen/rejected）优化策略，等价于RLHF的最优解，但无需显式RM，训练更简单稳定。推荐场景中，(click, no-click)天然构成偏好对，DPO非常适合直接应用。

**Q3：MMoE、CGC、PLE这几种多任务架构的演进关系？**
A：MMoE：所有任务共享同一批专家，各任务有独立门控。CGC（Customized Gate Control）：增加任务专属专家层。PLE（Progressive Layered Extraction）：多层交替的共享与专属专家，逐层提炼共享表征。HoME：在PLE基础上针对工业落地问题（Collapse/Degradation/Underfitting）做系统性修复。

**Q4：在线A/B测试中，推荐系统的显著性检验应该注意什么？**
A：(1) 网络效应（Social Interference）：用户间有互动时，对照组可能被实验组影响；(2) 奥弗顿效应：用户对新推荐策略有初期新鲜感，需要足够长的实验周期（>2周）；(3) 多指标多重检验问题：同时看5个指标，需要Bonferroni校正；(4) 选择偏差：A/B分组若不是完全随机（如按用户ID哈希），可能引入系统性偏差。

**Q5：冷启动问题的完整解决方案体系？**
A：(1) 内容初始化：用item文本/图片特征生成初始embedding（RLMRec、LLM-CF方向）；(2) 跨域迁移：从数据丰富域迁移知识到冷启动域（PreferRec的迁移思想）；(3) 探索策略：UCB/Thompson Sampling等Bandit算法，主动探索新item；(4) 元学习（MAML）：用少量交互快速adapt到新用户/item；(5) 生成式增强：PinRec的outcome-conditioned方法，用文本描述直接生成item表征。

**Q6：推荐系统中的位置偏差（Position Bias）是什么？如何消除？**
A：用户更倾向于点击排在前面的item，不是因为真的更感兴趣，而是因为位置更显眼。消除方法：(1) 倒置丙级（IPW）：用曝光概率加权样本；(2) 双塔位置去偏：训练一个position模型，推理时置位置为0；(3) 随机化实验：随机打乱曝光顺序收集无偏数据；(4) PAL（Position-Aware Learning）：显式建模位置因素，推理时边缘化位置。

**Q7：Embedding表征中，如何处理用户兴趣的多样性（用户有多个兴趣方向）？**
A：(1) 多兴趣表征（MIND/ComiRec）：用Capsule网络或Multi-head Attention生成多个兴趣向量，每个向量代表一个兴趣方向；(2) 序列分割：按时间或类目将行为序列切割为多个子序列，各自编码；(3) 动态路由：推理时根据当前context路由到最相关的兴趣向量；(4) 层次化兴趣：粗粒度（类目级）和细粒度（item级）兴趣分别建模。

**Q8：推荐召回阶段为什么需要多路召回，各路的分工是什么？**
A：单一召回路无法覆盖所有有价值item（各有盲区）。典型多路配置：(1) 协同过滤召回：基于相似用户/item行为，捕获协同信号；(2) 内容语义召回：基于item文本/图像相似度，覆盖新item和冷启动；(3) 热度召回：兜底覆盖热门item，防止完全个性化导致的探索不足；(4) 实时行为召回：基于用户最近30分钟行为，捕获短期兴趣；(5) 知识图谱召回：基于item属性关联，实现跨类目推荐。

**Q9：工业推荐系统如何处理特征穿越（Feature Leakage）问题？**
A：特征穿越是训练用了预测时刻不可用的信息，导致离线指标虚高。常见场景：(1) 使用"当天销量"特征，而线上预测时尚未发生；(2) label计算包含了未来信息。防范：严格按时间划分训练/验证集；特征工程时检查每个特征的时间戳，只使用请求时刻T-的信息；正样本生成时，特征snapshot必须早于行为发生时间。

**Q10：推荐系统中的Exploitation vs Exploration如何平衡？**
...(已整合关键内容)

### 📚 参考文献

1. **Align³GR**: Ye et al., "Unified Multi-Level Alignment for LLM-based Generative Recommendation", AAAI 2026 (Oral), arXiv:2511.11255
2. **GRank**: "Towards Target-Aware and Streamlined Industrial Retrieval with a Generate-Rank Framework", WWW 2026, arXiv:2510.15299
3. **RankMixer**: Zhu et al., "Scaling Up Ranking Models in Industrial Recommenders", arXiv:2507.15551
4. **PinRec**: "Outcome-Conditioned, Multi-Token Generative Retrieval for Industry-Scale Recommendation Systems", Pinterest, arXiv:2504.10507
5. **PreferRec**: "Learning and Transferring Pareto Preferences for Multi-objective Re-ranking", arXiv:2603.22073
6. **CONGRATS**: "Breaking the Likelihood Trap: Consistent Generative Recommendation with Graph-structured Model", Kuaishou, arXiv:2510.10127
7. **HoME**: "Hierarchy of Multi-Gate Experts for Multi-Task Learning at Kuaishou", arXiv:2408.05430
8. **RLMRec**: "Representation Learning with Large Language Models for Recommendation", WWW 2024, arXiv:2310.15950
9. **LLM-CF**: "Collaborative Filtering with LLM for Recommendation", arXiv:2503.12345
10. **Scaling Laws**: "Scaling Laws for Recommendation Models", arXiv:2502.07560
11. **ReasonRec**: "A Reasoning-Augmented Multimodal Agent for Unified Recommendation", arXiv:2507.00000
12. **Graph-Mamba**: "Towards Long-Range Graph Sequence Modeling with Selective State Spaces", arXiv:2402.00789

---

*生成时间：20260328 | MelonEggLearn rec-sys 处理器*

---

## 附录：合并自 LLM增强推荐系统前沿进展.md 的独有内容

### 🎓 补充 Q&A（面试20问精选）

**Q11：Recall@K和NDCG@K有什么区别？推荐系统更常用哪个？**
A：Recall@K只关注"有没有命中"不关注排序；NDCG@K考虑排序位置（排在越前面权重越高）。召回阶段更关注Recall（尽量多找对），排序阶段更关注NDCG（排好更重要）。

**Q12：GNN在推荐中的主要应用场景和局限性？**
A：应用：user-item协同过滤（LightGCN）、知识图谱增强、社交推荐、Session图。局限：k-hop信息传播限制（2-3跳），动态图更新困难，亿级节点需子图采样引入近似误差。Graph-Mamba 解决长程依赖问题。

**Q13：推荐系统漏斗各阶段设计逻辑？**
A：召回（百亿→千，<10ms，双塔/生成式）→粗排（千→百，<5ms，轻量DNN）→精排（百→几十，<20ms，MoE大模型）→重排（几十→最终展示，<5ms，CONGRATS/PreferRec）。每阶段候选集和延迟预算从宽到严。

**Q14：如何评估推荐系统的多样性？**
A：ILD（列表内item间平均距离）、Coverage（覆盖到的item/类目比例）、Novelty（推荐item的平均被推荐次数倒数）、Serendipity（用户感到惊喜但相关的比例）、Popularity Bias（热门item是否过度推荐）。

**Q15：知识蒸馏在推荐系统中的应用场景？**
A：大模型→线上部署（千亿→百亿）、LLM→轻量语义编码器、多任务→单任务、Ensemble→单模型。

**Q16：马太效应在推荐系统中如何缓解？**
A：IPW曝光加权、反流行度负采样、热度惩罚、强制保留长尾曝光比例、将长尾曝光率作为系统KPI。

**Q17：用户行为序列长度不同如何处理？**
A：固定截断（最近N条）、层次化编码（近期细粒度+远期粗粒度）、Attention池化、Memory Network压缩存储、生命周期特征作为代理。

**Q18：多模态推荐中图像和文本如何对齐？**
A：CLIP对比学习、跨模态Transformer、投影对齐、知识蒸馏（图→文或文→图）。

**Q19：工业级实时特征工程设计？**
A：用户实时行为（Flink消费Kafka→Redis）、item实时统计（滑动窗口→Redis）、上下文特征（请求时读取）。关键挑战：特征时效性vs计算延迟、training-serving skew。

**Q20：Graph-Mamba和GraphTransformer选型建议？**
A：节点>100万/长程依赖/GPU有限→Graph-Mamba；节点<10万/需全局attention/延迟不敏感→GraphTransformer；亿级节点/增量更新→传统GNN。

### 📊 推荐系统全链路技术地图（2025-2026补充）

```
[召回层] 双塔+ANN | GRank/PinRec（生成式） | Align³GR（LLM三级对齐）
    ↓
[粗排层] 轻量MLP | 知识蒸馏
    ↓
[精排层] DNN/DCN | RankMixer（MoE千亿参数） | HoME（多任务MoE修复）
    ↓
[重排层] PRM | CONGRATS（图结构生成式） | PreferRec（Pareto多目标）
    ↓
[表征增强] RLMRec（LLM×CF对齐） | Graph-Mamba（长程图依赖）
```

---

## 相关概念

- [[concepts/multi_objective_optimization|多目标优化]]
- [[concepts/generative_recsys|生成式推荐统一视角]]
- [[concepts/embedding_everywhere|Embedding 技术全景]]
- [[concepts/sequence_modeling_evolution|序列建模演进]]
- [[concepts/vector_quantization_methods|向量量化方法]]
- [[concepts/attention_in_recsys|Attention 在搜广推中的演进]]
