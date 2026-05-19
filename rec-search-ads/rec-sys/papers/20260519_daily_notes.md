# 推荐系统论文笔记 — 2026-05-19

> 来源：MelonEgg 每日学习（automated daily-cron）
> 范围：rec-sys 领域 10 篇

---

## 1. Multi-Behavior Generative Recommendation (MBGen)

**来源：** https://arxiv.org/abs/2405.16871 （CIKM 2024）
**领域：** 多行为 × 生成式推荐
**核心定位：** 首个把 MBSR（multi-behavior sequential recommendation）统一到生成式范式的工作

**两步生成式建模：**
1. 给定历史 item 序列 → 先预测下一个 behavior type（点击 / 加购 / 购买等），用来"圈定"用户意图
2. 给定历史 + 目标 behavior → 再生成下一个 item token

**关键设计：**
- 把 behavior 和 item 都 tokenize 成 token，交错排成 single sequence
- Position-routed sparse architecture：利用 token sequence 的异质性，在 scale up 时高效路由不同位置类型到不同专家
- 天然支持多任务（预测行为 + 预测物品）

**面试考点：** 多行为推荐为什么不能简单 concat、生成式范式如何统一异构行为、稀疏路由 vs MoE 的区别

---

## 2. Multi-Behavior Recommender Systems: A Survey

**来源：** https://arxiv.org/abs/2503.06963 （IJDSA 2026）
**领域：** 多行为推荐综述（首个系统性综述）

**三步分类法：**
1. **Data Modeling：** 输入层如何表示多行为（统一序列 / 多视图 / 异构图）
2. **Encoding：** 把多行为转成 embedding（行为感知 attention、行为间对比学习、行为图卷积）
3. **Training：** 多目标 loss、辅助任务、行为蒸馏

**研究态势：** 2019–2020 共 37 篇 → 2021–2022 共 79 篇 → 2023–2024 共 164 篇，热度持续上升

**核心 insight：** 真实场景下用户在 click / cart / fav / buy 等多种行为间穿梭，单一行为信号稀疏且 biased；多行为既能缓解稀疏性，也能更准确刻画 funnel 上的意图迁移

**面试考点：** 主辅行为如何区分、行为图建模的常见 GNN 范式、cascade 与 MMoE 在多行为下的取舍

---

## 3. EST: Towards Efficient Scaling Laws in CTR via Unified Modeling

**来源：** https://arxiv.org/abs/2602.10811 （Feb 2026）
**领域：** CTR × Transformer × Scaling Law
**核心问题：** CTR 模型与 LLM 的根本差异 — 候选打分要求毫秒级 latency，难以直接套用大 Transformer

**EST 架构：**
- **完全统一建模：** 所有原始输入（user features / item features / behavior history）拼成 single sequence，**不做有损 aggregation**
- **LCA（Lightweight Cross-Attention）：** 剪掉冗余的 self-interaction，只算高影响的 cross-feature 依赖
- **CSA（Content Sparse Attention）：** 基于内容相似度动态选择"高信号"的行为，避免对长序列做全连接 attention

**Scaling Law 结果：** EST 的 loss vs (参数, 算力) 呈现 **幂律** 关系，且斜率优于现有 hierarchical / partially-unified 范式 — 说明"无损统一序列"是 CTR scaling 的正确方向

**面试考点：** CTR 模型为什么难以 scale up、Lossless vs lossy aggregation、Cross-attention 在排序中的 latency 工程优化

---

## 4. Deep Mutual Learning across Task Towers for Multi-Task Rec

**来源：** https://arxiv.org/abs/2309.10357
**领域：** 多任务推荐 × 知识蒸馏

**核心论点：** 传统 MMoE / PLE 中每个 task tower 独立，知识共享只发生在下层 expert，task tower 之间缺乏 positive transfer，导致 negative transfer 严重

**两个新组件：**
1. **CTFM（Cross Task Feature Mining）：** 在 task tower 入口处共享、互传输入信息
2. **GKD（Global Knowledge Distillation）：** 用上层 task tower 的全局结果做软标签，蒸馏给其他 tower

**与主流方案对比：**
- Shared bottom：底层共享但 tower 独立 → 易冲突
- MMoE/PLE：用 gating 控制 expert 分配 → 仍未解决 tower 间隔阂
- MTL-DML：在 tower 之间显式做 mutual learning → 缓解 negative transfer

**面试考点：** Negative transfer 的根因诊断、Mutual learning vs Distillation 的区别、Task tower 共享策略的权衡

---

## 5. Predictive Models in Sequential Rec: Performance Laws & Data Quality

**来源：** https://arxiv.org/abs/2412.00430
**领域：** Sequential Rec × Scaling Law × 数据质量

**核心贡献：** 不只研究数据"量"，更研究数据"质"

**Performance Law：**
- 把 HR / NDCG 拟合成 (模型规模, 数据规模, 数据质量) 的函数
- 提出 **Approximate Entropy (ApEn)** 作为数据质量度量
- 验证：高质量数据上小模型可超过低质量数据上大模型

**Insight：** 数据扩展到一定规模后开始包含 repetitive / inefficient 样本，盲目堆量收益递减；用 ApEn 筛过的子集能在更少 token 下达到同等精度

**工业实践：** 用 Performance Law 指导"算力 / 数据 / 模型规模"三者的最优配比，避免无谓的 GPU 浪费

**面试考点：** Chinchilla scaling law 在推荐场景如何改写、序列数据的熵如何度量、训练数据去重 / 去噪的工程做法

---

## 6. Scaling Law of Large Sequential Recommendation Models

**来源：** https://arxiv.org/abs/2311.11351 （RecSys 2024）
**领域：** ID-only Sequential Rec × Scaling Law

**实验设定：** 纯 ID 形式（不加 item text / 多模态），把 user history 当 chronological item ID 序列，类似 GPT 的 next-token prediction

**关键发现：**
- 模型性能 vs 模型规模 / 数据规模呈 **power-law**
- 大模型更"数据高效"：相同 1B tokens，13B 模型从中提取的信息多于 200M 模型
- ID-only 范式即可观察到 scaling 现象 — 说明 scaling 与多模态信息无强绑定，本质是序列容量

**与 NLP 对比：** NLP scaling 来自 token 的语义压缩，推荐 scaling 来自 user behavior 的长尾覆盖与 collaborative signal

**面试考点：** 推荐 scaling 与 LLM scaling 的本质差异、ID embedding 表征容量瓶颈、HSTU / OneRec 等大推荐模型的演进脉络

---

## 7. VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language

**来源：** https://arxiv.org/abs/2512.10942 （Yann LeCun 等，Dec 2025）
**领域：** 表征学习 × JEPA × 多模态（与推荐的关系：item / content 表征底座）

**核心思想：** 不做 autoregressive token generation，而是在 abstract embedding space 里做"预测下一段文本的 embedding"

**架构：**
- Vision encoder + Text predictor 都在 embedding 空间工作
- 推理时才按需调用轻量 text decoder 把 embedding 翻译回 token

**实测：**
- 同 vision encoder + 同训练数据下，参数量仅为 token-space VLM 的 50%，效果更好
- 支持 selective decoding，decoding 操作数 −2.85×
- 天然适配 open-vocabulary classification / video retrieval / discriminative VQA

**对推荐的意义：** Item 多模态表征若用 JEPA 范式（而非 token-level CLIP 对齐），可能在 cold-start 与跨域迁移上更鲁棒

**面试考点：** JEPA vs Contrastive / Autoregressive 的本质区别、为什么 embedding-space prediction 抗冗余、推荐场景的多模态对齐方案

---

## 8. Multi-Task Recommendations with Reinforcement Learning (RMTL)

**来源：** https://arxiv.org/abs/2302.03328 （WWW 2023）
**领域：** 多任务 × 强化学习

**两大痛点：**
1. 现有 MTL 忽略 session-wise 的交互序列性
2. 多目标 loss 的权重难调，静态权重在不同用户 / 不同 session 上次优

**RMTL 方案：**
- 把 session 视为 RL trajectory，每个 step 的 reward 来自具体任务
- 用 actor-critic 学一个 **动态 task-weight policy**，根据 session state 自适应调整 (w_ctr, w_cvr, w_play, ...)
- 与 MMoE / PLE / ESMM 兼容，作为 plug-in 替换静态 weight

**实验：** 多个公开 / 工业数据集上 AUC 与 GAUC 同时提升

**面试考点：** 多任务静态 weighting 的失败模式（GradNorm / Uncertainty Weighting）、session-aware RL 的工程落地难点、actor-critic 与 PPO 的工业选型

---

## 9. Multi-Task Deep Recommender Systems: A Survey

**来源：** https://arxiv.org/abs/2302.03525
**领域：** MTL 推荐综述

**任务关系三分：**
- **Parallel：** CTR / CVR / 时长 同时预测（典型 MMoE）
- **Cascaded：** 后置任务依赖前置（ESMM：CVR 依赖 CTR）
- **Auxiliary with main：** 主任务 + 辅助任务（点击为主、关注 / 点赞为辅）

**方法学三分：**
1. **参数共享：** Shared bottom → Cross-stitch → MMoE → PLE → CGC
2. **优化策略：** Uncertainty Weighting、GradNorm、PCGrad、CAGrad
3. **训练机制：** 课程学习、知识蒸馏、对比学习辅助

**Negative Transfer 处理思路：** 任务隔离（PLE）、梯度投影（PCGrad）、动态权重（GradNorm）、跨塔蒸馏（MTL-DML，#4）

**面试考点：** MMoE vs PLE 的演进动机、为什么 ESMM 解决了 SSB 与 DS 问题、PCGrad 的梯度冲突缓解原理

---

## 10. EMPRA: Embedding Perturbation Rank Attack against Neural Ranking

**来源：** https://arxiv.org/abs/2412.16382 （TOIS 2026）
**领域：** 排序鲁棒性 / 对抗攻击

**威胁模型：** 黑盒 NRM，攻击者只能看到排序结果，目标是让某个 target document 排名前移

**EMPRA 方法：**
- 在 **句子级 embedding 空间** 做扰动（不是 token 级），保持语义可读性
- 用 transporter function 迭代地把目标文档每个句子的 embedding 推向一个 query-relevant anchor text
- 文本端通过 paraphrase / 同义重写"投影"回自然语言，对人类不可察

**威力：** 把原始排名 51–100 的 96% 文档推到 Top-10；不依赖 surrogate model，对各种 victim NRM 都鲁棒

**对推荐 / 搜索的启示：** 生产排序系统应增加：句子级 embedding 异常检测、对 anchor text 类语义投影的检测、defensive training 加入 EMPRA 风格扰动样本

**面试考点：** 黑盒 vs 白盒 ranking attack 的差异、对抗鲁棒排序的训练 trick、内容安全审核中 anchor text 检测的工程方案

---

## 当日小结

- **主线 1：Scaling Law in Rec** — 三篇（#3 EST、#5 Performance Law、#6 ID-only Scaling）共同描绘了推荐侧 scaling 的图景：模型规模与数据规模呈幂律，但数据质量比单纯数据量更关键，CTR 场景下"无损统一序列 + 稀疏 attention"是关键工程化路径
- **主线 2：Multi-Task / Multi-Behavior** — 四篇（#1 MBGen、#2 MBR Survey、#4 DML、#8 RMTL、#9 MTL Survey）系统覆盖了多任务多行为推荐的范式：从 MMoE/PLE 到生成式多行为，再到 RL 驱动的动态权重
- **副线：表征 / 鲁棒性** — VL-JEPA（#7）提示 item 表征的新方向；EMPRA（#10）警示排序系统的对抗脆弱性
