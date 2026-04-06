# 推荐系统论文笔记 — 2026-04-07

## 1. Beyond Interleaving: Causal Attention Reformulations for Generative Recommender Systems

**来源：** arXiv（近期）
**领域：** 生成式推荐 / 因果注意力
**核心问题：** 传统生成式推荐中，用户历史与候选 item 的 token 序列交错排列（interleaving），导致 attention 模式混乱，因果性难以保证。

**核心贡献：**
- 提出 Causal Attention Reformulation，将用户历史序列与候选 item SID token 明确解耦
- 通过因果掩码重新设计 attention 矩阵，保证历史 token 对候选 token 的单向信息流
- 解决 interleaving 带来的"位置污染"问题，提升生成一致性

**技术要点：**
- 将输入序列拆分为 context segment（用户历史）和 target segment（候选 SID）
- Context segment 使用完整双向 attention；target segment 使用因果（单向）attention
- 可插拔到现有 TIGER/HSTU 等生成式推荐框架

**面试考点：** 生成式推荐的 attention 设计、因果性 vs 双向性的 trade-off

---

## 2. Fidelity-Aware Recommendation Explanations via Stochastic Path Integration

**来源：** arXiv:2511.18047
**领域：** 可解释推荐
**核心问题：** 推荐系统解释的"忠实度"（fidelity）——解释是否真实反映模型决策过程

**核心贡献：**
- 提出 SPINRec（Stochastic Path Integration for Neural Recommender Explanations）
- 改进 Integrated Gradients，用随机基线采样替代固定基线，更适应推荐数据的稀疏性
- 从经验数据分布采样多个合理用户画像作为基线，选择最忠实的归因路径

**技术要点：**
```
IG(x) = (x - x') × ∫₀¹ ∂F(x' + α(x-x'))/∂x dα
SPINRec: x' ~ P_empirical，选 argmax fidelity(path)
```
- 评估三种架构（MF, VAE, NCF），多数据集（ML1M, Yahoo! Music, Pinterest）
- 引入反事实忠实度指标（counterfactual fidelity metrics）

**面试考点：** 推荐系统可解释性方法、Integrated Gradients 原理、忠实度 vs 可读性

---

## 3. RASTP: Representation-Aware Semantic Token Pruning for Generative Recommendation

**来源：** arXiv:2511.16943
**领域：** 生成式推荐效率优化
**核心问题：** SID（Semantic Identifier）用多个 token 表示 item，导致输入序列超长，训练推理开销大

**核心贡献：**
- 提出 RASTP，对 SID token 序列做智能剪枝
- Token 重要性 = 语义显著性（representation magnitude）+ 注意力中心性（cumulative attention weights）
- 剪枝后训练时间减少 **26.7%**，推荐性能不降甚至略有提升

**技术要点：**
```
importance(t) = α × ||h_t||₂ + (1-α) × Σ_l attn_l(t)
```
- 开源代码：https://github.com/Yuzt-zju/RASTP
- 适用于 TIGER、P5 等 SID-based 生成式推荐系统

**面试考点：** SID 生成式推荐原理、token pruning 策略、推理效率优化

---

## 4. Optimizing Recall or Relevance? A Multi-Task Multi-Head Approach for Item-to-Item Retrieval

**来源：** arXiv:2506.06239，KDD 2025
**领域：** I2I（Item-to-Item）召回
**核心问题：** 工业 I2I 检索主要基于 co-engagement 数据优化 recall，导致过拟合短期共现，忽视语义相关性

**核心贡献：**
- 提出 MTMH（Multi-Task Multi-Head I2I retrieval）
- 双头架构：一个头优化 co-engagement recall，另一个头优化语义 relevance
- 多任务 loss 正式建模 recall-relevance trade-off

**效果：**
- Recall 提升 **14.4%**，语义相关性提升 **56.6%**
- 线上验证：短期消费指标 + 长期用户体验指标双提升

**技术要点：**
```
L = λ₁·L_recall + λ₂·L_relevance
两路 embedding head 共享底层表示，独立优化目标
```

**面试考点：** I2I 召回设计、多任务学习平衡、recall vs precision 在工业场景的权衡

---

## 5. An Industrial-Scale Sequential Recommender for LinkedIn Feed Ranking

**来源：** arXiv:2602.12354
**领域：** 序列推荐 / 工业实践
**核心问题：** LinkedIn Feed 从 DCNv2 迁移到 Transformer-based 序列排序模型的工程实践

**核心贡献：**
- Feed-SR：生产级 Transformer 序列排序模型，替代 DCNv2 ranker
- 线上 A/B：成员时长增加 **+2.10%**
- 关键工程优化：dense embeddings 作为 late-fused context features，不增加 Transformer 维度

**技术要点：**
- Profile embeddings 作为 late-fused dense feature，对低交互用户（<10 历史行为）Long-Dwell AUC 提升 **+2% AUC**
- 在 LinkedIn 规模下的 serving 优化策略（模型量化、特征压缩）
- 严格的生产约束：延迟、吞吐量、内存

**面试考点：** 工业序列推荐落地挑战、Transformer 排序模型的 serving 优化、冷启动用户处理

---

## 6. Reason4Rec: LLMs for Recommendation with Deliberative User Preference Alignment

**来源：** arXiv:2502.02061
**领域：** LLM 推荐 / 推理增强
**核心问题：** 对齐后的推荐 LLM 在复杂场景表现差——因为只优化直接输出，缺乏"深思熟虑"（deliberation）

**核心贡献：**
- 提出 Deliberative Recommendation 任务：生成用户反馈前先显式推理用户偏好
- 三阶段推理框架：Preference Distillation → Preference Matching → Feedback Prediction
- 各阶段独立训练策略，从 verbalized user feedback 提取不同信号

**技术要点：**
```
步骤1: Distill(history) → aspect-level preferences
步骤2: Match(preferences, item) → rationale
步骤3: Predict(rationale, history, item) → feedback
```
- 三个真实数据集验证，预测精度和推理质量双提升

**面试考点：** LLM 推荐的对齐方法、CoT 推理在推荐中的应用、preference distillation

---

## 7. Cold-Starts in Generative Recommendation: A Reproducibility Study

**来源：** arXiv:2603.29845，SIGIR 2026
**领域：** 生成式推荐 / 冷启动
**核心问题：** 生成式推荐（基于 PLM）能否真正解决用户冷启动和 item 冷启动？

**核心贡献：**
- 系统性可重现性研究：在 8 块 A800 GPU 上复现多个生成式推荐方法
- 严格遵循原始超参配置，评估 user-side 和 item-side cold-start
- 发现：PLM-based 生成式推荐在某些冷启动场景**并不优于**传统方法

**关键发现：**
- Item cold-start：生成式推荐利用语义 SID 有优势
- User cold-start：依赖历史行为序列的生成式方法在极端冷启动下退化明显
- 可重现性差距：多个方法无法达到论文声称的指标

**面试考点：** 冷启动策略对比、生成式推荐的局限性、可重现性危机

---

## 8. GLASS: Generative Recommender for Long-sequence Modeling via SID-Tier and Semantic Search

**来源：** arXiv:2602.05663
**领域：** 生成式推荐 / 长序列建模
**核心问题：** 生成式推荐在长序列用户历史建模上的瓶颈

**核心贡献：**
- 双 Tier 架构：SID-Tier（长期兴趣 → 统一兴趣向量）+ Semantic Hard Search（语义检索增强）
- SID-Tier 将长期历史映射为兴趣向量，增强初始 SID token 预测
- Semantic Hard Search：用粗粒度 SID 作动态 key，检索相关历史行为，门控融合

**技术要点：**
```
SID-Tier: h_long = Aggregate(h_{t-L}...h_{t-K}) → 增强第1个SID token预测
SemanticHardSearch: query = coarse_SID, keys = history_SIDs
数据稀疏解决：语义邻居增强 + codebook resizing
```
- 在 TAOBAO-MM 和 KuaiRec 大规模数据集上超越 SOTA

**面试考点：** 生成式推荐的长序列问题、SID codebook 设计、双层检索架构
