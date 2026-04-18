# 推荐系统论文笔记 — 2026-04-17

## 1. HSTU: Actions Speak Louder than Words — Trillion-Parameter Sequential Transducers for Generative Recommendations

**来源：** https://arxiv.org/abs/2402.17152 (Meta AI)
**领域：** 生成式推荐 × 超大规模模型
**核心定位：** 首个万亿参数级序列转导架构，专为推荐系统的高基数、非平稳流数据设计

**核心贡献：**
- 引入 HSTU（Hierarchical Sequential Transduction Units），将推荐重新定义为生成式序列转导任务
- 使用 **Pointwise Normalization** 替代 softmax，解决非平稳词汇表问题
- 在长序列（8192 tokens）上比 FlashAttention2-based Transformers 快 5.3x-15.2x

**关键技术：**
- Pointwise normalization 处理动态变化的候选集分布
- 层级转导单元优化流数据处理
- 在 Meta 多个产品面部署，服务数十亿用户

**工业验证：**
- 在线 A/B 测试：+12.4% 核心指标提升
- 公开数据集：+65.8% NDCG 提升

**面试考点：** pointwise vs softmax normalization 在非平稳场景的区别、万亿参数推荐模型的工程挑战、生成式推荐的序列建模范式

---

## 2. GPR: Towards a Generative Pre-trained One-Model Paradigm for Large-Scale Advertising Recommendation

**来源：** https://arxiv.org/abs/2511.10138 (Tencent)
**领域：** 生成式推荐 × 统一模型
**核心定位：** 首个统一单模型框架，用端到端生成方法替代传统级联多阶段广告系统

**核心贡献：**
- 统一输入 schema、tokenization 和多阶段训练
- Heterogeneous Hierarchical Decoder (HHD)：双解码器架构解耦用户意图建模与广告生成
- 三阶段训练：Multi-Token Prediction (MTP) → Value-Aware Fine-Tuning (VAFT) → Hierarchy Enhanced Policy Optimization (HEPO)

**关键公式：**
- VAFT loss 加权：L = Σ(weight_i × loss_i)，weight 融合行为类型和归一化 eCPM
- Token schema：U-Token（用户）、O-Token（场景）、E-Token（行为）、I-Token（物品）

**工业验证：** 部署于腾讯微信视频号广告系统，CTCVR 和 GMV 显著提升

**面试考点：** 单模型统一范式 vs 级联系统的 trade-off、VAFT 如何对齐商业价值与预测目标、Semantic ID 空间设计

---

## 3. OneRanker: Unified Generation and Ranking with One Model

**来源：** https://arxiv.org/abs/2603.02999 (Tencent)
**领域：** 生成+排序统一 × 工业广告推荐
**核心问题：** 兴趣目标与商业价值之间的失配、生成阶段无目标感知、生成与排序断裂

**核心贡献：**
- **价值感知多任务解耦：** 共享表征中分离兴趣覆盖和价值优化
- **粗到精协同目标感知：** Fake Item Tokens（生成阶段隐式感知）+ 排序解码器（显式价值对齐）
- **Key/Value 直通 + Distribution Consistency (DC) Loss：** 端到端优化

**工业验证：** 部署于微信视频号，GMV +1.34%

**面试考点：** Fake Item Token 的隐式目标感知机制、DC Loss 如何保证生成-排序一致性、兴趣覆盖 vs 价值优化的张力

---

## 4. RecGPT: Next-Generation Recommendation System with LLM

**来源：** https://arxiv.org/abs/2507.22879 (Taobao/Alibaba)
**领域：** LLM × 推荐系统
**核心转变：** 从日志拟合到意图中心的推荐范式

**核心方法（两阶段 LLM Pipeline）：**
1. **User-Interest LLM：** 分析用户全生命周期行为，生成自然语言兴趣画像
2. **Item-Tag LLM：** 基于兴趣推理生成细粒度物品标签

**关键创新：**
- 用自然语言兴趣画像作为中间表征（而非嵌入向量）
- RecGPT-V2 引入 Agentic 意图推理
- 强化学习在淘宝十亿级行为数据上训练

**工业验证：**
- V1: +6.96% CICD, +4.82% DT
- V2: +3.40% IPV, +4.68% CTR, +4.05% TV, +11.46% NER

**面试考点：** 自然语言画像 vs 嵌入向量的优劣、两阶段 LLM 设计防止误差传播、RL 训练数据构建

---

## 5. MTMH: Multi-Task Multi-Head Approach for Item-to-Item Retrieval

**来源：** KDD 2025
**领域：** I2I 召回 × 多任务学习
**核心问题：** 传统 I2I 过度拟合短期共消费模式，忽略语义相关性和多样性

**核心贡献：**
- 多任务学习 loss 形式化平衡 recall vs semantic relevance
- 多头 I2I 架构同时检索共消费和语义相关物品
- 解耦共消费优化与语义匹配

**工业验证：** recall +14.4%，semantic relevance +56.6%

**面试考点：** recall-relevance trade-off 的形式化定义、多头架构优势、长期用户体验指标改善

---

## 6. MTORL: Multi-task Offline Reinforcement Learning for Advertising

**来源：** https://arxiv.org/abs/2506.23090 (KDD 2025)
**领域：** 离线 RL × 广告推荐
**核心贡献：** 将离线 RL 应用于同时优化渠道推荐和预算分配

**关键技术：**
- MDP 框架建模广告决策
- **因果状态编码器：** 通过条件序列建模捕获动态用户兴趣和时序依赖
- 多任务解码：联合渠道和预算动作
- 因果注意力机制处理用户序列

**核心挑战：** 离线 RL 中的过估计、分布偏移、预算约束

**面试考点：** 因果注意力 vs 标准注意力在序列建模中的区别、广告场景的分布偏移问题、多任务 RL 梯度冲突

---

## 7. OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer

**来源：** https://arxiv.org/abs/2510.26104 (ByteDance & NTU)
**领域：** 统一 Transformer × 特征交互 × 序列建模
**核心贡献：** 单一 Transformer 同时完成特征交互和用户行为序列建模，替代分离的专门模块

**关键技术：**
- **统一 Tokenizer：** 序列和非序列属性转换为统一 token 序列
- 相似序列 token 共享参数，非序列特征使用 token-specific 参数
- Causal attention + cross-request KV caching 降低推理成本

**工业验证：**
- 在线 A/B: per-user GMV +5.68%
- 离线: CTR AUC +1.53%, CTR UAUC +2.79%, CVR AUC +1.14%, CVR UAUC +3.23% (vs DCNv2+DIN)

**面试考点：** 统一 tokenization 实现信息交流的原理、token-specific 参数平衡效率与表达力、KV caching 跨请求优化

---

## 8. E2E Semantic ID Generation for Generative Recommendation Systems

**来源：** Snap Research, Spotify Research
**领域：** 生成式推荐 × 语义 ID
**核心贡献：** 将物品语义表征编码为离散、紧凑、可解释的标识符

**关键技术：**
- 层级聚类 Tokenizer：物品嵌入 → 稀疏 ID 序列
- 有意义的碰撞（相似物品共享 ID 组件）
- 同时具备记忆性（离散）和泛化性（共享语义）

**工业验证：** 新/长尾物品 +16% 准确率（LLM-driven POI 推荐）

**面试考点：** Semantic ID vs one-hot 编码的泛化优势、层级聚类过程、有意义碰撞的分布外推荐改善

---

## 9. Multi-Task Learning with Task-Aware Routing for Recommendation

**来源：** CIKM/ICCV（多篇综合）
**领域：** 多任务学习 × 自适应路由
**核心问题：** MMoE 隐式路由导致的负迁移

**核心方法：**
- **TIDM：** Task Information Decoupling Module (MIA + AIS)
- **GRec：** 任务-句子级路由的动态专家选择
- **逐样本自适应路由：** 动态选择模态路径和任务共享策略

**面试考点：** MMoE 负迁移的具体模式、显式 vs 隐式路由区别、逐样本 vs 任务级路由决策
