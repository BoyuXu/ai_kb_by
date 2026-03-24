# 广告系统综合总结（20260323）

> 📚 参考文献
> - [Action Is All You Need Dual-Flow Generative Ran...](../../ads/papers/20260323_action_is_all_you_need_dual-flow_generative_ranking.md) — Action is All You Need: Dual-Flow Generative Ranking Netw...
> - [Wukong Towards A Scaling Law For Large-Scale Recom](../../ads/papers/20260323_wukong_towards_a_scaling_law_for_large-scale_recom.md) — Wukong: Towards a Scaling Law for Large-Scale Recommendation
> - [Esmm-Cvr](../../ads/papers/20260317_esmm-cvr.md) — ESMM：全空间多任务 CVR 预估
> - [Multiple-Hypothesis-Bias-Ctr](../../ads/papers/20260319_multiple-hypothesis-bias-ctr.md) — Addressing Multiple Hypothesis Bias in CTR Prediction for...
> - [Est-Ctr-Scaling](../../ads/papers/20260316_est-ctr-scaling.md) — EST: Efficient Scaling Laws in CTR Prediction via Unified...
> - [Din Deep Interest Network](../../ads/papers/20260322_din_deep_interest_network.md) — DIN: Deep Interest Network for Click-Through Rate Prediction
> - [Onerec-Think In-Text Reasoning For Generative R...](../../ads/papers/20260323_onerec-think_in-text_reasoning_for_generative_recom.md) — OneRec-Think: In-Text Reasoning for Generative Recommenda...


> 本批论文：10篇 | 主题：广告系统的LLM融合与多任务优化

## 本批核心主题（4个）

### 1. LLM知识融合到广告系统（LEARN / PLUM / HLLM）
三篇工作分别探索了将LLM知识迁移到广告推荐的不同路径：知识蒸馏（LEARN）、PLM适配（PLUM）、层级LLM（HLLM）。共同发现：LLM对冷启动广告和长尾广告的提升最显著（约1-2% AUC）。**关键insight：LLM知识应通过离线蒸馏而非在线推理融入广告系统，保证latency。**

### 2. 多任务学习的精细化（GNOLR / ESMM延伸）
GNOLR利用用户行为的自然序关系（曝光→点击→转化）设计有序逻辑回归约束，在CTR/CVR多任务建模上取得提升。**关键insight：在多任务模型中加入行为序关系约束（概率单调性），可以减少约85%的概率违序，提升模型可靠性。**

### 3. 推荐Scaling Law（Wukong）
Meta的工作系统验证广告推荐的Scaling Law：参数量从百万到万亿时，效果幂律增长；Embedding Table的scaling收益高于MLP。**关键insight：给定固定计算预算，优先增大Embedding Table（稀疏参数），而非加深MLP。**

### 4. 多模态广告理解（Qarm / Action Dual-Flow）
快手Qarm将视频的视觉/音频/文本多模态特征量化对齐用于广告推荐；双流生成排序网络在行为流和内容流的信息融合上取得CTR+1.2%的提升。**关键insight：短视频广告场景下，视觉内容特征和用户行为信号的对齐质量是CTR提升的关键。**

## 技术演进趋势

**广告系统正在完成LLM化转型**：从"LLM辅助特征"（知识蒸馏/embedding增强）到"LLM作为主干"（生成式CTR/推理增强排序），整个广告serving架构在重构。时间线：2023年引入LLM辅助特征 → 2024年生成式广告排序实验 → 2025年工业生产部署。

**多目标优化从人工设权到自动化**：Wukong用Scaling Law指导资源分配，Pantheon用Pareto优化自动发现多目标最优权重，OneRec-Think用推理理解用户意图。广告目标函数设计正从艺术变为科学。

**广告生成式召回可行性验证**：TBGRecall在电商广告场景验证了生成式召回的可行性，长尾类目召回提升约15%，弥补了向量召回的语义表达不足。

## 面试高频考点（8条）

1. **Q: 广告系统的多任务学习（MTL）主流方案？** A: MMoE（多门混合专家）处理任务相关性；ESMM（完整空间）解决CVR样本选择偏差；PLE（渐进分层提取）进一步隔离任务间干扰

2. **Q: CVR预估为什么比CTR困难？** A: 正样本稀疏（转化率通常<5%）；样本选择偏差（只有点击才有CVR标签，全量空间无观测）；ESMM通过CTCVR=CTR×CVR在全曝光空间建模解决

3. **Q: LoRA在广告模型适配中的工程考量？** A: rank=16-64适合大多数广告任务；可以为不同广告主/垂类维护独立LoRA adapter；推理时adapter合并到主模型，无额外latency

4. **Q: 短视频广告与图文广告的技术差异？** A: 短视频需要视频特征提取（GPU密集，离线批处理）；时效性更强（爆款视频1-3天消亡）；完播率是核心指标而非单纯CTR

5. **Q: 广告Scaling Law的工程意义？** A: 为资源分配提供理论依据：扩Embedding比加深MLP ROI更高；万亿参数Embedding需要分布式Parameter Server（PS）架构

6. **Q: 广告冷启动的主要技术方案？** A: LLM语义特征（LEARN/PLUM）提供无行为先验；类目迁移（相似类目的点击历史）；内容特征（图片/文字的CLIP/BERT编码）

7. **Q: 广告排序中的位置偏差（position bias）如何处理？** A: Propensity Weighting（加权纠偏）、双塔去偏（用位置单独建一个塔）、在线日志随机化实验校准

8. **Q: 生成式CTR预估相比判别式的主要优势？** A: 可以建模CTR以外的生成性分布特征；冷启动广告无需历史数据（通过文本语义生成预测）；但推理成本高5-10x，需要蒸馏

## 与已有知识的连接

- **与推荐系统的关联**：ads和rec-sys本批论文高度重叠（OneRec-Think同时出现），说明推荐和广告系统技术栈正在合并
- **与LLM推理能力**：RLVR/DeepSeek-R1等推理能力论文将影响广告系统的意图理解（用推理链理解复杂购买意图）
- **与搜索广告**：生成式CTR预估同时适用于搜索广告（query-ad匹配）和展示广告（user-ad匹配），技术共通
