# 广告系统知识库导航 🎯

## 📊 领域概览

| 分类 | 文档数 | 描述 |
|------|--------|------|
| **Papers** (学术论文笔记) | 82篇 | CTR预估、竞拍机制、自动出价 |
| **Practices** (工业实践案例) | 2篇 | 大厂广告系统设计 |
| **Synthesis** (提炼总结) | 4篇 | 广告系统演进、框架总结 |
| **总计** | 88篇 | - |

---

## 🚀 快速导航

### 📚 按学习阶段查找

1. **新手入门** → 广告系统基础概览（待补充）
2. **CTR 预估** → [CTR 预估综述](./papers/20260319_ctr-prediction-comprehensive-survey.md)
3. **竞拍机制** → [GSP & VCG 竞拍](./papers/20260316_gsp-vcg-auction.md)
4. **自动出价** → [自动出价基础](./papers/20260318_bidding_basics.md)
5. **LLM 融合** → [LLM 集成框架](./synthesis/llm_integration_framework.md)

### 🎯 按研究方向查找

#### 💰 竞拍与拍卖机制
- [GSP & VCG 竞拍](./papers/20260316_gsp-vcg-auction.md)
- [RTB 拍卖机制](./papers/20260318_rtb_auction_mechanism.md)
- [自动出价基础](./papers/20260318_bidding_basics.md)
- [无遗憾自动出价](./papers/20260319_no-regret-autobidding-first-price.md)

#### 💻 CTR 预估
- [CTR 预估综述](./papers/20260319_ctr-prediction-comprehensive-survey.md)
- [ESMM - CVR](./papers/20260317_esmm-cvr.md)
- [EST - CTR 缩放](./papers/20260316_est-ctr-scaling.md)
- [CADET - CTR 上下文](./papers/20260313_cadet_context_ctr.md)
- [多假设与偏差 CTR](./papers/20260319_multiple-hypothesis-bias-ctr.md)
- [RQ-GMM - 多模态 CTR](./papers/20260313_rq_gmm_multimodal_ctr.md)
- [GEnCI - CTR 队列](./papers/20260316_genci-cohort-ctr.md)
- [生成式 CTR 范式](./papers/20260313_generative_ctr_paradigm.md)

#### 🖼️ 创意优化
- [CTR 与图像生成](./papers/20260313_ctr_image_gen.md)

#### 💵 成本与预算
- [适应性预算规划](./papers/20260318_AdaptableBudgetPlanner.md)
- [多目标在线广告](./papers/20260322_multi_objective_online_advertising.md)
- [多目标优化在线广告](./papers/20260321_multi-objective-optimization-for-online-advertising-balancing-revenue-and-user-experience.md)

#### 🎯 出价与优化
- [OCPC & OCPA 深度学习](./papers/20260318_ocpc_ocpa_deep.md)
- [扩散模型自动出价](./papers/20260318_DiffusionAutoBidding.md)

#### 🌐 特征与系统
- [展示广告特征工程](./papers/20260316_display-ads-feature-engineering.md)

#### ❄️ 冷启动问题
- [IDProxy - 冷启动](./papers/20260313_idproxy_coldstart.md)

#### 📊 推理与决策
- [P3 - LTR 到 RL](./papers/p3_ltr_to_rl_ranking.md)
- [P5 - 工业实践](./papers/p5_industrial_practice.md)
- [抖音 RL 混合](./papers/douyin_rl_mixing.md)
- [Wukong - 大规模推荐缩放律](./papers/20260323_wukong_towards_a_scaling_law_for_large-scale_recom.md)

---

## 📚 完整文档列表

### 📄 Papers (82篇 学术论文笔记)

#### 竞拍与机制设计
- [GSP & VCG 竞拍](./papers/20260316_gsp-vcg-auction.md)
- [RTB 拍卖机制](./papers/20260318_rtb_auction_mechanism.md)

#### CTR 预估
- [CTR 预估综述](./papers/20260319_ctr-prediction-comprehensive-survey.md)
- [EST - CTR 缩放](./papers/20260316_est-ctr-scaling.md)
- [CADET - 上下文 CTR](./papers/20260313_cadet_context_ctr.md)
- [多假设偏差 CTR](./papers/20260319_multiple-hypothesis-bias-ctr.md)
- [RQ-GMM - 多模态 CTR](./papers/20260313_rq_gmm_multimodal_ctr.md)
- [GEnCI - CTR 队列](./papers/20260316_genci-cohort-ctr.md)
- [生成式 CTR 范式](./papers/20260313_generative_ctr_paradigm.md)
- [ESMM - CVR](./papers/20260317_esmm-cvr.md)

#### 创意与图像
- [CTR 与图像生成](./papers/20260313_ctr_image_gen.md)

#### 出价与优化
- [自动出价基础](./papers/20260318_bidding_basics.md)
- [无遗憾自动出价](./papers/20260319_no-regret-autobidding-first-price.md)
- [OCPC & OCPA 深度](./papers/20260318_ocpc_ocpa_deep.md)
- [扩散模型自动出价](./papers/20260318_DiffusionAutoBidding.md)
- [适应性预算规划](./papers/20260318_AdaptableBudgetPlanner.md)

#### 多目标与预算
- [多目标在线广告](./papers/20260322_multi_objective_online_advertising.md)
- [多目标优化在线广告](./papers/20260321_multi-objective-optimization-for-online-advertising-balancing-revenue-and-user-experience.md)
- [自适应竞拍非平稳预算](./papers/20260320_adaptive-bidding-budget-nonstationary.md)

#### 冷启动与特征
- [IDProxy - 冷启动](./papers/20260313_idproxy_coldstart.md)
- [展示广告特征工程](./papers/20260316_display-ads-feature-engineering.md)

#### 工业实践与排序
- [P3 - LTR 到 RL](./papers/p3_ltr_to_rl_ranking.md)
- [P5 - 工业实践](./papers/p5_industrial_practice.md)
- [抖音 RL 混合](./papers/douyin_rl_mixing.md)
- [Wukong 缩放律](./papers/20260323_wukong_towards_a_scaling_law_for_large-scale_recom.md)

#### 升降与转化
- [无偏 CTR 升降优惠券](./papers/20260313_debiased_ctr_uplift_coupon.md)

#### LLM 项目创意
- [LLM 项目创意](./papers/llm_project_ideas.md)

#### 知识库文档
- [广告系统知识库](./papers/广告系统知识库.md)

### 🏢 Practices (2篇 工业实践案例)

当前包含：
- [P3 - LTR 到 RL 工业实践](./papers/p3_ltr_to_rl_ranking.md)
- [P5 - 广告系统工业实践](./papers/p5_industrial_practice.md)

### 📖 Synthesis (4篇 提炼总结)

- [LLM 集成框架](./synthesis/llm_integration_framework.md)
- [自动出价演进](./synthesis/auto_bidding_evolution.md)
- [创意优化 LLM](./synthesis/creative_optimization_with_llm.md)
- [广告系统概览](./synthesis/ads_overview.md) (待补充)

---

## 💡 使用指南

### 对于不同角色

**🎓 学生/初学者**
1. 了解竞拍机制：[GSP & VCG](./papers/20260316_gsp-vcg-auction.md)
2. 学习 CTR 预估：[CTR 综述](./papers/20260319_ctr-prediction-comprehensive-survey.md)
3. 了解自动出价：[基础](./papers/20260318_bidding_basics.md)

**🔬 研究者**
1. CTR 建模创新
2. 多目标优化
3. 在线学习与强化学习

**👨‍💼 工程师**
1. 查看 [LLM 集成框架](./synthesis/llm_integration_framework.md)
2. 了解工业实践
3. 实现系统组件

---

## 🔗 快速链接

| 主题 | 关键论文 |
|------|---------|
| **竞拍** | [GSP-VCG](./papers/20260316_gsp-vcg-auction.md) \| [RTB](./papers/20260318_rtb_auction_mechanism.md) |
| **CTR** | [综述](./papers/20260319_ctr-prediction-comprehensive-survey.md) \| [CADET](./papers/20260313_cadet_context_ctr.md) |
| **出价** | [基础](./papers/20260318_bidding_basics.md) \| [扩散模型](./papers/20260318_DiffusionAutoBidding.md) |
| **工业** | [P3-LTR](./papers/p3_ltr_to_rl_ranking.md) \| [P5](./papers/p5_industrial_practice.md) |

---

## 📝 最后更新

- **最后更新**: 2026-03-24
- **总文档数**: 88 篇
- **近期更新**: 补充 LLM 时代的广告应用
