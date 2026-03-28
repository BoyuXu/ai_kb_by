# Ads 领域论文索引

> 最后更新：20260328 | 共 21 篇

## 特征交叉与 CTR 建模

| 文件 | 论文 | 关键方法 |
|-----|------|---------|
| [dcn_v2_deep_cross_network_feature_cross.md](dcn_v2_deep_cross_network_feature_cross.md) | DCN V2 | 矩阵权重 cross layer，Low-rank 近似，Parallel/Stacked 结构 |
| [masknet_feature_wise_multiplication_ctr.md](masknet_feature_wise_multiplication_ctr.md) | MaskNet | Instance-guided mask，element-wise 乘法门控，Serial/Parallel MaskBlock |
| [dhen_deep_hierarchical_ensemble_ctr.md](dhen_deep_hierarchical_ensemble_ctr.md) | DHEN | 层次 Ensemble，多模块并行（Cross+Bilinear+Attn+MLP），Meta 广告系统 |
| [wukong_ctr_scalable_deep_parallel_training.md](wukong_ctr_scalable_deep_parallel_training.md) | Wukong CTR | Pre-LN 深层架构，大规模并行训练，Embedding+Dense 混合并行 |
| [value_aware_finetuning_advertising_CTR.md](value_aware_finetuning_advertising_CTR.md) | Value-Aware Fine-tuning | 价值感知 CTR 微调 |

## 召回与向量检索

| 文件 | 论文 | 关键方法 |
|-----|------|---------|
| [unified_embedding_personalized_retrieval_etsy.md](unified_embedding_personalized_retrieval_etsy.md) | Etsy Unified Embedding | 统一 query/user/item embedding，多任务对比学习，Hard Negative Mining |
| [IDProxy_online_embedding_identity_decomposition.md](IDProxy_online_embedding_identity_decomposition.md) | IDProxy | 在线 embedding 身份分解 |

## 重排与列表优化

| 文件 | 论文 | 关键方法 |
|-----|------|---------|
| [generative_reasoning_reranker.md](generative_reasoning_reranker.md) | Generative Reasoning Re-ranker | CoT 推理驱动重排，RL 优化 NDCG |
| [llm_explainable_reranker_recommendation.md](llm_explainable_reranker_recommendation.md) | LLM Explainable Re-Ranker | LLM 重排 + 可解释输出 + 蒸馏部署 |
| [higr_generative_slate_recommendation.md](higr_generative_slate_recommendation.md) | HiGR | 层次规划生成式 Slate 推荐，类别→Item 两阶段生成 |

## 强化学习与长期优化

| 文件 | 论文 | 关键方法 |
|-----|------|---------|
| [hierarchy_enhanced_policy_optimization_ad_ranking.md](hierarchy_enhanced_policy_optimization_ad_ranking.md) | HEPO | 层次化 RL，High/Low Level Policy，长期用户留存优化 |
| [MTORL_multi_task_offline_RL_advertising.md](MTORL_multi_task_offline_RL_advertising.md) | MTORL | 多任务离线 RL 广告优化 |
| [real_time_bidding_deep_reinforcement_learning.md](real_time_bidding_deep_reinforcement_learning.md) | RTB DRL | 实时竞价深度强化学习 |
| [RTBAgent_LLM_real_time_bidding.md](RTBAgent_LLM_real_time_bidding.md) | RTBAgent | LLM 驱动的实时竞价 Agent |

## LLM 与广告系统

| 文件 | 论文 | 关键方法 |
|-----|------|---------|
| [dapo_open_source_llm_rl_system.md](dapo_open_source_llm_rl_system.md) | DAPO | 非对称 Clip-Higher，Token-level Loss，去 KL 约束 RL |
| [banner_agency_multimodal_llm_ads_design.md](banner_agency_multimodal_llm_ads_design.md) | BannerAgency | 多 Agent 广告 Banner 生成，Director/Copy/Visual/QA 角色 |
| [GPR_generative_pretraining_ads_ranking.md](GPR_generative_pretraining_ads_ranking.md) | GPR | 生成式预训练广告排序 |
| [creative_generation_LLM_ad_recommendation.md](creative_generation_LLM_ad_recommendation.md) | LLM 广告创意生成 | 广告文案自动生成 |
| [enhancing_generative_autobidding_prompt_constraints.md](enhancing_generative_autobidding_prompt_constraints.md) | Generative Autobidding | 生成式自动出价 + Prompt 约束 |

## 多场景与参数效率

| 文件 | 论文 | 关键方法 |
|-----|------|---------|
| [meta_lattice_model_space_redesign_ads.md](meta_lattice_model_space_redesign_ads.md) | Meta Lattice | 全局 Backbone + 低秩场景 Adapter，多场景参数共享 |
| [CTCVR_heterogeneous_hierarchical_decoder_weixin.md](CTCVR_heterogeneous_hierarchical_decoder_weixin.md) | CTCVR Weixin | 异构层次解码器 CTCVR 建模 |

---

## 综合总结

- [CTR预估模型工业级实践进展.md](../synthesis/CTR预估模型工业级实践进展.md) — 20260328，覆盖 12 篇论文的综合分析、公式汇总、Q&A
