# AI 知识库导航 (ai-kb)

> Karpathy 原则：深度优先、从简到繁、每步可验证、用自己的话教。
> 本文件是全库唯一入口。200行以内，放进 LLM context window。

---

## ⭐ 整合知识精华 `synthesis/` ← 去重后的核心层

### 推荐系统 `synthesis/rec/` (9篇)
| # | 主题 | 文件 | 一句话 |
|---|------|------|--------|
| 01 | 语义ID与生成式召回 | [01_语义ID与生成式召回演进.md](synthesis/rec/01_语义ID与生成式召回演进.md) | CF→Embedding→SID→TIGER/HSTU/OneRec-V2/GLIDE |
| 02 | 序列建模演进 | [02_序列建模演进_DIN到Mamba.md](synthesis/rec/02_序列建模演进_DIN到Mamba.md) | DIN→SIM→HSTU→FuXi-Linear/SIGMA/M2Rec |
| 03 | 精排系统 | [03_精排系统_CTR到多目标生成.md](synthesis/rec/03_精排系统_CTR到多目标生成.md) | Wide&Deep→DCN-V2→HSTU，全链路架构 |
| 04 | LLM增强推荐 | [04_LLM增强推荐全景.md](synthesis/rec/04_LLM增强推荐全景.md) | LLM五阶段：特征→排序→生成→推理→原生 |
| 05 | 表示学习 | [05_表示学习_Embedding到特征交互.md](synthesis/rec/05_表示学习_Embedding到特征交互.md) | Word2Vec→EGES→DCN-V2→Wukong Scaling Law |
| 06 | 重排与多样性 | [06_重排与多样性.md](synthesis/rec/06_重排与多样性.md) | MMR/DPP + 生成式重排 + LLM推理重排 |
| 07 | 多任务与MoE | [07_多任务学习与MoE.md](synthesis/rec/07_多任务学习与MoE.md) | MMoE→PLE→SMES→DHEN，梯度冲突解法 |
| 08 | 冷启动/GNN/因果 | [08_冷启动_GNN_因果推断.md](synthesis/rec/08_冷启动_GNN_因果推断.md) | DropoutNet/PinSage/IPS/A/B测试 |
| 09 | Scaling Laws+生成召回+冷启动 | [20260420_scaling_laws_and_cold_start.md](rec-search-ads/rec-sys/synthesis/20260420_scaling_laws_and_cold_start.md) | LLaTTE/Kunlun/TokenMixer-Large/HyFormer/MixFormer/EmerG/ULIM 10篇 |
| 10 | 生成式召回+序列推荐效率 | [20260421_generative_retrieval_and_seqrec.md](rec-search-ads/rec-sys/synthesis/20260421_generative_retrieval_and_seqrec.md) | SID对齐/SynerGen/GRank/LONGER/PerSRec/DLLM2Rec/LIGER/GAMER 9篇 |

### 广告系统 `synthesis/ads/` (12篇)
| # | 主题 | 文件 | 一句话 |
|---|------|------|--------|
| 01 | CTR/CVR预估 | [01_CTR_CVR预估与校准全景.md](synthesis/ads/01_CTR_CVR预估与校准全景.md) | FM→DCN-V2→DHEN，ESMM CVR，EST Scaling |
| 02 | 排序系统演进 | [02_广告排序系统演进.md](synthesis/ads/02_广告排序系统演进.md) | 简单→多目标→约束→LTR→生成式五阶段 |
| 03 | LLM驱动广告 | [03_LLM驱动广告系统.md](synthesis/ads/03_LLM驱动广告系统.md) | OneRanker/ThinkRec/R2ec/TBGRecall |
| 04 | 多目标与混排 | [04_广告多目标与混排.md](synthesis/ads/04_广告多目标与混排.md) | eCPM公式/DPP多样性/RL混排 |
| 05 | 竞价与预算 | [05_竞价与预算优化.md](synthesis/ads/05_竞价与预算优化.md) | GSP/VCG→AutoBidding→RL→Pacing |
| 06 | 冷启动与偏差 | [06_冷启动与偏差治理.md](synthesis/ads/06_冷启动与偏差治理.md) | IDProxy/UCB探索/位置偏差因果框架 |
| 07 | 归因与Uplift | [07_效果归因与Uplift.md](synthesis/ads/07_效果归因与Uplift.md) | Shapley归因/S-T-X Learner/ZILN LTV |
| 08 | 创意优化 | [08_广告创意优化.md](synthesis/ads/08_广告创意优化.md) | DCO/多模态CTR/AIGC创意 |
| 09 | 延迟转化预估 | [09_延迟转化预估处理方案.md](synthesis/ads/09_延迟转化预估处理方案.md) | DFM/生存分析/全空间延迟反馈 |
| 10 | 模型校准方案全景 | [10_模型校准方案全景.md](synthesis/ads/10_模型校准方案全景.md) | 负采样/位置偏差/Neural Cal/分桶校准选型 |
| 11 | 基础模型+冷启动+评估 | [20260420_ads_ctr_foundation_and_evaluation.md](rec-search-ads/ads/synthesis/20260420_ads_ctr_foundation_and_evaluation.md) | CADET/LFM4Ads/PAM/Bench-CTR 10篇综合 |
| 12 | LLM拍卖+延迟反馈 | [20260421_llm_auction_and_delayed_feedback.md](rec-search-ads/ads/synthesis/20260421_llm_auction_and_delayed_feedback.md) | LLM-Auction/TESLA/READER/CFR-DF 10篇 |

### 搜索算法 `synthesis/search/` (6篇)
| # | 主题 | 文件 | 一句话 |
|---|------|------|--------|
| 01 | 检索范式 | [01_检索范式_稀疏到混合到稠密.md](synthesis/search/01_检索范式_稀疏到混合到稠密.md) | BM25/DPR/ColBERT三角 + 混合检索RRF |
| 02 | LLM+RAG | [02_LLM增强检索与RAG.md](synthesis/search/02_LLM增强检索与RAG.md) | Naive→Agentic RAG + GraphRAG |
| 03 | 推理增强检索 | [03_推理增强检索与重排.md](synthesis/search/03_推理增强检索与重排.md) | LREM/ReasonEmbed + Rank-R1/zELO |
| 04 | 生成式搜索 | [04_生成式搜索与Query理解.md](synthesis/search/04_生成式搜索与Query理解.md) | DSI/TIGER/OneSearch + Query理解 |
| 05 | Reranker与LTR | [05_Reranker演进与LTR.md](synthesis/search/05_Reranker演进与LTR.md) | Pointwise/Pairwise/Listwise + LLM重排 |
| 06 | 系统架构 | [06_搜索系统综合架构.md](synthesis/search/06_搜索系统综合架构.md) | 全链路三段式 + 工业实践 |
| 07 | Agent重排与RAG检索 | [20260421_reranking_in_search_agents.md](rec-search-ads/search/synthesis/20260421_reranking_in_search_agents.md) | RGS/ETC/KG-RAG/Reranking演进 10篇 |

### LLM基础设施 `synthesis/llm/` (9篇)
| # | 主题 | 文件 | 一句话 |
|---|------|------|--------|
| 01 | 推理优化全景 | [01_LLM推理优化全景.md](synthesis/llm/01_LLM推理优化全景.md) | KV Cache/FlashAttn/投机解码/vLLM |
| 02 | 对齐与后训练 | [02_LLM对齐与后训练全景.md](synthesis/llm/02_LLM对齐与后训练全景.md) | PPO→DPO→GRPO→DAPO 20+方法全景 |
| 02b | GRPO变体演进 | [GRPO变体演进](llm-agent/llm-infra/synthesis/GRPO变体演进_从OnPolicy到TrainingFree.md) | 原始→Off-Policy→Training-Free→Critique-GRPO |
| 03 | RAG全景 | [03_RAG系统全景与决策框架.md](synthesis/llm/03_RAG系统全景与决策框架.md) | Naive→Agentic RAG + RAG vs FT决策 |
| 04 | Agent系统 | [04_Agent系统完整指南.md](synthesis/llm/04_Agent系统完整指南.md) | 框架/记忆/多Agent/失败模式 |
| 05 | 微调与压缩 | [05_高效微调与模型压缩.md](synthesis/llm/05_高效微调与模型压缩.md) | LoRA→QLoRA→DoRA + PTQ/QAT量化 |
| 06 | 知识蒸馏 | [06_知识蒸馏技术全景.md](synthesis/llm/06_知识蒸馏技术全景.md) | 10大蒸馏模式×搜广推全景 |
| 07 | MoE架构 | [07_MoE架构与稀疏激活.md](synthesis/llm/07_MoE架构与稀疏激活.md) | 专家路由/负载均衡/MegaScale |
| 08 | 预训练演进 | [08_LLM预训练与架构演进.md](synthesis/llm/08_LLM预训练与架构演进.md) | GPT→LLaMA→DeepSeek-V3 |
| 09 | 常见误区 | [09_LLM常见认知误区.md](synthesis/llm/09_LLM常见认知误区.md) | 5大误区面试防坑 |

---

## 横切概念 `concepts/`
| 概念 | 文件 | 横跨领域 |
|------|------|---------|
| Attention in RecSys | [concepts/attention_in_recsys.md](concepts/attention_in_recsys.md) | 推荐/搜索/广告/LLM |
| Embedding全景 | [concepts/embedding_everywhere.md](concepts/embedding_everywhere.md) | 推荐/搜索/广告/LLM |
| 多目标优化 | [concepts/multi_objective_optimization.md](concepts/multi_objective_optimization.md) | 推荐/广告/搜索 |
| 序列建模演进 | [concepts/sequence_modeling_evolution.md](concepts/sequence_modeling_evolution.md) | 推荐/搜索/LLM |
| 生成式推荐 | [concepts/generative_recsys.md](concepts/generative_recsys.md) | 推荐/搜索/广告 |
| 向量量化方法 | [concepts/vector_quantization_methods.md](concepts/vector_quantization_methods.md) | 推荐/多模态/LLM |

---

## 基础知识 `fundamentals/`
| 主题 | 文件 | 难度 |
|------|------|------|
| Attention与Transformer | [fundamentals/attention_transformer.md](fundamentals/attention_transformer.md) | 入门 |
| BM25稀疏检索 | [fundamentals/bm25_sparse_retrieval.md](fundamentals/bm25_sparse_retrieval.md) | 入门 |
| Embedding与ANN | [fundamentals/embedding_ann.md](fundamentals/embedding_ann.md) | 入门 |
| DIN序列建模 | [fundamentals/din_sequence_modeling.md](fundamentals/din_sequence_modeling.md) | 入门 |
| FM/DeepFM CTR | [fundamentals/ctr_calibration.md](fundamentals/ctr_calibration.md) | 入门 |
| MMoE多任务 | [fundamentals/mmoe_multitask.md](fundamentals/mmoe_multitask.md) | 入门 |
| LoRA/PEFT | [fundamentals/lora_peft.md](fundamentals/lora_peft.md) | 入门 |
| KV Cache推理 | [fundamentals/kv_cache_inference.md](fundamentals/kv_cache_inference.md) | 入门 |
| 对比学习 | [fundamentals/contrastive_learning.md](fundamentals/contrastive_learning.md) | 入门 |
| RLHF | [fundamentals/rlhf_reward_modeling.md](fundamentals/rlhf_reward_modeling.md) | 入门 |
| 拍卖理论 | [fundamentals/auction_theory.md](fundamentals/auction_theory.md) | 入门 |
| Uplift建模 | [fundamentals/uplift_modeling.md](fundamentals/uplift_modeling.md) | 进阶 |
| Loss函数全景 | [fundamentals/loss_functions.md](fundamentals/loss_functions.md) | 入门 |
| 优化器演进 | [fundamentals/optimizers.md](fundamentals/optimizers.md) | 入门 |
| 激活函数全景 | [fundamentals/activation_functions.md](fundamentals/activation_functions.md) | 入门 |
| 正则化技术 | [fundamentals/regularization.md](fundamentals/regularization.md) | 入门 |
| 梯度问题与解法 | [fundamentals/gradient_issues.md](fundamentals/gradient_issues.md) | 入门 |
| 学习率策略 | [fundamentals/lr_scheduling.md](fundamentals/lr_scheduling.md) | 入门 |
| Transformer架构演进 | [fundamentals/transformer_evolution.md](fundamentals/transformer_evolution.md) | 进阶 |
| 搜索算法工程师完整知识储备 | [rec-search-ads/fundamentals/搜索算法工程师完整知识储备.md](rec-search-ads/fundamentals/搜索算法工程师完整知识储备.md) | 综合 |

---

## 金融量化 `quant-finance/`
| 主题 | 文件 | 难度 | 一句话 |
|------|------|------|--------|
| 因子投资 | [synthesis/factor_investing.md](quant-finance/synthesis/factor_investing.md) | 进阶 | CAPM→多因子、Barra模型 |
| 策略开发框架 | [synthesis/strategy_development.md](quant-finance/synthesis/strategy_development.md) | 进阶 | 动量/均值回归/回测框架 |
| ML在量化中 | [synthesis/ml_in_quant.md](quant-finance/synthesis/ml_in_quant.md) | 进阶 | 特征工程、树模型选股 |

---

## 速查入口
- **全库精华速查** → [CORE_TECH_DIGEST.md](CORE_TECH_DIGEST.md)
- **学习路径** → [LEARNING_PATH.md](LEARNING_PATH.md)
- **面试题库** → [rec-search-ads/interview/](rec-search-ads/interview/)
- **跨域统一视角** → [rec-search-ads/unified/](rec-search-ads/unified/)
- **原始散落synthesis** → [rec-search-ads/](rec-search-ads/) + [llm-agent/llm-infra/synthesis/](llm-agent/llm-infra/synthesis/)
- **粗排蒸馏专题** → [粗排蒸馏模型专题.md](rec-search-ads/rec-sys/02_rank/synthesis/粗排蒸馏模型专题.md) — 6种蒸馏方法 + COLD/快手/美团/Google/百度案例 + 面试Q&A
