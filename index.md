# AI 知识库导航 (ai-kb)

> Karpathy 原则：深度优先、从简到繁、每步可验证、用自己的话教。
> 本文件是全库唯一入口。200行以内，放进 LLM context window。

---

## 一、推荐系统 `rec-search-ads/rec-sys/`

### 召回 `01_recall/` ⭐入门→进阶
| 主题               | 文件                                                                                                            | 难度  | 一句话                                         |
| ---------------- | ------------------------------------------------------------------------------------------------------------- | --- | ------------------------------------------- |
| 召回范式演进           | [synthesis/推荐系统召回范式演进.md](rec-search-ads/rec-sys/01_recall/synthesis/推荐系统召回范式演进.md)                           | 入门  | 双塔→图→序列→生成式，四代召回                            |
| 工业最佳实践           | [synthesis/召回系统工业界最佳实践.md](rec-search-ads/rec-sys/01_recall/synthesis/召回系统工业界最佳实践.md)                         | 进阶  | 负样本、多路融合、ANN索引实战                            |
| Semantic ID      | [synthesis/SemanticID从论文到Spotify部署.md](rec-search-ads/rec-sys/01_recall/synthesis/SemanticID从论文到Spotify部署.md) | 高阶  | RQ-VAE离散化→自回归生成召回 + Meta Prefix Ngram 排序稳定性 |
| Semantic ID 知识图谱 | [synthesis/Semantic_ID演进知识图谱.md](rec-search-ads/rec-sys/01_recall/synthesis/Semantic_ID演进知识图谱.md)             | 高阶  | ID体系从Hash到语义的完整演进                           |
| 生成式推荐全景          | [synthesis/生成式推荐系统技术全景_2026.md](rec-search-ads/rec-sys/01_recall/synthesis/生成式推荐系统技术全景_2026.md)               | 高阶  | TIGER/HSTU/UniGRec/Mender统一视角               |

### 排序 `02_rank/` ⭐入门→高阶
| 主题 | 文件 | 难度 | 一句话 |
|------|------|------|--------|
| CTR模型深度解析 | [synthesis/CTR模型深度解析.md](rec-search-ads/rec-sys/02_rank/synthesis/CTR模型深度解析.md) | 入门 | LR→FM→DeepFM→DCN，特征交叉演进 |
| Embedding学习 | [synthesis/Embedding学习_推荐系统表示基石.md](rec-search-ads/rec-sys/02_rank/synthesis/Embedding学习_推荐系统表示基石.md) | 入门 | ID→特征交叉→图→语义，表示学习基石 |
| 排序范式演进 | [synthesis/推荐系统排序范式演进.md](rec-search-ads/rec-sys/02_rank/synthesis/推荐系统排序范式演进.md) | 进阶 | 判别式→生成式排序的范式转变 |
| 用户行为序列建模 | [synthesis/用户行为序列建模.md](rec-search-ads/rec-sys/02_rank/synthesis/用户行为序列建模.md) | 进阶 | DIN→DIEN→SIM→Transformer序列建模 |
| 精排进阶深度解析 | [synthesis/精排模型进阶深度解析.md](rec-search-ads/rec-sys/02_rank/synthesis/精排模型进阶深度解析.md) | 高阶 | 工业精排的tricks和工程细节 |
| 特征工程体系 | [synthesis/推荐系统特征工程体系.md](rec-search-ads/rec-sys/02_rank/synthesis/推荐系统特征工程体系.md) | 进阶 | 特征体系设计、实时特征、特征重要性 |
| 全链路架构概览 | [synthesis/推荐系统全链路架构概览.md](rec-search-ads/rec-sys/02_rank/synthesis/推荐系统全链路架构概览.md) | 入门 | 召回→粗排→精排→重排全链路串联 |
| Scaling Law | [synthesis/推荐系统ScalingLaw_Wukong.md](rec-search-ads/rec-sys/02_rank/synthesis/推荐系统ScalingLaw_Wukong.md) | 高阶 | Meta HSTU 1.5T参数，推荐的Scaling定律 |

### 重排 `03_rerank/` 进阶
| 主题 | 文件 | 难度 | 一句话 |
|------|------|------|--------|
| 重排与多样性 | [synthesis/推荐系统重排与多样性.md](rec-search-ads/rec-sys/03_rerank/synthesis/推荐系统重排与多样性.md) | 进阶 | MMR/DPP/滑窗，多样性与相关性平衡 |
| 生成式重排 | [synthesis/生成式重排与LLM推理增强.md](rec-search-ads/rec-sys/03_rerank/synthesis/生成式重排与LLM推理增强.md) | 高阶 | LLM直接输出排列序列 |

### 多任务 `04_multi-task/` 进阶→高阶
| 主题 | 文件 | 难度 | 一句话 |
|------|------|------|--------|
| 多任务学习与MoE | [synthesis/推荐广告系统多任务学习与MoE专家混合.md](rec-search-ads/rec-sys/04_multi-task/synthesis/推荐广告系统多任务学习与MoE专家混合.md) | 进阶 | MMoE→PLE→SMES，任务冲突与解法 |
| LLM增强推荐 | [synthesis/LLM增强推荐系统前沿综述.md](rec-search-ads/rec-sys/04_multi-task/synthesis/LLM增强推荐系统前沿综述.md) | 高阶 | LLM作为推荐组件的各种范式 |
| GNN推荐 | [synthesis/图神经网络在推荐中的应用.md](rec-search-ads/rec-sys/04_multi-task/synthesis/图神经网络在推荐中的应用.md) | 高阶 | PinSAGE→LightGCN→异构图 |
| 冷启动 | [synthesis/推荐系统冷启动.md](rec-search-ads/rec-sys/04_multi-task/synthesis/推荐系统冷启动.md) | 进阶 | 内容特征/Meta-Learning/探索策略 |
| 因果推断 | [synthesis/推荐系统因果推断.md](rec-search-ads/rec-sys/04_multi-task/synthesis/推荐系统因果推断.md) | 高阶 | IPS/DR/因果图在推荐中的应用 |
| AB测试 | [synthesis/推荐广告AB测试与在线实验.md](rec-search-ads/rec-sys/04_multi-task/synthesis/推荐广告AB测试与在线实验.md) | 进阶 | 假设检验、分流、指标体系 |

### 长序列 `long-sequence/` 高阶
| 主题 | 文件 | 难度 | 一句话 |
|------|------|------|--------|
| 长序列建模演进 | [synthesis/长序列用户行为建模技术演进.md](rec-search-ads/rec-sys/long-sequence/synthesis/长序列用户行为建模技术演进.md) | 高阶 | SIM/ETA/SDIM/SparseCTR，万级序列工业方案 |

### 顶层综合 `synthesis/` 高阶
| 主题 | 文件 | 难度 | 一句话 |
|------|------|------|--------|
| 生成式推荐统一 | [synthesis/生成式推荐范式统一_20260403.md](rec-search-ads/rec-sys/synthesis/生成式推荐范式统一_20260403.md) | 高阶 | 判别式→生成式范式转变全景 |
| LLM for RecSys | [synthesis/2026-04-09_llm_for_recsys_landscape.md](rec-search-ads/rec-sys/synthesis/2026-04-09_llm_for_recsys_landscape.md) | 高阶 | LLM×推荐的技术全景 |
| 推荐基础设施 | [synthesis/recsys-infrastructure-landscape-2025-2026.md](rec-search-ads/rec-sys/synthesis/recsys-infrastructure-landscape-2025-2026.md) | 高阶 | 推荐系统基础设施演进 |
| 序列+生成式前沿 | [synthesis/20260411_sequential_and_generative_rec.md](rec-search-ads/rec-sys/synthesis/20260411_sequential_and_generative_rec.md) | 高阶 | Linear Attn/Mamba SeqRec + SORT/OneRec-V2/GLIDE/NEO |
| 全链路实战 | [synthesis/20260411_industrial_recsys_fullstack.md](rec-search-ads/rec-sys/synthesis/20260411_industrial_recsys_fullstack.md) | 进阶 | 召回→粗排→精排→重排工业全链路 |

---

## 二、广告系统 `rec-search-ads/ads/`

### 排序/CTR `02_rank/` ⭐入门→高阶
| 主题 | 文件 | 难度 | 一句话 |
|------|------|------|--------|
| CTR/CVR预估与校准 | [synthesis/广告CTR_CVR预估与校准.md](rec-search-ads/ads/02_rank/synthesis/广告CTR_CVR预估与校准.md) | 入门 | CTR建模+校准全流程 |
| ESMM系列CVR | [synthesis/ESMM系列CVR估计演进.md](rec-search-ads/ads/02_rank/synthesis/ESMM系列CVR估计演进_从整体空间到因果推断.md) | 进阶 | 整体空间→因果推断解CVR偏差 |
| 排序演进路线图 | [synthesis/广告排序系统演进路线图.md](rec-search-ads/ads/02_rank/synthesis/广告排序系统演进路线图.md) | 进阶 | 广告排序从LR到LLM增强 |
| 多目标优化 | [synthesis/广告系统多目标优化.md](rec-search-ads/ads/02_rank/synthesis/广告系统多目标优化.md) | 进阶 | 点击/转化/LTV多目标权衡 |
| 偏差治理 | [synthesis/广告系统偏差治理三部曲.md](rec-search-ads/ads/02_rank/synthesis/广告系统偏差治理三部曲.md) | 高阶 | 曝光/选择/位置偏差系统性解法 |
| 冷启动 | [synthesis/广告系统冷启动.md](rec-search-ads/ads/02_rank/synthesis/广告系统冷启动.md) | 进阶 | 新广告/新广告主冷启动策略 |
| LTV预测 | [synthesis/LTV预测技术演进与工业实践.md](rec-search-ads/ads/02_rank/synthesis/LTV预测技术演进与工业实践.md) | 高阶 | BG/NBD→ZILN→Deep LTV |
| 效果归因 | [synthesis/广告效果归因.md](rec-search-ads/ads/02_rank/synthesis/广告效果归因.md) | 高阶 | Last-click→多触点→因果归因 |

### 出价 `04_bidding/` 进阶→高阶
| 主题 | 文件 | 难度 | 一句话 |
|------|------|------|--------|
| AutoBidding演进 | [synthesis/AutoBidding技术演进_从规则到RL.md](rec-search-ads/ads/04_bidding/synthesis/AutoBidding技术演进_从规则到RL.md) | 进阶 | 手动→oCPC→RL自动出价 |
| RTB架构全景 | [synthesis/广告系统RTB架构全景.md](rec-search-ads/ads/04_bidding/synthesis/广告系统RTB架构全景.md) | 入门 | 实时竞价系统架构 |
| 预算Pacing | [synthesis/广告预算Pacing算法全景.md](rec-search-ads/ads/04_bidding/synthesis/广告预算Pacing算法全景.md) | 进阶 | PID控制→预测型→双层优化 |

### 顶层综合 `synthesis/` 高阶
| 主题 | 文件 | 难度 | 一句话 |
|------|------|------|--------|
| 生成式广告革命 | [synthesis/工业广告系统生成式革命_20260403.md](rec-search-ads/ads/synthesis/工业广告系统生成式革命_20260403.md) | 高阶 | 竞价/全链路/基础模型/冷启动生成式重构 |
| LLM时代广告演进 | [synthesis/LLM时代广告系统技术演进.md](rec-search-ads/ads/synthesis/LLM时代广告系统技术演进.md) | 高阶 | CTR/出价/创意/拍卖全链路LLM融合 |
| LLM推理+生成召回+基础设施 | [synthesis/20260411_LLM驱动推荐推理_生成式召回_工业基础设施.md](rec-search-ads/ads/synthesis/20260411_LLM驱动推荐推理_生成式召回_工业基础设施.md) | 高阶 | R2ec/ThinkRec/OneRanker/QuaSID/DeepRec/HugeCTR |

### 重排/混排 `03_rerank/` 进阶 | 创意 `05_creative/` 高阶 | Uplift `uplift/` 高阶
| 主题 | 文件 | 难度 |
|------|------|------|
| 混排演进 | [ads/03_rerank/synthesis/广告系统混排演进路线.md](rec-search-ads/ads/03_rerank/synthesis/广告系统混排演进路线.md) | 进阶 |
| 创意优化 | [ads/05_creative/synthesis/广告创意优化.md](rec-search-ads/ads/05_creative/synthesis/广告创意优化.md) | 高阶 |
| Uplift建模 | [ads/uplift/synthesis/Uplift建模技术演进与工业实践.md](rec-search-ads/ads/uplift/synthesis/Uplift建模技术演进与工业实践.md) | 高阶 |

---

## 三、搜索 `rec-search-ads/search/`

### 召回 `01_recall/` 入门→进阶
| 主题 | 文件 | 难度 | 一句话 |
|------|------|------|--------|
| 检索三角 | [synthesis/检索三角_Dense_Sparse_LateInteraction.md](rec-search-ads/search/01_recall/synthesis/检索三角_Dense_Sparse_LateInteraction.md) | 入门 | BM25/DPR/ColBERT三种范式对比 |
| 混合检索 | [synthesis/混合检索的工业化演进.md](rec-search-ads/search/01_recall/synthesis/混合检索的工业化演进.md) | 进阶 | 稠密+稀疏融合的工业实践 |
| LLM赋能搜索 | [synthesis/LLM赋能搜索系统前沿综述.md](rec-search-ads/search/01_recall/synthesis/LLM赋能搜索系统前沿综述.md) | 高阶 | LLM改造搜索系统各环节 |

### Query理解 `04_query/` 进阶 | 重排 `03_rerank/` 进阶→高阶
| 主题 | 文件 | 难度 |
|------|------|------|
| Query理解 | [search/04_query/synthesis/搜索Query理解.md](rec-search-ads/search/04_query/synthesis/搜索Query理解.md) | 进阶 |
| Reranker演进 | [search/03_rerank/synthesis/搜索Reranker演进.md](rec-search-ads/search/03_rerank/synthesis/搜索Reranker演进.md) | 进阶 |
| LLM增强重排 | [search/03_rerank/synthesis/LLM增强信息检索与推理重排序综合总结.md](rec-search-ads/search/03_rerank/synthesis/LLM增强信息检索与推理重排序综合总结.md) | 高阶 |

### 搜索顶层综合 `synthesis/` 高阶
| 主题 | 文件 | 难度 |
|------|------|------|
| 生成式检索 | [search/synthesis/2026-04-09_generative_retrieval_evolution.md](rec-search-ads/search/synthesis/2026-04-09_generative_retrieval_evolution.md) | 高阶 |
| RAG系统演进 | [search/synthesis/2026-04-09_rag_systems_evolution.md](rec-search-ads/search/synthesis/2026-04-09_rag_systems_evolution.md) | 高阶 |
| 端到端生成式搜索 | [search/synthesis/端到端生成式搜索前沿_20260403.md](rec-search-ads/search/synthesis/端到端生成式搜索前沿_20260403.md) | 高阶 |
| 稠密检索与重排序前沿 | [search/synthesis/20260411_dense_retrieval_and_reranking_advances.md](rec-search-ads/search/synthesis/20260411_dense_retrieval_and_reranking_advances.md) | 高阶 |

---

## 四、LLM基础设施 `llm-agent/llm-infra/`

### synthesis/ 入门→高阶（31篇，选重点）
| 主题 | 文件 | 难度 | 一句话 |
|------|------|------|--------|
| KV Cache与推理优化 | [synthesis/KVCache与LLM推理优化全景.md](llm-agent/llm-infra/synthesis/KVCache与LLM推理优化全景.md) | 入门 | GQA/PagedAttention/投机解码 |
| FlashAttention | [synthesis/FlashAttention3与LLM推理基础设施.md](llm-agent/llm-infra/synthesis/FlashAttention3与LLM推理基础设施.md) | 进阶 | Tiling+Online Softmax原理 |
| LoRA与PEFT | [synthesis/LoRA与PEFT高效微调技术进展.md](llm-agent/llm-infra/synthesis/LoRA与PEFT高效微调技术进展.md) | 入门 | 低秩适配器微调全景 |
| MoE架构 | [synthesis/MoE架构设计与推理优化.md](llm-agent/llm-infra/synthesis/MoE架构设计与推理优化.md) | 进阶 | 稀疏专家路由设计 |
| RAG系统全景 | [synthesis/RAG系统全景.md](llm-agent/llm-infra/synthesis/RAG系统全景.md) | 入门 | 检索增强生成全链路 |
| LLM对齐 | [synthesis/LLM对齐方法演进.md](llm-agent/llm-infra/synthesis/LLM对齐方法演进.md) | 进阶 | SFT→RLHF→DPO→GRPO |
| GRPO算法 | [synthesis/GRPO大模型推理RL算法.md](llm-agent/llm-infra/synthesis/GRPO大模型推理RL算法.md) | 高阶 | 组内归一化替代Critic |
| 量化技术 | [synthesis/LLM_quantization_evolution_20260408.md](llm-agent/llm-infra/synthesis/LLM_quantization_evolution_20260408.md) | 进阶 | PTQ/QAT/NF4量化 |
| Agent框架 | [synthesis/agent-frameworks-landscape-2025-2026.md](llm-agent/llm-infra/synthesis/agent-frameworks-landscape-2025-2026.md) | 进阶 | 多Agent框架对比 |
| 知识蒸馏10大模式 | [synthesis/知识蒸馏技术整体总结.md](llm-agent/llm-infra/synthesis/知识蒸馏技术整体总结.md) | 进阶 | 10大蒸馏模式×搜广推全景×面试考点 |
| Serving实践 | [synthesis/LLMServing系统实践.md](llm-agent/llm-infra/synthesis/LLMServing系统实践.md) | 进阶 | vLLM/TGI部署实战 |

---

## 五、横切概念 `concepts/` ← NEW
| 概念 | 文件 | 横跨领域 |
|------|------|---------|
| Attention in RecSys | [concepts/attention_in_recsys.md](concepts/attention_in_recsys.md) | 推荐/搜索/广告/LLM |
| Embedding全景 | [concepts/embedding_everywhere.md](concepts/embedding_everywhere.md) | 推荐/搜索/广告/LLM |
| 多目标优化 | [concepts/multi_objective_optimization.md](concepts/multi_objective_optimization.md) | 推荐/广告/搜索 |
| 序列建模演进 | [concepts/sequence_modeling_evolution.md](concepts/sequence_modeling_evolution.md) | 推荐/搜索/LLM |
| 生成式推荐 | [concepts/generative_recsys.md](concepts/generative_recsys.md) | 推荐/搜索/广告 |
| 向量量化方法 | [concepts/vector_quantization_methods.md](concepts/vector_quantization_methods.md) | 推荐/多模态/LLM |

---

## 六、基础知识 `fundamentals/`
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

---

## 七、速查入口
- **全库精华速查** → [CORE_TECH_DIGEST.md](CORE_TECH_DIGEST.md)（表格+公式，面试前30分钟速览）
- **学习路径** → [LEARNING_PATH.md](LEARNING_PATH.md)（从零到精通，Karpathy风格）
- **面试题库** → [rec-search-ads/interview/](rec-search-ads/interview/)
- **跨域统一视角** → [rec-search-ads/unified/](rec-search-ads/unified/) + [rec-search-ads/cross-domain/](rec-search-ads/cross-domain/)
- **技术演进时间线** → [tech-evolution/](tech-evolution/)
