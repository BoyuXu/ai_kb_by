# 推荐系统基础设施全景分析 (2025-2026)

## 技术演进轨迹

推荐系统基础设施正经历三个维度的深刻变革：

**计算架构**：从单机训练 → 分布式训练 → 端到端GPU加速。单机时代受限于内存和计算力，难以处理十亿级别的用户-物品交互。分布式训练通过数据并行和模型并行解决规模问题，但通信开销成为瓶颈。当前趋势是GPU原生框架（NVIDIA Merlin、TorchRec）直接在GPU上完成embedding分片和通信，消除CPU-GPU数据搬运。

**Serving模式**：从离线评分 → 实时排序 → 流式决策。离线评分以小时为粒度生成候选集，响应延迟难以优化。实时排序框架（Vespa、X算法）支持毫秒级重排序，融合用户实时信号。流式决策则通过在线学习和contextual bandits动态调整策略，适应快速变化的用户兴趣。

**模型范式**：从浅层特征交叉 → 深度学习 → Transformer序列建模。CTR模型标准化（FuxiCTR基准库）确立了评测规范，但浅层模型计算高效仍有应用空间。Transformers4Rec引入自注意力机制捕捉序列依赖，相比RNN有更强的长期记忆能力和更好的并行化特性。

## 核心框架生态对比

| 框架 | 核心定位 | 特色优势 | 适用场景 |
|------|---------|---------|---------|
| **NVIDIA Merlin** | 端到端GPU加速平台 | 数据预处理→特征工程→训练→推理完整管道GPU化 | 大规模实时推荐系统 |
| **TensorFlow Recommenders (TFRS)** | 灵活的模型构建框架 | Two-tower架构、近似最近邻检索、离线-在线一致性 | 学术研究、中等规模系统 |
| **TorchRec** | 分布式embedding训练 | 细粒度embedding分片策略、异构设备支持 | 超大规模embedding表 |
| **RecBole 2.0** | 统一基准库（130+模型） | 模块化设计、标准化评估、快速迭代 | 算法研究、模型选型 |
| **FuxiCTR** | CTR预测标准化基准 | 严格的数据处理规范、可复现的结果 | CTR模型对标、特征工程优化 |
| **X算法** | 业界级联系统 | 候选生成→排序→重排序完整pipeline | 超大规模社交平台 |

## Embedding分布式切分策略

**TorchRec的三层切分模型**：
- Table-wise分片：不同embedding表分配到不同GPU，优点是通信最少但负载均衡困难
- Row-wise分片：按用户/物品ID范围分片，适合embedding向量维度高的场景
- Column-wise分片：按embedding维度分片，需要全局通信但更细粒度的负载均衡

**HugeCTR的跨GPU/跨节点扩展**：借由All-to-All通信模式和梯度累积，实现单个embedding表跨多个节点的训练，突破单GPU内存限制。

**TFRS的近似最近邻策略**：Two-tower架构预先离线计算item embedding，候选阶段通过ANN（如ScaNN）快速检索相似item，排序阶段才进行细粒度的用户-item交互计算。

## 工业级系统架构范式

**X算法的三阶段管道**：
- 候选生成（Embedding-based）：数百万物品通过embedding相似度初步筛选至数千候选
- 排序（深度学习模型）：融合用户-物品-上下文多维特征，精细化评分
- 重排序（业务规则+强化学习）：多目标优化（点击率、停留时间、分享）、多样性约束、实时干预

**Vespa的搜索-推荐统一平台**：将搜索和推荐视为同一问题——在向量空间中精准检索。支持BM25关键词搜索、向量相似度搜索、混合排序，一个系统承载多种检索需求。

**Merlin的端到端加速**：GPU上直接完成ETL→特征工程→模型训练→批量推理，避免了Python序列化、CPU-GPU数据搬运等开销，比传统CPU+GPU流程快3-5倍。

## DeepRec 深度解析 (2026-04-11 补充)

> 来源: GitHub DeepRec-AI/DeepRec | LF AI & Data Foundation | 阿里PAI团队

**核心定位**: 基于 TensorFlow 的高性能推荐专用框架，2016年起支撑淘宝搜索/推荐/广告核心业务。

**关键技术优势**:
- **超大规模训练**: 支持万亿样本、十万亿参数的推荐模型
- **Embedding Variable (EV)**: 动态维度、多级存储（GPU→CPU→SSD），比 TF 原生 Embedding 内存效率高 3-5x
- **在线深度学习**: 分钟级增量模型更新，支持 10TB+ 模型的 serving
- **Session Group**: 多 session 并行执行优化 GPU 利用率

**vs HugeCTR 选型**:
- DeepRec: TF 生态 + 在线学习 + 全链路优化 → 阿里系/TF 用户首选
- HugeCTR: CUDA 原生 + 极致训练速度 + Merlin 全栈 → GPU 密集型离线训练首选

## Kamae: Training-Serving 一致性 (2026-04-11 补充)

> 来源: arxiv 2507.06021 | RecSys 2025 (Expedia)

**解决问题**: 特征预处理在 Spark（训练）和 Keras（推理）之间不一致导致 Training-Serving Skew。

**技术方案**: 在 Keras 内统一实现所有预处理逻辑，pipeline 导出为 Keras 模型 bundle，训练和推理共用。

**工业价值**: Expedia Learning-to-Rank 场景，消除序列特征（酒店列表、房间列表）处理不一致问题。

## 面试核心考点

**分布式训练**：对比数据并行与模型并行的通信成本，embedding分片的三种策略选择依据，异步梯度下降的收敛性权衡。

**Embedding优化**：特征交叉的稀疏性如何指导embedding维度设计，大规模embedding表的显存管理方案，embedding初始化与正则化对收敛的影响。DeepRec 的 EV 如何实现动态维度和多级存储。

**搜索-推荐融合**：向量检索的精度-召回权衡，候选生成与排序的协同优化（exposure bias问题），实时特征如何在推理中集成。

**CTR标准化评测**：FuxiCTR的数据处理规范为什么重要，不同数据集上的模型可比性如何保证，离线指标与在线A/B测试的gap原因。

**Training-Serving Skew**: Kamae 的 Pipeline-as-Model 设计如何消除特征处理不一致？为什么这是工业推荐系统的常见陷阱？

**关联综述**: [[20260411_LLM驱动推荐推理_生成式召回_工业基础设施.md]]

---

## 相关概念

- [[concepts/generative_recsys|生成式推荐统一视角]]
- [[concepts/embedding_everywhere|Embedding 技术全景]]
- [[concepts/attention_in_recsys|Attention 在搜广推中的演进]]
- [[concepts/multi_objective_optimization|多目标优化]]
- [[concepts/sequence_modeling_evolution|序列建模演进]]
