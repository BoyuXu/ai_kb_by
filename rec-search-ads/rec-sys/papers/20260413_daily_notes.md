# 推荐系统论文/资源笔记 — 2026-04-13

## 1. Milvus: 高性能云原生向量数据库

**来源：** https://github.com/milvus-io/milvus
**领域：** 向量检索基础设施 / 推荐召回
**核心定位：** 为大规模向量近似最近邻搜索（ANN）构建的高性能云原生向量数据库

**核心架构：**
- 计算存储分离的分布式架构，四层独立（接入层、协调层、计算层、存储层）
- 无状态微服务部署在 K8s 上，支持水平扩展和故障快速恢复
- 读写分离：读密集场景增加 query node，写密集场景增加 data node

**关键特性：**
- 支持 HNSW、IVF、FLAT、SCANN、DiskANN 等主流向量索引
- 混合检索：原生支持 BM25 全文检索 + 稀疏向量（SPLADE、BGE-M3）+ 稠密向量
- GPU/CPU 硬件加速，Go + C++ 实现
- RBAC + TLS 安全机制

**2025-2026 创新（Milvus 2.6）：**
- RaBitQ 1-bit 量化：主索引压缩至 1/32，配合 SQ8 精炼保持 95% recall，仅用 1/4 内存
- 分层存储：按访问模式自动分类数据，降低 50% 存储成本
- Woodpecker：零磁盘 WAL 系统，日志直接持久化到对象存储

**面试考点：** ANN 索引选型（HNSW vs IVF vs DiskANN）、向量量化技术（PQ/SQ/RaBitQ）、分布式向量检索架构设计

---

## 2. RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

**来源：** arXiv:2005.11401 (Lewis et al., 2020)
**领域：** 检索增强生成 / 推荐系统知识补充
**核心问题：** 大型预训练语言模型虽存储了大量知识，但在知识密集型任务中精确获取和操控知识的能力有限

**核心贡献：**
- 提出 RAG 范式：结合参数化记忆（seq2seq transformer）与非参数化记忆（Wikipedia 向量索引）
- 使用预训练的神经检索器（DPR）访问外部知识
- 两种变体：RAG-Sequence（整个序列用同一文档）和 RAG-Token（每个 token 可用不同文档）

**核心公式：**
```
P_RAG-Seq(y|x) = Σ_z P(z|x) × Π_i P(y_i|x, z, y_{1:i-1})
P_RAG-Token(y|x) = Π_i Σ_z P(z|x) × P(y_i|x, z, y_{1:i-1})
```

**实验结果：** 在三个开放域 QA 任务上达 SOTA，生成更具体、多样、事实性的文本

**面试考点：** RAG 的检索-生成联合训练、RAG-Seq vs RAG-Token 的区别、如何在推荐系统中应用 RAG

---

## 3. EasyRec: 阿里巴巴工业级推荐框架

**来源：** arXiv:2209.12766 / https://github.com/alibaba/EasyRec
**领域：** 推荐系统工程框架
**核心定位：** 易用、可扩展、高效的工业级推荐系统框架

**关键特性：**
- 模块化插件式设计：降低自定义模型开发成本
- 支持候选生成（matching）、评分排序（ranking）、多任务学习
- 自动超参优化和特征选择
- 在线学习支持，快速适应数据分布变化
- 多平台：MaxCompute、DataScience(K8s)、DLC、Alink、本地
- 基于 PyTorch 重构，GPU 加速 + 混合并行

**工业验证：** 在 20+ 业务场景验证（商品推荐、信息流广告、社交、直播、视频等）

**面试考点：** 推荐系统工程化最佳实践、在线学习 vs 离线学习、多平台部署方案

---

## 4. DeepCTR: 深度学习 CTR 预测模型库

**来源：** https://github.com/shenweichen/DeepCTR
**领域：** CTR 预测 / 排序模型
**核心定位：** 易用、模块化、可扩展的深度学习 CTR 预测模型工具包

**核心架构（四层）：**
1. **输入模块：** 处理稀疏/稠密特征，自动处理缺失值
2. **嵌入模块：** 高维稀疏特征映射到低维稠密空间
3. **特征提取模块：**
   - 低阶提取器（FM 系列）：向量间乘积学习特征交互
   - 高阶提取器（MLP、Cross Net）：复杂神经网络学习特征组合
4. **预测输出模块**

**支持模型：** PNN、WDL、DeepFM、MLR、DCN、AFM、NFM、DIN、DIEN、xDeepFM、AutoInt 等 10+ 模型
**兼容性：** tf.keras 接口 + tf.estimator 接口，兼容 TF 1.x/2.x

**面试考点：** 各 CTR 模型的核心差异（特征交互方式）、DeepFM vs DCN vs DIN 对比

---

## 5. ChatRec: 基于 LLM 的对话式推荐系统

**来源：** arXiv:2303.14524 (Gao et al., 2023)
**领域：** LLM + 推荐系统 / 对话式推荐
**核心问题：** 传统推荐系统交互性差、可解释性低

**核心贡献：**
- 将用户画像和历史交互转化为 prompt，利用 LLM 的 in-context learning 能力
- 实现交互式、可解释的推荐过程
- 支持跨域推荐：用户偏好可在不同商品类别间迁移
- 缓解冷启动问题：通过 prompt 注入信息

**面试考点：** LLM 作为推荐系统的优劣势、prompt engineering 在推荐中的应用、冷启动解决方案

---

## 6. RecBole: 统一推荐算法库

**来源：** arXiv:2011.01731 / https://github.com/RUCAIBox/RecBole
**领域：** 推荐系统研究框架
**核心定位：** 统一、全面、高效的推荐算法研究库

**关键数据：**
- 实现 94 个推荐算法，覆盖 4 大类别
- 支持 44 个基准推荐数据集
- 统一灵活的数据文件格式

**四大类别：**
1. 通用推荐（General Recommendation）
2. 序列推荐（Sequential Recommendation）
3. 上下文感知推荐（Context-aware Recommendation）
4. 知识图谱推荐（Knowledge-based Recommendation）

**扩展生态：** RecBole 2.0、RecBole-MetaRec、RecBole-DA、RecBole-Debias、RecBole-FairRec、RecBole-CDR、RecBole-TRM、RecBole-GNN、RecBole-PJF（共 11 个子项目）

**面试考点：** 推荐算法分类体系、各类推荐算法代表模型、如何选择合适的推荐算法

---

## 7. Microsoft Recommenders: 推荐系统最佳实践

**来源：** https://github.com/recommenders-team/recommenders
**领域：** 推荐系统工程最佳实践
**核心定位：** 以 Jupyter Notebook 形式提供推荐系统的端到端最佳实践

**五大任务：**
1. Prepare Data（数据准备）
2. Model（建模：经典算法 + 深度学习，如 ALS、xDeepFM）
3. Evaluate（评估）
4. Model Select and Optimize（模型选择与优化）
5. Operationalize（Azure 上的生产化部署）

**价值：** GitHub 最多星标的开源推荐系统项目（10k+ stars），企业实施效率提升 10 倍以上

---

## 8. Torch-RecHub: PyTorch 推荐算法框架

**来源：** https://github.com/datawhalechina/torch-rechub
**领域：** 推荐系统工程框架
**核心定位：** 轻量级 PyTorch 推荐模型框架，10 行代码构建生产级推荐系统

**关键特性：**
- 30+ 主流推荐算法（匹配、排序、多任务、生成式推荐）
- 统一数据加载、训练、评估工作流
- 一键 ONNX 导出，支持生产部署
- PySpark 数据处理集成

---

## 9. FunRec: 推荐系统入门到实战教程

**来源：** Datawhale 社区
**领域：** 推荐系统学习资源
**核心定位：** 帮助具有机器学习基础的学习者快速进入推荐系统领域

**学习路径：** 系统概述 → 算法基础（召回+排序经典算法）→ 实战竞赛 → 项目实现 → 面试准备

---

## 10. LLM4Rec 系列资源综述

**来源：** 多个 GitHub 仓库（Awesome-LLM-for-RecSys, LLM4Rec-Awesome-Papers, Awesome-LLM4Rec）
**领域：** LLM + 推荐系统

**行业趋势（RecSys 2025）：**
- LLM 集成推荐系统成为主流方向
- 周期性微调 + RAG 混合方法在 YouTube 大规模测试中取得可衡量性能提升
- 主要研究范式：预训练、微调、提示（prompting）

**面试考点：** LLM 在推荐各环节（召回/排序/解释/对话）的应用方式

---

## 11. Gorse: Go 语言开源推荐引擎

**来源：** https://github.com/gorse-io/gorse
**领域：** 推荐系统引擎
**核心特性：**
- 多源推荐：热门、最新、UserCF、ItemCF、协同过滤 + CTR 排序
- AutoML 自动模型选择
- 分布式架构：单节点训练 + 分布式预测
- RESTful API 接口
- 支持多模态内容嵌入

---

## 12. Awesome-Recommender-System 资源合集

**来源：** 多个 GitHub 仓库
**核心价值：** 汇集推荐系统领域的论文、框架、工具、会议、研究者信息，持续更新
- loserChen/Awesome-Recommender-System: 每周更新，追踪顶会顶刊论文
- jihoo-kim/awesome-RecSys: 书籍、会议、论文、视频等多维资源

---

## 13. Deep-Learning-Papers-for-Search-Recommendation-Advertisements

**来源：** https://github.com/guyulongcs/Awesome-Deep-Learning-Papers-for-Search-Recommendation-Advertising
**领域：** 搜索/推荐/广告深度学习论文集
**覆盖：** Embedding、Matching、Pre-Ranking、Ranking（CTR/CVR）、Post-Ranking、Relevance、LLM、RL
**收录公司：** Google、Alibaba、Amazon、Airbnb、Facebook、Microsoft 等

---

## 14. PapersWithCode 推荐系统排行榜

**来源：** paperswithcode.com/task/recommendation-systems
**状态：** PapersWithCode 于 2025 年 7 月被 Meta 关闭，重定向至 Hugging Face
**历史数据：** 1862 篇有代码论文、54 个基准、57 个数据集
**替代方案：** BARS 基准（8000+ 实验，70+ 模型，6 个数据集）

---

## 15. AlgoNotes: 算法工程师面试笔记

**来源：** GitHub 资源合集
**核心价值：** 覆盖算法、数据结构、系统设计的面试准备资料
**适用场景：** 推荐/搜索/广告方向的算法工程师求职准备
