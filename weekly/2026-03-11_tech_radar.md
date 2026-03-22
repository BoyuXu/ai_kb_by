# 技术雷达 2026-03-11 | 第1期

> MelonEggLearn 每周技术雷达 - 搜广推+AI前沿动态

---

## 🔥 本周最值得关注（Top 3）

### 1. 阿里巴巴 RecGPT-V2 推荐系统突破
- **核心价值**: 成本降低60%，推荐质量显著提升
- **关键创新**: 动态提示生成 + 约束强化学习 + 多维度智能评委机制
- **工程意义**: 在淘†"猜你喜欢"模块1%流量AB测试中验证有效，为工业界推荐系统提供新思路

### 2. ChunkKV: 长上下文LLM推理新突破
- **核心价值**: 语义感知的KV Cache压缩，解决长文本推理内存瓶颈
- **关键创新**: 从token级压缩转向语义chunk级压缩，层间索引复用
- **工程意义**: 在10万+token场景下，相比H2O/SnapKV等方法提升约10%准确率

### 3. 字节跳动 Primus 大规模推荐训练系统
- **核心价值**:  unified training system for DLRM，日均处理160TB数据
- **关键创新**: 统一资源调度 + 统一数据编排 + 混合训练范式（记忆塔+适应塔）
- **工程意义**: 被USENIX ATC 2025收录，支撑900万核CPU+数万GPU的异构训练

---

## 📄 推荐系统论文

### 1. Towards An Efficient LLM Training Paradigm for CTR Prediction
- **机构**: Texas A&M University, Meta
- **链接**: arXiv:2503.01001
- **核心创新**: 
  - 提出 Dynamic Target Isolation (DTI) 训练范式，将LLM用于CTR预测的训练时间从70.5小时降至5.31小时（平均减少92%）
  - 解决传统"滑动窗口"范式O(mn²)复杂度问题
  - 通过并行化k个目标交互的训练，同时解决hidden-state leakage和positional bias overfitting
- **工程价值**: 使LLM-based CTR预测在工业规模数据上可行

### 2. HLLM: 分层大语言模型推荐系统 (字节跳动)
- **机构**: 字节跳动
- **核心创新**:
  - 两层架构: Item LLM提取内容特征 → User LLM预测用户兴趣
  - 冷启动场景R@5达6.129，远超SASRec的5.142
  - 仅需1/6-1/4数据量即可达到HSTU同等性能
- **工程价值**: 线上AB测试提升0.705%，为LLM在推荐系统的规模化应用提供可行路径

### 3. Vision-Language Model for E-commerce Recommendations (Mercari)
- **机构**: Mercari, Inc.
- **会议**: ACM RecSys 2025 (Spotlight)
- **链接**: arXiv:2510.13359
- **核心创新**:
  - 基于SigLIP微调视觉语言模型用于商品推荐
  - 使用100万商品图像-标题对进行训练
  - 离线nDCG@5提升9.1%，线上CTR提升50%，转化率提升14%
- **工程价值**: 验证了VLM在电商推荐中的实际落地价值

### 4. Cross-Domain Recommendation Survey 2025
- **机构**: 中科大等
- **链接**: arXiv:2503.18xxx
- **核心创新**: 系统性综述跨域推荐技术，提出新的分类体系
- **工程价值**: 为跨域推荐技术选型提供全面参考

### 5. RAU: Regularized Alignment and Uniformity for Representation Learning in Recommendation
- **核心创新**: 针对推荐中表征学习的对齐性和均匀性进行正则化优化
- **工程价值**: 提升推荐表征质量，改善冷启动和长尾部item推荐

---

## 🔍 搜索算法

### 1. Dense Passage Retrieval in Conversational Search
- **链接**: arXiv:2503.21xxx
- **核心创新**: 将稠密检索应用于对话式搜索场景，使用双编码器构建上下文嵌入
- **工程价值**: 改善多轮对话中的检索质量，支持高效索引和聚类

### 2. SUNAR: Semantic Uncertainty based Neighborhood Aware Retrieval
- **核心创新**: 基于语义不确定性的邻域感知检索，解决复杂QA中的bounded-recall问题
- **工程价值**: 提升多面查询(multi-faceted queries)的检索召回率

### 3. Neural Retrieval + LLM 融合趋势
- **关键洞察**: 基于Eugene Yan等行业专家分析，2025年搜索推荐系统呈现以下趋势:
  - **Semantic IDs**: YouTube使用RQ-VAE将内容嵌入压缩为离散语义ID，解决冷启动问题
  - **M3CSR** (快手): 多模态内容嵌入通过K-means聚类为可训练类别ID
  - **FLIP** (华为): 表格数据与语言数据联合学习，实现跨模态对齐
  - **beeFormer**: 纯文本Transformer训练user-item交互数据，桥接语义相似度和交互相似度

---

## 🤖 大模型工程

### 1. KV Cache优化最新进展

#### ChunkKV (2025)
- **核心创新**: 语义感知的KV Cache压缩，将连续token分组为语义chunk进行评分
- **效果**: 在LongBench和Needle-In-A-Haystack基准上，相同压缩比下准确率提升约10%
- **特点**: 层间索引复用，训练无关，模型无关

#### NVFP4 KV Cache (NVIDIA)
- **核心创新**: 4-bit浮点数量化KV Cache，减少内存占用和计算成本
- **工程价值**: 支持长上下文和大batch size推理

#### llm-d: KV Cache感知路由
- **核心创新**: 基于Gateway API Inference Extension的智能路由，实现87.4% cache命中率
- **效果**: TTFT降低88%，GPU计算时间减少70%

### 2. 推理框架对比: vLLM vs SGLang

| 维度 | SGLang | vLLM |
|------|--------|------|
| 核心技术 | RadixAttention | PagedAttention |
| 擅长场景 | 多轮对话、结构化输出、复杂工作流 | 高吞吐单轮推理、批量任务 |
| 性能特点 | 多轮场景快10-20% | 简单任务快1.1倍，内存效率4倍 |
| 吞吐量 | 5000+ tokens/s | 5000+ tokens/s |
| 适用模型 | DeepSeek-R1, Qwen3, LLaMA | GPT-4, Mixtral, Llama 70B+ |

**关键趋势**: 
- Hugging Face TGI于2025年12月进入维护模式，官方推荐迁移至vLLM或SGLang
- SGLang 0.4.3支持DeepSeek-R1/V3多token预测，长文本生成效率质的飞跃
- vLLM支持torch.compile，性能持续优化

### 3. 多模态RAG进展
- **Self-RAG**: 学习自反思的检索增强生成
- **ClassRAG**: 基于分类器的检索增强文本分类
- **FairRAG**: 公平性导向的检索增强生成

---

## 💻 开源项目

### 1. ktransformers (15.5k ⭐)
- **定位**: LLM推理优化框架
- **亮点**: 灵活的优化策略，模块化hooks，支持kernels和caching实验
- **适用**: 需要低延迟高吞吐LLM推理的场景

### 2. LocalAI (37.9k ⭐)
- **定位**: 本地优先的OpenAI兼容服务器
- **亮点**: 无需GPU，支持多种模型，隐私优先
- **适用**: 私有化LLM部署，边缘计算

### 3. agent-lightning (7.7k ⭐)
- **定位**: AI Agent训练器
- **亮点**: 结构化训练循环，微软背书
- **适用**: Agent训练和评估研究

### 4. skyvern (17.2k ⭐)
- **定位**: AI浏览器工作流自动化
- **亮点**: Agentic web自动化，可重复运行，支持RPA场景
- **适用**: 网页数据抓取，自动化测试

### 5. mindsdb (37.1k ⭐)
- **定位**: 联邦AI查询引擎
- **亮点**: 跨数据源统一AI查询，MCP Server支持
- **适用**: 企业数据整合，AI驱动决策

### 6. UI-TARS-desktop (14.7k ⭐)
- **机构**: 字节跳动
- **定位**: 原生GUI智能体
- **亮点**: 10+项SOTA，超越Claude/GPT-4o在OSWorld等基准
- **适用**: 桌面自动化，UI测试

---

## 📝 业界文章

### 阿里技术
1. **RecGPT-V2推荐系统** (2025-12)
   - 成本降低60%，GMV +7.43%
   - 元提示技术实现动态解释生成
   - 约束奖励塑形实现持续学习

2. **多模态行为序列建模**
   - 增加辅助任务预测用户兴趣分布
   - 监督学习提升多模态兴趣编码能力

### 字节跳动技术
1. **Primus: 统一大规模DLRM训练系统** (USENIX ATC 2025)
   - 日均训练160TB数据
   - 900万核CPU + 数万GPU异构资源
   - 混合训练范式（记忆塔+适应塔）

2. **Magnus: 机器学习数据管理方案** (VLDB 2025)
   - 5EB数据规模
   - 高性能MOR更新与Upsert机制
   - 推荐大模型和多模态大模型训练优化

3. **HLLM分层LLM推荐方案**
   - Item LLM + User LLM两层架构
   - 冷启动场景显著优势

4. **GPU Scale-up互联技术白皮书 (2025)**
   - 自研EthLink网络方案
   - 支持Load/Store和RDMA语义

### 行业趋势
1. **Improving Recommendation Systems & Search in the Age of LLMs** (Eugene Yan, 2025-03)
   - 系统性总结LLM时代搜索推荐架构演进
   - 涵盖Semantic IDs、多模态融合、统一框架等方向
   - 链接: https://eugeneyan.com/writing/recsys-llm/

2. **DeepSeek-V3.2-Exp发布**
   - 稀疏注意力优化长上下文效率
   - SGLang和vLLM day-0支持
   - FlashMLA稀疏注意力内核开源

---

## 📊 趋势总结

### 推荐系统趋势
1. **LLM化**: 从ID-based向语义理解转变，HLLM、RecGPT等方案验证可行
2. **多模态**: 视觉-语言模型在电商推荐中取得显著效果提升
3. **冷启动**: Semantic IDs、内容特征提取等技术成为重点
4. **训练效率**: DTI等训练范式大幅降低LLM推荐训练成本

### 搜索算法趋势
1. **稠密检索普及**: 双编码器架构成为标配
2. **对话式搜索**: 多轮上下文感知检索成为研究热点
3. **Neural+LLM融合**: 传统神经检索与大模型深度结合

### 大模型工程趋势
1. **KV Cache优化**: 从简单量化到语义感知压缩，长上下文支持成为关键
2. **推理框架分化**: vLLM主打吞吐，SGLang主打多轮对话和结构化输出
3. **确定性推理**: SGLang支持完全确定性推理，支持可复现RL训练
4. **端侧优化**: INT4量化、KV Cache压缩推动端侧大模型部署

### 开源生态趋势
1. **Agent框架爆发**: 多智能体系统、GUI自动化等方向活跃
2. **本地部署**: LocalAI等项目推动私有化部署
3. **推理优化**: ktransformers等高性能推理框架受关注

---

*本期技术雷达由MelonEggLearn整理，数据来源涵盖arXiv、GitHub、企业技术博客等公开渠道。*

*更新时间: 2026-03-11*
