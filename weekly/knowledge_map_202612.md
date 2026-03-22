# 🗺️ MelonEgg 周天知识地图 — 2026-W12（03/16–03/22）

> 生成时间：2026-03-22 23:00 JST | 知识库：~/Documents/ai-kb/

---

## A. 本周学了什么（快速全景）

### 📊 数量统计
- **总计处理：~210篇** 论文/项目笔记（7天 × 30篇/天）
- 覆盖领域：推荐系统 / 广告系统 / 搜索 / LLM工程 / 开源项目

---

### 🔮 推荐系统（Rec-Sys）

- **生成式推荐范式全面铺开**：RankGR（列表级DPO）、GEMs（多流解码长序列）、DiffGRM（扩散式生成）、ActionPiece（行为序列Tokenization）——生成式推荐正从学术原型走向工业可用
- **Semantic ID 生态完整成型**：RQ-VAE离散化 → Variable-Length动态长度 → Spotify双论文工业落地，标志着生成式检索的完整工程闭环首次公开
- **多行为 + 多任务仍是主战场**：MMoE/PLE框架复习、CascadingRank多行为图、AutoIFS多场景多任务——工业推荐的核心复杂度依然在任务协调
- **LLM与推荐深度融合**：LLM Universal Retriever、Web-scale LLM Rec、Spotify Unified LM（统一搜索+推荐+推理）——双塔正在被LLM作为backbone部分替代
- **经典基础补全**：DIN深度兴趣网络、双塔召回、对比学习、冷启动——面试知识地基全面夯实

---

### 📢 广告系统（Ads）

- **竞价机制体系全覆盖**：GSP/VCG机制原理 → oCPC/oCPA优化 → First-Price Auction出价策略（含non-stationary环境下的regret minimization）——从理论到工程完整打通
- **偏差治理三部曲**：位置偏差（IPS反事实学习）+ 样本选择偏差（ESMM全空间建模）+ 兴趣偏差（DIN激活机制）——广告系统数据偏差问题的系统化认知
- **自动出价进化史**：HALO事后增强RL → AutoBid约束RL → Diffusion AutoBidding → BAT评测基准——从规则出价到LLM/扩散模型的演进路线图清晰
- **预算管理工程**：Pacing（Throttling/Online Primal-Dual）、多目标约束优化（帕累托 vs 加权求和）、多平台跨预算分配——广告系统长期博弈视角
- **创意生成新前沿**：LLM广告创意生成、LLM-Auction原生竞价、多模态CTR预估——生成式AI入侵广告全链路

---

### 🔍 搜索（Search）

- **稀疏 vs 稠密大和解**：BM25/SPLADE-v3（稀疏语义升级）+ DPR/BGE-M3（稠密多语言）+ 混合RRF融合——二者不是竞争而是互补，混合检索成标配
- **查询理解与改写**：RewriteGen（RL查询优化）、Intent-Aware Neural Reformulation、Query-as-Anchor（LLM场景自适应用户表示）——查询端的智能化是搜索提升的核心杠杆
- **多模态检索崛起**：LLandMark视频检索、Visual Document Retrieval综述（ColPali框架）、KGMEL知识图谱实体链接——纯文本搜索向多模态迁移加速
- **RAG工程深化**：LURE轻量重排加速、MA-RAG多智能体协作、DRAMA多样化数据增强——RAG不再是简单的"检索+生成"，而是复杂的多阶段工程
- **工业搜索新方向**：DLLM-Searcher（扩散LLM搜索Agent）、SUNAR语义不确定性检索——搜索正在向Agent化和不确定性感知演进

---

### ⚙️ LLM工程（LLM Infra）

- **推理加速三重奏**：FlashAttention-3（硬件感知注意力）+ PagedAttention/FlashInfer（KV Cache管理）+ 投机解码/Recurrent Drafter——生产级推理加速的黄金组合
- **KV Cache压缩大爆发**：KVSharer（层间共享）、ZSMerge/ZeroMerge（零样本压缩）、CacheClip/HA-RAG（RAG场景复用）——KV Cache已成推理工程的核心战场
- **GRPO开创RLHF新路**：无Critic的组内相对奖励，DeepSeek-R1的核心技术，中小团队做RLHF的实用方案
- **MoE架构工业化**：MoE-LLaMA serving、MegaScale-Infer Expert Parallelism、DeepSeek-V3/Qwen3技术报告——MoE + KV压缩是大模型规模化部署主流方向
- **长上下文综合进展**：Efficient Long-Context LLMs综述、LONGER超长序列推荐、LLMOrbit分类学——长上下文从单点技术扩展到系统性工程挑战

---

### 🛠️ 开源项目（Repo）

- **推荐基础设施**：TorchRec（TB级Embedding分片）、RecBole 2.0（100+算法）、Gorse（一体化推荐引擎）、Microsoft Recommenders
- **向量搜索生态**：Milvus（云原生）、Qdrant（Rust高性能FilteredHNSW）、Chroma 2025（Chroma Cloud）
- **LLM工程框架**：vLLM（PagedAttention v2）、LlamaIndex 2025、Haystack 2.0、LightRAG、OpenRAG
- **Agent框架涌现**：Open-SWE（异步编码Agent）、DeepAgents（LangGraph）、OpenViking（上下文数据库）、UI-TARS（多模态GUI自动化）
- **前沿推理基础设施**：BitNet 1-bit LLM、verl（字节RL训练框架）、FATE联邦学习

---

## B. 技术关系图（本周知识网络）

```
                    ┌─────────────────────────────────────────────────────┐
                    │              本周核心技术主线                         │
                    └─────────────────────────────────────────────────────┘

【主线1：生成式推荐范式演进】
                   传统协同过滤
                        │
              双塔召回（相似度计算）
                        │
                   DIN/MMoE精排
                        │
                        ▼
             Semantic ID（RQ-VAE离散化）
                  /           \
    Variable-Length SID    Constrained Beam Search
         │                        │
   冷启动友好                  线上可用的约束生成
         │                        │
    Spotify工业落地（播客发现 + 统一LM）
         │
         ▼
   GEMs多流解码（解决长序列瓶颈）
         │
   DiffGRM（扩散式生成，多样性↑）
         │
   RankGR（列表级DPO，排序对齐）

【主线2：广告偏差治理闭环】
   位置偏差 ──IPS纠偏──→ 公平排序
        │
   样本选择偏差 ──ESMM──→ 全空间CVR建模
        │
   兴趣偏差 ──DIN注意力──→ 目标感知激活
        │
   多目标冲突 ──约束优化──→ 长期LTV博弈
        │
   自动出价 ──GRPO/RL──→ 约束满足的智能出价

【主线3：LLM推理效率三角】
            GRPO（训练效率）
           /
  推理效率三角
           \
            MoE（架构效率）
           /
          ×
           \
            Spec-Decode + FlashAttn3（解码效率）
                    |
               KV Cache压缩（内存效率）

【主线4：检索范式融合】
   BM25（词频精确）──┐
                    ├──RRF混合──→ 混合检索（工业标配）
   DPR/BGE（语义）──┘
        │
   SPLADE-v3（稀疏语义升级，可复用倒排索引）
        │
   ColBERT（Late Interaction，精度/效率平衡）
        │
   RAG管道（Chunk→检索→重排→生成）
        │
   Agent化搜索（MA-RAG/DLLM-Searcher，规划+推理）

【跨领域连接】
   LLM工程 ──→ 推荐系统（LLM作为推荐backbone）
   广告精排 ──→ 搜索重排（IPS、位置偏差共用方法论）
   生成式推荐 ──→ LLM解码（Beam Search、Speculative Decoding共享基础设施）
   向量数据库 ──→ 推荐召回（ANN检索是双塔的基础设施）
```

**基础层（Prerequisites）：**
- FM/DeepFM → DIN → MMoE → ESMM（CTR/CVR经典演进）
- BM25 → DPR → ColBERT → SPLADE（检索模型演进）
- PPO → GRPO（RLHF高效化）
- Dense Transformer → MoE → 长上下文（LLM架构演进）

**进阶层（Advanced）：**
- Semantic ID全链路、Variable-Length SID、生成式推荐系统
- KV Cache压缩系列（KVSharer/ZSMerge/CacheClip）
- First-Price Auction非平稳环境出价策略
- 多目标约束优化（帕累托前沿、拉格朗日对偶）

**并列方案（Alternatives）：**
- 稀疏检索 vs 稠密检索（→ 最优解：混合）
- PPO vs GRPO（→ 资源受限选GRPO）
- 双塔 vs LLM检索（→ 融合架构）
- Throttling Pacing vs 出价调整Pacing（→ 场景依赖）

---

## C. 本周最重要的 3 个工业落地附加技能点

### 1. 🔧 Semantic ID 的版本管理与冷启动工程

**论文层面**：RQ-VAE离散化 → 生成式检索  
**工业落地的隐藏复杂度**：
- **ID空间漂移**：每次重训RQ-VAE，商品的Semantic ID都会改变 → 需要版本锁定机制 + 灰度切换
- **新商品实时编码**：线上新上架商品需要独立的轻量encoder service，而非离线批量 → 需要独立serving基础设施
- **Trie约束解码**：没有约束的Beam Search会生成不存在的ID序列 → 线上必须维护实时Trie树，随商品库更新而更新
- **Spotify实践**：每日重建ID索引，p99解码延迟 <50ms，音频+文本+元数据三路多模态融合

**落地结论**：Semantic ID不只是训练问题，而是一套完整的在线serving+索引管理工程体系

---

### 2. 🎯 广告系统的LTV（长期价值）核算框架

**论文层面**：多目标优化平衡收入和用户体验  
**工业落地的隐藏复杂度**：
- **短期实验的陷阱**：7天A/B实验可能看到eCPM +2%，但用户留存率-0.1%，折算6个月LTV是亏损的
- **体验指标的滞后性**：点击率/CTR立竿见影，但用户疲劳/留存变化是3-6个月才显现的长期信号
- **约束 vs 加权的工程优势**：帕累托约束（体验 ≥ 阈值）比加权求和（α×收入 + β×体验）更易于业务解释和调参
- **冷热区隔**：新广告主需要保护性流量（避免马太效应），老广告主约束可放松 → 分群约束才是工业实践

**落地结论**：广告多目标优化的核心是建立长期收益的可度量框架，而非追求单次曝光的局部最优

---

### 3. ⚡ KV Cache压缩的工程优先级矩阵

**论文层面**：多种KV Cache压缩方法（量化/剪枝/共享/合并）  
**工业落地的隐藏复杂度**：
- **场景区分**：RAG场景（Context高度重复）→ Prefix Cache最优；长对话场景（历史不重复）→ 量化压缩最优；MoE场景 → Expert并行 + 共享Cache
- **压缩 vs 精度权衡**：KVSharer层间共享可以压缩至1/20，但相邻层语义差异大的任务（如数学推理）精度损失显著
- **基础设施依赖**：vLLM的PagedAttention、SGLang的RadixAttention是Prefix Cache的工程实现，需要配套升级
- **压缩组合拳**：实际生产推荐 FlashAttention-3（计算加速）+ PagedAttention（内存管理）+ 量化（带宽节省）三层叠加，而非选一种

**落地结论**：KV Cache优化没有银弹，需要根据业务场景（RAG/对话/推理）和硬件配置（A100/H100/H200）选择最优组合

---

## D. 面试导向：本周内容高频考法

### 🔴 必背（出现概率 >80%）

| 考点 | 核心回答要点 |
|------|-------------|
| DIN vs 传统Pooling | Attention权重 + 不用Softmax（局部激活避免竞争）+ GAUC评估 |
| ESMM为什么解决SSB | 全空间建模 + 共享Embedding + CTCVR联合监督 |
| GRPO vs PPO | 无Critic + 组内相对归一化 + 显存-40% + 可验证reward场景最优 |
| 混合检索为什么比单一好 | 精确查询BM25胜 + 语义查询Dense胜 + 长尾最优解是Hybrid + RRF融合 |
| GSP为什么不是激励相容 | 赢家付第二高价变体，存在Nash均衡但非dominant strategy |
| KV Cache是什么 | Transformer推理复用历史K/V，避免重复计算；PagedAttention虚拟内存管理 |

### 🟡 重要（出现概率 50-80%）

| 考点 | 核心回答要点 |
|------|-------------|
| Semantic ID原理 | RQ-VAE层次化离散 → 生成式检索 → 冷启动友好 |
| 广告位置偏差如何纠正 | IPS逆倾向评分 + 随机化实验估计propensity + Clipped IPS防高方差 |
| MoE工作原理 | 稀疏激活（top-K专家）+ 路由函数 + 辅助负载均衡损失 |
| 投机解码原理 | 小草稿模型生成 → 大模型并行验证 → 3-5x解码加速 |
| oCPC/oCPA本质 | 广告主设目标转化价，系统自动出价；等价于在约束下最优化 |
| 双塔In-batch负采样偏差 | 热门item被过多采样为负 → logQ correction修正 |

### 🟢 加分（深度理解展示）

| 考点 | 差异化亮点 |
|------|-----------|
| Semantic ID工业落地挑战 | ID漂移、实时Trie、多模态融合（不止论文原理） |
| GRPO适用边界 | 可验证reward才有效，开放对话reward难设计是短板 |
| First-Price拍卖 vs Second-Price | Google 2019年切换后，truthful bidding不再最优，需要学习bid shading |
| KV Cache压缩组合拳 | FlashAttention + PagedAttention + 量化三层叠加，场景驱动选择 |
| 广告LTV博弈 | 短期实验看不出用户留存损失，需要长期holdout实验 |

---

## E. 下周自主学习计划

### 本周自我评估劣势

通过本周学习识别到的薄弱点：

1. **推荐系统**：生成式推荐的评估指标（Recall@K vs NDCG vs 多样性指标）理解不够深，线上A/B设计有盲区
2. **广告系统**：First-Price Auction的bid shading数学（最优shading系数推导、regret bound证明）尚不扎实
3. **搜索**：多模态搜索（ColPali、VDR框架）的工程实现细节了解不足
4. **LLM**：FlashAttention-3的硬件感知优化原理（Hopper架构异步流水线、warp-specialization）还没深入

### 下周学习重点

#### 🎯 重点1：推荐系统评估体系 + 在线实验设计

**为什么**：生成式推荐改变了评估范式，传统Recall/NDCG不能反映生成质量

**学习内容**：
- 离线评估：Recall@K、NDCG、Novelty、Coverage、Diversity的权衡
- 在线实验：曝光位分层 vs 流量分桶、双边市场的互干扰问题（Cannibalization）
- 生成式特有：生成Hit Rate、Constrained Generation成功率
- 工业实践：美团/淘宝/Spotify如何设计线上A/B for Gen-Rec

**目标产出**：`synthesis/evaluation_framework_for_gen_rec.md`

---

#### 🎯 重点2：多模态搜索与检索系统深化

**为什么**：本周搜索领域涉及了多模态检索，但工程细节没吃透

**学习内容**：
- ColPali：Late Interaction在视觉文档中的实现（patch-level token交互）
- 多模态Embedding：CLIP/SigLIP vs E5-Mistral在检索任务上的差距
- 图文联合检索的工程挑战：多模态Index存储、跨模态相似度归一化
- 工业案例：Pinterest视觉搜索、Alibaba商品图搜

**目标产出**：`synthesis/multimodal_retrieval_engineering.md`

---

#### 附加目标（如时间允许）

- 补完 FlashAttention-3 的硬件原理笔记（Hopper SM90架构）
- 阅读 DeepSeek-V3 完整技术报告的数学部分（MLA / MTP）
- 整理面试卡片：First-Price Auction数学推导 + GRPO推导

---

## 本周总结

**这是搬推广告搜 + LLM工程知识密度最高的一周之一。**

三条核心主线完整成型：
1. **生成式推荐**：从RQ-VAE → Semantic ID → 可变长度 → Spotify工业落地，工业范式转移已清晰
2. **广告偏差治理**：位置/样本/兴趣三类偏差的系统化认知建立，对面试+工业理解都有直接价值
3. **LLM推理三角**：GRPO（训练）+ MoE（架构）+ 投机解码/FlashAttention（解码）的完整技术体系

**一句话评价**：本周是「打地基」周——经典方法（DIN/MMoE/双塔/BM25）和前沿进展（Semantic ID/GRPO/KV压缩）同时覆盖，为后续面试准备和工业理解打下了扎实基础。

---

*生成工具: MelonEggLearn Agent | 知识库: ~/Documents/ai-kb/ | 覆盖笔记: ~210篇*
