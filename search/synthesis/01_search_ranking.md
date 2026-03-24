# 搜索排序专项笔记 - MelonEggLearn

> 📚 参考文献
> - [Dense-Retrieval-Vs-Sparse-Retrieval-A-Unified-E...](../../search/papers/20260321_dense-retrieval-vs-sparse-retrieval-a-unified-evaluation-framework-for-large-scale-product-search.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Bm25S Orders Of Magnitude Faster Lexical Search...](../../search/papers/20260323_bm25s_orders_of_magnitude_faster_lexical_search_via.md) — BM25S: Orders of Magnitude Faster Lexical Search via Eage...
> - [Colbert V3 Efficient Neural Retrieval With Late...](../../search/papers/20260323_colbert_v3_efficient_neural_retrieval_with_late_int.md) — ColBERT v3: Efficient Neural Retrieval with Late Interaction
> - [Dense Retrieval Vs Sparse Retrieval A Unified Eval](../../search/papers/20260323_dense_retrieval_vs_sparse_retrieval_a_unified_eval.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Colbert-Serve-Efficient-Memory-Mapped-Scoring](../../search/papers/20260320_ColBERT-serve-Efficient-Memory-Mapped-Scoring.md) — ColBERT-serve: Efficient Multi-Stage Memory-Mapped Scoring
> - [Dense-Retrieval-Vs-Sparse-Retrieval-Unified-Eva...](../../search/papers/20260321_dense-retrieval-vs-sparse-retrieval-unified-evaluation-framework.md) — Dense Retrieval vs Sparse Retrieval: A Unified Evaluation...
> - [Flashrag A Modular Toolkit For Efficient Retrie...](../../search/papers/20260323_flashrag_a_modular_toolkit_for_efficient_retrieval-.md) — FlashRAG: A Modular Toolkit for Efficient Retrieval-Augme...
> - [Query-As-Anchor-Scenario-Adaptive-User-Represen...](../../search/papers/20260321_query-as-anchor-scenario-adaptive-user-representation-via-large-language-model-for-search.md) — Query as Anchor: Scenario-Adaptive User Representation vi...


> 深入研究：从传统检索到神经检索

---

## 📚 参考资料与引用

本文是对搜索系统发展的总结提炼，引用了以下研究与实践案例：

### 学术基础论文
- [BM25 & 语义混合检索](../papers/20260316_bm25-semantic-hybrid-retrieval.md)
- [DPR - 密集通道检索](../papers/20260316_dpr-dense-retrieval.md)
- [ColBERT v2 - 后期交互检索](../papers/20260313_colbert_v2.md)
- [ColBERT v3 - 高效后期交互](../papers/20260323_colbert_v3_efficient_neural_retrieval_with_late_int.md)
- [SPLADE v3 - 稀疏检索新基准](../papers/20260322_splade_v3_sparse_retrieval.md)
- [稀疏检索 vs 密集检索统一评估](../papers/20260323_dense_retrieval_vs_sparse_retrieval_a_unified_eval.md)

### 混合检索与融合
- [混合搜索 LLM 重排](../papers/20260320_Hybrid-Search-LLM-Re-ranking.md)
- [BM25S - 快速稀疏搜索](../papers/20260323_bm25s_orders_of_magnitude_faster_lexical_search_via.md)

### LLM 时代集成
- [LLM for IR 综述](../papers/20260316_llm-for-ir-survey.md)
- [RAG 检索优化](../papers/20260316_rag-retrieval-optimization.md)
- [LLM 集成框架](./llm_integration_framework.md)

### 工程实现参考
- [FlashRAG - 检索工具包](../papers/20260323_flashrag_a_modular_toolkit_for_efficient_retrieval-.md)
- [ColBERT Serve - 高效内存映射评分](../papers/20260320_ColBERT-serve-Efficient-Memory-Mapped-Scoring.md)

---

## 目录
1. [BM25原理与局限](#1-bm25原理与局限)
2. [Dense Retrieval (DPR/ANCE)](#2-dense-retrieval-dprance)
3. [ColBERT多向量检索](#3-colbert多向量检索)
4. [混合检索 (Sparse + Dense)](#4-混合检索-sparsedense)

---

## 1. BM25原理与局限

### 1.1 核心公式

BM25 (Best Match 25) 是信息检索领域最经典、使用最广泛的词袋模型评分函数。

```
                     ┌─────────────────────────────────────────────────────────────┐
                     │  BM25(q, d) = Σᵢ IDF(qᵢ) · TF(qᵢ, d)                        │
                     │                                                             │
                     │  where:                                                     │
                     │    IDF(qᵢ) = log( (N - n(qᵢ) + 0.5) / (n(qᵢ) + 0.5) )       │
                     │                                                             │
                     │    TF(qᵢ, d) = f(qᵢ,d)·(k₁+1) / [f(qᵢ,d)+k₁·(1-b+b·|d|/avgdl)]│
                     └─────────────────────────────────────────────────────────────┘
```

**参数说明**：
| 参数 | 含义 | 典型值 | 作用 |
|-----|------|-------|------|
| N | 文档总数 | - | 计算IDF |
| n(qᵢ) | 包含词qᵢ的文档数 | - | 计算IDF |
| f(qᵢ,d) | 词qᵢ在文档d中的词频 | - | 基础TF |
| k₁ | TF饱和参数 | 1.2-2.0 | 控制词频上限 |
| b | 长度归一化参数 | 0.75 | 控制文档长度影响 |
| avgdl | 平均文档长度 | - | 长度归一化基准 |

### 1.2 设计思想拆解

#### TF饱和 (Saturation)
```
TF贡献随词频增长趋于饱和：

TF分数
  │        ╭──── k₁=1.2
  │       ╱
  │      ╱  ╭─── k₁=2.0 (更平缓)
  │     ╱  ╱
  │    ╱  ╱
  │   ╱  ╱
  │  ╱  ╱
  │ ╱  ╱
  │╱__╱____________
  └────────────────► 词频

直觉: 一个词出现100次 vs 10次，相关性不会差10倍
```

#### 长度归一化
```
长文档天然有更多词，需要惩罚：

BM25-TF = raw_TF / [length_penalty]

当文档长度 = avgdl 时，length_penalty = 1
当文档长度 > avgdl 时，length_penalty > 1 (惩罚)
当文档长度 < avgdl 时，length_penalty < 1 (奖励)

参数b控制惩罚强度:
- b=0: 无长度归一化
- b=1: 完全归一化
- b=0.75: 经验最佳值
```

### 1.3 BM25变种

#### BM25+
```
解决BM25对长文档过度惩罚的问题:

TF(qᵢ, d) = f(qᵢ,d)·(k₁+1) / [f(qᵢ,d)+k₁·(1-b+b·|d|/avgdl)] + δ

δ = 1.0 (经验值)，确保每个匹配词至少有一定贡献
```

#### BM25F (BM25 with Field Weights)
```
适用于多字段文档 (标题、正文、标签等):

BM25F(q, d) = Σᵢ IDF(qᵢ) · TF_F(qᵢ, d)

TF_F = Σⱼ [ f(qᵢ, fieldⱼ) · boost(fieldⱼ) ] · saturation / length_norm
```

#### BM25 with Proximity
```
考虑词项位置接近度:

Score = BM25_base + α · ProximityScore

ProximityScore = 1 / (1 + 词项间平均距离)
```

### 1.4 局限性与改进方向

| 局限 | 说明 | 改进方向 |
|-----|------|---------|
| **词汇鸿沟** | "手机" ≠ "移动电话" | 同义词扩展、语义向量 |
| **语义缺失** | 无法理解词义和上下文 | 预训练语言模型 |
| **权重调参** | k₁、b需人工调参 | 学习排序(LTR)优化 |
| **领域依赖** | 通用参数不适合所有领域 | 领域自适应BM25 |
| **长尾Query** | 生僻词IDF不稳定 | 平滑技术 |

### 1.5 实现优化

#### 倒排索引结构
```python
# 简化版倒排索引
inverted_index = {
    "手机": {
        "doc_freq": 10000,  # 包含该词的文档数
        "postings": [
            (doc_id_1, term_freq_1, [pos1, pos2, ...]),
            (doc_id_2, term_freq_2, [pos3, pos4, ...]),
            ...
        ]
    },
    "苹果": {
        ...
    }
}
```

#### WAND算法 (Weak AND)
```
快速跳表算法，避免计算全部文档：

1. 维护各词项的postings list指针
2. 计算当前组合的分数上界
3. 如果上界 < 当前TopK最低分，快速跳过
4. 否则深入计算

复杂度: 从O(N)降低到O(√N)级别
```

---

## 2. Dense Retrieval (DPR/ANCE)

### 2.1 从稀疏到密集：范式转变

```
稀疏表示 (Sparse)              密集表示 (Dense)
    │                              │
    ▼                              ▼
┌─────────┐                  ┌─────────┐
│ 高维    │                  │ 低维    │
│ (Vocab  │                  │ (768-d) │
│  Size)  │                  │         │
│         │                  │         │
│ 大多数为0│                 │ 稠密向量 │
│ (稀疏)   │                 │         │
└─────────┘                  └─────────┘
    │                              │
精确匹配                       语义相似
"手机"=1                       "手机"≈"移动电话"
其他=0                         语义相近词向量相近
```

### 2.2 DPR (Dense Passage Retrieval)

#### 核心思想
```
双塔架构 + 独立编码 + 点积相似度

Query ──────► [BERT] ──────► q_vector ──┐
                                         ├──► sim(q,d) = qᵀd
Doc ─────────► [BERT] ──────► d_vector ──┘
                              (预计算索引)
```

#### 训练方法
```python
# 正样本: 人工标注的相关文档
# 负样本: In-batch negatives (同一batch内的其他文档)

class DPR(nn.Module):
    def __init__(self):
        self.query_encoder = BertModel()
        self.doc_encoder = BertModel()
        
    def forward(self, query, pos_doc, neg_docs):
        q_emb = self.query_encoder(query).pooler_output  # [batch, dim]
        d_emb = self.doc_encoder(pos_doc).pooler_output  # [batch, dim]
        
        # 计算相似度
        similarities = q_emb @ d_emb.T  # [batch, batch]
        
        # InfoNCE Loss (对角线为正样本)
        labels = torch.arange(batch_size)
        loss = F.cross_entropy(similarities / temperature, labels)
        
        return loss
```

#### 负采样策略
| 策略 | 方法 | 效果 |
|-----|------|------|
| Random | 随机采样 | 太简单，学不到什么 |
| BM25 Top | BM25返回的top但标注不相关 | 难负样本，效果好 |
| In-batch | 同一batch内的其他正样本 | 高效，免费获得 |
| Cross-batch | 跨batch缓存负样本 | ANCE使用，增大负样本池 |

### 2.3 ANCE (Approximate Nearest Neighbor Negative Contrastive Learning)

#### 动机
```
DPR的问题: 训练时负样本 vs 推理时负样本分布不一致

训练: 使用batch内随机负样本
推理: 从全部文档中ANN检索topK

ANCE解决: 让训练负样本分布逼近推理分布
```

#### 异步训练流程
```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: 用当前模型编码全部文档，构建ANN索引                       │
│  └── 使用FAISS/HNSW建立向量索引                                   │
├─────────────────────────────────────────────────────────────────┤
│  Step 2: 对训练Query，从ANN索引中检索topK                         │
│  └── 这些就是"难负样本" (模型认为相似但实际不相关)                 │
├─────────────────────────────────────────────────────────────────┤
│  Step 3: 使用难负样本训练模型                                     │
│  └── 正样本: 标注相关文档                                          │
│  └── 负样本: ANN返回的不相关文档                                   │
├─────────────────────────────────────────────────────────────────┤
│  Step 4: 定期更新ANN索引 (如每1000步)                             │
│  └── 确保负样本分布跟上模型进化                                    │
└─────────────────────────────────────────────────────────────────┘
```

#### 算法优势
| 特点 | 说明 |
|-----|------|
| 动态难负样本 | 随着模型变强，负样本也变难 |
| 训练-推理一致 | 都是ANN检索的top结果 |
| 收敛更快 | 更难样本带来更强信号 |
| 效果提升 | 相比DPR有显著改进 |

### 2.4 向量索引技术

#### FAISS (Facebook AI Similarity Search)
```
常用索引类型:

1. Flat Index (暴力搜索)
   IndexFlatIP / IndexFlatL2
   └── 精确但慢，适合小规模 (<100K)

2. IVF (Inverted File Index)
   IndexIVFFlat
   └── 聚类分桶，搜索时只查最近几个桶
   └── nlist参数控制桶数量，越大越准越慢

3. PQ (Product Quantization)
   IndexPQ / IndexIVFPQ
   └── 向量压缩，内存友好
   └── 有损压缩，精度换速度

4. HNSW (Hierarchical Navigable Small World)
   IndexHNSWFlat
   └── 图索引，当前最佳速度和精度平衡
   └── efConstruction: 构建时邻居数
   └── M: 每节点最大邻居数
```

#### 索引选择指南
| 数据规模 | 推荐索引 | 召回率 | 延迟 |
|---------|---------|--------|------|
| <100K | Flat | 100% | <10ms |
| 100K-1M | IVF-Flat (nlist=1024) | ~99% | <5ms |
| 1M-10M | HNSW or IVF-PQ | ~95% | <10ms |
| 10M-100M | IVF-PQ 或分片 | ~90% | <20ms |
| >100M | 分片 + 多级索引 | 可调 | 可调 |

### 2.5 Dense Retrieval vs BM25

| 维度 | BM25 | Dense Retrieval |
|-----|------|-----------------|
| 匹配方式 | 精确词匹配 | 语义相似度 |
| 词汇鸿沟 | 无法处理 | 天然支持 |
| 新词/长尾 | IDF不稳定 | 依赖预训练 |
| 可解释性 | 强 | 弱 |
| 计算成本 | 低 | 高 (需GPU) |
| 存储成本 | 低 (稀疏) | 高 (稠密向量) |
| 增量更新 | 容易 | 需重建索引 |
| 领域适配 | 需调参 | 需领域训练 |

---

## 3. ColBERT多向量检索

### 3.1 动机：单向量 vs 多向量

```
单向量表示的问题:
┌─────────────────────────────────────────────────────────────┐
│  "苹果发布了新iPhone"                                        │
│                                                             │
│  压缩为单个[CLS]向量: [0.1, -0.3, 0.5, ...] (768-d)          │
│                                                             │
│  问题: 细粒度信息丢失，无法区分"苹果"(公司)vs"苹果"(水果)    │
│        长文档信息稀释                                        │
└─────────────────────────────────────────────────────────────┘

多向量表示 (ColBERT):
┌─────────────────────────────────────────────────────────────┐
│  "苹果发布了新iPhone"                                        │
│                                                             │
│  苹果 ──► [0.2, -0.1, ...]  (128-d per token)               │
│  发布 ──► [-0.3, 0.5, ...]                                  │
│  了   ──► [0.1, 0.2, ...]                                   │
│  新   ──► [0.4, -0.2, ...]                                  │
│  iPhone ► [0.5, 0.3, ...]                                   │
│                                                             │
│  保留每个token的细粒度表示                                   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 ColBERT架构

#### Late Interaction机制
```
Query Tokens (maxlen=32)      Doc Tokens (maxlen=180)
       │                              │
       ▼                              ▼
   [BERT]                         [BERT]
       │                              │
       ▼                              ▼
   E_q1, E_q2, ...                E_d1, E_d2, ...
   (每个token一个向量)              (每个token一个向量)
       │                              │
       │    (离线预计算，存入索引)      │
       │                              ▼
       │                        FAISS索引
       │                    (每个向量单独索引)
       │
       ▼
   MaxSim计算 (在线)
   
Score(q,d) = Σᵢ maxⱼ(E_qᵢ · E_dⱼ)  
              └── 对每个Query token，找Doc中最相似的token
```

#### 评分公式详解
```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   S(q, d) = Σᵢ maxⱼ ( E_qᵢᵀ · E_dⱼ )                                │
│             ────┘                                                   │
│              └── Late Interaction: 延迟到最后一刻才做交互           │
│                                                                     │
│   直观理解:                                                          │
│   - "苹果"在Query中 → 匹配Doc中任意位置的"苹果"/"iPhone"/"手机"    │
│   - 取最大相似度，然后所有Query token求和                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 工程优化

#### 向量压缩
```
原始: 每个token 768-d float32 = 3KB per token
文档180 token → 540KB per doc
1亿文档 → 54TB (不可行)

优化1: 降维投影 (ColBERTv1)
  768-d → 128-d = 87%压缩
  
优化2: 向量量化 (ColBERTv2)
  Residual Compression:
  - 每个向量用 centroid + residual 表示
  - 通常 1-2 bytes per dimension
  - 最终 ~100-200 bytes per token

优化3: 剪枝 (PLAID)
  - 只保留重要token的向量
  - 去除停用词、低频词
```

#### 索引结构 (ColBERTv2)
```
┌─────────────────────────────────────────────────────────────────┐
│  1. 编码全部文档，获得token embeddings                            │
│                                                                 │
│  2. 聚类量化 (K-means, 256 centroids)                           │
│     └── 每个向量表示为: centroid_id (1 byte) + residual         │
│                                                                 │
│  3. 倒排索引: term → [包含该term的doc_ids]                       │
│     └── 利用BERT的tokenization特性                              │
│                                                                 │
│  4. 检索时:                                                     │
│     a. 编码Query                                                │
│     b. 对Query每个token，在对应倒排列表中找最相似文档            │
│     c. 汇总各token贡献，计算最终分数                             │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 ColBERT优势

| 优势 | 说明 |
|-----|------|
| 细粒度匹配 | Token-level匹配，比文档级更精确 |
| 可解释性 | 可以可视化Query-Doc的token对齐 |
| 长文档友好 | 没有信息稀释问题 |
| 延迟交互 | 文档编码离线完成，在线只做轻量计算 |
| 效果领先 | 多数据集SOTA |

### 3.5 局限
```
1. 存储开销大
   └── 即使压缩后，仍比单向量大10-100倍
   
2. 索引构建慢
   └── 需要编码每个token
   
3. 检索延迟
   └── 虽然比单向量慢，但<100ms仍可接受
   
4. 只适配短文档
   └── 长文档需截断或分块
```

---

## 4. 混合检索 (Sparse + Dense)

### 4.1 为什么需要混合

```
BM25擅长:                    Dense擅长:
  - 精确匹配                  - 语义匹配
  - 罕见词/专业术语            - 同义词/变体
  - 短查询                    - 长查询/自然语言
  - ID明确的实体              - 概念理解

Query: "apple juice recipe"
       │            │
       ▼            ▼
    BM25高分      Dense高分
    "apple"精确   "fruit drink how to make"
    匹配文档       语义相关文档
```

### 4.2 融合策略

#### 1. 线性加权融合
```
S_final = α · S_sparse + β · S_dense

参数调优:
- 网格搜索: α + β = 1，尝试不同组合
- 学习排序: 用LR/GDBT学习最优权重
- 查询自适应: 不同Query类型用不同权重
```

#### 2. RRF (Reciprocal Rank Fusion)
```
S_RRF(q,d) = Σᵢ 1/(k + rankᵢ(d))

其中:
- rankᵢ(d): 文档d在第i个检索器中的排名
- k: 常数，通常60

优点:
- 无需调参，对分数分布不敏感
- 不同检索器可比
- 对排名更关注，对绝对分数不敏感
```

#### 3. 学习排序融合
```
把各检索器的分数作为特征，训练LTR模型:

特征:
- BM25分数
- Dense分数  
- BM25排名
- Dense排名
- BM25匹配词数
- Dense最大相似度
...

模型: GBDT / DNN
目标: 最终相关性排序
```

### 4.3 SPLADE (Learned Sparse Retrieval)

#### 核心思想
```
结合两者优点:
  BM25的优势: 稀疏、高效、可解释、精确匹配
  Dense的优势: 语义理解、学习能力强

SPLADE: 学习稀疏表示，但包含语义扩展
```

#### 模型架构
```
Query/Doc ───► BERT ───► 每个token输出logits ───► 扩展为vocab分布
                              │
                              ▼
                      term_importance = max(0, logit)
                      
最终表示: 稀疏向量，维度=vocab_size
         非零元素=有贡献的term及其权重
         
例如Query "cell phone":
  原始: cell(1.0), phone(1.0)
  SPLADE: cell(0.9), phone(1.0), mobile(0.8), telephone(0.7), ...
```

#### 优势
| 特性 | SPLADE |
|-----|--------|
| 表示 | 稀疏向量 (可倒排索引) |
| 语义 | 可扩展同义词 |
| 存储 | 比Dense小得多 |
| 效率 | 可用传统倒排索引加速 |
| 效果 | 接近Dense Retrieval |

### 4.4 混合检索架构实践

```
┌─────────────────────────────────────────────────────────────────┐
│                        Query输入                                 │
└──────────────────────┬──────────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │  Lexical │  │ Dense   │  │ SPLADE  │
    │  (BM25)  │  │ (DPR)   │  │         │
    └────┬────┘  └────┬────┘  └────┬────┘
         │            │            │
         ▼            ▼            ▼
    Top-K (100)  Top-K (100)  Top-K (100)
         │            │            │
         └────────────┼────────────┘
                      ▼
              ┌─────────────┐
              │   结果融合   │
              │  (RRF/LTR)  │
              └──────┬──────┘
                     ▼
              Top-K Re-rank
                     │
                     ▼
               最终结果
```

### 4.5 方法对比总结

| 方法 | 表示 | 索引 | 语义能力 | 效率 | 适用场景 |
|-----|------|------|---------|------|---------|
| BM25 | 稀疏 | 倒排 | 弱 | 极高 | 关键词匹配、长尾 |
| DPR/ANCE | 稠密 | FAISS | 强 | 高 | 语义搜索、FAQ |
| ColBERT | 多向量 | 特殊 | 很强 | 中 | 高精度需求 |
| SPLADE | 学习稀疏 | 倒排 | 强 | 高 | 平衡方案 |
| 混合 | 多路 | 混合 | 最强 | 中 | 生产环境首选 |

### 4.6 生产环境选型建议

```
小规模 (<100万文档):
  └── Dense (DPR) 或 纯BM25

中等规模 (100万-1000万):
  └── BM25 + DPR 混合 (RRF融合)

大规模 (1000万+):
  └── SPLADE 或 BM25 + 轻量Dense

极高精度需求:
  └── ColBERT 或 多阶段 (召回+精排)

资源受限:
  └── BM25 + 同义词扩展
```

---

## 附录：关键论文

| 论文 | 年份 | 贡献 |
|-----|------|------|
| BM25 | 1994 | 经典词袋模型 |
| DPR | 2020 | 稠密检索开山之作 |
| ANCE | 2020 | 难负样本训练 |
| ColBERT | 2020 | 多向量延迟交互 |
| ColBERTv2 | 2021 | 高效索引 |
| SPLADE | 2021 | 学习稀疏检索 |
| PLAID | 2022 | ColBERT高效检索 |

---

*MelonEggLearn - 搜索排序专项笔记*
