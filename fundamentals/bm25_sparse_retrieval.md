# 稀疏检索：从 TF-IDF 到 BM25 到神经稀疏

> 标签：#BM25 #TFIDF #稀疏检索 #SPLADE #混合检索 #RRF #Elasticsearch #广告搜索

---

## 1. TF-IDF 完整推导

### 1.1 词频（TF）

**直觉**：一个词在文档中出现次数越多，该词对这篇文档越重要。

**定义**：词 $t$ 在文档 $d$ 中的词频（Term Frequency）：

$$
\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}
$$

- $f_{t,d}$：词 $t$ 在文档 $d$ 中的原始出现次数
- 分母：文档 $d$ 中所有词的出现次数之和（归一化，消除长文档偏差）

**TF 的变体**：

| 变体 | 公式 | 特点 |
|------|------|------|
| 原始词频 | $f_{t,d}$ | 简单，但受文档长度影响 |
| 归一化 TF | $\frac{f_{t,d}}{\max_{t'} f_{t',d}}$ | 除以最高频词，缩放到 [0,1] |
| 对数 TF | $1 + \log(f_{t,d})$ | 抑制高频词的超线性增长 |
| 布尔 TF | $\mathbb{1}\left[f_{t,d} > 0\right]$ | 只看是否出现，不看次数 |

### 1.2 逆文档频率（IDF）

**直觉**：如果一个词在很多文档中都出现（如"的"、"是"），它对区分文档没有帮助；如果一个词只在少数文档中出现，它对区分文档很有价值。

**定义**：词 $t$ 的逆文档频率（Inverse Document Frequency）：

$$
\text{IDF}(t) = \log \frac{N}{|\{d : t \in d\}|}
$$

- $N$：文档集合的总数量
- $|\{d : t \in d\}|$：包含词 $t$ 的文档数量

**IDF 的作用**：出现在 10% 文档中的词 $\text{IDF} = \log(10) \approx 2.3$；出现在 50% 文档中的词 $\text{IDF} = \log(2) \approx 0.69$；出现在 99% 文档中的词 $\text{IDF} \approx 0.01$（近零，几乎无贡献）。

**IDF 平滑**（防止除以零）：

$$
\text{IDF}(t) = \log \frac{N + 1}{|\{d : t \in d\}| + 1} + 1
$$

### 1.3 TF-IDF 组合

$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
$$

**查询-文档相关性**：

$$
\text{Score}(q, d) = \sum_{t \in q} \text{TF-IDF}(t, d)
$$

### 1.4 TF-IDF 的向量空间模型

每个文档表示为 $|V|$ 维稀疏向量（$V$ 是词汇表大小），第 $i$ 维是词 $v_i$ 的 TF-IDF 值。查询同样表示为稀疏向量，通过余弦相似度计算相关性：

$$
\text{cosine\_sim}(q, d) = \frac{q \cdot d}{\|q\| \|d\|}
$$

---

## 2. BM25 改进点

### 2.1 TF-IDF 的两个不足

1. **TF 超线性增长问题**：词 $t$ 出现 100 次的文档，TF 是出现 10 次的文档的 10 倍，但实际信息量增加远没有这么多（边际效益递减）。
2. **文档长度偏差**：长文档中同一个词出现更多次，但并不一定更相关；短文档中每次出现意味着更高的词汇密度。

### 2.2 BM25 公式

$$
\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f_{t,d} \cdot (k_1 + 1)}{f_{t,d} + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
$$

其中：
- $k_1$：词频饱和参数（通常 1.2-2.0）
- $b$：文档长度归一化参数（通常 0.75）
- $|d|$：文档长度（词数）
- $\text{avgdl}$：文档集合的平均文档长度

**BM25 的 IDF 公式**（更稳定的变体）：

$$
\text{IDF}(t) = \log \frac{N - df_t + 0.5}{df_t + 0.5}
$$

其中 $df_t = |\{d : t \in d\}|$ 是文档频率。

### 2.3 BM25 对 TF-IDF 的改进逐步分析

**改进1：TF 饱和**

BM25 中 TF 部分 $\frac{f_{t,d} \cdot (k_1 + 1)}{f_{t,d} + k_1}$ 分析：

- 当 $f_{t,d} \to \infty$：趋近于 $(k_1 + 1)$（饱和上限）
- 当 $f_{t,d} = 0$：等于 0
- 当 $f_{t,d} = k_1$：等于 $\frac{k_1(k_1+1)}{2k_1} = \frac{k_1+1}{2}$（半饱和点）

```python
import numpy as np
import matplotlib.pyplot as plt

# 可视化 TF 饱和效果
tf_values = np.linspace(0, 20, 100)
k1 = 1.5

# TF-IDF 的 TF（线性增长）
tfidf_tf = tf_values

# BM25 的 TF（饱和增长）
bm25_tf = tf_values * (k1 + 1) / (tf_values + k1)

# BM25 饱和上限
saturation = k1 + 1  # = 2.5

# tf=1时 BM25 得分 = (k1+1)/(1+k1) = 1，与 TF-IDF 相比：更快达到高值
```

**改进2：文档长度归一化**

归一化因子：$1 - b + b \cdot \frac{|d|}{\text{avgdl}}$

- $|d| = \text{avgdl}$ 时，等于 1（标准文档，无调整）
- $|d| > \text{avgdl}$ 时，> 1（长文档，分母增大，TF 贡献减小）
- $|d| < \text{avgdl}$ 时，< 1（短文档，分母减小，TF 贡献增大）
- $b = 0$：无长度归一化；$b = 1$：完全归一化（等价于按词密度）

### 2.4 BM25 参数调优

```python
from rank_bm25 import BM25Okapi

# 构建 BM25 索引
corpus = [doc.split() for doc in documents]
bm25 = BM25Okapi(corpus, k1=1.5, b=0.75)

# 检索
query = "广告 点击率 预估"
tokenized_query = query.split()
scores = bm25.get_scores(tokenized_query)
top_k = np.argsort(scores)[::-1][:10]

# 参数调优建议
# k1=1.2: 短文档集合（如广告创意）
# k1=2.0: 长文档集合（如新闻、论文）
# b=0.75: 通用默认值（大多数场景）
# b=0.0: 文档长度差异不大时（如固定格式文档）
```

---

## 3. 稠密检索 vs 稀疏检索

### 3.1 两种范式的本质差异

| 维度 | 稀疏检索（BM25）| 稠密检索（DPR/BERT）|
|------|----------------|---------------------|
| 向量类型 | 高维稀疏（|V| 维，通常 99%+ 为 0）| 低维稠密（128-768 维，每维非零）|
| 索引结构 | 倒排索引（词→文档列表）| ANN 索引（HNSW/IVF）|
| 词汇匹配 | 精确词匹配 | 语义相似（"汽车"≈"轿车"）|
| 可解释性 | 强（知道哪些词匹配）| 弱（黑盒向量）|
| 训练数据 | 不需要（无监督）| 需要大量标注 |
| 词汇表外词（OOV）| 完全无法处理 | 通过 subword 处理 |
| 查询延迟 | 极快（倒排索引 O(df)）| 较快（ANN 近似）|

### 3.2 混合检索：RRF 融合

**Reciprocal Rank Fusion（互惠秩融合）**：将稀疏和稠密检索的结果融合：

$$
\text{RRF\_score}(d | q) = \sum_{r \in \{\text{sparse}, \text{dense}\}} \frac{1}{k + r_r(d)}
$$

其中：
- $r_r(d)$：文档 $d$ 在检索系统 $r$ 中的排名（从 1 开始）
- $k$：平滑参数（通常 60）

**RRF 的优点**：
- 不需要对两个系统的分数做归一化（分数量级可以差异极大）
- 只关注排名，对 outlier 分数鲁棒
- 简单有效，实测通常优于任意单一系统

```python
def reciprocal_rank_fusion(sparse_results, dense_results, k=60):
    """
    sparse_results: [(doc_id, score), ...] 按分数降序
    dense_results: [(doc_id, score), ...] 按分数降序
    """
    scores = {}
    
    # 稀疏检索贡献
    for rank, (doc_id, _) in enumerate(sparse_results, 1):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
    
    # 稠密检索贡献
    for rank, (doc_id, _) in enumerate(dense_results, 1):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
    
    # 按融合得分排序
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

---

## 4. 神经稀疏检索（SPLADE）

### 4.1 SPLADE 的核心思想

**目标**：保留稀疏检索的高效性（可用倒排索引），同时引入语义理解能力。

**方案**：用 BERT 为词汇表中每个词预测权重，非零权重的词构成稀疏向量：

$$
w_{i}^d = \log(1 + \text{ReLU}(E_d^T \cdot W_{MLM} + b)_{v_i})
$$

其中：
- $E_d$：文档通过 BERT 得到的 [CLS] 或 token 级表示
- $W_{MLM}$：BERT Masked LM 头的投影矩阵（将 hidden 映射到词汇表大小）
- ReLU：确保权重非负
- $\log(1 + \cdot)$：抑制高值，确保稀疏性

**关键性质**：
1. 输出是 $|V|$ 维稀疏向量（如 $|V| = 30K$，通常只有 100-200 维非零）
2. 仍可用倒排索引：非零维度视为"词"，构建倒排
3. 语义扩展："汽车"可能激活"车"、"轿车"、"automobile"等相关词

### 4.2 SPLADE 训练

训练目标（对比学习）：

$$
L = L_{contrastive}(q, d^+, d^-) + \lambda_q \cdot L_{FLOPS}(q) + \lambda_d \cdot L_{FLOPS}(d)
$$

- $L_{contrastive}$：InfoNCE 对比损失，使正例得分高于负例
- $L_{FLOPS}$：正则化项，鼓励稀疏（减少非零维度数量）

$$
L_{FLOPS} = \sum_{i=1}^{|V|} \left(\frac{1}{N} \sum_{j=1}^N |w_i^j|\right)^2
$$

**FLOPS 正则的直觉**：若某维度的权重均值很大，该维度对所有文档都重要（类似高频词），通过正则使其趋向 0，相当于 TF-IDF 的 IDF 惩罚。

### 4.3 与 BM25 的对比

| 维度 | BM25 | SPLADE |
|------|------|--------|
| 词汇泛化 | 无（精确匹配）| 有（语义扩展）|
| 训练数据 | 不需要 | 需要 |
| 索引结构 | 倒排索引 | 倒排索引（兼容）|
| 可解释性 | 强 | 中（可查看非零词）|
| 向量维度 | |V| 维稀疏 | |V| 维稀疏 |
| 检索速度 | 极快 | 快（同倒排索引）|
| 效果（BEIR）| 基准 | +10-20% |

---

## 5. 工业应用：广告搜索

### 5.1 广告 Query 理解中的稀疏检索

在广告搜索场景（如关键词广告），用户搜索 Query → 匹配相关广告：

**挑战**：
- 广告关键词由广告主手动设置，可能与用户 Query 用词不同
- "买手机" vs "购买智能手机" → 应该匹配
- 需要在精确匹配和语义匹配之间平衡

**方案**：BM25 + 稠密检索混合：
1. BM25 处理精确词匹配（快速过滤，高精确率）
2. 稠密检索处理语义扩展（提高召回率）
3. RRF 融合，再由精排模型精细排序

### 5.2 Elasticsearch 的 BM25 实现

```python
from elasticsearch import Elasticsearch

es = Elasticsearch()

# 创建索引，配置 BM25 参数
es.indices.create(index='ads', body={
    "settings": {
        "similarity": {
            "custom_bm25": {
                "type": "BM25",
                "k1": 1.2,      # 词频饱和参数
                "b": 0.75,      # 长度归一化参数
                "discount_overlaps": True
            }
        }
    },
    "mappings": {
        "properties": {
            "ad_title": {
                "type": "text",
                "similarity": "custom_bm25",
                "analyzer": "ik_max_word"  # 中文分词
            },
            "ad_keywords": {
                "type": "text",
                "similarity": "custom_bm25"
            }
        }
    }
})

# 检索广告
query = "买手机"
result = es.search(index='ads', body={
    "query": {
        "multi_match": {
            "query": query,
            "fields": ["ad_title^2", "ad_keywords"],  # 标题权重更高
            "type": "most_fields"
        }
    },
    "size": 100
})
```

### 5.3 与向量检索的互补性

在广告搜索中，稀疏和稠密检索各有优势：

**稀疏检索（BM25）优势场景**：
- 品牌词精确匹配："Nike 运动鞋" → 必须匹配 Nike 广告
- 产品型号："iPhone 15 Pro" → 精确型号检索
- 地域限定词："北京 律师" → 必须包含"北京"

**稠密检索优势场景**：
- 口语化 Query："便宜好用的手机" → 扩展到"高性价比智能手机"
- 意图理解："减肥" → 扩展到健康食品、健身课程广告
- 跨语言：英文 Query 匹配中文广告

**工业实践**：通常两路并行，候选合并后进入精排。BM25 负责保证精确匹配不漏召，稠密检索负责语义扩展提升泛化。

---

## 6. 面试考点

### Q1：BM25 的 k1 和 b 参数分别控制什么？如何调优？

k1 控制词频饱和速度：k1 大（如 2.0），词频效益持续时间长，适合长文档（新闻、论文）；k1 小（如 1.0），词频很快饱和，适合短文档（标题、广告文案）。b 控制长度归一化强度：b=0 不归一化（长文档有天然优势），b=1 完全按词密度（消除文档长度影响），b=0.75 是经验最优折中。调优方法：在验证集上网格搜索，以 MRR@10 或 NDCG@10 为优化目标。

### Q2：倒排索引的数据结构是什么？查询时间复杂度？

倒排索引是词到文档列表的映射：{词: [(doc_id, tf, position), ...]}。存储时按词排序（可用哈希表或 B-Tree），每个词对应的文档列表按 doc_id 排序（便于合并操作）。查询时间复杂度：O(|df_t|) 检索单词，O(|df_{t1}| + |df_{t2}|) 合并两个词（AND/OR 操作），总体 O(sum of df for query terms)，远优于全文扫描 O(N)。

### Q3：RRF 为什么比直接归一化分数融合更鲁棒？

直接分数融合（$s_{fusion} = \alpha \cdot s_{sparse} + (1-\alpha) \cdot s_{dense}$）的问题：BM25 分数可能是 20-50，余弦相似度是 0.7-0.9，量级不同，$\alpha$ 很难设置。另外，某些结果的 BM25 分数异常高（outlier），会主导融合结果。RRF 只用排名信息，对量级和 outlier 完全鲁棒，且研究表明 $k=60$ 对大多数场景都近似最优，无需调参。

### Q4：SPLADE 中的 FLOPS 正则是什么含义？

FLOPS 正则统计每个词汇表维度在训练集上的平均激活值，并惩罚那些被广泛激活的维度（类比 IDF 惩罚高频词）。这迫使模型只激活真正有区分度的词，产生更稀疏的表示。稀疏性很重要：若每个文档激活 10000 个词，倒排索引大小会比 BM25 大 100×，失去了稀疏检索的优势。FLOPS 正则将平均非零维度数控制在 50-200，保持可接受的索引大小。

### Q5：BM25 的最大缺陷是什么？有哪些解决方案？

最大缺陷：词汇不匹配（Vocabulary Mismatch），无法处理同义词、上下位词等语义关系。解决方案：(1) 查询扩展（Query Expansion）：用词典或模型扩展 Query 中的词（如添加同义词），传统方法；(2) 文档扩展（DocT5query）：用 T5 预测该文档可能被哪些 Query 检索，将生成的 Query 词添加到文档索引；(3) 神经稀疏（SPLADE）：端到端学习语义扩展；(4) 混合检索：BM25 + 稠密检索互补。

### Q6：如何评估检索系统的效果？

主要指标：(1) Recall@K：前 K 个结果中包含相关文档的比例，衡量召回能力；(2) Precision@K：前 K 个结果中相关文档的比例，衡量精确度；(3) MRR（Mean Reciprocal Rank）：第一个相关结果排名的倒数均值，强调最高排名结果；(4) NDCG@K（Normalized Discounted Cumulative Gain）：考虑多级相关性和位置折扣，是最常用的综合指标。广告场景：常用 Recall@100（精排输入量）和 Hit Rate（Query 是否至少召回一个相关广告）。

### Q7：为什么广告搜索需要同时维护稀疏和稠密索引？

单一索引各有盲区：(1) 只用 BM25：无法处理语义扩展，用户"帮我找一双跑步的鞋"无法匹配"Nike 跑步训练鞋"；(2) 只用稠密索引：无法保证精确品牌/型号匹配，"iPhone 15 Pro Max"可能被"安卓手机"的向量污染。双路召回后，精排模型（通常是 Cross-Encoder 或精细 MLP）再基于更丰富的特征精确排序。工程实现时，两路并行查询，延迟取最大值（非串行），通常控制在 20ms 以内。

---

## 参考资料

- Robertson & Zaragoza. "The Probabilistic Relevance Framework: BM25 and Beyond" (2009)
- Formal et al. "SPLADE: Sparse Lexical and Expansion Model for First Stage Retrieval" (2021)
- Cormack et al. "Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods" (RRF, 2009)
- Nogueira et al. "Document Expansion by Query Prediction" (DocT5query, 2019)
- Karpukhin et al. "Dense Passage Retrieval for Open-Domain Question Answering" (DPR, 2020)
