# 02 CTR加权向量检索

> 核心论文：Learning to Rank for Advertising Creatives (Google, 2022)、EBR in Facebook Search (KDD 2020)、ANN-Benchmarks

---

## 一、为什么要CTR加权

### 1.1 纯语义相似度的局限

**核心矛盾：语义相似 ≠ 效果好**

```
场景示例：
  查询：「保湿护肤品，适合敏感肌」

  检索结果A（语义相似度=0.95）：
    创意文案：「水润保湿，敏感肌专属，告别干燥」
    历史CTR：0.002（远低于均值0.05）
    → 文案很准确，但CTR极低（可能设计差、图片差）

  检索结果B（语义相似度=0.78）：
    创意文案：「敏感肌的温柔守护，1000万人选择」
    历史CTR：0.12（是均值的2.4倍）
    → 语义匹配度低，但效果极好

结论：应该参考B，而不是A！
```

**长尾创意的高方差问题：**
- 语义描述丰富的小众创意，因曝光量不足，CTR估计不可靠（样本量=10，CTR=0.5可能只是运气）
- 反之，曝光百万次的CTR=0.08才是真实表现

**广告主个体差异：**
- 同一条护肤品创意，A广告主（面向高消费女性）CTR=0.15，B广告主（面向男性）CTR=0.01
- 需要**个性化CTR估计**，而非全局CTR

---

## 二、CTR加权向量的构建公式

### 2.1 方法一：直接线性加权

**公式：**

$$
\text{score}(q, d) = \alpha \cdot \cos(q, d) + (1-\alpha) \cdot \text{CTR}_{	ext{norm}}(d)
$$

其中 $\text{CTR}_{	ext{norm}}(d) = \frac{\text{CTR}(d)}{\text{CTR}_{	ext{max}}}$ 将CTR归一化到 [0,1]

**代码实现：**
```python
def direct_weighted_score(
    query_emb: np.ndarray,
    doc_emb: np.ndarray,
    ctr: float,
    ctr_max: float,
    alpha: float = 0.7
) -> float:
    cos_sim = np.dot(query_emb, doc_emb) / (
        np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
    )
    ctr_norm = ctr / ctr_max
    return alpha * cos_sim + (1 - alpha) * ctr_norm
```

**优缺点：**
| 项 | 评估 |
|----|------|
| 实现简单 | ✅ |
| 灵活调α | ✅ |
| CTR需要归一化 | ⚠️ 分布偏移时max值不稳定 |
| ANN索引失效 | ❌ 无法用近似最近邻，必须暴力搜索 |

---

### 2.2 方法二：CTR加权的向量空间（**主要方法**）

**核心思想：** 不改变检索逻辑，而是在向量空间中放大高CTR创意的"引力"

**加权公式：**

$$
\tilde{v}_d = v_d \cdot \sqrt{\frac{\text{CTR}(d)}{\bar{\text{CTR}}}}
$$

**检索时的等效得分：**

$$
\text{score}(q, d) = q \cdot \tilde{v}_d = \|q\| \cdot \|v_d\| \cdot \cos(q, v_d) \cdot \sqrt{\frac{\text{CTR}(d)}{\bar{\text{CTR}}}}
$$

**公式含义：**
- 若 $\text{CTR}(d) > \bar{\text{CTR}}$（高于均值），则权重 $> 1$，向量被放大，与查询的点积得分更高
- 若 $\text{CTR}(d) < \bar{\text{CTR}}$（低于均值），则权重 $< 1$，向量被缩小，得分更低
- 几何意义：高CTR创意在向量空间中"更靠近"所有查询点

**完整实现：**
```python
import numpy as np
from typing import List, Dict

class CTRWeightedVectorStore:
    def __init__(self, index, mean_ctr: float = 0.05):
        self.index = index          # FAISS/Milvus索引
        self.mean_ctr = mean_ctr
    
    def build_weighted_index(
        self,
        creatives: List[Dict],
        embedding_model
    ):
        """构建CTR加权向量索引"""
        weighted_embeddings = []
        
        for creative in creatives:
            # 1. 编码原始向量
            v = embedding_model.encode(creative["text"])
            
            # 2. 计算CTR权重
            ctr_smooth = self._smooth_ctr(
                creative["clicks"],
                creative["impressions"]
            )
            weight = np.sqrt(ctr_smooth / self.mean_ctr)
            
            # 3. 向量加权（放大/缩小）
            v_weighted = v * weight
            weighted_embeddings.append(v_weighted)
        
        # 4. 建立ANN索引
        embeddings_matrix = np.array(weighted_embeddings).astype('float32')
        self.index.add(embeddings_matrix)
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Dict]:
        """检索时直接计算点积（内积 = cos_sim * ctr_weight）"""
        # 注意：使用内积搜索，不是余弦距离
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            top_k
        )
        return indices[0], distances[0]
    
    def _smooth_ctr(
        self,
        clicks: int,
        impressions: int,
        alpha: float = 5.0,   # Beta先验参数
        beta: float = 95.0
    ) -> float:
        """贝叶斯平滑CTR（见第三节）"""
        return (clicks + alpha) / (impressions + alpha + beta)
```

**工程优势：** 向量空间加权后仍然可以使用 FAISS InnerProduct / Milvus IP距离做ANN搜索，保持检索效率

---

### 2.3 方法三：CTR作为后处理重排

**流程：**
```
语义检索 top-N（N=100）
    ↓
CTR加权重排（全量计算N条的加权分）
    ↓
返回 top-K（K=10）
```

**重排公式：**

$$
\text{final}_{	ext{score}}(q, d) = \text{sem}_{	ext{score}}(q, d)^{\gamma} \cdot \text{CTR}_{	ext{smooth}}(d)^{1-\gamma}
$$

使用乘法（而非加法）的好处：语义完全不相关的创意，无论CTR多高，分数也接近0

**代码：**
```python
def ctr_rerank(
    candidates: List[Dict],  # 包含 sem_score 和 ctr_smooth
    gamma: float = 0.6       # 语义权重
) -> List[Dict]:
    for c in candidates:
        c["final_score"] = (
            c["sem_score"] ** gamma *
            c["ctr_smooth"] ** (1 - gamma)
        )
    return sorted(candidates, key=lambda x: x["final_score"], reverse=True)
```

### 2.4 三种方法对比

| 方法 | ANN兼容 | 实时更新CTR | 实现复杂度 | 适用场景 |
|------|---------|------------|-----------|---------|
| 直接线性加权 | ❌（暴力搜索） | 🔄 需重新检索 | 低 | 小规模（<10万） |
| 向量空间加权 | ✅（InnerProduct） | 需重建索引 | 中 | 百万级，CTR变化慢 |
| 后处理重排 | ✅（先语义ANN） | ✅（只更新重排层） | 低 | **推荐：生产环境首选** |

**生产推荐：** 向量空间加权（离线索引） + 后处理重排（在线实时CTR）的混合方案

---

## 三、CTR平滑处理

### 3.1 问题背景

```
创意A：曝光10次，点击3次 → CTR = 30%（样本太少，不可信）
创意B：曝光100万次，点击5万次 → CTR = 5%（可信）

直接用CTR排序：A会排在B前面，结果灾难性
```

### 3.2 贝叶斯平滑（Beta-Binomial）

**公式：**

$$
\text{CTR}_{	ext{smooth}} = \frac{\text{click} + \alpha}{\text{impression} + \alpha + \beta}
$$

其中 $\alpha, \beta$ 是全局Beta分布的参数，由历史数据MLE估计：

$$
\alpha = \bar{\text{CTR}} \cdot \left(\frac{\bar{\text{CTR}}(1-\bar{\text{CTR}})}{\text{Var}(\text{CTR})} - 1\right)
$$

$$
\beta = (1 - \bar{\text{CTR}}) \cdot \left(\frac{\bar{\text{CTR}}(1-\bar{\text{CTR}})}{\text{Var}(\text{CTR})} - 1\right)
$$

**代码实现：**
```python
import numpy as np
from scipy.optimize import minimize

def estimate_beta_params(ctrs: np.ndarray) -> tuple:
    """从历史CTR数据估计Beta分布参数"""
    mean_ctr = ctrs.mean()
    var_ctr = ctrs.var()
    
    concentration = mean_ctr * (1 - mean_ctr) / var_ctr - 1
    alpha = mean_ctr * concentration
    beta = (1 - mean_ctr) * concentration
    
    return alpha, beta

def bayesian_smooth_ctr(
    clicks: int,
    impressions: int,
    alpha: float,  # 由estimate_beta_params得到
    beta: float
) -> float:
    """
    贝叶斯平滑CTR
    
    效果：
    - 曝光少时：向全局均值收缩（α/(α+β)）
    - 曝光多时：接近实际CTR
    """
    return (clicks + alpha) / (impressions + alpha + beta)

# 示例
historical_ctrs = np.array([0.03, 0.05, 0.08, 0.02, 0.12, ...])
alpha, beta = estimate_beta_params(historical_ctrs)

# 新创意：10次曝光，3次点击
smooth_ctr = bayesian_smooth_ctr(3, 10, alpha, beta)
print(f"原始CTR: 0.300, 平滑CTR: {smooth_ctr:.4f}")
# 输出：原始CTR: 0.300, 平滑CTR: 0.0523（向均值收缩）
```

### 3.3 Wilson置信区间下界（保守估计）

**公式：**

$$
\text{lower} = \frac{\hat{p} + \frac{z^2}{2n} - z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}
$$

其中 $\hat{p} = \text{click/impression}$，$z=1.96$（95%置信度），$n=\text{impression}$

**特点：** 比贝叶斯平滑更保守，对低曝光创意惩罚更重，适合风险厌恶场景

```python
from scipy.stats import norm

def wilson_lower_bound(
    clicks: int,
    impressions: int,
    confidence: float = 0.95
) -> float:
    if impressions == 0:
        return 0.0
    
    z = norm.ppf(1 - (1 - confidence) / 2)
    p_hat = clicks / impressions
    n = impressions
    
    numerator = (p_hat + z**2 / (2*n)
                 - z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)))
    denominator = 1 + z**2 / n
    
    return numerator / denominator
```

**贝叶斯 vs Wilson 选择：**
- 贝叶斯平滑：当有历史先验分布时使用，更准确
- Wilson下界：当需要保守估计（避免高估新创意）时使用

---

## 四、向量库更新策略

### 4.1 增量 vs 全量重建

**百万级创意库面临的挑战：**
- 每天新增约1万条创意（向量化 + 插入）
- CTR权重每小时变化（需要更新向量）
- 不能停服（广告系统7×24）

**Milvus 增量更新策略：**

```python
class IncrementalUpdateStrategy:
    """
    增量更新策略
    - 日常：增量插入新创意
    - 每周：全量重建防止索引退化
    """
    
    def daily_update(self, new_creatives: List[Dict]):
        """每日增量插入（耗时 < 1分钟）"""
        embeddings = self.embed_batch(new_creatives)
        ctr_weights = [self.smooth_ctr(c) for c in new_creatives]
        weighted_embs = [e * np.sqrt(w/self.mean_ctr)
                        for e, w in zip(embeddings, ctr_weights)]
        
        self.milvus_client.insert(
            collection_name="creatives",
            data=weighted_embs
        )
    
    def hourly_ctr_update(self, ctr_updates: Dict[str, float]):
        """每小时更新CTR权重（指数移动平均）"""
        for creative_id, new_ctr in ctr_updates.items():
            old_emb = self.milvus_client.get_embedding(creative_id)
            old_ctr = self.ctr_store.get(creative_id)
            
            # 指数移动平均
            ema_ctr = 0.1 * new_ctr + 0.9 * old_ctr
            
            # 更新向量权重
            new_weight = np.sqrt(ema_ctr / self.mean_ctr)
            old_weight = np.sqrt(old_ctr / self.mean_ctr)
            
            # 重新缩放向量
            new_emb = old_emb / old_weight * new_weight
            self.milvus_client.upsert(creative_id, new_emb)
    
    def weekly_rebuild(self):
        """每周全量重建（耗时约2小时，蓝绿部署）"""
        # 1. 新集合构建
        new_collection = self.build_full_index()
        # 2. 流量切换（不停服）
        self.milvus_client.swap_collection(new_collection)
```

**CTR权重更新频率选择：**
| 更新频率 | 优点 | 缺点 |
|---------|------|------|
| 实时 | CTR最准确 | 向量库频繁写入，影响查询性能 |
| 每小时（推荐） | 延迟可接受，性能影响小 | 热门活动期间CTR滞后 |
| 每天 | 索引最稳定 | CTR权重明显滞后 |

---

## 五、系统架构

```
【离线流水线】（每天运行）
商品数据库 → 文本向量化(BGE/bge-m3) → CTR平滑计算
                                              ↓
点击日志 → CTR汇总统计 → 贝叶斯平滑 → 加权向量 → Milvus索引

【在线检索流水线】（毫秒级）
用户请求 → Query构造 → Embedding服务(3ms)
                              ↓
                        Milvus ANN检索(8ms, k=100)
                              ↓
                        CTR实时重排(10ms, 从100选K)
                              ↓
                        返回 top-K 创意
```

---

## 六、常见考点 Q&A

**Q1：CTR加权会不会导致马太效应（热门越来越热）？**

A：会，这是典型的正反馈问题。初始高CTR创意被更频繁推荐 → 更多曝光 → CTR进一步提升 → 挤压新创意空间。解决方案：(1) UCB探索策略——给低曝光创意额外加分，鼓励探索；(2) 曝光衰减——对曝光次数超过阈值的创意降低权重（避免"审美疲劳"）；(3) 时效加权——近期CTR权重 > 历史CTR权重，避免古老爆款长期占据。可以在重排公式中加入`novelty_bonus = 1 / (1 + log(1 + impressions/1e6))`。

**Q2：如何处理CTR为0的新创意？**

A：三层策略。首先，CTR为0 ≠ 效果差，可能只是还未曝光。用贝叶斯平滑：CTR_smooth = α/(α+β)，即全局均值的收缩估计（约等于全局CTR，不是0）。其次，对新创意额外加UCB探索加分：`score += β * sqrt(log(total_impressions) / (creative_impressions + 1))`，β是探索系数（0.01~0.1）。最后，设置曝光门槛：曝光 < 100次的创意使用全局CTR代替，不参与CTR加权，等积累足够样本再启用真实CTR。

**Q3：α参数（语义 vs CTR权重）怎么调？**

A：α不应该是全局固定值，而应根据业务场景动态调整。高语义要求场景（如专业工具广告）：α偏高（0.8），确保文案准确性；高转化要求场景（如电商促销）：α偏低（0.4~0.5），更看重历史效果。调参方法：用离线评估——按(语义得分, CTR)构建二维评估矩阵，找到CTR提升最大化且语义不下降的α值。线上A/B测试验证，每个广告主类型单独调α最佳。

**Q4：Milvus vs Faiss vs Pinecone 怎么选？**

A：三者定位不同。Faiss是纯粹的向量检索库（C++），无服务端，适合嵌入进程内部，性能最强，但运维复杂、不支持实时更新；Milvus是开源向量数据库，支持实时插入/删除/过滤（标量+向量联合查询），适合百万级以上需要运维管理的生产环境；Pinecone是托管云服务，无运维，但数据出境+按用量收费，数据隐私要求高的公司不适用。广告创意场景推荐Milvus：需要实时更新CTR权重、需要品类过滤（WHERE category=XX）、需要开源自部署。

**Q5：HNSW索引的M值和ef_search怎么设置？**

A：HNSW的M值控制图的连接度，ef_search控制搜索范围。经验公式：M=16~32（向量维度768用32，维度256用16）。ef_search是查询时精度与速度的权衡：ef_search=64时，召回率95%，延迟约8ms（推荐用于在线服务）；ef_search=128时，召回率99%，延迟约15ms（用于离线评估）。百万量级建议：构建时ef_construction=200（慢，但索引质量好），查询时ef_search=64（快，召回率满足业务需求）。

**Q6：向量空间加权和重排加权哪个更好？**

A：各有优劣。向量空间加权一次索引搞定，ANN直接返回融合得分，延迟最低（节省重排时间10ms），但CTR更新需要修改向量，索引更新成本高；重排加权工程最灵活，CTR变更只需更新重排层，不触碰向量索引，适合CTR频繁变化的场景，代价是额外10ms重排延迟。生产推荐混合策略：向量空间加权使用T-1日（昨天）的平滑CTR（慢变量，一天更新一次），重排加权使用实时CTR（快变量，每小时更新），两层叠加效果最好。

**Q7：如何评估CTR加权的效果（离线评估方法）？**

A：用历史数据构建评估集。将某时间段内的高CTR创意作为"金标准答案"，然后用不同检索策略（纯语义、CTR加权）检索，计算：(1) Precision@K：检索top-K中有多少历史高CTR创意；(2) NDCG@K（归一化折损累计增益）：用CTR值作为相关性分数，评估排序质量；(3) 覆盖率：不同品类的高CTR创意被覆盖比例。注意：用留出法，用T-14~T-7天的数据预测T-7~T的高CTR，避免数据穿越。

**Q8：多目标检索：既要CTR高又要多样性怎么做？**

A：多目标优化的经典MMR（最大边际相关性）变体。将MMR中的相关性项替换为CTR加权得分：`score(d) = λ * CTR_weighted(q, d) - (1-λ) * max_sim(d, S)`，其中S是已选集合。λ控制CTR vs 多样性权衡（λ=0.7效果较好）。另一种方法是先用CTR加权检索top-50，再用聚类（K-means, K=5）按簇均匀抽样，每簇取top-1~3，保证类型多样性同时确保每条都有历史效果背书。

---

*参考文献：*
- *Learning to Rank for Advertising Creatives, Google (2022)*
- *Embedding-based Retrieval in Facebook Search, Huang et al., KDD 2020*
- *ANN-Benchmarks, Aumuller et al., Information Systems 2020*
