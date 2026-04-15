# 01 动态K值自适应检索

> 核心论文：FLARE (2023)、Self-RAG (2023)、Adaptive-RAG (2024)、Dynamic RAG (EMNLP 2023)

---

## 一、为什么需要动态K

### 1.1 固定K的根本问题

传统RAG系统使用固定的检索数量K（如K=5或K=10），存在以下缺陷：

**K太小的问题（信息不足）：**
- LLM缺乏足够的参考多样性，容易生成模板化内容
- 对于复杂产品（多卖点、多人群），5条参考无法覆盖全面
- 在创意生成中表现为：所有输出风格单一，千篇一律

**K太大的问题（冗余干扰）：**
- 冗余信息混入上下文，LLM需要"忽略"噪声，增加推理难度
- Token消耗剧增：K=20时 context 可能多出 2000+ tokens
- 在线延迟上升：更多token → 更慢的LLM推理
- 在创意生成中表现为：LLM陷入过拟合历史创意，创新性下降

**不同查询的信息密度差异：**
```
热门品类（护肤品）：
  - 历史创意 > 10万条，语义空间密集
  - K=5 已足够代表性，K=20 严重冗余

长尾品类（小众乐器配件）：
  - 历史创意 < 100条，语义空间稀疏
  - K=5 可能全部相似，K=20 才能覆盖有限多样性
```

### 1.2 广告场景的特殊挑战

**百万级创意库，密度分布极度不均匀：**
```python
# 典型的创意库品类分布（长尾分布）
category_counts = {
    "护肤品": 150000,      # 占总量15%
    "服装": 120000,        # 占总量12%
    "家电": 80000,         # ...
    "宠物用品": 5000,      # 稀疏品类
    "小众运动器材": 200,   # 极度稀疏
    "新品类X": 3,          # 冷启动
}
```

**季节性密度骤变（双十一期间）：**
- 促销活动文案大量涌入，某些品类密度在1周内增加10倍
- 固定K=5可能全检索到活动文案，失去日常参考多样性

**新品类冷启动：**
- 新品牌/品类上线时，历史创意为0~10条
- 此时K=5可能超过实际可用量，需要降级策略

---

## 二、动态K的核心算法

### 2.1 方法一：基于置信度的动态停止（FLARE思想）

**论文来源：** FLARE: Active Retrieval Augmented Generation (Jiang et al., 2023, arXiv:2305.06983)

**FLARE核心思想：**
FLARE不是一次性检索K条，而是在生成过程中**主动判断何时需要检索、需要多少**。核心机制：
1. 用低置信度token（概率 < θ）作为触发信号
2. 用这些token构造检索query
3. 用检索结果更新上下文，继续生成

**广告场景适配的动态停止算法：**

```python
import numpy as np
from typing import List, Tuple

def dynamic_k_flare(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,  # shape: [N, D]
    candidate_scores: np.ndarray,      # 向量相似度得分
    confidence_threshold: float = 0.05, # top-1和top-k分差阈值
    min_k: int = 3,
    max_k: int = 20
) -> int:
    """
    基于置信度的动态K确定
    停止条件：当增加一条检索结果带来的边际信息增益小于阈值时停止
    """
    sorted_scores = np.sort(candidate_scores)[::-1]
    
    current_k = min_k
    while current_k < max_k:
        # 计算当前k和k+1的得分差
        if current_k < len(sorted_scores) - 1:
            marginal_gain = sorted_scores[current_k] - sorted_scores[current_k + 1]
            
            # 停止条件：边际增益小于阈值（信息增量可忽略）
            if marginal_gain < confidence_threshold:
                break
        
        # 计算diversity score（避免冗余）
        top_k_embeddings = candidate_embeddings[:current_k]
        diversity = compute_diversity(top_k_embeddings)
        
        # 如果多样性足够高，停止增加K
        if diversity > 0.7:  # 平均两两距离 > 0.7
            break
            
        current_k += 1
    
    return current_k

def compute_diversity(embeddings: np.ndarray) -> float:
    """计算检索结果集的多样性（平均两两余弦距离）"""
    if len(embeddings) <= 1:
        return 1.0
    
    # 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    
    # 余弦相似度矩阵
    sim_matrix = normalized @ normalized.T
    
    # 取上三角（排除对角线）
    k = len(embeddings)
    upper_tri = sim_matrix[np.triu_indices(k, k=1)]
    
    # 多样性 = 1 - 平均相似度
    diversity = 1.0 - upper_tri.mean()
    return diversity
```

**置信度公式：**

$$
\text{confidence}_{	ext{score}}(k) = \alpha \cdot \text{sim}(q, d_k) + (1-\alpha) \cdot \text{diversity}(D_{1:k})
$$

停止条件：

$$
\text{stop if} \quad \text{sim}(q, d_1) - \text{sim}(q, d_k) > \theta
$$

其中 θ 为超参数（推荐0.05~0.1）

---

### 2.2 方法二：基于信息密度的K估计

**核心思想（Dynamic RAG, EMNLP 2023）：**
先用小K检索，通过测量结果集的**信息熵/多样性**判断是否需要更多参考。

**算法步骤：**

```python
def density_based_dynamic_k(
    query_embedding: np.ndarray,
    retriever,
    min_k: int = 3,
    max_k: int = 20,
    high_similarity_threshold: float = 0.85,  # 同质化阈值
    low_similarity_threshold: float = 0.40,   # 稀疏阈值
    step: int = 3
) -> Tuple[int, List]:
    """
    基于信息密度的动态K估计
    
    Returns:
        k: 最终选定的K值
        results: 检索结果列表
    """
    current_k = min_k
    results = retriever.search(query_embedding, top_k=current_k)
    
    while current_k < max_k:
        # 计算当前结果集的平均内部相似度
        embeddings = np.array([r.embedding for r in results])
        avg_similarity = compute_avg_pairwise_similarity(embeddings)
        
        if avg_similarity > high_similarity_threshold:
            # 结果同质化：需要更多多样性，增加K
            current_k = min(current_k + step, max_k)
            results = retriever.search(query_embedding, top_k=current_k)
            
        elif avg_similarity < low_similarity_threshold:
            # 结果稀疏：说明该品类信息少，但已覆盖全部可用信息
            # 检查是否还有更多结果可用
            if len(results) < current_k:
                # 已经把全部结果都检出了，停止
                break
            break
            
        else:
            # 正常密度区间（0.40 ~ 0.85），当前K合适
            break
    
    return current_k, results

def compute_avg_pairwise_similarity(embeddings: np.ndarray) -> float:
    """计算向量集合的平均两两余弦相似度"""
    if len(embeddings) <= 1:
        return 1.0
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    sim_matrix = normalized @ normalized.T
    k = len(embeddings)
    # 取上三角平均（不含对角线）
    indices = np.triu_indices(k, k=1)
    return sim_matrix[indices].mean()
```

**信息密度公式（余弦相似度矩阵）：**

$$
\text{density}(D_k) = \frac{2}{k(k-1)} \sum_{i < j} \cos(d_i, d_j)
$$

决策规则：
- $\text{density} > 0.85$ → 同质化，增大K（或切换多样性策略）
- $0.40 \leq \text{density} \leq 0.85$ → 正常，保持当前K
- $\text{density} < 0.40$ → 信息稀疏（新品类），增大K或切换策略

---

### 2.3 方法三：Self-RAG的检索决策

**论文来源：** Self-RAG (Asai et al., 2023, arXiv:2310.11511)

**Self-RAG核心机制：**
训练4种特殊token，让模型**自主决策**检索时机和质量：

| Token | 含义 | 示例 |
|-------|------|------|
| `[Retrieve]` | 是否需要检索 | `[Retrieve]=Yes / No` |
| `[IsREL]` | 检索结果是否相关 | `[IsREL]=Relevant / Irrelevant` |
| `[IsSUP]` | 检索结果是否支持生成内容 | `[IsSUP]=Fully / Partially / No` |
| `[IsUSE]` | 整体回复是否有用 | `[IsUSE]=5 / 4 / 3 / 2 / 1` |

**在广告创意生成中的适配：**

```python
# Self-RAG风格的广告创意生成流程（伪代码）
def self_rag_creative_generation(product_info, user_profile, llm):
    context = f"商品：{product_info}\n用户：{user_profile}"
    
    # Step 1: 模型判断是否需要检索参考创意
    decision = llm.generate(f"{context}\n[Retrieve]?")
    
    if "[Retrieve]=Yes" in decision:
        # Step 2: 检索参考创意
        query = construct_query(product_info, user_profile)
        candidates = retriever.search(query, k=10)
        
        # Step 3: 模型评估每条检索结果的相关性
        relevant_candidates = []
        for cand in candidates:
            relevance = llm.generate(
                f"参考创意：{cand}\n商品：{product_info}\n[IsREL]?"
            )
            if "[IsREL]=Relevant" in relevance:
                relevant_candidates.append(cand)
        
        # 动态K：实际使用的K = 通过相关性过滤的数量
        dynamic_k = len(relevant_candidates)
        
    else:
        # 不需要检索（通常是热门品类，模型已有足够先验）
        relevant_candidates = []
        dynamic_k = 0
    
    # Step 4: 生成创意
    creative = llm.generate(
        context + format_references(relevant_candidates)
    )
    
    # Step 5: 自我评分
    usefulness = llm.generate(f"创意：{creative}\n[IsUSE]?")
    
    return creative, dynamic_k, usefulness
```

---

## 三、广告创意生成的具体实现

### 3.1 查询构造策略

```python
def construct_retrieval_query(
    product_title: str,
    product_category: str,
    user_profile: dict,
    platform: str = "douyin"
) -> str:
    """
    构造检索query：融合商品信息和用户画像
    """
    # 方式1：简单拼接（快速）
    simple_query = f"{product_title} {product_category}"
    
    # 方式2：结构化query（推荐）
    age_group = user_profile.get("age_group", "通用")
    interest = user_profile.get("top_interest", "")
    
    structured_query = (
        f"{product_title} "
        f"适合{age_group} "
        f"{interest}人群 "
        f"{platform}平台推广文案"
    )
    
    # 方式3：LLM扩展query（高质量但有延迟）
    # expanded_query = llm.expand_query(product_title, user_profile)
    
    return structured_query
```

### 3.2 完整动态K流程

```python
class DynamicKRetriever:
    def __init__(
        self,
        vector_store,           # Milvus/FAISS
        embedding_model,
        min_k: int = 3,
        max_k: int = 20,
        cache_ttl: int = 3600   # 1小时缓存
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.min_k = min_k
        self.max_k = max_k
        self.k_cache = {}        # {category: k_value}
    
    def retrieve(
        self,
        product_info: dict,
        user_profile: dict
    ) -> dict:
        category = product_info.get("category")
        
        # 1. 检查缓存（相同品类K值缓存1小时）
        cached_k = self._get_cached_k(category)
        
        # 2. 构造query并向量化
        query_text = construct_retrieval_query(
            product_info["title"], category, user_profile
        )
        query_embedding = self.embedding_model.encode(query_text)
        
        # 3. 粗检索（FAISS，k=100）
        raw_results = self.vector_store.search(
            query_embedding, top_k=100
        )
        
        # 4. 计算结果集密度
        embeddings = np.array([r.embedding for r in raw_results[:20]])
        avg_sim = compute_avg_pairwise_similarity(embeddings)
        
        # 5. 动态确定K
        if cached_k:
            optimal_k = cached_k
        else:
            optimal_k = self._determine_k(avg_sim, len(raw_results))
            self._cache_k(category, optimal_k)
        
        # 6. 精排（BGE-reranker，从100中选optimal_k）
        reranked = self._rerank(query_text, raw_results[:50], optimal_k)
        
        return {
            "results": reranked,
            "k": optimal_k,
            "density": avg_sim,
            "strategy": self._get_strategy(avg_sim)
        }
    
    def _determine_k(self, avg_similarity: float, available: int) -> int:
        """根据信息密度确定最优K"""
        if avg_similarity > 0.85:
            # 高同质化：减小K，避免冗余
            return max(self.min_k, int(self.max_k * 0.3))
        elif avg_similarity < 0.40:
            # 信息稀疏（新品类）：扩大K获取更多覆盖
            return min(self.max_k, available)
        else:
            # 正常密度：线性映射
            # avg_sim 越低（多样性高），K越小；越高（多样性低），K越大
            ratio = (avg_similarity - 0.40) / (0.85 - 0.40)
            return int(self.min_k + ratio * (self.max_k - self.min_k))
    
    def _get_strategy(self, avg_similarity: float) -> str:
        if avg_similarity > 0.85:
            return "homogeneous - reduce_k"
        elif avg_similarity < 0.40:
            return "sparse - cold_start_fallback"
        else:
            return "normal - dynamic_k"
```

---

## 四、工程优化

### 4.1 两阶段检索架构

```
粗检索（FAISS，毫秒级）
  └─► k=100，基于HNSW向量索引
  └─► 目标：高召回，不要漏掉好创意

精排（BGE-reranker，10~30ms）
  └─► 从100中选5~20
  └─► 考虑语义相关性 + CTR权重
  └─► 目标：高精度，选出最佳参考
```

**延迟分析：**
| 阶段 | 延迟 | 备注 |
|------|------|------|
| Query向量化 | 3~5ms | 本地BERT/bge-m3 |
| FAISS粗检索(k=100) | 5~8ms | HNSW ef_search=64 |
| 密度计算 | 1ms | numpy矩阵运算 |
| BGE精排(top-50→top-K) | 10~20ms | GPU推理 |
| **总计** | **19~34ms** | 满足在线要求 |

### 4.2 批量化与缓存

```python
# 品类K值缓存（Redis实现）
import redis
import json
from datetime import timedelta

class KValueCache:
    def __init__(self, redis_client: redis.Redis, ttl: int = 3600):
        self.redis = redis_client
        self.ttl = ttl
    
    def get(self, category: str) -> int | None:
        val = self.redis.get(f"dynamic_k:{category}")
        return int(val) if val else None
    
    def set(self, category: str, k: int, density: float):
        data = {"k": k, "density": density}
        self.redis.setex(
            f"dynamic_k:{category}",
            timedelta(seconds=self.ttl),
            json.dumps(data)
        )
```

**批量化查询合并：**
- 同一品类的多个并发查询，合并为一次检索后分别返回
- 节省 60~80% 的向量检索时间（高并发场景）

---

## 五、评估指标

| 指标 | 计算方式 | 目标 |
|------|---------|------|
| 创意CTR提升 | A/B实验，对比固定K=10 | +5%以上 |
| 检索召回率 | 相关创意被检索到的比例 | >90% |
| Token消耗节省 | (固定K - 动态K) / 固定K | >20% |
| 平均检索延迟 | 端到端P99 | <50ms |
| 品类覆盖率 | 有有效检索结果的品类比例 | >95% |

---

## 六、常见考点 Q&A

**Q1：为什么不直接用大K（比如K=50）？**

A：大K有三个核心问题。第一，Token成本剧增，K=50的参考上下文可能消耗5000+ tokens，API成本是K=5的10倍；第二，质量反而下降，过多冗余参考让LLM注意力分散，实验表明K>15后质量开始下降（"lost in the middle"问题）；第三，延迟增加，更长的上下文使LLM推理时间从1s增到3s+。动态K的目标是找到信息增益最大化的最小K。

**Q2：动态K和固定K的A/B测试怎么做？**

A：将流量按品类分桶而非随机分桶，原因是不同品类密度差异悬殊，随机分桶会掩盖效果。控制组：全部品类用K=10（固定）。实验组：动态K（稀疏品类K=15，热门品类K=5~8）。核心指标：CTR提升（广告效果）和P99延迟（工程质量）。建议观察窗口2周，消除周末效应。同时埋点记录每次请求的实际K值，方便后续分析。

**Q3：如何处理新品类冷启动时K=0的极端情况？**

A：三级降级策略。第一级：跨类检索——从语义最近的相关品类借调参考创意（例如"无人机配件"冷启动时，借用"无人机"品类的高CTR创意）。第二级：通用模板——使用平台Top-1000全局高CTR创意作为无品类参考（确保基础质量）。第三级：Zero-shot生成——只用商品信息做纯LLM生成，不提供参考（质量最低但总比失败好）。同时触发异步任务，为该新品类持续积累真实创意数据。

**Q4：FLARE vs Self-RAG 的本质区别？**

A：FLARE是**在生成过程中插入检索**——生成某个token的置信度低时暂停，去检索后再继续生成，是一种迭代交错式框架，不需要额外训练。Self-RAG是**端到端训练模型来内化检索决策**——通过特殊token让模型自主判断"是否检索"、"检索结果是否有用"，需要专门微调，但推理时更自然高效。广告场景推荐FLARE，因为无需训练成本，易于工程集成；大规模部署且有微调能力时考虑Self-RAG。

**Q5：信息密度如何量化？最佳实践？**

A：核心是计算检索结果集的**平均两两余弦相似度**。实现时注意：(1) 只计算向量化后的embedding相似度，不是文本字面相似度；(2) 取上三角矩阵均值（避免重复计算和自相似）；(3) 密度 > 0.85说明同质化（需减K或增diversity），< 0.4说明信息稀疏（品类冷启动信号）。另一个有用指标是结果集的**熵**：把检索分数归一化为概率分布，计算Shannon熵，熵越高说明信息越分散（需要大K），熵越低说明信息集中（小K即可）。

**Q6：在线延迟如何控制在50ms内？**

A：四个关键优化：(1) 向量化用量化模型（int8）+本地部署，压缩到3ms内；(2) FAISS粗检索的HNSW参数调优——ef_search=64时精度95%/延迟8ms是最佳平衡点；(3) 密度计算用numpy批量矩阵运算，避免Python循环；(4) K值缓存——同品类1小时内复用上次计算结果，跳过密度计算步骤（覆盖70%+的请求）。极限情况下，预热阶段可以对全品类预计算K值并缓存，使在线路径完全绕过动态K计算。

**Q7：动态K对模型幻觉有什么影响？**

A：动态K对幻觉有双向影响。正面影响：当品类信息稀疏时，通过降级到跨品类检索而非直接Zero-shot，提供了更多真实参考，减少LLM"编造"历史创意效果的概率。负面影响：如果动态K的上限设得太高（如K=30），过多不相关参考反而会混淆LLM，导致生成内容偏离商品实际特点。最佳实践是设置max_k=20，并在精排阶段严格过滤相关性得分 < 0.5的候选，宁可K少也不引入低质量参考。

**Q8：如何防止K值频繁抖动？**

A：三种稳定化方法。(1) **滑动平均平滑**：记录最近N次请求的K值，用指数移动平均（EMA, α=0.3）更新，避免单次密度异常导致K骤变；(2) **变化阈值**：只有当新计算的K与缓存值差异超过2时才更新缓存；(3) **基于时间段的稳定性**：同品类在同一小时内使用相同K值，跨小时才重新计算（TTL=1小时的缓存本质上解决了这个问题）。可以额外引入异常检测：如果K值在10分钟内变化超过50%，触发告警，人工确认是正常的活动密度骤变还是数据异常。

---

*参考文献：*
- *FLARE: Jiang et al., arXiv:2305.06983 (2023)*
- *Self-RAG: Asai et al., arXiv:2310.11511 (2023)*
- *Adaptive-RAG: Jeong et al., NAACL 2024*
- *Dynamic RAG: EMNLP 2023*
