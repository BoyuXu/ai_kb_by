# 项目2：多样性混排系统（Feed 优化）

## 项目概览

这个项目核心解决社交 Feed 平台的"多样性 vs 点击率"的矛盾。在原有逐条排序系统的基础上，我实现了基于 DPP（Determinantal Point Process）的**列表级别联合优化**，在保证用户体验多样性的前提下，将用户留存率提升 8%，整体 RPM 增加 5%。项目历时 10 个月，涉及算法设计、工程实现、AB 测试三个阶段。

---

## 一、问题发现：多样性衰竭

### 1.1 现象观察

某一天，数据分析团队发现了一个诡异的现象：

```
2022 年 Q1：
  用户 Feed 点击率：3.5%（正常）
  
2022 年 Q2（热门广告增加）：
  用户 Feed 点击率：3.8%（+8.5%，好像提升了？）
  用户 1 天活跃度（DAU）：不变
  用户 7 日留存率：下降 2.3% ← 警告信号！
```

这很奇怪：点击率提升了，但用户留存反而下降。推荐系统的黄金法则是什么？**长期用户价值**。

### 1.2 根本原因分析

深入分析后，我们发现了问题：

**同类广告堆积现象**

用户滑过一个 Feed 的 10 条内容：

```
基础排序（纯按点击率）：
├─ 广告1（品类：电商，类型：手机）- 点击率 5%
├─ 广告2（品类：电商，类型：手机） - 点击率 5%
├─ 广告3（品类：电商，类型：衣服）- 点击率 4.8%
├─ 广告4（品类：电商，类型：衣服） - 点击率 4.8%
├─ [5-8] 都是电商类...
└─ 全是同一品类、同一创意风格 → 用户感到 "疲劳"，滑完就关掉 APP

多样化排序（我们的方案）：
├─ 广告1（品类：电商，手机）
├─ 广告2（品类：旅游，酒店）← 换品类
├─ 广告3（品类：电商，衣服）
├─ 广告4（品类：美妆，彩妆）← 完全不同
├─ [继续交错...]
└─ 用户看到各种类型的广告，保持新鲜感，继续滑动更多
```

数据证实了这个假设：
- **多样性低的 Feed**：用户平均看 8 条就离开
- **多样性高的 Feed**：用户平均看 12 条才离开（+50%！）

虽然单条点击率可能下降（因为把热门广告分散开了），但**总点击数反而增加**（因为用户看的条数更多）。

### 1.3 商业影响

```
留存率下降的连锁反应：

用户留存 ↓
  ↓ 
每日活跃用户 ↓
  ↓
每用户日均查看广告 ↓（虽然单条 CTR ↑，但总数量 ↓）
  ↓
总展示数 ↓
  ↓
平台 RPM（Revenue Per Mille）↓（虽然 eCPM ↑）
```

目标：在保证 CTR 的前提下，提升 Feed 多样性，最终提升长期留存率和 RPM。

---

## 二、多样性度量

不能优化一个度量不出来的东西。首先需要定义"多样性"。

### 2.1 香农熵（Shannon Entropy）

最简单的多样性度量：

$$
H = -\sum_{i=1}^{C} p_i \log p_i
$$

其中 $C$ 是品类数，$p_i$ 是品类 $i$ 在 Feed 中的占比。

例子：

```
Feed 1（单一品类）：
  100% 电商
  p = [1.0]
  H = -1.0 * log(1.0) = 0
  
Feed 2（均匀分布）：
  33% 电商 + 33% 旅游 + 33% 美妆
  p = [0.33, 0.33, 0.33]
  H = -3 * 0.33 * log(0.33) = 1.10
  
Feed 3（偏斜分布）：
  50% 电商 + 30% 旅游 + 20% 美妆
  p = [0.5, 0.3, 0.2]
  H = -0.5*log(0.5) - 0.3*log(0.3) - 0.2*log(0.2) = 1.03
```

香农熵越高，多样性越好。范围 [0, log(C)]。

### 2.2 广告主均衡度（Source Diversity）

仅看品类是不够的。还要考虑**广告主的公平性**。

假设某热门广告主 A 占据了 40% 的流量，而小广告主 B 只有 1%。这不公平。

定义：

$$
D_{source} = 1 - \frac{\text{top-1 广告主流量占比}}{100\%}
$$

或者更精细的方式：用广告主的 Herfindahl 指数

$$
HHI = \sum_{i=1}^{N} s_i^2
$$

其中 $s_i$ 是广告主 $i$ 的流量占比。HHI 范围 [1/N, 1]，越小越均衡。

### 2.3 内容多样性（Content Diversity）

除了品类，还要考虑：

- **创意多样性**：文案长度、关键词、配图风格
- **格式多样性**：图文、视频、纯文本
- **格式内容多样性**：如果都是视频，也应该有不同风格

衡量方式：计算 Feed 中任意两个广告的内容相似度，然后取反。

```python
def content_diversity(feed_ads):
    """计算 Feed 的内容多样性"""
    similarities = []
    for i in range(len(feed_ads)):
        for j in range(i+1, len(feed_ads)):
            # 用 TF-IDF + 余弦相似度计算文本相似度
            sim = cosine_similarity(feed_ads[i].text, feed_ads[j].text)
            similarities.append(sim)
    
    # 相似度越高，多样性越低
    avg_similarity = np.mean(similarities)
    content_diversity_score = 1 - avg_similarity
    return content_diversity_score
```

### 2.4 综合多样性度量

不能只看一个维度，需要加权组合：

$$
D_{total} = w_1 \cdot D_{topic} + w_2 \cdot D_{source} + w_3 \cdot D_{content}
$$

我们实验中的权重是：$w_1 = 0.5$, $w_2 = 0.3$, $w_3 = 0.2$（前期通过网格搜索得出）。

---

## 三、排序算法演进

### 3.1 v0：硬规则混排（最简单）

```python
def hard_rule_mixing(ranked_list, max_consecutive_same_category=2):
    """
    硬规则：同品类广告不能超过 2 个连续
    """
    result = []
    category_count = {}
    last_category = None
    consecutive = 0
    
    for ad in ranked_list:
        cat = ad.category
        
        if cat == last_category:
            consecutive += 1
            if consecutive >= max_consecutive_same_category:
                # 跳过这个广告，寻找下一个不同品类的
                continue
        else:
            consecutive = 1
            last_category = cat
        
        result.append(ad)
    
    return result
```

**优点**：简单，易于理解和调试

**缺点**：死板，无法根据广告质量灵活调整。例如，如果两个电商广告的 CTR 分别是 8% 和 2%，硬规则会简单粗暴地把 2% 的也排进来。

**效果**：多样性 Entropy 从 1.1 → 1.4，但 CTR 从 3.5% → 3.3%（-5%），不划算。

---

### 3.2 v1：启发式混排（加入权衡）

引入"多样性惩罚项"：

$$
Score_i = CTR\_score_i - \lambda \times D\_penalty_i
$$

其中：
- $CTR\_score_i$ = 标准化后的点击率估计
- $D\_penalty_i$ = 该广告与 Feed 前 k 个广告的相似度均值
- $\lambda$ = 权衡参数（可调整）

```python
def heuristic_mixing(ranked_list, k=5, lambda_=0.1):
    """
    启发式混排：加入多样性惩罚
    """
    result = []
    processed = set()
    
    # 标准化 CTR 分数
    ctr_scores = np.array([ad.ctr for ad in ranked_list])
    ctr_scores = (ctr_scores - ctr_scores.min()) / (ctr_scores.max() - ctr_scores.min())
    
    for idx, ad in enumerate(ranked_list):
        if ad.id in processed:
            continue
        
        # 计算与已选广告的相似度
        diversity_penalty = 0
        if result:
            similarities = [
                compute_similarity(ad, result_ad)
                for result_ad in result[-k:]  # 只看最后 k 个
            ]
            diversity_penalty = np.mean(similarities)
        
        # 调整分数
        adjusted_score = ctr_scores[idx] - lambda_ * diversity_penalty
        
        # 记录（分数, 广告）
        yield (adjusted_score, ad)
```

**效果**：
- 多样性 Entropy：1.4 → 1.65
- CTR：3.3% → 3.35%（基本持平，好！）
- 用户留存：+3% ✓

**问题**：这个方法虽然简单，但是**贪心的**。每次只考虑当前最高分的广告，无法看到全局最优解。

例子：如果当前最高分的广告是"电商"，加上多样性惩罚后得分还是 0.8，那就选它。但如果我们往前看，可能发现把这个"电商"排后面，选一个"旅游"广告（虽然分数 0.7），反而能更优地平衡点击率和多样性。

---

### 3.3 v2：DPP 混排（突破点）

#### 3.3.1 DPP 的数学原理

DPP（Determinantal Point Process）是一个概率模型，用于从集合中采样多样的子集。核心思想是：**使用行列式来衡量多样性**。

定义 $L$ 矩阵（kernel matrix）：

$$
L_{ij} = q_i \cdot \text{sim}(i, j) \cdot q_j
$$

其中：
- $q_i$ = 广告 $i$ 的质量分数（如点击率）
- $\text{sim}(i, j)$ = 广告 $i$ 和 $j$ 的相似度（范围 [0, 1]）

DPP 的概率分布定义为：

$$
P(S) \propto \det(L_S)
$$

即：选择子集 $S$ 的概率与 $L_S$ 的行列式成正比。

**直观理解**：行列式衡量的是矩阵列向量的"线性独立性"。列向量越独立（越不相似），行列式越大。所以，DPP 自然地倾向于选择差异大的元素。

#### 3.3.2 为什么 DPP 比贪心更优

**贪心方法的陷阱**：

```
广告集合 {A, B, C}：
  A（电商）：CTR 5%，与前面相似度 0.9
  B（旅游）：CTR 4%，与前面相似度 0.1
  C（美妆）：CTR 3%，与前面相似度 0.1

贪心（多样性惩罚）：
  选 A（5% - 0.1*0.9 = 4.91）
  → 然后 B（4% - 0.1*0.1 = 3.99）
  → 最后 C（3% - 0.1*0.1 = 2.99）
  结果：A, B, C，预期 CTR = (5% + 4% + 3%) / 3 = 4%

DPP：
  考虑所有可能的组合
    {A, B}：det = q_A * sim(A,B) * q_B = 0.05 * 0.1 * 0.04 = 0.00002
    {A, C}：det = q_A * sim(A,C) * q_C = 0.05 * 0.1 * 0.03 = 0.000015
    {B, C}：det = q_B * sim(B,C) * q_C = 0.04 * 0.1 * 0.03 = 0.000012
  
  （假设多样性子集长度为 2）
  选 {A, B}（det 最大）
  预期 CTR = (5% + 4%) / 2 = 4.5% ✓ 更好
```

DPP 不是逐个贪心选择，而是**考虑全局最优**。

#### 3.3.3 算法实现

直接计算行列式（求解最大概率子集）是 NP-hard 的。但有两种实用的采样算法：

**方法 1：贪心采样（Greedy Sampling）**

```python
def greedy_dpp_sampling(L, k):
    """
    贪心采样：每次选择使行列式增长最多的元素
    时间复杂度：O(k^3)
    """
    n = L.shape[0]
    selected = set()
    
    # 初始化：选择 q 值最高的元素
    q_diag = np.diag(L)
    selected.add(np.argmax(q_diag))
    
    # 迭代选择 k-1 个元素
    for _ in range(k - 1):
        best_idx = -1
        best_det_increase = -np.inf
        
        for i in range(n):
            if i in selected:
                continue
            
            # 计算加入元素 i 后的行列式增长
            S_new = sorted(selected | {i})
            det_S_new = np.linalg.det(L[np.ix_(S_new, S_new)])
            det_S_old = np.linalg.det(L[np.ix_(selected, selected)])
            
            det_increase = det_S_new - det_S_old
            
            if det_increase > best_det_increase:
                best_det_increase = det_increase
                best_idx = i
        
        selected.add(best_idx)
    
    return sorted(selected)
```

**方法 2：快速采样（Fast Sampling）**

贪心采样时间复杂度还是 O(k³)，对于大规模 Feed（k=100）可能太慢。可以用快速采样（O(k²)）：

利用 Cholesky 分解的性质，可以在 O(k²) 时间内完成采样。这里涉及比较复杂的线性代数，我简化说明：

```python
def fast_dpp_sampling(L, k):
    """
    基于特征值分解的快速采样：O(k^2)
    """
    # L 矩阵的特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # 按特征值排序，选择 top-k 对应的特征向量
    idx = np.argsort(eigenvalues)[::-1][:k]
    V = eigenvectors[:, idx]  # (n, k)
    
    # 采样
    selected = []
    remaining = np.arange(n)
    
    for _ in range(k):
        # 计算每个剩余元素的 "归一化长度"
        norms = np.linalg.norm(V[remaining], axis=1) ** 2
        probs = norms / np.sum(norms)  # 归一化为概率
        
        # 按概率采样
        i = np.random.choice(remaining, p=probs)
        selected.append(i)
        
        # 更新 V（正交化）
        # ...（省略细节）
    
    return selected
```

我在实现中选择了**贪心采样**（更直观，易于调试），因为 Feed 长度 k=10-20，O(k³) 也能接受（每次 <10ms）。

#### 3.3.4 L 矩阵的构造

这是 DPP 最关键、最需要调优的部分。

```python
def construct_L_matrix(ads, quality_scores, similarity_matrix):
    """
    构造 DPP 的 L 矩阵
    
    L_ij = sqrt(q_i) * sim_ij * sqrt(q_j)
    """
    n = len(ads)
    L = np.zeros((n, n))
    
    # 标准化质量分数到 [0, 1]
    q = np.array(quality_scores)
    q_norm = (q - q.min()) / (q.max() - q.min() + 1e-8)
    
    for i in range(n):
        for j in range(n):
            L[i, j] = np.sqrt(q_norm[i]) * similarity_matrix[i, j] * np.sqrt(q_norm[j])
    
    return L

# 质量分数：可以用 CTR + 其他因素
def compute_quality_score(ad):
    ctr = ad.predicted_ctr
    freshness = 1.0 if ad.created_time > 24*3600 else 0.7  # 新广告加权
    advertiser_credit = ad.advertiser.credit_score / 100.0  # 广告主信誉
    quality = 0.6 * ctr + 0.2 * freshness + 0.2 * advertiser_credit
    return quality

# 相似度矩阵：可以综合多个维度
def compute_similarity(ad_i, ad_j):
    # 品类相似度
    category_sim = 1.0 if ad_i.category == ad_j.category else 0.0
    
    # 创意相似度（文本）
    text_sim = cosine_similarity(ad_i.text_embedding, ad_j.text_embedding)
    
    # 广告主相似度
    advertiser_sim = 1.0 if ad_i.advertiser_id == ad_j.advertiser_id else 0.0
    
    # 综合
    sim = 0.4 * (1 - category_sim) + 0.4 * text_sim + 0.2 * advertiser_sim
    return sim
```

#### 3.3.5 DPP 效果

```
基线（v1 启发式）：
  多样性 Entropy：1.65
  CTR：3.35%
  用户留存：+3%

DPP（v2）：
  多样性 Entropy：2.1（+27%！）
  CTR：3.25%（只下降 3%）
  用户留存：+8%（❤ 大幅提升）
```

DPP 的魔力在于：虽然平均 CTR 微降（因为把一些热门广告挤了出去），但**多样性大幅提升，用户留存反而增加**。

---

### 3.4 v3：多目标优化（帕累托前沿）

DPP 虽然很好，但参数很多（L 矩阵的构造方式、特征权重等），难以调优。

有没有更优雅的方式？用**多目标优化**明确地最大化两个目标：

1. **最大化 CTR**：$\max \sum_{i \in S} \text{CTR}}_{\text{i$
2. **最大化多样性**：$\max D(S)$

这是个矛盾的目标（Pareto Frontier）。可以用加权组合：

$$
F(S) = \alpha \cdot \text{CTR}}(S) + (1 - \alpha) \cdot D(S)
$$

其中 $\alpha \in [0, 1]$ 控制权重。

```python
def pareto_ranking(candidates, alpha=0.5):
    """
    多目标优化排序
    """
    # 候选集合是所有可能的 Feed 组合（指数爆炸，需要剪枝）
    best_solution = None
    best_score = -np.inf
    
    # 枚举（实际中用启发式搜索）
    for S in all_possible_subsets(candidates, max_size=20):
        ctr = compute_avg_ctr(S)
        diversity = compute_diversity(S)
        
        # 归一化
        ctr_norm = (ctr - min_ctr) / (max_ctr - min_ctr)
        diversity_norm = (diversity - min_diversity) / (max_diversity - min_diversity)
        
        score = alpha * ctr_norm + (1 - alpha) * diversity_norm
        
        if score > best_score:
            best_score = score
            best_solution = S
    
    return best_solution
```

**多目标优化的好处**：
- 直观：直接权衡 CTR 和多样性
- 灵活：可以通过 $\alpha$ 调整策略
- 可解释：下游业务团队容易理解和接受

**与 DPP 的对比**：
- DPP 是隐式地通过 L 矩阵结构来平衡（更高级，但黑盒）
- 多目标优化是显式的权衡（更透明，更易调）

实际上，我在项目中**同时运行了 DPP 和多目标优化**，AB 测试对比效果：

| 方法 | 多样性 | CTR | 留存 | 延迟 |
|------|--------|-----|------|------|
| 启发式（v1） | 1.65 | 3.35% | +3% | 8ms |
| DPP（v2） | 2.1 | 3.25% | +8% | 25ms ← 慢 |
| 多目标（v3） | 2.05 | 3.28% | +7.5% | 12ms ← 快 |

**选择**：虽然 DPP 的多样性最高，但延迟太高（25ms 用掉了一半的排序预算）。最终选择**多目标优化**（v3）作为线上方案，平衡了精度、延迟、效果。

---

## 四、工程实现

### 4.1 实时性要求

Feed 排序需要在 <50ms 内完成（包括特征提取、排序、结果返回）。

```
用户请求
  ↓ (0-5ms)
特征获取（从缓存 / RPC 调用）
  ↓ (5-20ms)
排序算法（DPP 或多目标）
  ↓ (20-40ms)
后处理（去重、审核过滤）
  ↓ (40-50ms)
返回结果给客户端
```

### 4.2 特征缓存策略

DPP/多目标排序需要大量的相似度计算。为了加速，我们预先计算并缓存：

```python
class FeatureCache:
    def __init__(self):
        self.ad_embeddings = {}  # 广告创意的文本/视觉 embedding
        self.similarity_matrix = {}  # 广告对之间的相似度（稀疏矩阵）
    
    def get_or_compute_embedding(self, ad_id, embedding_type='text'):
        """
        获取或计算 embedding
        """
        key = f"{ad_id}_{embedding_type}"
        if key in self.ad_embeddings:
            return self.ad_embeddings[key]
        
        # 缓存未命中，在线计算
        if embedding_type == 'text':
            embedding = text_encoder(ad.text)
        elif embedding_type == 'image':
            embedding = image_encoder(ad.image_url)
        
        # 写回缓存
        redis_client.set(key, embedding, ex=24*3600)  # 24h 过期
        self.ad_embeddings[key] = embedding
        return embedding
    
    def get_or_compute_similarity(self, ad_i_id, ad_j_id):
        """
        获取或计算两个广告的相似度
        """
        key = f"sim_{min(ad_i_id, ad_j_id)}_{max(ad_i_id, ad_j_id)}"
        
        if key in self.similarity_matrix:
            return self.similarity_matrix[key]
        
        # 计算
        emb_i = self.get_or_compute_embedding(ad_i_id)
        emb_j = self.get_or_compute_embedding(ad_j_id)
        sim = cosine_similarity(emb_i, emb_j)
        
        # 缓存
        redis_client.set(key, sim, ex=24*3600)
        return sim
```

### 4.3 候选集合的预筛选

不能对所有 Feed 候选（通常有 1000+ 个）都做 DPP 排序。需要先筛选到 20-30 个高质量候选，再进行多样性优化。

```python
def two_stage_ranking(candidates, k=20):
    """
    两阶段排序：
    第一阶段：快速筛选（基于 CTR 预估）
    第二阶段：多样性优化（基于 DPP/多目标）
    """
    # 第一阶段：按 CTR 取 top-k'（k' = 3k）
    sorted_by_ctr = sorted(candidates, key=lambda x: x.ctr, reverse=True)
    candidates_filtered = sorted_by_ctr[:3*k]
    
    # 第二阶段：多目标优化
    selected = pareto_ranking(candidates_filtered, alpha=0.5)
    
    return selected[:k]
```

---

## 五、AB 测试与线上验证

### 5.1 AB 测试设计

```
对照组（v1 启发式）：50% 流量
实验组（v3 多目标）：50% 流量

时长：2 周
样本量：每组 5000 万 用户-Session
统计功效：80%（能检测 1% 的留存提升）
```

### 5.2 主要指标

| 指标 | 对照 | 实验 | 差异 | p 值 |
|------|------|------|------|------|
| 日均点击 CTR | 3.35% | 3.28% | -2.1% | <0.01 |
| 多样性 Entropy | 1.65 | 2.05 | +24% | <0.001 |
| 7 日留存率 | 40.2% | 43.4% | **+8.0%** | <0.001 |
| 日均 Session 长度 | 12.5 | 14.2 | +13.6% | <0.001 |
| **RPM** | 0.85 元 | 0.89 元 | **+5.9%** | <0.001 |

最关键的指标是：虽然单条 CTR 下降了 2%，但因为用户 Session 长度增加 13.6%（看更多广告了），整体展示数增加，**RPM 增加 5.9%**。

### 5.3 分层分析（Cohort Analysis）

不同用户群体的反应是否一致？

```python
# 按用户等级分层
new_users (< 7 days):
  7 日留存提升：+25%（新用户最受益）
  RPM：+12%

active_users (7-30 days):
  7 日留存提升：+8%
  RPM：+6%

veteran_users (> 30 days):
  7 日留存提升：+2%
  RPM：+2%
```

发现：**多样性优化对新用户效果最好**。这符合直觉：新用户需要快速了解平台的多样性，避免用户疲劳；而老用户已经有偏好了，多样性调整影响有限。

---

## 六、效果总结

### 6.1 关键成果

```
线上运行 3 个月数据：
├─ 多样性指标
│   ├─ Topic Entropy：1.2 → 2.1（+75%）
│   ├─ Source Diversity：0.58 → 0.71（+22%）
│   └─ Content Diversity：0.42 → 0.65（+55%）
│
├─ 用户体验
│   ├─ 日均 Session 长度：8.5 → 10.2 条（+20%）
│   ├─ 7 日留存率：40.2% → 43.4%（+8%）
│   └─ 日均使用时长：22 分钟 → 25 分钟（+14%）
│
├─ 商业指标
│   ├─ 日展示数：50 亿 → 55 亿（+10%）
│   ├─ RPM：0.85 → 0.89 元（+5.9%）
│   └─ **月广告收入：+2.9 亿**
│
└─ 小广告主公平性
    ├─ Top-1 广告主流量占比：35% → 28%（下降，更公平）
    └─ Top-10 广告主流量占比：68% → 61%
```

### 6.2 技术指标

| 指标 | 目标 | 实际 |
|------|------|------|
| 排序延迟（p99） | <50ms | 12ms ✓ |
| 多样性改进 | +50% | +75% ✓ |
| 留存提升 | +5% | +8% ✓ |
| RPM 提升 | +3% | +5.9% ✓ |

---

## 七、关键技术洞察

### 7.1 局部最优 ≠ 全局最优

最初，我们以为"最大化每条广告的点击率"就能最大化整体收益。但事实正好相反：

```
贪心策略（逐条最优）：
  ├─ 优点：单条 CTR 高（3.35%）
  └─ 缺点：用户疲劳，早期离开，总展示数少

列表级别优化（全局最优）：
  ├─ 优点：单条 CTR 低（3.28%），但用户看更多（Session +13.6%）
  └─ 结果：总展示数反而增加，RPM 更高
```

推荐系统的本质是**用户长期价值最大化**，不是短期点击率最大化。

### 7.2 DPP 的理论很美，但工程化很难

DPP 理论上很优雅：用行列式天然地建模多样性和质量的权衡。

但在实践中，问题繁多：
- L 矩阵的构造方式：用什么相似度度量？权重怎么分配？
- 采样算法的选择：贪心 O(k³) vs 快速采样 O(k²)？
- 超参调优：alpha, beta, lambda 等参数怎么调？

花了 2 个月才调出效果好的 DPP。最后发现，**多目标优化（显式权衡）反而更高效**——参数少，易调，延迟也更低。

### 7.3 小广告主的问题比想象严重

一开始，我们觉得"只要优化多样性就行"。但问题更深层：

```
原始情况（纯 CTR 排序）：
  Top-1 电商广告主：占 35% 流量
  → 其他 9999 个广告主：瓜分剩余 65%
  → 小广告主无机会，流失
```

通过多样性优化，我们把 Top-1 的占比降到 28%，**给小广告主更多曝光机会**。这不仅提升了公平性，还有意外好处：**小广告主虽然 CTR 低，但他们往往更新颖**（因为规模小，创意更自由），这正好补充了热门广告主的"乏味"。

---

## 八、讲故事要点

### 8.1 30 秒电梯演讲

> "我在某社交平台做 Feed 优化，发现一个反直觉的现象：单条点击率高的 Feed 反而让用户留存率下降。根本原因是同类广告堆积导致用户疲劳。我用 DPP（行列式点过程）来重新建模 Feed 混排问题，从'最大化单条 CTR'转变为'最大化列表多样性'。结果虽然单条 CTR 下降 2%，但用户留存提升 8%，最终 RPM 增加 5.9%，月增收 2.9 亿。"

### 8.2 完整版讲述

**问题**：用户在看 Feed 的时候，如果都是同类广告，会感到疲劳，滑完几条就离开。虽然单条点击率高，但整体留存和收益反而下降。

**解决方案**：我采用了 DPP（Determinantal Point Process）来建模 Feed 多样性。DPP 的核心思想是用行列式来衡量多样性——如果两个广告差异大（不相似），它们的行列式就大，被同时选中的概率就高。这样，系统会自动平衡"质量"和"多样性"。

**结果**：线上 AB 测试显示，新方案虽然单条 CTR 下降 2%，但：
- 用户每次看更多广告（Session 长度 +14%）
- 7 日留存率提升 8%
- RPM 增加 5.9%，月增收 2.9 亿

**学到的**：最关键的洞察是：**推荐系统优化的目标不是短期点击率，而是长期用户价值**。有时候，为了长期价值，需要在短期指标上做出权衡。这个思路后来也应用到其他项目上，效果都很好。

### 8.3 可能的追问

**Q: "为什么不直接用多目标优化，要用 DPP？"**

A: "好问题。实际上，我们尝试了两种方法：
- DPP：理论更优雅（用行列式天然建模多样性），但参数多，调优困难，延迟也高（25ms）
- 多目标优化：参数少，易调，延迟低（12ms），最终效果也不差（留存 +7.5% vs DPP 的 +8%）
最终选择多目标，因为在工程化和可维护性上优势更大。"

**Q: "有没有伤害某些广告主？"**

A: "这是个很重要的问题。确实，热门广告主（原来占 35% 流量）现在只占 28%。但我们的数据显示：
- 虽然展示数下降了，但他们的转化也相对降低不多（因为他们的广告质量确实好）
- 小广告主的受益更大（流量从 1% → 3%），整体平台生态更健康
通过与商业团队的协作，我们也给热门广告主补偿（如增加在其他场景的展示机会），所以没有引起不满。"

---

## 九、总结

这个项目让我深刻理解了：

1. **问题定义的重要性**：最初我们以为问题是"CTR 低"，但根本问题其实是"用户体验单调"。花时间诊断真实问题，比盲目优化更值钱。

2. **全局视角**：不能只看单条广告，要看整个 Feed、整个 Session、整个用户生命周期。局部最优往往意味着全局次优。

3. **理论与工程的平衡**：DPP 是漂亮的理论，但工程化有成本。有时候，"丑陋"但有效的多目标优化，比"优雅"但难以调控的 DPP 更实用。

4. **用户体验的长期性**：短期看，我们损失了 2% 的 CTR。但长期看（7 天、30 天），用户留存和收益都提升了。这让我认识到，**推荐系统优化需要长期视角**。
