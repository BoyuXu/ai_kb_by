# 04 创意多样性监控

> 核心论文：BLEU (Papineni et al., ACL 2002)、MMR (Li et al., 2016)、SentEval (LREC 2018)、CheckList (ACL 2020)

---

## 一、为什么多样性监控重要

### 1.1 广告疲劳效应（Ad Fatigue）

用户对重复内容的CTR衰减规律：
```
首次看到创意：CTR = 5%（基准）
第2次：CTR = 4.5%（-10%）
第5次：CTR = 3.0%（-40%）
第10次：CTR = 1.5%（-70%）
第20次：CTR ≈ 0.5%（-90%）
```

**如果系统生成的10条文案全部相似：**
- 广告主会看到的是10个几乎相同的选项，对系统丧失信任
- 即使上线，用户很快进入疲劳期，CTR衰减加速
- 无法覆盖不同偏好的用户群（有人喜欢理性诉求，有人喜欢情感诉求）

### 1.2 平台合规风险

- 同一广告账户下大量相似创意被平台判定为"spam"，触发审核限流
- 某些平台要求创意相似度不超过阈值（如抖音要求同一广告组下文案相似度 < 0.7）

### 1.3 广告主体验

```
广告主的合理期望：
  系统生成10条 → 情感型、功效型、故事型、价格型各有2-3条
  每条有不同的切入角度，提供真实参考价值

实际失败案例：
  10条全是"XXX产品，多重功效，立即购买"的变体
  → 广告主一眼就看出是"废话生成器"，对系统评分极低
```

---

## 二、n-gram重复率评估

### 2.1 BLEU原理详解

**BLEU公式（Papineni et al., ACL 2002）：**

$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

各项含义：
- $p_n$：n-gram精确率，候选文本中n-gram出现在参考文本中的比例
- $w_n = 1/N$：各阶n-gram的权重（通常N=4，均匀分配）
- $BP$：简短惩罚因子（Brevity Penalty）

**简短惩罚因子：**

$$BP = \begin{cases} 1 & \text{if } c > r \\ e^{1 - r/c} & \text{if } c \leq r \end{cases}$$

其中 $c$ 是候选文本长度，$r$ 是最近参考文本长度

**BLEU在创意监控中的用法：**

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List
import jieba  # 中文分词

def compute_inter_bleu(creatives: List[str]) -> float:
    """
    计算批次内部BLEU（inter-sample BLEU）
    值越高 = 批次内创意越相似 = 多样性越差
    """
    if len(creatives) < 2:
        return 0.0
    
    total_bleu = 0.0
    count = 0
    smoother = SmoothingFunction().method1
    
    for i, candidate in enumerate(creatives):
        # 将其他创意作为参考
        references = [creatives[j] for j in range(len(creatives)) if j != i]
        
        # 中文分词
        cand_tokens = list(jieba.cut(candidate))
        ref_tokens = [list(jieba.cut(ref)) for ref in references]
        
        bleu = sentence_bleu(ref_tokens, cand_tokens,
                             smoothing_function=smoother)
        total_bleu += bleu
        count += 1
    
    return total_bleu / count
```

### 2.2 Self-BLEU（批量多样性更好的指标）

**Self-BLEU的优势：** 直接度量生成集合的内部多样性，不需要"参考文本"，是生成系统自评的标准方法

**定义：** 对集合中的每条创意，以其他所有创意为参考，计算BLEU得分，取平均

$$\text{Self-BLEU} = \frac{1}{N} \sum_{i=1}^{N} \text{BLEU}(d_i, \{d_j\}_{j \neq i})$$

**实现代码：**

```python
def self_bleu(
    creatives: List[str],
    max_n: int = 3  # 通常用2-gram或3-gram
) -> dict:
    """
    计算Self-BLEU
    返回各n-gram阶次的得分
    
    解读：
    Self-BLEU = 0.0  → 完全不同（最多样）
    Self-BLEU = 1.0  → 完全相同（最差）
    Self-BLEU < 0.3  → 良好多样性
    Self-BLEU > 0.6  → 告警：同质化严重
    """
    results = {}
    
    for n in range(1, max_n + 1):
        weights = tuple([1.0/n] * n)  # 只用n-gram
        scores = []
        
        for i, candidate in enumerate(creatives):
            references = [creatives[j] for j in range(len(creatives)) if j != i]
            
            cand_tokens = list(jieba.cut(candidate))
            ref_tokens = [list(jieba.cut(ref)) for ref in references]
            
            score = sentence_bleu(
                ref_tokens, cand_tokens,
                weights=weights,
                smoothing_function=SmoothingFunction().method1
            )
            scores.append(score)
        
        results[f"self_bleu_{n}gram"] = sum(scores) / len(scores)
    
    return results

# 使用示例
creatives = [
    "轻松护肤，一步到位，敏感肌也放心",
    "专为敏感肌设计，温和不刺激，轻松搞定护肤",  # 高相似
    "工作压力大，皮肤还是要爱护自己",             # 低相似
    "10万女生的护肤秘密，你还不知道？",           # 低相似
]

scores = self_bleu(creatives)
print(scores)
# {'self_bleu_1gram': 0.42, 'self_bleu_2gram': 0.28, 'self_bleu_3gram': 0.15}
```

---

## 三、语义多样性（聚类方法）

### 3.1 SBERT向量化 + K-means聚类

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import numpy as np

class SemanticDiversityAnalyzer:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
    
    def analyze(
        self,
        creatives: List[str],
        n_clusters: int = 3    # 期望的多样性簇数
    ) -> dict:
        # 1. 语义向量化
        embeddings = self.model.encode(creatives, normalize_embeddings=True)
        
        # 2. 聚类
        if len(creatives) < n_clusters:
            n_clusters = len(creatives)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # 3. 计算ILD（Intra-list Diversity）
        ild = self._compute_ild(embeddings)
        
        # 4. 簇分布均匀度
        cluster_counts = np.bincount(labels)
        uniformity = 1 - (cluster_counts.std() / cluster_counts.mean())
        
        # 5. 最小簇间距离（检测极端同质化）
        cluster_centers = kmeans.cluster_centers_
        inter_cluster_dists = pairwise_distances(cluster_centers, metric='cosine')
        np.fill_diagonal(inter_cluster_dists, np.inf)
        min_inter_dist = inter_cluster_dists.min()
        
        return {
            "ild": ild,
            "cluster_labels": labels.tolist(),
            "cluster_uniformity": uniformity,
            "min_inter_cluster_dist": min_inter_dist,
            "n_effective_clusters": len(set(labels))
        }
    
    def _compute_ild(self, embeddings: np.ndarray) -> float:
        """
        Intra-list Diversity (ILD)
        ILD = 平均两两余弦距离
        """
        k = len(embeddings)
        if k <= 1:
            return 0.0
        
        # 余弦距离矩阵（距离 = 1 - 余弦相似度）
        sim_matrix = embeddings @ embeddings.T
        dist_matrix = 1 - sim_matrix
        
        # 取上三角平均
        indices = np.triu_indices(k, k=1)
        return dist_matrix[indices].mean()
```

**ILD公式：**

$$\text{ILD} = \frac{2}{K(K-1)} \sum_{i<j} d(d_i, d_j)$$

其中 $d(d_i, d_j) = 1 - \cos(d_i, d_j)$

**评判标准：**
| ILD 值 | 解读 | 动作 |
|--------|------|------|
| > 0.6 | 多样性优秀 | ✅ 正常输出 |
| 0.4 ~ 0.6 | 多样性可接受 | ⚠️ 关注 |
| < 0.4 | 多样性不足 | ❌ 触发重新生成 |

---

## 四、MMR最大边际相关性

### 4.1 完整公式推导

**MMR（Maximum Marginal Relevance，Li et al., 2016）：**

$$\text{MMR} = \arg\max_{d_i \in C \setminus S} \left[\lambda \cdot \text{sim}(d_i, q) - (1-\lambda) \cdot \max_{d_j \in S} \text{sim}(d_i, d_j)\right]$$

**各项含义：**
- $C$：候选集合（全部生成的创意，如50条）
- $S$：已选中的结果集（初始为空）
- $\text{sim}(d_i, q)$：候选 $d_i$ 与查询 $q$ 的相关性（质量）
- $\max_{d_j \in S} \text{sim}(d_i, d_j)$：$d_i$ 与已选中创意的最大相似度（惩罚重复）
- $\lambda$：权衡相关性 vs 多样性（0~1）

**在广告创意中的适配：**
- 用 **CTR预估得分** 替代 $\text{sim}(d_i, q)$（相关性 → 效果期望）
- 用 **语义余弦相似度** 计算 $\text{sim}(d_i, d_j)$（惩罚语义重复）

```python
import numpy as np
from typing import List, Tuple

def mmr_select(
    candidates: List[dict],      # 每条包含 text, ctr_score, embedding
    n_select: int = 5,           # 最终选取条数
    lambda_: float = 0.7         # 质量 vs 多样性权衡
) -> List[dict]:
    """
    MMR算法：在保证效果的前提下最大化多样性
    """
    if len(candidates) <= n_select:
        return candidates
    
    # 预计算相似度矩阵
    embeddings = np.array([c["embedding"] for c in candidates])
    # 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_embs = embeddings / (norms + 1e-8)
    sim_matrix = norm_embs @ norm_embs.T
    
    selected_indices = []
    remaining_indices = list(range(len(candidates)))
    
    while len(selected_indices) < n_select and remaining_indices:
        best_score = -np.inf
        best_idx = None
        
        for idx in remaining_indices:
            # 相关性得分（用CTR预估）
            relevance = candidates[idx]["ctr_score"]
            
            if not selected_indices:
                # 第一条：只考虑CTR，选最高
                mmr_score = relevance
            else:
                # 后续：CTR - 与已选集合的最大相似度
                max_sim_to_selected = max(
                    sim_matrix[idx][sel_idx]
                    for sel_idx in selected_indices
                )
                mmr_score = (lambda_ * relevance
                             - (1 - lambda_) * max_sim_to_selected)
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
    
    return [candidates[i] for i in selected_indices]
```

### 4.2 λ参数的经验设置

| λ值 | 偏向 | 适用场景 |
|-----|------|---------|
| λ = 0.9 | 强调质量（CTR） | 追求转化率，接受相似文案 |
| λ = 0.7 | 平衡（推荐） | 大多数广告场景 |
| λ = 0.5 | 强调多样性 | 广告主要求看更多样的选项 |
| λ = 0.3 | 极度多样 | 创意探索、头脑风暴 |

---

## 五、实时监控系统设计

### 5.1 监控架构

```
广告创意生成服务
      ↓（每次生成后）
多样性计算服务
  ├── Self-BLEU计算（2-gram）
  ├── ILD计算（SBERT向量）
  └── 簇分布均匀度
      ↓
指标写入 TimescaleDB（时序数据库）
      ↓
Grafana 仪表板展示
      ↓
告警规则引擎（Prometheus AlertManager）
  ├── Self-BLEU > 0.6 → 中等告警
  └── Self-BLEU > 0.8 → 严重告警 + 自动调参
```

### 5.2 告警与反馈闭环

```python
class DiversityMonitor:
    THRESHOLDS = {
        "self_bleu_2gram": {"warn": 0.5, "critical": 0.7},
        "ild": {"warn": 0.4, "critical": 0.25},
    }
    
    def check_and_respond(
        self,
        metrics: dict,
        generation_params: dict
    ) -> dict:
        """
        检查多样性指标，返回调整后的生成参数
        """
        actions = []
        
        # 检查Self-BLEU
        sb = metrics.get("self_bleu_2gram", 0)
        if sb > self.THRESHOLDS["self_bleu_2gram"]["critical"]:
            # 严重同质化：大幅提高temperature，并重新采样
            generation_params["temperature"] = min(
                generation_params.get("temperature", 0.7) + 0.4, 1.5
            )
            generation_params["regenerate"] = True
            actions.append(f"CRITICAL: Self-BLEU={sb:.2f}, 提高温度并重新生成")
            
        elif sb > self.THRESHOLDS["self_bleu_2gram"]["warn"]:
            # 轻度同质化：小幅提高temperature
            generation_params["temperature"] = min(
                generation_params.get("temperature", 0.7) + 0.2, 1.3
            )
            actions.append(f"WARN: Self-BLEU={sb:.2f}, 温度微调")
        
        # 检查ILD
        ild = metrics.get("ild", 1.0)
        if ild < self.THRESHOLDS["ild"]["critical"]:
            # ILD过低：启用强制多样化策略
            generation_params["force_diverse_styles"] = True
            actions.append(f"CRITICAL: ILD={ild:.2f}, 强制多样化风格")
        
        return {
            "updated_params": generation_params,
            "actions": actions,
            "alert_level": "critical" if any("CRITICAL" in a for a in actions)
                          else "warn" if actions else "ok"
        }
```

### 5.3 关键监控指标仪表板

```
┌─────────────────── 创意多样性监控 ───────────────────────┐
│                                                        │
│  Self-BLEU (2-gram)     ILD              Cluster Dist  │
│  ████████░░  0.42       ░░████████  0.68  均匀度: 0.82  │
│  [正常]                 [良好]           [良好]          │
│                                                        │
│  Top3 同质化品类：                                       │
│  1. 女装 (Self-BLEU=0.71) ⚠️                            │
│  2. 美妆 (Self-BLEU=0.65) ⚠️                            │
│  3. 家电 (Self-BLEU=0.38) ✅                            │
│                                                        │
│  最近1小时生成量：12,847 条   告警：2 条                  │
└────────────────────────────────────────────────────────┘
```

---

## 六、面试考点 Q&A

**Q1：BLEU的缺点是什么？为什么不够用于创意评估？**

A：BLEU有三个根本性缺陷。第一，只评估字面n-gram重叠，忽略语义——「我很高兴」和「我非常开心」BLEU=0但语义相同；第二，不区分重要词和停用词——「的、了、吗」这些高频词的n-gram重叠会虚高BLEU；第三，召回率被完全忽略，只看precision——一句话只说了一个词但这个词在参考中出现，BLEU依然可以很高。在广告创意评估中，两条语义完全相同但词汇不同的文案，BLEU会误判为多样（Self-BLEU低），而这正好相反。因此推荐Self-BLEU + ILD（语义）联合评估。

**Q2：语义多样性和文本多样性（n-gram）有什么区别？哪个更重要？**

A：两者互补，缺一不可。文本多样性（n-gram/BLEU）衡量表面词汇差异，容易检测"同一内容换词"的伪多样性；语义多样性（ILD）衡量深层含义差异，能识别「换汤不换药」的创意。哪个更重要取决于场景：对用户而言语义多样性更重要（看着不同才有参考价值）；对平台合规而言文本多样性更重要（字面重复会被判spam）。实际系统中必须双指标同时满足：Self-BLEU < 0.5 AND ILD > 0.4，缺一不可。

**Q3：MMR的λ参数怎么设置？有没有自适应方法？**

A：λ不应固定，可以根据业务场景自适应。基于候选集CTR方差的自适应方法：若候选CTR方差大（说明CTR分化明显，有明显优劣），增大λ偏向高CTR；若候选CTR方差小（说明候选质量相近），减小λ更偏向多样性。公式：`λ = 0.5 + 0.4 * min(ctr_std / ctr_mean, 1.0)`。另一种方法是基于广告主历史偏好：如果广告主历史上倾向于选多样风格，自动降低λ。

**Q4：如何在保证多样性的同时不损失CTR？**

A：关键是将多样性约束转化为"多样性保底"而非"多样性最大化"。具体做法：第一步，先用CTR排序取top-N（如top-20）；第二步，在top-20中应用MMR（λ=0.7），选最终K=5条——这样最终5条都来自CTR较高的候选集，保证了质量下限；第三步，多样性检查——若5条中有2条Self-BLEU > 0.7，用top-20中CTR次高且不相似的替换。这样既保证每条CTR不会太低（都在top-20内），又强制多样性。

**Q5：Self-BLEU和inter-BLEU的区别和使用场景？**

A：Self-BLEU是每条创意对其他所有创意计算BLEU，取平均；inter-BLEU是指所有创意两两之间BLEU的均值（本质相同，计算方式略有差异）。Self-BLEU更常见，因为它符合机器翻译中"用参考集评估候选"的直觉。使用场景：Self-BLEU适合实时监控单次生成批次（10条）的内部多样性；如果要跨批次评估（今天生成的10批共100条），用inter-BLEU全量计算更精确。

**Q6：大规模监控（每天百万条生成结果）的计算怎么优化？**

A：四个关键优化：(1) 采样监控——不对所有创意计算，每个品类按1%采样；(2) 批次级而非创意级——对每次生成的K条（一个批次）计算一次多样性，写入一条指标记录，而非对每条创意单独计算；(3) 轻量化指标优先——Self-BLEU用2-gram（而非4-gram），ILD用维度256的轻量SBERT（而非大模型），计算时间从100ms降到5ms；(4) 预聚合——每小时按品类聚合一次统计，Grafana展示聚合数据而非原始数据，减少查询压力。

**Q7：创意多样性和广告主转化率之间有什么关系？**

A：多项研究和内部实验表明，多样性与转化率呈倒U型关系。多样性太低（同质化）：广告主看到相似文案，只选一条上线，无法进行A/B测试，错过最优创意；多样性太高（随机多样）：质量保证不足，高多样性可能包含低质量创意。最优点（ILD=0.5~0.65）：既有足够多样性供广告主选择，又保证每条有历史效果支撑。量化关系：根据内部A/B测试，ILD从0.3提升到0.6时，广告主选择率（在10条中点击"使用"的比例）从35%提升到58%，间接带动广告消耗提升约22%。

---

*参考文献：*
- *BLEU: Papineni et al., ACL 2002*
- *Self-BLEU: Zhu et al., arXiv:1802.01886 (2018)*
- *SentEval: Conneau & Kiela, LREC 2018*
- *CheckList: Ribeiro et al., ACL 2020*
- *MMR原始应用: Carbonell & Goldstein, SIGIR 1998*
