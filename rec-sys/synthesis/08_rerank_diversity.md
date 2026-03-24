# 08. 重排与多样性 (Reranking & Diversity)

> 📚 参考文献
> - [A-Unified-Language-Model-For-Large-Scale-Search...](../../rec-sys/papers/20260321_a-unified-language-model-for-large-scale-search-recommendation-and-reasoning-at-spotify.md) — A Unified Language Model for Large Scale Search, Recommen...
> - [Gems-Breaking-The-Long-Sequence-Barrier-In-Gene...](../../rec-sys/papers/20260321_gems-breaking-the-long-sequence-barrier-in-generative-recommendation-with-a-multi-stream-decoder.md) — GEMs: Breaking the Long-Sequence Barrier in Generative Re...
> - [A Generative Re-Ranking Model For List-Level Multi](../../rec-sys/papers/20260323_a_generative_re-ranking_model_for_list-level_multi.md) — A Generative Re-ranking Model for List-level Multi-object...
> - [Deploying-Semantic-Id-Based-Generative-Retrieva...](../../rec-sys/papers/20260321_deploying-semantic-id-based-generative-retrieval-for-large-scale-podcast-discovery-at-spotify.md) — Deploying Semantic ID-based Generative Retrieval for Larg...
> - [Linear-Item-Item-Session-Rec](../../rec-sys/papers/20260319_linear-item-item-session-rec.md) — Linear Item-Item Model with Neural Knowledge for Session-...
> - [Etegrec Generative Recommender With End-To-End Lea](../../rec-sys/papers/20260323_etegrec_generative_recommender_with_end-to-end_lea.md) — ETEGRec: Generative Recommender with End-to-End Learnable...
> - [Gpr Generative Personalized Recommendation With E](../../rec-sys/papers/20260323_gpr_generative_personalized_recommendation_with_e.md) — GPR: Generative Personalized Recommendation with End-to-E...
> - [Diffgrm-Diffusion-Based-Generative-Recommendati...](../../rec-sys/papers/20260321_diffgrm-diffusion-based-generative-recommendation-model.md) — DiffGRM: Diffusion-based Generative Recommendation Model


> MelonEggLearn 整理 | 推荐系统重排层核心技术与工业实践

---

## 目录
1. [重排的目标：从精排到展示的差距](#1-重排的目标从精排到展示的差距)
2. [经典重排模型](#2-经典重排模型)
3. [多样性算法](#3-多样性算法)
4. [新颖性 vs 多样性 vs 准确性的权衡](#4-新颖性-vs-多样性-vs-准确性的权衡)
5. [生成式重排（LLM排序）](#5-生成式重排llm排序)
6. [Context-aware重排](#6-context-aware重排)
7. [工业界实践](#7-工业界实践)

---

## 1. 重排的目标：从精排到展示的差距

### 1.1 为什么需要重排层？

```
推荐系统漏斗架构：
┌─────────────────────────────────────────────────────────────┐
│  召回 (Recall)     →  百万级物品 → 千级候选                  │
│  粗排 (Pre-rank)   →  千级候选   → 百级候选                  │
│  精排 (Rank)       →  百级候选   → 十级候选 (Point-wise)     │
│  重排 (Rerank)     →  十级候选   → 最终展示 (List-wise)      │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 精排 vs 重排的本质差异

| 维度 | 精排 (Ranking) | 重排 (Reranking) |
|------|----------------|------------------|
| **优化目标** | Point-wise (单个物品CTR/CVR) | List-wise (整体列表效果) |
| **假设** | 物品间相互独立 | 物品间存在相互影响 |
| **位置因素** | 通常作为特征输入 | 显式建模位置效应 |
| **多样性** | 通常不考虑 | 核心优化目标之一 |
| **新鲜度** | 次要因素 | 重要调控手段 |
| **业务规则** | 纯模型打分 | 融合策略与规则 |

### 1.3 精排到展示的gap来源

```python
# 精排的问题：独立打分导致的不一致性
def pointwise_ranking_issue():
    """
    示例：精排模型给每个物品独立打分
    
    物品A: "iPhone 15"          score=0.95
    物品B: "iPhone 15 Pro"      score=0.93  
    物品C: "iPhone 15 Pro Max"  score=0.92
    物品D: "华为手机"            score=0.85
    
    问题：Top3全是iPhone，用户可能只需要看一个
    精排忽略了：物品间的相似性、信息的冗余性
    """
    pass

# 重排要解决的问题：
rerank_objectives = {
    "多样性": "避免结果同质化，覆盖用户多兴趣",
    "新颖性": "给用户带来惊喜，探索新兴趣", 
    "公平性": "流量分配，冷启动扶持",
    "上下文感知": "列表整体的连贯性和协调性",
    "业务目标": "GMV、时长、互动等复合目标"
}
```

### 1.4 重排层的核心挑战

1. **曝光偏差 (Exposure Bias)**：用户只能看到展示的物品，训练数据有偏
2. **位置偏差 (Position Bias)**：用户更倾向点击靠前的物品
3. **物品间关系建模**：相似性、互补性、竞争性
4. **实时性要求**：在线响应延迟要求极高 (<50ms)
5. **多目标平衡**：准确性、多样性、新颖性的帕累托最优

---

## 2. 经典重排模型

### 2.1 PRM (Personalized Re-ranking Model)

> 论文: *Personalized Re-ranking for Recommendation* (Alibaba, RecSys 2019)

**核心思想**：使用Transformer建模列表中物品间的相互影响

```python
import torch
import torch.nn as nn

class PRM(nn.Module):
    """
    PRM: Personalized Re-ranking Model
    
    输入:
    - 精排分数 (初始排序信息)
    - 物品特征
    - 用户特征
    - 位置编码
    
    输出: 每个位置的物品重排分数
    """
    def __init__(self, item_dim, user_dim, hidden_dim, num_heads=8, num_layers=2):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, item_dim)
        self.user_embedding = nn.Embedding(num_users, user_dim)
        
        # 输入变换层
        self.input_transform = nn.Linear(item_dim + user_dim + 1, hidden_dim)
        
        # Transformer Encoder建模物品间关系
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, item_ids, user_id, initial_scores, positions):
        """
        Args:
            item_ids: [B, N] - N个候选物品
            user_id: [B] - 用户ID
            initial_scores: [B, N] - 精排分数
            positions: [B, N] - 位置信息
        """
        B, N = item_ids.shape
        
        # 特征拼接
        item_emb = self.item_embedding(item_ids)  # [B, N, item_dim]
        user_emb = self.user_embedding(user_id).unsqueeze(1).expand(-1, N, -1)  # [B, N, user_dim]
        
        # 拼接特征: 物品 + 用户 + 精排分数 + 位置
        score_feat = initial_scores.unsqueeze(-1)  # [B, N, 1]
        pos_emb = self.position_embedding(positions)  # [B, N, pos_dim]
        
        input_feat = torch.cat([item_emb, user_emb, score_feat], dim=-1)
        input_feat = self.input_transform(input_feat)  # [B, N, hidden_dim]
        
        # Transformer建模物品间关系
        transformer_out = self.transformer(input_feat)  # [B, N, hidden_dim]
        
        # 预测每个位置的CTR
        ctr_logits = self.output_layer(transformer_out).squeeze(-1)  # [B, N]
        
        return ctr_logits
```

**PRM的关键创新点**：
1. **Pre-LN Transformer**: 解决训练不稳定问题
2. **个性化位置编码**: 不同用户对位置的敏感度不同
3. **初始分数作为输入**: 保留精排模型的先验知识

**公式表达**：
$$\text{PRM}(\mathbf{x}) = \text{Transformer}(\mathbf{E}_{item} + \mathbf{E}_{user} + \mathbf{E}_{pos} + \mathbf{s}_{initial})$$

### 2.2 DLCM (Deep Listwise Context Model)

> 论文: *Learning a Deep Listwise Context Model for Ranking Refinement* (SIGIR 2018)

**核心思想**：使用RNN（GRU）建模列表上下文，逐位置 refine 分数

```python
class DLCM(nn.Module):
    """
    DLCM: Deep Listwise Context Model
    
    使用GRU顺序读取列表，每个位置的表示受前面位置影响
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.score_predictor = nn.Linear(hidden_dim, 1)
        
    def forward(self, item_features):
        """
        Args:
            item_features: [B, N, D] - N个物品的特征
        """
        # GRU顺序编码
        gru_out, _ = self.gru(item_features)  # [B, N, hidden_dim]
        
        # 每个位置预测新分数
        refined_scores = self.score_predictor(gru_out).squeeze(-1)  # [B, N]
        
        return refined_scores
    
    """
    工作流程:
    
    位置1: h_1 = GRU(x_1, h_0)
    位置2: h_2 = GRU(x_2, h_1)  ← 受位置1影响
    位置3: h_3 = GRU(x_3, h_2)  ← 受位置1、2影响
    ...
    
    问题: 单向GRU，后面的位置不能影响前面（需要双向或attention改进）
    """
```

**DLCM vs PRM**：
| 特性 | DLCM | PRM |
|------|------|-----|
| 结构 | RNN/GRU | Transformer |
| 信息流动 | 单向顺序 | 双向全局 |
| 计算复杂度 | O(N) | O(N²) |
| 捕捉范围 | 局部依赖 | 全局依赖 |
| 并行性 | 顺序计算 | 可并行 |

### 2.3 SetRank (Learning to Rank as a Set)

> 论文: *SetRank: Learning a Permutation-Invariant Ranking Model* (SIGIR 2020)

**核心思想**：将排序视为集合问题，追求置换不变性 (Permutation Invariant)

```python
class SetRank(nn.Module):
    """
    SetRank: 置换不变的排序模型
    
    关键洞察：排序结果不应该依赖于输入顺序
    """
    def __init__(self, feat_dim, hidden_dim, num_layers=2):
        super().__init__()
        # 使用Induced Set Attention Block实现置换不变性
        self.isab_layers = nn.ModuleList([
            InducedSetAttentionBlock(feat_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.score_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, item_features):
        """
        Args:
            item_features: [B, N, D] - 物品特征（顺序无关）
        """
        h = item_features
        for isab in self.isab_layers:
            h = isab(h)
        
        scores = self.score_layer(h).squeeze(-1)
        return scores

class InducedSetAttentionBlock(nn.Module):
    """
    ISAB: 通过诱导点(inducing points)实现置换不变性
    避免O(N²)的attention计算
    """
    def __init__(self, dim, hidden_dim, num_induce=32):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(1, num_induce, dim))
        
        # Self-attention on inducing points
        self.attn1 = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        # Cross-attention: items -> inducing points
        self.attn2 = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        
    def forward(self, x):
        B, N, D = x.shape
        
        # Expand inducing points
        h = self.inducing_points.expand(B, -1, -1)  # [B, num_induce, D]
        
        # Step 1: 诱导点聚合信息
        h, _ = self.attn1(h, x, x)  # [B, num_induce, D]
        
        # Step 2: 物品从诱导点获取信息
        out, _ = self.attn2(x, h, h)  # [B, N, D]
        
        return out
```

**SetRank的数学表达**：

$$f(\{x_1, x_2, ..., x_n\}) = f(\{x_{\pi(1)}, x_{\pi(2)}, ..., x_{\pi(n)}\})$$

其中 $\pi$ 是任意置换函数，体现**集合的无序性**。

### 2.4 模型对比与演进

```
演进路线:

DLCM (2018)        PRM (2019)          SetRank (2020)
    │                  │                    │
    ▼                  ▼                    ▼
┌─────────┐      ┌─────────┐         ┌─────────┐
│  RNN    │  →   │Transformer│  →    │ Set Attn │
│ 局部依赖 │      │ 全局依赖  │        │ 置换不变  │
│ O(N)    │      │ O(N²)    │        │ O(N×M)   │
└─────────┘      └─────────┘         └─────────┘
                    │
                    ▼
            ┌─────────────┐
            │  后续工作    │
            ├─────────────┤
            │ MIRAGE (2020)│  - 多轮重排
            │ Seq2Slate (2019)│ - 序列生成
            │ PIER (2021)  │  - 个性化探索
            └─────────────┘
```

---

## 3. 多样性算法

### 3.1 MMR (Maximal Marginal Relevance)

> 论文: *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries* (1998)

**核心思想**：在相关性和多样性之间做贪心平衡

```python
def mmr_reranking(candidate_items, query, lambda_param=0.5, top_k=10):
    """
    MMR: Maximal Marginal Relevance
    
    Args:
        candidate_items: 候选物品列表
        query: 查询/用户表示
        lambda_param: 相关性权重 (0-1)
        top_k: 返回结果数
    
    Returns:
        重排后的物品列表
    """
    selected = []
    remaining = candidate_items.copy()
    
    while len(selected) < top_k and remaining:
        best_mmr_score = -float('inf')
        best_item = None
        
        for item in remaining:
            # 相关性分数 (与query的相似度)
            relevance = similarity(item, query)
            
            # 最大相似度 (与已选物品的相似度)
            if selected:
                max_sim_to_selected = max(similarity(item, s) for s in selected)
            else:
                max_sim_to_selected = 0
            
            # MMR分数: λ·Relevance - (1-λ)·max_sim
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_item = item
        
        selected.append(best_item)
        remaining.remove(best_item)
    
    return selected

# MMR的优缺点:
mmr_analysis = """
优点:
- 简单直观，易于实现
- 可解释性强
- 可调控lambda参数平衡相关性/多样性

缺点:
- 贪心策略，非全局最优
- 相似度函数需要精心设计
- 无法建模物品间的高阶关系
- 位置效应未考虑
"""
```

**MMR公式**：
$$\text{MMR}(D_i) = \lambda \cdot \text{Sim}(D_i, Q) - (1-\lambda) \cdot \max_{D_j \in S} \text{Sim}(D_i, D_j)$$

### 3.2 DPP (Determinantal Point Process) 行列式点过程

> 论文: *Determinantal Point Processes for Machine Learning* (2012)

**核心思想**：通过核矩阵的行列式来建模子集多样性

```python
import numpy as np
from scipy.linalg import det, solve

class DiversifiedReranking:
    """
    DPP (Determinantal Point Process) 基础实现
    """
    def __init__(self, items, quality_scores, feature_matrix):
        """
        Args:
            items: 物品列表
            quality_scores: 每个物品的质量分数 q_i
            feature_matrix: 物品特征矩阵 [N, D]
        """
        self.items = items
        self.N = len(items)
        
        # 构建核矩阵 L = diag(q) · S · S^T · diag(q)
        # 其中 S 是归一化的特征相似度矩阵
        q = np.array(quality_scores).reshape(-1, 1)
        
        # 特征相似度 (余弦相似度)
        S = feature_matrix @ feature_matrix.T
        S = S / (np.linalg.norm(feature_matrix, axis=1, keepdims=True) + 1e-8)
        S = S / (np.linalg.norm(feature_matrix, axis=1, keepdims=True).T + 1e-8)
        
        # DPP核矩阵
        self.L = (q * S) * q.T
        
    def greedy_map_inference(self, k):
        """
        贪心MAP推断：选择使det(L_Y)最大的k个物品子集
        
        P(Y) ∝ det(L_Y)  概率与子集多样性成正比
        """
        selected = []
        remaining = list(range(self.N))
        
        for _ in range(k):
            best_item = None
            best_score = -float('inf')
            
            for idx in remaining:
                candidate_set = selected + [idx]
                L_Y = self.L[np.ix_(candidate_set, candidate_set)]
                score = np.log(det(L_Y) + 1e-10)
                
                if score > best_score:
                    best_score = score
                    best_item = idx
            
            selected.append(best_item)
            remaining.remove(best_item)
        
        return [self.items[i] for i in selected]
    
    def fast_greedy_dpp(self, k):
        """
        快速贪心DPP: 利用矩阵求逆的增量更新
        时间复杂度从 O(k·N³) 降到 O(k·N·D²)
        """
        selected = []
        remaining = set(range(self.N))
        
        # 预计算对角线元素
        diags = np.diag(self.L).copy()
        
        for _ in range(k):
            if not selected:
                # 第一步：选择质量分数最高的
                best_idx = max(remaining, key=lambda i: diags[i])
            else:
                # 计算边际增益
                best_idx = max(remaining, 
                              key=lambda i: diags[i] - 
                              self.L[i, selected] @ solve(self.L[np.ix_(selected, selected)], 
                                                           self.L[selected, i]))
            
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return [self.items[i] for i in selected]


# DPP的直观理解
dpp_intuition = """
DPP核矩阵 L 的结构:

L_ij = q_i · φ(i)^T φ(j) · q_j = q_i · S_ij · q_j

其中:
- q_i: 物品i的质量分数 (对应精排分数)
- S_ij: 物品i和j的相似度
- L的行列式 det(L_Y) 衡量子集Y的多样性

几何解释:
- 把每个物品看作特征空间中的一个向量
- det(L_Y) 等于这些向量张成的平行多面体的体积
- 相似的物品 → 向量夹角小 → 体积小 → 概率低
- 不相似的物品 → 向量夹角大 → 体积大 → 概率高
"""
```

**DPP的概率公式**：
$$P(Y) = \frac{\det(\mathbf{L}_Y)}{\det(\mathbf{L} + \mathbf{I})}$$

其中 $Y$ 是物品的子集，$\mathbf{L}_Y$ 是核矩阵在子集上的限制。

### 3.3 xQuAD (Explicit Query Aspect Diversification)

```python
def xquad_reranking(query, candidates, aspects, top_k=10, lambda_param=0.5):
    """
    xQuAD: 基于显式维度的多样化
    
    假设：每个物品可以映射到若干个预定义的维度/方面
    """
    selected = []
    remaining = candidates.copy()
    
    # 计算query对每个aspect的覆盖
    aspect_coverage = {a: 0 for a in aspects}
    
    while len(selected) < top_k and remaining:
        best_item = None
        best_score = -float('inf')
        
        for item in remaining:
            # 相关性
            relevance = item.relevance_score
            
            # 新颖性: 覆盖未被充分表示的aspects
            novelty = 0
            for aspect in item.aspects:
                # 未被选中的aspect贡献更高
                novelty += (1 - aspect_coverage[aspect]) * item.aspect_weights[aspect]
            
            # xQuAD分数
            score = lambda_param * relevance + (1 - lambda_param) * novelty * relevance
            
            if score > best_score:
                best_score = score
                best_item = item
        
        # 更新aspect覆盖
        for aspect in best_item.aspects:
            aspect_coverage[aspect] += best_item.aspect_weights[aspect]
        
        selected.append(best_item)
        remaining.remove(best_item)
    
    return selected
```

### 3.4 多样性算法对比

| 算法 | 核心机制 | 复杂度 | 优点 | 缺点 |
|------|----------|--------|------|------|
| **MMR** | 贪心选择，最大化边际相关 | O(k·N) | 简单、可解释 | 只考虑两两相似 |
| **DPP** | 核矩阵行列式 | O(k·N³)→O(k·N·D²) | 数学优雅、考虑高阶关系 | 核函数设计难 |
| **xQuAD** | 显式维度覆盖 | O(k·N·A) | 可解释性强、可控 | 需要预定义aspects |
| **MSD** | 子模函数最大化 | O(k·N) | 理论保证 | 需要子模性假设 |
| **Intent-aware** | 用户意图建模 | 较高 | 个性化强 | 意图识别难度大 |

---

## 4. 新颖性 vs 多样性 vs 准确性的权衡

### 4.1 三个维度的定义

```
┌─────────────────────────────────────────────────────────────────┐
│                        推荐效果的三维空间                          │
│                                                                  │
│    准确性 (Accuracy)                                             │
│         ↑                                                        │
│         │  理想区域                                               │
│         │     ★ (帕累托前沿)                                      │
│         │    /│\                                                  │
│         │   / │ \                                                 │
│         │  /  │  \                                                │
│         │ /   │   \                                               │
│         │/    │    \                                              │
│  多样性 ←─────┼─────→ 新颖性                                       │
│    (Diversity)│   (Novelty)                                       │
│               │                                                  │
│         随机推荐                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

| 维度 | 定义 | 衡量指标 | 优化目标 |
|------|------|----------|----------|
| **准确性** | 推荐物品符合用户已知兴趣 | CTR, CVR, NDCG | 最大化点击/转化 |
| **多样性** | 推荐列表覆盖用户多兴趣 | ILD, DivRatio, α-NDCG | 避免同质化 |
| **新颖性** | 推荐用户未知的惊喜内容 | 新颖度比率, 意外性 | 探索新兴趣 |

### 4.2 常用的多目标融合策略

```python
class MultiObjectiveReranking:
    """
    多目标重排框架
    """
    def __init__(self):
        self.objectives = {
            'accuracy': AccuracyScorer(),
            'diversity': DiversityScorer(),
            'novelty': NoveltyScorer()
        }
    
    # 1. 线性加权融合
    def linear_fusion(self, candidates, weights):
        """
        score = w1·accuracy + w2·diversity + w3·novelty
        
        问题: 不同指标量纲不同，需要归一化
        """
        scores = {}
        for item in candidates:
            score = sum(
                weights[obj] * self.objectives[obj].score(item)
                for obj in weights
            )
            scores[item] = score
        return sorted(candidates, key=lambda x: scores[x], reverse=True)
    
    # 2. 约束优化
    def constrained_optimization(self, candidates, min_diversity=0.3):
        """
        max: accuracy
        s.t: diversity >= threshold
        """
        # 从准确性的Top-K开始，逐步用多样性替换
        sorted_by_acc = sorted(candidates, 
                              key=lambda x: self.objectives['accuracy'].score(x),
                              reverse=True)
        
        result = []
        for item in sorted_by_acc:
            if len(result) < top_k:
                # 检查多样性约束
                test_list = result + [item]
                if self.objectives['diversity'].score_list(test_list) >= min_diversity:
                    result.append(item)
        
        return result
    
    # 3. 帕累托优化 (进化算法)
    def pareto_optimization(self, candidates, population_size=100):
        """
        使用NSGA-II等进化算法寻找帕累托前沿
        """
        # 简化的帕累托选择
        # 实际使用需要完整的遗传算法实现
        pass
    
    # 4. 多臂老虎机 (MAB)
    def mab_exploration(self, candidates, context, epsilon=0.1):
        """
        ε-贪婪策略平衡探索和利用
        """
        if random.random() < epsilon:
            # 探索: 随机选择新颖性高的
            return self.explore_novel_items(candidates)
        else:
            # 利用: 选择准确性高的
            return self.exploit_known_interests(candidates)
```

### 4.3 评估指标

```python
# 多样性指标
def intra_list_diversity(recommendations, similarity_fn):
    """
    ILD: 列表内平均两两距离
    ILD = (1/|R|(|R|-1)) Σ_{i,j∈R, i≠j} (1 - sim(i,j))
    """
    n = len(recommendations)
    total_dist = 0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            total_dist += 1 - similarity_fn(recommendations[i], recommendations[j])
            count += 1
    return total_dist / count if count > 0 else 0

def diversity_ratio(recommendations, categories):
    """
    类别覆盖率
    """
    covered = set(item.category for item in recommendations)
    return len(covered) / len(categories)

# 新颖性指标
def novelty(recommendations, item_popularity):
    """
    平均自信息: -log(p(i))
    越不热门的物品，新颖性越高
    """
    return sum(-np.log(item_popularity[item]) for item in recommendations) / len(recommendations)

def unexpectedness(recommendations, baseline_predictions):
    """
    意外性: 与基线推荐的重叠度越低，越意外
    """
    overlap = len(set(recommendations) & set(baseline_predictions))
    return 1 - overlap / len(recommendations)

# 准确性指标 (NDCG)
def dcg(scores, k):
    """Discounted Cumulative Gain"""
    return sum((2**s - 1) / np.log2(i + 2) for i, s in enumerate(scores[:k]))

def ndcg(recommendations, ideal_recommendations, k):
    """Normalized DCG"""
    actual_dcg = dcg([item.relevance for item in recommendations], k)
    ideal_dcg = dcg([item.relevance for item in ideal_recommendations], k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0
```

### 4.4 工业界的权衡策略

```python
industrial_strategies = {
    "快手": {
        "方法": "分层重排",
        "策略": "先保准确性的Top-N，再用多样性填充剩余位置",
        "细节": "流量实验发现：多样性提升20%，人均消费时长+3%"
    },
    
    "淘宝": {
        "方法": "多目标Pareto优化",
        "策略": "GMV × CTR × Diversity 的多目标优化",
        "细节": "使用NSGA-II离线求解Pareto前沿，在线查表"
    },
    
    "Netflix": {
        "方法": "上下文感知多样性",
        "策略": "不同场景下动态调整多样性强度",
        "细节": "Browse页面高多样性，Continue Watching低多样性"
    },
    
    "Spotify": {
        "方法": "新颖性探索",
        "策略": "30%位置固定给探索(Discovery Mode)",
        "细节": "Weekly Discovery播放列表就是纯新颖性优化"
    }
}
```

---

## 5. 生成式重排（LLM排序）

### 5.1 LLM for Reranking的基本范式

```python
"""
LLM重排的三种范式:

1. Point-wise: LLM判断单个物品是否相关
2. Pair-wise: LLM比较两个物品的相对顺序  
3. List-wise: LLM直接生成完整排序列表 ← 主流方向
"""

class LLMReranker:
    """
    List-wise LLM重排
    """
    def __init__(self, llm_model):
        self.llm = llm_model
    
    def build_prompt(self, user_profile, candidate_items, instruction):
        """
        构建重排prompt
        """
        prompt = f"""你是一个个性化推荐助手。请根据用户画像，从候选商品中选择最相关的{top_k}个，并按相关性排序。

用户画像:
{user_profile}

候选商品:
"""
        for i, item in enumerate(candidate_items):
            prompt += f"{i+1}. {item.title} - {item.category} - 价格:{item.price}\\n"
        
        prompt += f"""
{instruction}

请直接输出排序后的商品编号，格式如: [3, 1, 7, 2, ...]
"""
        return prompt
    
    def rerank(self, user_profile, candidate_items, top_k=10):
        prompt = self.build_prompt(user_profile, candidate_items, top_k)
        
        # 调用LLM
        response = self.llm.generate(prompt, temperature=0.3)
        
        # 解析输出
        ranked_indices = self.parse_ranking(response)
        
        return [candidate_items[i-1] for i in ranked_indices]
```

### 5.2 代表性工作

| 工作 | 年份 | 核心贡献 | 关键创新 |
|------|------|----------|----------|
| **PRP (Pairwise Reranking Prompting)** | 2023 | 发现pair-wise比point-wise效果更好 | 成对比较减少幻觉 |
| **LTR (Listwise T5)** | 2023 | 微调T5做列表式排序 | 生成式排序模型 |
| **RankGPT** | 2023 | 滑动窗口 + 归并排序 | 解决长列表问题 |
| **P-Llama** | 2024 | 参数高效微调 + 位置感知 | 冷启动场景适配 |
| **Set-LLM** | 2024 | 集合输入 + 置换不变输出 | 顺序无关的表示 |

### 5.3 RankGPT详解

```python
class RankGPT:
    """
    RankGPT: 处理长列表的滑动窗口策略
    
    核心问题: LLM上下文长度有限，无法一次性处理所有候选物品
    """
    def __init__(self, llm, window_size=20, step_size=10):
        self.llm = llm
        self.window_size = window_size
        self.step_size = step_size
    
    def sliding_window_sort(self, items, query):
        """
        滑动窗口排序:
        1. 用滑动窗口对局部排序
        2. 归并相邻窗口的结果
        """
        # 初始化为精排顺序
        current_order = items.copy()
        
        # 多轮滑动
        for iteration in range(max_iterations):
            has_change = False
            
            for start in range(0, len(items) - window_size, step_size):
                window = current_order[start:start + window_size]
                
                # 用LLM对这20个物品排序
                ranked_window = self.llm_sort_window(window, query)
                
                # 更新全局顺序
                current_order[start:start + window_size] = ranked_window
                
                if ranked_window != window:
                    has_change = True
            
            if not has_change:
                break
        
        return current_order
    
    def llm_sort_window(self, window_items, query):
        """
        用LLM对窗口内物品排序
        """
        prompt = f"""给定查询: {query}

请对以下商品按相关性从高到低排序:
{self.format_items(window_items)}

输出排序后的序号列表，如: [3,1,4,2,...]
"""
        response = self.llm.generate(prompt)
        indices = self.parse_indices(response)
        return [window_items[i-1] for i in indices]

# RankGPT的复杂度分析
rankgpt_complexity = """
时间复杂度: O((N/W) × W log W × I) ≈ O(N log W × I)
- N: 候选物品数
- W: 窗口大小
- I: 迭代轮数

优势:
- 可处理任意长度列表
- 利用LLM的语义理解能力
- 多轮迭代逐步refine

局限:
- 调用次数多，成本高
- 延迟问题 (需要优化到可在线)
- 一致性不能保证全局最优
"""
```

### 5.4 工业落地的挑战与解决方案

```python
llm_rerank_challenges = {
    "延迟问题": {
        "挑战": "LLM推理延迟100ms-数秒，无法满足在线要求",
        "解决方案": [
            "1. 蒸馏: 用大模型生成伪标签，训练小模型",
            "2. 缓存: 常见query的排序结果缓存",
            "3. 剪枝: 只用LLM处理头部疑难案例",
            "4. 异步: LLM重排作为异步优化，不阻塞主链路"
        ]
    },
    
    "成本问题": {
        "挑战": "API调用成本高，无法全量",
        "解决方案": [
            "1. 本地化部署小模型 (7B/13B)",
            "2. 分桶策略: 高价值用户才用LLM",
            "3. 离线预计算 + 在线Lookup"
        ]
    },
    
    "一致性问题": {
        "挑战": "LLM输出不稳定，多次调用结果不一致",
        "解决方案": [
            "1. temperature=0，greedy解码",
            "2. 多次采样取平均 (Self-consistency)",
            "3. 输出格式约束 (JSON mode)"
        ]
    },
    
    "可解释性": {
        "挑战": "需要解释为什么这样排序",
        "解决方案": [
            "1. Chain-of-thought prompting",
            "2. 让LLM生成推荐理由",
            "3. 与可解释模型结合"
        ]
    }
}
```

### 5.5 实践建议

```
LLM重排的适用场景:

✅ 适合:
- 内容理解复杂的场景 (如文章、视频)
- 长尾query/冷启动用户
- 需要强解释性的场景
- 离线实验和数据分析

❌ 不适合:
- 延迟敏感的核心链路
- 海量物品的粗排/召回
- 需要严格一致性的场景
- 成本敏感的初期阶段

推荐落地路径:
Phase 1: 离线A/B测试，验证效果
Phase 2: 小流量实验，收集bad case
Phase 3: 蒸馏小模型，逐步替换
Phase 4: 全量上线，持续监控
```

---

## 6. Context-aware重排

### 6.1 上下文感知的维度

```
上下文类型:
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  1. 用户上下文                                                │
│     ├── 会话历史 (Session History)                           │
│     ├── 实时行为 (Real-time Behavior)                        │
│     └── 设备/位置/时间 (Device/Location/Time)                │
│                                                              │
│  2. 物品上下文                                                │
│     ├── 列表内物品关系 (Intra-list Relations)                │
│     ├── 库存/价格变化 (Dynamic Features)                     │
│     └── 社交信号 (Social Signals)                            │
│                                                              │
│  3. 环境上下文                                                │
│     ├── 流量来源 (Traffic Source)                            │
│     ├── 促销活动 (Campaign Context)                          │
│     └── 竞争态势 (Competitive Context)                       │
│                                                              │
│  4. 位置上下文                                                │
│     ├── 页面位置 (Slot Position)                             │
│     ├── 曝光上下文 (Above/Below)                             │
│     └── 屏幕可见性 (Viewport Visibility)                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 Context-aware模型架构

```python
class ContextAwareReranker(nn.Module):
    """
    上下文感知重排模型
    """
    def __init__(self, item_dim, user_dim, context_dim, hidden_dim):
        super().__init__()
        
        # 物品编码器
        self.item_encoder = ItemEncoder(item_dim, hidden_dim)
        
        # 用户编码器
        self.user_encoder = UserEncoder(user_dim, hidden_dim)
        
        # 上下文编码器 (多模态)
        self.context_encoders = nn.ModuleDict({
            'session': SessionEncoder(hidden_dim),
            'realtime': RealtimeEncoder(hidden_dim),
            'position': PositionEncoder(hidden_dim),
            'environment': EnvironmentEncoder(hidden_dim)
        })
        
        # 交叉注意力: 物品间关系
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
        # 融合层
        self.fusion_layer = ContextFusionLayer(hidden_dim * 5, hidden_dim)
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, items, user, contexts):
        """
        Args:
            items: [B, N, item_dim] - N个候选物品
            user: [B, user_dim] - 用户特征
            contexts: dict - 各类上下文特征
        """
        B, N = items.shape[:2]
        
        # 编码各维度
        item_emb = self.item_encoder(items)  # [B, N, H]
        user_emb = self.user_encoder(user).unsqueeze(1)  # [B, 1, H]
        
        # 编码上下文
        context_embs = []
        for ctx_type, encoder in self.context_encoders.items():
            ctx_emb = encoder(contexts[ctx_type])  # [B, H]
            context_embs.append(ctx_emb.unsqueeze(1))
        
        # 物品间交叉注意力
        item_emb, _ = self.cross_attention(item_emb, item_emb, item_emb)
        
        # 融合所有信息
        combined = torch.cat([item_emb, user_emb.expand(-1, N, -1)] + 
                            [c.expand(-1, N, -1) for c in context_embs], dim=-1)
        fused = self.fusion_layer(combined)
        
        # 预测分数
        scores = self.output(fused).squeeze(-1)
        
        return scores
```

### 6.3 位置偏差建模 (Position Bias)

```python
class PositionBiasModeling:
    """
    位置偏差建模的几种方法
    """
    
    # 1. Position as Feature (最简单)
    def position_as_feature(self, features, position):
        """将位置作为离散/连续特征输入模型"""
        position_emb = self.position_embedding(position)
        return torch.cat([features, position_emb], dim=-1)
    
    # 2. Examine-based Model (基于检查概率)
    def examination_model(self, relevance_score, position):
        """
        P(click|pos) = P(examine|pos) × P(relevant)
        
        点击概率 = 检查概率 × 相关概率
        """
        examine_prob = self.examination_predictor(position)  # 位置越靠前，检查概率越高
        click_prob = examine_prob * relevance_score
        return click_prob
    
    # 3. PAL (Position-bias Aware Learning)
    def pal_model(self, item_features, position, user_features):
        """
        分别建模位置效应和相关性
        """
        # Tower 1: 位置效应 (与物品无关)
        position_effect = self.position_tower(position, user_features)
        
        # Tower 2: 物品相关性 (与位置无关)
        relevance = self.relevance_tower(item_features, user_features)
        
        # 组合
        click_prob = position_effect * relevance
        return click_prob
    
    # 4. IPW (Inverse Propensity Weighting)
    def ipw_correction(self, observed_clicks, position):
        """
        训练时用逆概率加权校正位置偏差
        """
        propensity = self.get_propensity_score(position)
        # 损失函数中使用: loss = (observed_click / propensity) × log_loss
        corrected_weight = observed_clicks / (propensity + 1e-6)
        return corrected_weight
```

### 6.4 会话上下文建模

```python
class SessionContextReranker(nn.Module):
    """
    基于会话历史的重排
    
    场景: 用户在当前会话中已浏览/点击了某些物品
    目标: 根据会话历史调整后续推荐
    """
    def __init__(self, hidden_dim):
        super().__init__()
        
        # 会话序列编码器 (GRU/Transformer)
        self.session_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # 候选物品与会话历史的交互
        self.interaction_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
        # 实时兴趣提取
        self.realtime_interest = RealtimeInterestExtractor(hidden_dim)
        
    def forward(self, candidates, session_history, current_intent):
        """
        Args:
            candidates: 候选物品 [B, N, D]
            session_history: 会话历史 [B, T, D]
            current_intent: 当前意图 [B, D]
        """
        # 编码会话历史
        session_repr, _ = self.session_encoder(session_history)  # [B, T, D]
        session_summary = session_repr[:, -1, :]  # 取最后时刻
        
        # 计算候选物品与会话历史的注意力
        candidate_attn, weights = self.interaction_attention(
            query=candidates,
            key=session_history,
            value=session_history
        )
        
        # 实时兴趣调整
        adjusted_scores = self.realtime_interest(
            candidates, session_summary, current_intent
        )
        
        return adjusted_scores
```

---

## 7. 工业界实践

### 7.1 淘宝序列检索重排

> 来源: 阿里巴巴技术分享 (2021-2023)

```
淘宝推荐系统架构:
┌─────────────────────────────────────────────────────────────────────┐
│                           淘宝首页推荐                                │
├─────────────────────────────────────────────────────────────────────┤
│  召回层                                                              │
│  ├── i2i (ItemCF, Swing)                                            │
│  ├── u2i (DIN, MIND)                                                │
│  ├── 向量召回 (MGN, Graph Embedding)                                 │
│  └── 实时触发 (实时行为触发)                                          │
├─────────────────────────────────────────────────────────────────────┤
│  粗排层 (Pre-rank)                                                   │
│  └── 轻量级模型 (浅层DNN, 知识蒸馏)                                   │
├─────────────────────────────────────────────────────────────────────┤
│  精排层 (Rank)                                                       │
│  └── 深度模型 (DIEN, BST, 多任务: MMoE)                              │
├─────────────────────────────────────────────────────────────────────┤
│  重排层 (Rerank)  ← 本节重点                                         │
│  ├── 序列生成重排 (Seq2Slate)                                        │
│  ├── 多样性控制 (DPP)                                                │
│  └── 业务规则融合                                                    │
└─────────────────────────────────────────────────────────────────────┘
```

**淘宝重排核心方案：序列生成模型 (Seq2Slate)**

```python
class TaobaoSeq2Slate:
    """
    淘宝序列生成重排
    
    核心思想: 把重排看作序列生成问题，使用Pointer Network生成排序序列
    """
    def __init__(self, hidden_dim=256):
        self.encoder = ItemSetEncoder(hidden_dim)
        self.decoder = PointerDecoder(hidden_dim)
        self.critic = ValueNetwork(hidden_dim)  # 用于RL训练
    
    def forward(self, candidate_items, user_profile, max_len=10):
        """
        生成排序序列
        """
        # 编码候选物品集合
        memory = self.encoder(candidate_items, user_profile)  # [N, H]
        
        # 自回归生成序列
        generated_sequence = []
        decoder_state = self.init_decoder_state(user_profile)
        
        for step in range(max_len):
            # Pointer Network: 选择下一个物品
            pointer_logits = self.decoder(decoder_state, memory, generated_sequence)
            
            # 采样或贪婪选择
            next_item_idx = self.select_next_item(pointer_logits, generated_sequence)
            generated_sequence.append(next_item_idx)
            
            # 更新decoder状态
            decoder_state = self.update_state(decoder_state, memory[next_item_idx])
        
        return generated_sequence
    
    def train_with_rl(self, batch_data):
        """
        使用强化学习训练 (REINFORCE / PPO)
        
        Reward设计:
        - 点击奖励: 用户是否点击
        - 转化奖励: 是否购买
        - 多样性奖励: 类别覆盖度
        - 新鲜度奖励: 物品年龄
        """
        sequences, rewards = batch_data
        
        # 计算策略梯度
        log_probs = self.compute_log_probs(sequences)
        baseline = self.critic(sequences)
        
        advantage = rewards - baseline
        loss = -(log_probs * advantage).mean()
        
        return loss

# 淘宝重排的关键设计
taobao_rerank_design = """
关键设计点:

1. 多目标Reward:
   R = α·R_click + β·R_convert + γ·R_diversity + δ·R_freshness

2. 位置感知:
   Decoder中显式融入位置embedding，不同位置的期望不同

3. 多样性约束:
   在生成过程中加入硬约束，确保每k个位置覆盖m个类目

4. 在线 serving:
   - 使用TVM优化模型推理
   - 序列生成改为并行beam search
   - 延迟控制在30ms以内
"""
```

**淘宝DPP多样性实践**

```python
class TaobaoDPPDiversity:
    """
    淘宝DPP多样性算法工程实践
    """
    def __init__(self):
        # 核矩阵分解优化
        self.use_low_rank_approx = True
        self.rank = 64  # 低秩近似
    
    def build_kernel_matrix_fast(self, items, q_scores, features):
        """
        快速构建核矩阵
        
        优化点:
        1. 预计算特征向量
        2. 低秩分解: L ≈ V^T V
        3. 增量行列式计算
        """
        N = len(items)
        
        # 低秩分解: L = D^{1/2} · V^T · V · D^{1/2}
        D_sqrt = np.diag(np.sqrt(q_scores))
        
        # 使用随机投影降维
        V = self.random_projection(features, self.rank)  # [rank, N]
        
        # L的近似
        L_approx = D_sqrt @ (V.T @ V) @ D_sqrt
        
        return L_approx
    
    def online_dpp_sample(self, candidates, top_k):
        """
        在线DPP采样 (优化版本)
        
        延迟: 10-15ms for 100 candidates
        """
        # 预筛选：只考虑精排Top-M
        m_candidates = candidates[:50]
        
        # 使用贪心DPP，而非精确采样
        selected = self.fast_greedy_map(m_candidates, top_k)
        
        return selected
```

### 7.2 快手重排演进

> 来源: 快手技术分享 (2020-2024)

```
快手重排演进路线:

2019 ──────────────────────────────────────────────────────────► 2024

Phase 1: 规则重排
  └── 启发式多样性规则
  └── MMR贪心选择
  └── 简单轮询打散

Phase 2: 模型化重排
  └── 引入PRM模型 (Transformer-based)
  └── 多目标融合 (观看时长 + 互动)
  └── 分层打散策略

Phase 3: 生成式重排  
  └── Seq2Slate序列生成
  ├── 强化学习训练 (PPO)
  └── 多样性约束

Phase 4: 全链路重排
  └── 粗排+精排+重排联合优化
  ├── 端云协同重排
  └── 实时反馈闭环
```

**快手生成式重排细节**

```python
class KuaishouGenerativeRerank:
    """
    快手生成式重排 (Kuaishou KDD 2022)
    """
    def __init__(self):
        self.encoder = MultiInterestEncoder()
        self.decoder = TransformerDecoder()
        self.diversity_controller = DiversityController()
    
    def multi_interest_encoding(self, user_history):
        """
        多兴趣编码: 用户可能有多个兴趣点
        """
        # 使用胶囊网络或Transformer提取多兴趣
        interests = self.encoder(user_history)  # [K, D], K个兴趣向量
        return interests
    
    def diversity_constrained_generation(self, candidates, interests, k=10):
        """
        带多样性约束的序列生成
        """
        sequence = []
        coverage_vector = torch.zeros(self.num_categories)
        
        for position in range(k):
            # 计算每个候选与当前兴趣的匹配度
            match_scores = self.compute_match_scores(candidates, interests)
            
            # 计算多样性增益
            diversity_gains = self.compute_diversity_gains(candidates, coverage_vector)
            
            # 融合分数
            combined_scores = (
                self.alpha * match_scores + 
                self.beta * diversity_gains +
                self.gamma * self.position_bias[position]
            )
            
            # 选择最优
            best_idx = torch.argmax(combined_scores)
            selected = candidates[best_idx]
            
            # 更新覆盖率
            coverage_vector += self.get_category_vector(selected)
            sequence.append(selected)
            candidates.pop(best_idx)
        
        return sequence
    
    def compute_diversity_gains(self, candidates, coverage_vector):
        """
        边际多样性增益
        """
        gains = []
        for cand in candidates:
            cat_vec = self.get_category_vector(cand)
            # 未被覆盖的类目增益更高
            gain = (1 - coverage_vector) @ cat_vec
            gains.append(gain)
        return torch.tensor(gains)
```

**快手实时反馈闭环**

```python
class RealtimeFeedbackLoop:
    """
    快手实时反馈闭环系统
    
    目标: 用户反馈(滑动/停留/点赞)实时影响后续推荐
    """
    def __init__(self):
        self.realtime_feature_extractor = RealtimeFeatureExtractor()
        self.fast_model_update = OnlineLearningModule()
    
    def on_user_action(self, action_type, item, context):
        """
        处理用户实时行为
        
        action_type: 'scroll', 'stay', 'like', 'share', 'comment'
        """
        # 1. 提取实时特征
        realtime_features = self.realtime_feature_extractor(
            action_type, item, context, timestamp=now()
        )
        
        # 2. 更新用户实时兴趣
        self.update_realtime_user_profile(user_id, realtime_features)
        
        # 3. 触发重排 (如果还在当前请求的生命周期内)
        if context.request_active:
            self.trigger_rerank(context.request_id)
    
    def trigger_rerank(self, request_id):
        """
        触发实时重排
        
        优化: 只对未曝光位置重排，避免闪烁
        """
        # 获取当前候选队列
        candidates = self.get_remaining_candidates(request_id)
        
        # 使用最新的用户画像重排
        new_ranking = self.reranker.rerank(
            candidates, 
            user_profile=self.get_latest_profile()
        )
        
        # 推送给客户端
        self.push_to_client(request_id, new_ranking)
```

### 7.3 其他工业界实践

```python
industry_practices = {
    "字节跳动 (抖音/头条)": {
        "特点": "超大规模实时特征",
        "重排策略": [
            "FTRL在线学习实时调整权重",
            "多场景统一重排模型",
            "流量调控：扶持新内容、作者"
        ],
        "创新点": "端上重排 (Edge Reranking) 减少延迟"
    },
    
    "美团": {
        "特点": "LBS强约束",
        "重排策略": [
            "地理位置聚合重排",
            "商户/商品混合重排",
            "配送时间预估融入排序"
        ],
        "创新点": "时空上下文建模"
    },
    
    "小红书": {
        "特点": "内容社区，强调发现",
        "重排策略": [
            "Explore & Exploit平衡",
            "笔记-商品跨域重排",
            "社交关系融入多样性"
        ],
        "创新点": "双列流的重排优化"
    },
    
    "Netflix": {
        "特点": "全球多样化内容",
        "重排策略": [
            "Page-level优化 (整个页面)",
            "多行结果的协调",
            "个性化行排序 + 行内排序"
        ],
        "创新点": "Contextual Bandit for exploration"
    },
    
    "YouTube": {
        "特点": "视频时长差异大",
        "重排策略": [
            "Expected Watch Time作为目标",
            "Session-based重排",
            "多目标: 点击、时长、订阅"
        ],
        "创新点": "Watched video排除，避免重复"
    }
}
```

### 7.4 工程实践要点

```python
production_tips = """
══════════════════════════════════════════════════════════════════
                    重排层工程实践要点
══════════════════════════════════════════════════════════════════

1. 延迟优化
   ├─ 模型量化 (INT8)
   ├─ 算子融合 (TensorRT/TVM)
   ├─ 批处理推理
   ├─ 缓存热点结果
   └─ 异步特征获取

2. AB实验设计
   ├─ 分层实验：不同桶独立测试不同策略
   ├─ Holdout组：长期观察重排效果
   ├─ 互斥实验：多样性 vs 准确性权衡
   └─ 长期效应：避免短期指标损害长期价值

3. 监控指标
   ├─ 业务指标: CTR, CVR, GMV, 观看时长
   ├─ 多样性指标: 类目分布, ILD, 覆盖率
   ├─ 公平性指标: 头部/长尾流量分布
   ├─ 新鲜度指标: 新内容曝光占比
   └─ 系统指标: 延迟, 错误率, 资源消耗

4. 问题排查
   ├─ 特征一致性: 离线/在线特征对齐
   ├─ 位置偏差: 不同位置的点击率校准
   ├─ 反馈循环: 模型bias累积检测
   └─ 异常检测: 自动降级策略

5. 冷启动处理
   ├─ 新物品探索: ε-贪婪, Thompson Sampling
   ├─ 新用户处理: 默认策略 + 快速学习
   ├─ 探索利用平衡: UCB, LinUCB
   └─ 人群泛化: 相似用户策略迁移
"""
```

---

## 参考资料

### 经典论文
1. **PRM** (2019): *Personalized Re-ranking for Recommendation* - 阿里
2. **DLCM** (2018): *Learning a Deep Listwise Context Model* - 清华
3. **SetRank** (2020): *SetRank: Learning a Permutation-Invariant Ranking Model* - 人大
4. **Seq2Slate** (2019): *Seq2Slate: Re-ranking and Slate Optimization with RNNs* - 微软
5. **DPP** (2012): *Determinantal Point Processes for Machine Learning* - 基础理论

### 多样性相关
6. **MMR** (1998): *Maximal Marginal Relevance* - 经典算法
7. **xQuAD** (2009): *Explicit Query Aspect Diversification* - 维度多样化
8. **Intent-aware** (2011): *Intent-aware Diversification* - 意图感知

### LLM重排
9. **RankGPT** (2023): *Is ChatGPT Good at Search?* - LLM排序
10. **PRP** (2023): *Large Language Models are Effective Text Rankers* - 成对比较
11. **P-Llama** (2024): *Parameter-Efficient LLM Reranking* - 高效微调

### 工业实践
12. 阿里妈妈技术博客: 淘宝重排实践系列
13. 快手技术博客: 重排算法演进
14. KDD/WWW/RecSys 工业track论文

---

*文档版本: 1.0 | 更新时间: 2024 | MelonEggLearn*
