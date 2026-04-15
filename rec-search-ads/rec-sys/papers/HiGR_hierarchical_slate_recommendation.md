# HiGR: Efficient Generative Slate Recommendation via Hierarchical Planning and Multi-Objective Preference Alignment

> 来源：https://arxiv.org/abs/2512.24787 | 领域：rec-sys | 学习日期：20260401
> 注：已在腾讯商业平台部署，服务数亿用户

## 问题定义

**Slate 推荐（列表推荐）**：与单物品推荐不同，Slate 推荐向用户一次性展示一个有序的物品列表（Slate），在主流内容平台（短视频 Feed、电商首页）中无处不在。Slate 推荐的核心挑战在于需要同时优化**列表整体质量**，而非单个物品的相关性。

**现有生成式 Slate 推荐的三大缺陷：**

1. **物品 Tokenization 纠缠（Entangled Tokenization）**：现有 SID 编码将重建（Content Reconstruction）与生成（Generation Control）混为一谈，无法灵活控制生成列表的多样性、新颖性等属性

2. **顺序解码低效（Inefficient Sequential Decoding）**：自回归逐 token 生成一个 Slate（如 10 个物品 × 每物品 3 个 token = 30 步解码），推理延迟极高

3. **缺乏整体 Slate 规划（No Holistic Planning）**：逐物品生成缺乏对整个列表的全局视角，导致列表内物品高度相似、缺乏互补性

**工业背景**：
- 主流在线平台日均亿级 Slate 推荐请求
- Slate 质量直接影响用户观看时长、点击率等核心业务指标
- 生成式方法的**推理延迟**是工业部署的主要障碍

## 核心方法与创新点

### 1. 解耦 Auto-Encoder：可控语义 ID

**创新**：设计了一个结合**残差量化（Residual Quantization, RQ）**和**对比约束（Contrastive Constraints）**的 Auto-Encoder，将物品 tokenize 为**语义结构化 ID（Semantically Structured IDs）**，支持可控生成。

**RQ 编码过程：**

$$
\mathbf{z}_0 = \text{Encoder}(\text{item}_{	ext{content}})
$$

$$
c_1 = \arg\min_k \|\mathbf{z}_0 - \mathbf{e}_k^{(1)}\|, \quad \mathbf{r}_1 = \mathbf{z}_0 - \mathbf{e}_{c_1}^{(1)}
$$

$$
c_2 = \arg\min_k \|\mathbf{r}_1 - \mathbf{e}_k^{(2)}\|, \quad \mathbf{r}_2 = \mathbf{r}_1 - \mathbf{e}_{c_2}^{(2)}
$$

$$
\text{SID}(item) = [c_1, c_2, c_3, \ldots, c_L]
$$

**对比约束**：在训练中加入对比损失，确保语义相近的物品在 SID 空间中的距离更小：

$$
\mathcal{L}_{contrast} = -\log \frac{\exp(\text{sim}(z_i, z_j^+)/\tau)}{\exp(\text{sim}(z_i, z_j^+)/\tau) + \sum_k \exp(\text{sim}(z_i, z_k^-)/\tau)}
$$

### 2. 分层规划（Hierarchical Planning）：列表级 + 物品级解耦

**核心创新**：将 Slate 生成解耦为两个阶段，大幅减少搜索空间，实现高效生成：

**阶段 1：列表级规划（List-Level Planning）**
- 生成当前 Slate 的**整体意图向量（Global Slate Intent）** $\mathbf{s}_{plan}$
- 输入：用户历史 + 上下文信号（时间、位置、设备）
- 输出：一个连续向量，表示"这个列表应该是什么风格/主题"
- **关键设计**：列表规划不直接生成物品，而是生成整体调控信号

$$
\mathbf{s}_{plan} = \text{ListPlanner}(\mathbf{h}_{user}, \mathbf{h}_{context})
$$

**阶段 2：物品级解码（Item-Level Decoding）**
- 以 $\mathbf{s}_{plan}$ 为条件，自回归生成每个物品的 SID
- 关键优化：有了整体规划后，物品的搜索空间从"全部候选集"缩小到"符合规划意图的子空间"

$$
P(item_k | user, \mathbf{s}_{plan}, item_{1:k-1}) = \prod_t P(c_t | \mathbf{s}_{plan}, c_{<t})
$$

**效率提升机制**：
- 列表规划 $\mathbf{s}_{plan}$ 的生成是一次前向计算（非自回归）
- 有了规划后，物品级解码的 Beam Search 搜索树大幅剪枝
- 相比无规划的端到端自回归生成，解码步数减少约 **5x**

### 3. 多目标列表级偏好对齐（Multi-Objective Listwise Preference Alignment）

**问题**：工业推荐需要同时满足多个业务目标（点击率、完播率、多样性、商业目标），传统的 MLE（最大似然估计）训练只优化序列生成概率，无法感知这些业务目标。

**解决方案**：基于隐式用户反馈（观看时长、完播率、点赞等）构建**列表级偏好对**，用 DPO（Direct Preference Optimization）思想做 Slate 级对齐：

$$
\mathcal{L}_{align} = -\log\sigma\left(\beta \log \frac{\pi_\theta(\text{Slate}_{	ext{win}} | u)}{\pi_{ref}(\text{Slate}_{	ext{win}} | u)} - \beta \log \frac{\pi_\theta(\text{Slate}_{	ext{lose}} | u)}{\pi_{ref}(\text{Slate}_{	ext{lose}} | u)}\right)
$$

其中 $\text{Slate}_{	ext{win}}$ 是用户反馈更好的列表（高完播率），$\text{Slate}_{	ext{lose}}$ 是反馈差的列表。

## 实验结论

**离线对比（vs SOTA 基线）：**

| 方法 | Recall@10 | NDCG@10 | 推理延迟 |
|------|-----------|---------|---------|
| SAR（Sequential Auto-Regressive） | baseline | baseline | 1x |
| PDPO（List-wise DPO） | +4.2% | +3.8% | 1x |
| **HiGR** | **+10.3%** | **+9.7%** | **0.2x（5x 加速）** |

**在线 A/B 测试（腾讯商业平台）：**

| 指标 | 提升幅度 |
|------|---------|
| 平均观看时长（Avg Watch Time） | **+1.22%** |
| 平均播放次数（Avg Video Plays） | **+1.73%** |

**关键结论：**
- HiGR 在离线指标上比 SOTA **提升超过 10%**，同时推理速度提升 **5x**
- 在腾讯服务亿级用户的平台上验证了工业可用性
- 分层规划是效率提升的核心，偏好对齐是质量提升的核心

## 工程落地要点

### 腾讯生产系统架构

```
实时用户请求（数亿 QPS）
        ↓
上下文特征抽取（用户画像 + Session 信号）
        ↓
HiGR 服务
  ├── 列表规划（List Planner）：1次前向，<10ms
  └── 物品解码（Item Decoder）：Beam Search with Trie，约 50ms
        ↓
生成式 Slate（10个物品）
        ↓
AB 实验框架分流 → 展示层
```

### 为什么 5x 推理加速至关重要

以 10 个物品的 Slate、每物品 3 个 SID token 为例：
- **无规划的端到端生成**：30 步自回归，每步全 Trie 搜索
- **HiGR 分层生成**：1 步规划 + 约 6 步受约束解码
- 延迟从 ~250ms 降至 ~50ms，满足在线服务 100ms 要求

### 偏好数据构建

```python
# 构建 Slate 偏好对
def build_slate_preference_pairs(user_sessions):
    pairs = []
    for session in user_sessions:
        slates = session['shown_slates']
        # 按完播率/观看时长排序
        slates_sorted = sorted(slates, key=lambda s: s['engagement_score'])
        
        # 取最好和最差的 Slate 构成偏好对
        if slates_sorted[-1]['engagement_score'] > slates_sorted[0]['engagement_score'] * 1.5:
            pairs.append({
                'user': session['user_id'],
                'win': slates_sorted[-1]['slate'],
                'lose': slates_sorted[0]['slate']
            })
    return pairs
```

## 常见考点

**Q1: Slate 推荐和单物品推荐的核心区别是什么？**
A: (1) **优化目标不同**：单物品推荐优化单个物品的点击率/相关性；Slate 推荐优化列表整体效果（总点击量、用户留存、多样性满足）；(2) **物品间依赖**：Slate 中物品间有曝光位置竞争和语义互补关系，不能独立打分；(3) **组合爆炸**：从 1000 候选中选 10 个物品的组合数为 C(1000,10)≈2.6×10²³，无法穷举；(4) **展示效果**：用户同时看到多个物品，会有比较行为，影响整体满意度。

**Q2: 分层规划的"列表意图向量"是如何影响物品解码的？**
A: 列表意图向量 $\mathbf{s}_{plan}$ 作为条件信号注入到物品解码器（通过交叉注意力或拼接到输入）。它告诉解码器"这个列表的整体调性是：时尚类、适合下午场景、强调多样性"，从而引导每个物品的生成朝向符合整体规划的方向。实质上是一个软约束，而非硬规则，允许物品解码在保持整体一致性的同时有局部灵活性。

**Q3: DPO 和 RLHF 在推荐系统偏好对齐中各有什么优劣？**
A: DPO 优势：(1) 无需单独训练 Reward Model，直接从偏好对数据优化策略；(2) 训练稳定，无 RL 的方差问题；(3) 实现简单，一个损失函数搞定。RLHF 优势：(1) 可以用奖励模型泛化到未见的 Slate；(2) 支持在线探索（Online RL），能应对分布漂移。推荐系统中 DPO 更常用，因为推荐日志数据天然是偏好对形式（展示了多个 Slate，收集了用户反馈）。

**Q4: HiGR 的"残差量化 + 对比约束"相比普通 VQ-VAE 有什么优势？**
A: 普通 VQ-VAE：(1) 仅优化重建损失，SID 不保证语义有序；(2) 第一层 code 可能不对应有意义的类别。残差量化（RQ）的层级结构确保第一层 code 捕获最重要的语义维度，后续 code 是残差细化。对比约束进一步强制语义相近的物品在 SID 空间中相邻，使 Trie 的层级结构具有真实的语义含义，从而使 Slate 的多样性控制（选择不同 depth-1 code 的物品）有实际意义。

**Q5: 服务亿级用户的生成式推荐系统有哪些工程挑战？**
A: (1) **延迟**：自回归生成延迟高，HiGR 用分层规划解决（5x 加速）；(2) **一致性**：生成式模型的推荐结果有随机性，需要稳定性保证（固定随机种子或确定性解码）；(3) **物品库更新**：每天有大量新品上架/老品下架，SID 和 Trie 需要增量更新；(4) **冷启动**：新用户无历史，需要专门的冷启动策略（基于人口属性初始化列表意图）；(5) **A/B 实验**：生成式推荐难以做公平的在线实验（生成结果相关性强，流量分割有污染）。
