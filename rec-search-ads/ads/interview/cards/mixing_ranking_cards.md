# 广告混排高频常见题卡片库 | 50 张卡片

> 覆盖概念题、设计题、优化题、工程题 | 面试级难度

---

## 📌 概念题（1-15）

### 题 1：什么是混排（Mixing）？
**背景**：面试官想知道你是否理解混排的基本概念。

**答案框架**：
```
混排（Reranking / Mixing）是在精排之后的最后一步优化，目标是在保持排序质量（CTR）的基础上，
增加结果的多样性和用户体验。

关键点：
1. 位置：召回 → 粗排 → 精排 → 混排
2. 核心问题：精排会自然倾向于推荐"高 eCPM"的广告主，导致用户看到重复的广告
3. 混排的作用：打破这种重复，同时保证推荐质量
4. 典型指标改善：RPM 持平或略升，留存 +2-5%，点击率 +1-3%
```

**追问**：混排和排序的区别是什么？

---

### 题 2：为什么需要多样性？从产品和算法角度解释。
**背案例**：某平台只推荐高 eCPM 的广告，RPM 最大，但用户留存下降 15%。

**答案框架**：
```
【产品角度】
- 用户体验：连续看同一广告主的广告会产生"疲劳"
- 用户留存：多样的内容能保持用户兴趣，减少流失

【算法角度】
- 反馈偏差：精排的 CTR 预估基于历史数据，但历史数据本身就被"高 eCPM 广告"主导
- 长期目标：短期 RPM 可能下降 1-2%，但长期（1 个月）用户留存提升导致整体收益 +5%

【数学角度】
- Bandit 问题：如果只选择已知的最优（exploitation），永远无法发现更优的新广告
- 需要主动探索（exploration）新广告、新创作者、新话题
```

**数字证据**：
- 多样性提升 10% → 停留时间 +8%，7 日留存 +3%

---

### 题 3：什么是 DPP（Determinantal Point Process）？
**难度**：★★★（需要数学直觉）

**答案框架**：
```
DPP 是一个概率模型，用行列式（determinant）来衡量集合的"多样性"。

【直觉】
P(S) ∝ det(L_S)

- S：候选的子集
- L：相似度矩阵（编码了候选之间的差异）
- det：行列式
  - 如果选中的候选很相似 → 矩阵秩不满 → det 小 → P(S) 小 ✗
  - 如果选中的候选很不同 → 矩阵秩满 → det 大 → P(S) 大 ✓

【核心公式】
L[i,j] = quality[i] × quality[j] × similarity[i,j]

L[i,i] = quality[i]² （对角线：候选的"好"程度）
L[i,j] （i≠j）：候选 i 和 j 的相似度（相似度越大，竞争越强）

【优势】
✓ 同时考虑相关性和多样性
✓ 有坚实的数学基础
✓ 贪心近似的近似比为 1/e ≈ 0.37

【局限】
✗ 相似度矩阵如何定义是关键（需要领域知识）
✗ 超大规模场景计算复杂度高（O(k³)）
```

---

### 题 4：Shannon Entropy 和 Herfindahl Index 哪个更适合衡量多样性？
**背景**：两个常见的多样性指标。

**答案框架**：
```
【Shannon Entropy】
H = -Σ p_i × log(p_i)
范围：[0, log(n)]
越高越多样

【Herfindahl Index】
HI = Σ p_i²
范围：[1/n, 1]
越低越多样

【对比】

指标        │ Shannon  │ Herfindahl
────────────┼──────────┼───────────
敏感度      │ 中等     │ 对大p_i敏感
计算简度    │ 简单     │ 非常简单
常见场景    │ 学术     │ 商业
推荐用途    │ 多样性分析 │ 广告主覆盖率

【选择原则】
- 关心"所有话题都有覆盖"：用 Shannon（全局平衡）
- 关心"不要让大广告主垄断"：用 Herfindahl（对不均衡敏感）
- 工业应用通常用 Herfindahl（计算快）
```

---

### 题 5：Topic Diversity 和 Source Diversity 如何平衡？
**背景**：两个维度的多样性可能冲突。

**答案框架**：
```
【冲突场景】
假设：
- 电商话题有 3 个高质量的广告，广告主 A、B、C
- 美妆话题只有 1 个广告，广告主 D

选择方案 1：最大化 Topic Diversity
→ 选 3 个电商 + 1 个美妆
→ Topic Diversity 高，但 Source Diversity 低（只有 4 个广告主）

选择方案 2：最大化 Source Diversity
→ 选 1 个电商（A）+ 1 个电商（B）+ 1 个电商（C）+ 1 个美妆（D）
→ Source Diversity 高，但 Topic 有点重复

【解决方案】
加权融合：Score = α × Topic_Div + (1-α) × Source_Div

α 的选择：
- 新用户：α = 0.6（话题多样性更重要，保护体验）
- 老用户：α = 0.4（广告主多样性更重要，给中小广告主机会）

【最佳实践】
不是同时最大化两个指标，而是在两者之间找平衡点（Pareto frontier）
```

---

### 题 6：什么是 Ad Ratio（广告比例）？它如何动态调整？
**背景**：Feed 中应该有多少比例的广告？

**答案框架**：
```
【定义】
Ad Ratio = (广告数) / (总内容数)
通常范围：10-40%（取决于平台类型）

【平台对比】
平台类型        │ 广告比例  │ 原因
────────────────┼──────────┼─────────────────────
新闻平台        │ 10-20%   │ 内容体验至上
小红书          │ 15-25%   │ 内容 + 商业平衡
抖音            │ 20-35%   │ UGC 充足，广告空间大
Facebook        │ 25-35%   │ 用户习惯接受高广告比例

【动态调整策略】
用户粘性高        → ad_ratio = 0.35   （用户容忍度高）
用户粘性中等      → ad_ratio = 0.20   （平衡）
用户粘性低        → ad_ratio = 0.10   （保护体验）
新用户            → ad_ratio = 0.05   （避免流失）

【实现方式】
- 基于用户历史 DAU 频率判断粘性
- 基于用户反感率（广告跳过、反感）动态调整
- A/B 测试找到每个用户群体的最优 ratio
```

---

### 题 7：什么是 Learning-to-Rank（LTR）？它与混排的关系是什么？
**难度**：★★★

**答案框架**：
```
【LTR 是什么】
Learning-to-Rank 是一类机器学习方法，用于优化排序问题。

【典型方法】
1. Pointwise：预测单个候选的分数
   - 问题：忽视了相对排序关系
   
2. Pairwise：学习"A 应该排在 B 前面"
   - 损失函数：对比损失（contrast loss）
   - 代表：RankNet, LambdaRank
   
3. Listwise：直接优化列表排序质量
   - 损失函数：NDCG、MAP 的可微近似
   - 代表：LambdaMART, ListNet

【与混排的关系】

精排（Ranking）:
- 目标：最大化 CTR / eCPM
- 输出：Top-50 的排序
- 方法：DNN CTR 预估（Pointwise）

混排（Reranking）:
- 目标：最大化 CTR + 多样性
- 输入：精排的 Top-50
- 方法：LTR 加上多样性损失（Listwise）

【混排中的 LTR 应用】
损失函数 = λ₁ × L_NDCG + λ₂ × L_diversity + λ₃ × L_smoothness

- L_NDCG：排序质量（与精排的 Top-K 一致性）
- L_diversity：多样性约束
- L_smoothness：避免排序波动
```

---

### 题 8：什么是多目标优化（Multi-Objective Optimization）？
**背景**：混排通常需要平衡多个指标。

**答案框架**：
```
【问题定义】
同时优化多个目标，通常这些目标是冲突的：
- 目标 1：最大化 eCPM
- 目标 2：最大化多样性
- 目标 3：最大化用户留存

【解决方案】

方案 1：加权融合（Weighted Sum）
f(x) = w₁ × f₁(x) + w₂ × f₂(x) + w₃ × f₃(x)
优点：简单
缺点：权重难以手调，对不同用户群体的适用性差

方案 2：约束优化（Constrained Optimization）
maximize f₁(x)
subject to f₂(x) ≥ θ₂, f₃(x) ≥ θ₃

优点：明确的约束条件
缺点：需要事先设定约束值

方案 3：Pareto 优化
找到 Pareto frontier（没有一个目标能单独改进而不伤害其他目标）

优点：获得多个平衡点，可选择
缺点：计算复杂

【在混排中的应用】
最常见的是"约束优化"：
maximize RPM
subject to diversity ≥ 0.65, retention_rate ≥ 50%

这避免了"为了 RPM 完全放弃多样性"的极端情况。
```

---

### 题 9：广告频次限制（Frequency Cap）的实现方式有哪些？
**背景**：防止同一广告主的广告重复出现。

**答案框架**：
```
【常见规则】
- 7 天内同一品牌最多 3 次
- 当前 Feed 中同一品牌最多 1 个
- 同一创意（素材）7 天内最多 1 次

【实现方式】

方式 1：硬约束（规则）
```python
if brand in recent_brands[-3:]:
    skip this ad
```
优点：简单快速
缺点：可能丢弃高质量的广告

方式 2：软约束（惩罚）
```
score -= 0.3 × (frequency - 1)
```
优点：灵活，允许在必要时超频
缺点：需要调参

方式 3：概率约束
```
prob_show = 1.0 / (1 + exp(frequency - 1))
```
优点：平滑衰减，自然
缺点：可能在边界有不稳定

【何时使用】
- 广告库存充足：硬约束（保证多样性）
- 广告库存稀缺：软约束（允许灵活）
- 新广告优化：概率约束（自动控制露出）
```

---

### 题 10：什么是"冷启动"问题在混排中如何体现？
**背景**：新用户、新广告、新话题的推荐困难。

**答案框架**：
```
【三种冷启动】

1. 用户冷启动（新用户）
   - 问题：无历史浏览数据，无法判断用户兴趣
   - 解决：
     a) 基于人口统计特征（年龄、地区）的协同过滤
     b) 高覆盖度的推荐（优先推荐热门、高质量内容）
     c) 降低广告比例（保护体验）

2. 内容冷启动（新视频、新广告）
   - 问题：无点击反馈，无法估计 CTR
   - 解决：
     a) 基于内容特征（话题、创作者）的推荐
     b) Exploration（主动推荐新内容获得反馈）
     c) Upper Confidence Bound (UCB) 算法

3. 多样性冷启动（新话题、新创作者）
   - 问题：缺乏多样性的基线
   - 解决：
     a) 显式推荐新话题/创作者
     b) 给予新话题的"多样性红利"
     c) 强化学习的探索机制

【混排中的处理】
新用户：多样性权重 +30%（要让用户尽早发现兴趣）
新广告：从约束条件中优先放出（给予机会）
新话题：多样性奖励翻倍（鼓励用户尝试）
```

---

### 题 11-15：概念题快速版

**题 11**：什么是 Contextual Bandits？如何用于混排？
- 答：Thompson Sampling / UCB 算法，实时调整推荐策略而不需要完整的 RL 训练

**题 12**：什么是 Submodular 函数？为什么它在混排中重要？
- 答：一个候选的边际收益随着已选择的候选增加而递减，适合多样性建模

**题 13**：什么是"探索利用权衡"（Exploration-Exploitation Tradeoff）？
- 答：选择已知最优的动作 vs 尝试未知动作，两者的平衡

**题 14**：什么是 RPM 和 eCPM？它们在混排中如何变化？
- 答：RPM=千次展示收益，eCPM=单次展示出价。混排通常会保持 eCPM 不变或略增

**题 15**：什么是"用户疲劳指数"？如何测量？
- 答：用户对重复内容的厌烦程度，可用"跳过率"或"反感数"衡量

---

## 📋 设计题（16-35）

### 题 16：设计一个完整的混排系统。从零开始。

**背景**：面试官想看你的系统设计能力。

**答案框架**：
```
【需求分析】
- 日活：1000 万
- QPS：10 万
- 广告库存：10 万+
- 混排延迟要求：< 50ms

【系统架构】

输入：精排的 Top-50（已有初步排序）
      用户信息（ID, 兴趣标签, 性别, 地区）

处理流程：
┌─────────────────────────────────────┐
│ 1. 加载用户历史（最近 7 天）        │
│    - 最近看过的广告                  │
│    - 点击/点赞/分享情况             │
│    - 多样性指标（话题分布）         │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ 2. 应用频次约束（规则）             │
│    - 去除重复广告主的候选           │
│    - 应用品类限制                    │
│    - 结果：Top-30 候选              │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ 3. 计算混排分数                      │
│    score = α × quality +            │
│            β × diversity +           │
│            γ × user_affinity        │
│    结果：30 个候选的分数            │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ 4. 选择 Top-20                      │
│    按分数排序，选最高的 20 个       │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ 5. 交错排列                          │
│    广告与内容交错排列               │
│    (根据位置偏好调整)               │
└─────────────────────────────────────┘
         ↓
输出：最终 20 条 Feed（排序已优化）

【关键模块】

模块 A：多样性计算
def compute_diversity_score(ads_selected):
    topic_dist = compute_topic_distribution(ads_selected)
    creator_dist = compute_creator_distribution(ads_selected)
    return 0.6 * entropy(topic_dist) + 0.4 * entropy(creator_dist)

模块 B：用户亲和力
def compute_user_affinity(user, ad):
    topic_match = cosine_sim(user_interests, ad_topics)
    price_match = 1 - abs(user_price_segment - ad_price_segment)
    return 0.6 * topic_match + 0.4 * price_match

模块 C：在线更新
def update_on_user_feedback(user_id, ad_id, feedback):
    # 实时更新用户的多样性指标
    # 用于下一次请求时的混排

【性能指标】
- P95 延迟：< 50ms
- 精准度（与理想混排的一致性）：NDCG@20 > 0.8
- 多样性分数：entropy > 0.65
- 广告主覆盖率：> 60%
```

---

### 题 17：如果 RPM 下降 5%，多样性提升 20%，如何判断这个混排方案是否值得上线？

**答案框架**：
```
【首先，分析成分】
RPM 下降 5% 的原因：
- 可能是排除了一些高 eCPM 的广告
- 可能是用户多看了一些低 eCPM 的新广告

多样性提升 20% 的潜在收益：
- 用户留存可能提升 2-5%
- 新创作者/新广告的曝光增加

【定量分析】
假设：
- 日均广告收入：1000 万元
- RPM 下降 5%：-50 万元/天

但如果：
- 用户留存提升 3%
- 这意味着"用户的 7 日生命周期价值"增加 3%
- 通常 LTV 是日均收入的 7-10 倍
- LTV 增加 3% = 额外 210-300 万元（在用户生命周期内）

长期来看：
- 第 1 天：-50 万元
- 第 7 天累积：-350 万元（流量损失）
- 但 LTV 增加：+210-300 万元（用户更留存）
- 净收益：≈ 0 万元 → +200 万元（取决于精确数字）

【上线决策】
✓ 值得上线，原因：
  1. 短期 RPM 损失有限（5%）
  2. 长期 LTV 收益可能很大（多样性通常带来留存改善）
  3. 生态价值：新创作者有机会，平台更健康

✗ 但需要进一步验证：
  - 运行 2+ 周，确保不是"新鲜效应"
  - 监控 7 日留存，确认假设成立
  - 如果 2 周后留存没有改善，考虑回滚或调参
```

---

### 题 18：如何处理广告与内容的混排（不仅是广告之间的混排）？

**背景**：小红书等平台的特色问题。

**答案框架**：
```
【核心问题】
精排可能倾向于"高质量的广告"，导致 Feed 中广告比例过高。

【解决方案】

方案 1：分离排序
- 内容和广告分别排序（用不同的模型和目标）
- 然后按固定比例交错
- 优点：简单
- 缺点：无法全局优化

方案 2：统一排序 + 约束
```python
def mixed_ranking(contents, ads, user_context):
    # 统一计算所有候选（内容 + 广告）的分数
    all_candidates = contents + ads
    
    scores = []
    for candidate in all_candidates:
        if candidate['type'] == 'content':
            score = predict_content_ctr(candidate) * weight_content
        else:
            score = predict_ad_ctr(candidate) * weight_ad
        scores.append(score)
    
    # 在约束下选择
    selected = greedy_select_with_constraint(
        all_candidates,
        scores,
        constraint={'ad_ratio': 0.2}  # 广告不超过 20%
    )
    
    return selected
```
优点：全局优化
缺点：约束需要动态调整

方案 3：列表级优化
```python
def list_level_mixing(contents, ads, user_context):
    result = []
    remaining_contents = list(contents)
    remaining_ads = list(ads)
    
    for position in range(20):
        # 计算"当前位置插入内容或广告"的收益
        best_candidate = None
        best_score = -inf
        
        for content in remaining_contents:
            content_score = predict_content_ctr(content)
            content_diversity = compute_diversity_bonus(content, result[-3:])
            score = content_score + 0.2 * content_diversity
            
            if score > best_score:
                best_score = score
                best_candidate = content
        
        for ad in remaining_ads:
            ad_score = predict_ad_ctr(ad)
            ad_diversity = compute_diversity_bonus(ad, result[-3:])
            
            # 广告的约束：不能超过比例限制
            ad_count = sum(1 for x in result if x['type'] == 'ad')
            if ad_count >= position * 0.2:  # 广告不超过 20%
                continue
            
            score = ad_score + 0.2 * ad_diversity
            
            if score > best_score:
                best_score = score
                best_candidate = ad
        
        result.append(best_candidate)
    
    return result
```
优点：精细控制，最优结果
缺点：计算复杂，延迟高

【推荐使用】
- 小流量测试：方案 1（快速验证）
- 中等规模：方案 2（好效果，中等复杂）
- 大规模平台：方案 3（最优，需要优化）
```

---

### 题 19：如何为新用户设计混排策略？

**答案框架**：
```
【新用户的特点】
- 无历史数据，无法判断兴趣
- 流失风险高，需要快速激活
- 容忍度低，广告过多容易流失

【差异化策略】

维度 1：广告比例
- 新用户（< 3 天）：ad_ratio = 0.1  （最保守）
- 早期用户（3-7 天）：ad_ratio = 0.15
- 活跃用户（> 7 天）：ad_ratio = 0.25

维度 2：内容多样性
- 新用户：多样性权重 +30%  （尽早发现兴趣）
- 推荐新话题：多样性奖励翻倍

维度 3：广告质量
- 新用户只展示：Top-tier 广告（品牌信誉好）
- 过滤掉：低质量、高风险的广告

维度 4：用户基画像
```python
def new_user_mixing_strategy(user_id, age_in_platform):
    base_config = {
        'ad_ratio': 0.1,
        'diversity_weight': 0.4,
        'topic_diversity_bonus': 2.0,
        'min_ad_quality_score': 0.8,
    }
    
    if age_in_platform > 3:  # 3 天后
        base_config['ad_ratio'] = 0.15
        base_config['diversity_weight'] = 0.35
    
    if age_in_platform > 7:  # 7 天后
        base_config['ad_ratio'] = 0.25
        base_config['diversity_weight'] = 0.3
    
    # 如果新用户看了某类内容，立即推荐该话题的其他内容
    if user_engagement_score > 0.7:
        base_config['diversity_weight'] = 0.2  # 降低多样性，坚持兴趣
    
    return base_config
```

【监控指标】
- DAU/MAU 转化率（新用户 → 活跃用户）
- 日 1/3/7 留存
- 广告点击率（新用户的容忍度指标）
- 反感率（关键监控，过高说明广告太多）
```

---

### 题 20：如果有 100 万个广告和 1000 万个用户，混排系统如何设计（考虑规模）？

**答案框架**：
```
【规模问题分解】
- 100 万广告 × 1000 万用户 = 10^13 对（无法全部计算）
- 需要"多层过滤"架构

【架构设计】

第 1 层：广告召回（离线/近线）
- 输入：用户 ID
- 处理：根据用户定向获得候选广告集
  - 地理位置过滤：只看该地区的广告
  - 年龄/性别过滤：符合定向的广告
  - 兴趣过滤：基于用户标签匹配
- 输出：~ 1000 个候选（1000 万 → 1000）

第 2 层：粗排（快速评分）
- 输入：1000 个候选
- 处理：用轻量级模型（GBDT/简单 DNN）评分
  - 特征：基础特征（广告主、话题、创意类型）
  - 目标：快速筛选，保留质量
  - 时间：< 10ms
- 输出：Top-50 ~ Top-100

第 3 层：精排（精确评分）
- 输入：50-100 个候选
- 处理：用复杂模型（Deep DNN）评分
  - 特征：交叉特征、深度特征、上下文特征
  - 时间：10-30ms
- 输出：Top-20 ~ Top-50

第 4 层：混排（最后优化）
- 输入：Top-50
- 处理：
  - 多样性计算：10ms
  - 频次约束：5ms
  - 重排：5ms
- 输出：Top-20，排序已优化

【总延迟】
粗排：10ms
精排：30ms
混排：20ms
总计：< 60ms ✓

【在线服务**】
```python
class ScalableMixingSystem:
    def __init__(self):
        self.embedding_cache = LRUCache(size=10M)  # 用户 embedding 缓存
        self.ad_recall_model = trained_model()
        self.coarse_ranker = GBDT()
        self.ranker = DNN()
        self.mixer = MixingEngine()
    
    def process_request(self, user_id, context):
        # 1. 召回
        ad_candidates = self.ad_recall_model.recall(user_id)  # 1000
        
        # 2. 粗排
        coarse_scores = self.coarse_ranker.predict(ad_candidates)
        top_100 = sorted(ad_candidates, key=lambda x: coarse_scores[x])[:100]
        
        # 3. 精排
        fine_scores = self.ranker.predict(top_100, user_id, context)
        top_50 = sorted(top_100, key=lambda x: fine_scores[x])[:50]
        
        # 4. 混排
        final_ranking = self.mixer.mix(top_50, user_id)
        
        return final_ranking[:20]
```

【缓存策略】
- 用户 embedding：LRU 缓存（10M 条）
- 广告特征：写入缓存（广告不经常变化）
- 用户-广告相似度：局部计算（不缓存，节省空间）

【并行化】
- 粗排和精排可以在不同的 GPU 服务器上并行
- 混排在 CPU 上处理（计算量小）
```

---

### 题 21-35：设计题快速版

**题 21**：如何设计一个 A/B 测试框架来验证混排的效果？
- 分层分析、样本量计算、多目标检验

**题 22**：如何处理"时间衰减"（用户兴趣随时间变化）？
- 添加时间衰减因子到相似度计算

**题 23**：如何平衡"创意疲劳"和"内容新鲜度"？
- 频次限制 + 新鲜度奖励

**题 24**：跨域混排如何设计（电商/内容/广告混合）？
- 统一评分函数 + 多域约束

**题 25**：如何实现"用户个性化的多样性权重"？
- 基于用户行为聚类，分组设置不同权重

**题 26**：如何处理"广告主竞争关系"（A 和 B 是竞争品牌）？
- 竞争关系图 + Feed 中不同时出现的约束

**题 27**：如何设计混排的"降级方案"（当推荐系统故障时）？
- 回退到简单规则混排、离线缓存最近的排序

**题 28**：如何利用"用户反感数据"优化混排？
- 反感信号 → 负奖励 → 避免推荐类似广告

**题 29**：新广告如何获得曝光机会（在竞争激烈的环境中）？
- Thompson Sampling 分配初始流量

**题 30**：如何处理"话题突变"（某个话题突然热门）？
- 动态调整多样性权重，给热点话题降权

**题 31**：如何监控混排系统的"健康度"？
- 多样性指标、点击率、留存率、广告主满意度

**题 32**：如何处理多语言、多地区的混排差异？
- 地区/语言特定的参数配置

**题 33**：如何设计"隐私友好"的混排（不存储个人数据）？
- 本地化计算、差分隐私、联邦学习

**题 34**：如何评估混排模型的"公平性"（是否歧视某些广告主）？
- 分层分析，监控不同广告主的曝光率、RPM

**题 35**：混排系统的"容错"和"熔断"机制如何设计？
- 实时监控，偏离基线 > 阈值时自动回退

---

## 📊 优化题（36-45）

### 题 36：如何从 RPM -2% 优化到 RPM +3%（同时保持多样性）？

**答案框架**（省略细节）：
```
第 1 步：根因分析
- 用 RPM -2% 的原因：排除了哪些广告？
- 这些广告的特点：高 eCPM？高点击率？

第 2 步：优化目标函数权重
从：score = 0.7 × ecpm + 0.3 × diversity
改为：score = 0.6 × ecpm + 0.2 × diversity + 0.2 × user_affinity

理由：
- 降低 ecpm 权重，避免"贪心选择高 ecpm 广告"
- 增加用户亲和力，选择用户可能点击的广告
- 结果：eCPM 保持，点击率提升，整体 RPM 改善

第 3 步：精细化多样性约束
从：diversity >= 0.65（全局约束）
改为：diversity >= 0.65 且 top_5_ecpm >= 平均值（局部约束）

理由：
- 允许前 5 个位置选择高 ecpm 广告（用户注意力最高的位置）
- 后面的位置保证多样性
- 结果：兼顾收益和用户体验

第 4 步：A/B 测试与优化
- 测试多个权重组合
- 找到"RPM + 多样性 + 留存"的最优点
- 通常这个点在 RPM 保持甚至小幅增长的地方
```

---

### 题 37：如何优化混排的"计算延迟"（从 100ms 降至 30ms）？

**答案框架**：
```
【瓶颈分析】
100ms = 特征提取(40ms) + 多样性计算(30ms) + 排序(20ms) + 其他(10ms)

【优化方案】

优化 1：特征预计算（40ms → 5ms）
- 不在请求时计算用户特征，而是实时维护
- 缓存用户最近 7 天的广告、多样性指标
- 请求时只需加载缓存，不需要重新计算

优化 2：相似度矩阵缓存（多样性计算 30ms → 10ms）
- 离线预计算所有广告对的相似度
- 请求时直接查表（O(1)）而不是计算（O(n²)）

优化 3：并行处理（排序 20ms → 5ms）
- 多个广告的排序可以并行
- 用 SIMD 指令加速

优化 4：早停（混排贪心 15ms → 8ms）
- 不计算所有 50 个候选的分数
- 只考虑前 10 个最有竞争力的
- 贪心选择

【最终延迟分布】
特征：5ms
多样性：10ms
排序：5ms
贪心：8ms
其他：2ms
总计：30ms ✓
```

---

### 题 38：如何平衡"新广告的探索"和"已验证广告的利用"？

**答案框架**：
```
【经典的 Exploration-Exploitation 问题】

纯粹的利用（Exploitation）：
- 总是选择已知最优的广告
- 问题：无法发现新的好广告，长期效果变差

纯粹的探索（Exploration）：
- 随机选择广告，包括低质量的
- 问题：短期 RPM 和用户体验差

【解决方案】

方案 1：Epsilon-Greedy
```
if random() < epsilon:
    action = random_ad()  # 探索
else:
    action = best_ad()   # 利用

epsilon 根据广告新旧度调整：
- 新广告（< 1 周）：epsilon = 0.3（30% 流量探索）
- 中等（1-4 周）：epsilon = 0.1
- 老广告（> 4 周）：epsilon = 0.05
```

方案 2：Thompson Sampling
```
对每个广告维护一个"好的程度"的后验分布
每轮从分布中采样，选择最好的
自动平衡探索和利用
```

方案 3：Contextual Bandits
```
不仅基于广告本身，还基于用户上下文决策
新广告对于某些用户可能是最优的（个性化探索）
```

【推荐使用】
- 库存充足：Thompson Sampling
- 库存稀缺：Contextual Bandits
- 快速上线：Epsilon-Greedy
```

---

### 题 39：如何优化"不同用户群体"的混排参数？

**答案框架**：
```
【用户分群策略】

群组 1：高价值用户（VIP）
- 日均花费 > 100 元
- 策略：ad_ratio = 0.1，多样性权重 = 0.2
- 原因：VIP 容忍度低，重质量
- 期望指标：CTR 最高，满意度最高

群组 2：活跃用户
- DAU >= 70%，点赞率 > 5%
- 策略：ad_ratio = 0.25，多样性权重 = 0.3
- 原因：用户粘性高，可以承受更多广告
- 期望指标：RPM 最高

群组 3：低活跃用户（流失风险）
- DAU < 30%，最近 3 天无活动
- 策略：ad_ratio = 0.05，多样性权重 = 0.5
- 原因：需要保护体验，激活用户
- 期望指标：留存率改善

群组 4：新用户（< 7 天）
- 策略：ad_ratio = 0.1，多样性权重 = 0.4
- 原因：尽早发现兴趣，降低广告骚扰
- 期望指标：7 日留存率

【实现】
```python
def get_user_segment(user_id):
    user_profile = load_user_profile(user_id)
    
    if user_profile['vip_level'] >= 2:
        return 'vip'
    elif user_profile['dau_ratio'] >= 0.7:
        return 'active'
    elif user_profile['dau_ratio'] < 0.3:
        return 'churn_risk'
    elif user_profile['age_days'] < 7:
        return 'new_user'
    else:
        return 'default'

def get_mixing_config(segment):
    config_map = {
        'vip': {'ad_ratio': 0.1, 'diversity_weight': 0.2},
        'active': {'ad_ratio': 0.25, 'diversity_weight': 0.3},
        'churn_risk': {'ad_ratio': 0.05, 'diversity_weight': 0.5},
        'new_user': {'ad_ratio': 0.1, 'diversity_weight': 0.4},
        'default': {'ad_ratio': 0.2, 'diversity_weight': 0.3},
    }
    return config_map[segment]
```

【A/B 测试】
- 对照组：所有用户同一配置
- 实验组：用户分群，差异化配置
- 预期改善：+2-3% DAU，+5-10% 群组留存
```

---

### 题 40-45：优化题快速版

**题 40**：如何从"离线 A/B 测试"升级到"在线实验框架"？
- 多臂老虎机框架，实时流量分配优化

**题 41**：如何优化"冷启动状态下的推荐"？
- 基于内容相似度的转移学习

**题 42**：如何利用"用户反馈"快速优化混排？
- 反馈循环，用户行为直接影响下次推荐

**题 43**：如何优化"多目标函数的权重"（自动调参）？
- 多目标优化框架，Pareto 搜索

**题 44**：如何处理"季节性变化"（不同季节的兴趣差异）？
- 时间序列特征，季节性参数调整

**题 45**：如何优化"跨平台混排"（App/Web/小程序 差异）？
- 平台特定参数，设备适配约束

---

## 🔧 工程题（46-50）

### 题 46：实现混排系统的"监控和告警"体系，包括哪些关键指标？

**答案框架**：
```
【监控指标分类】

1. 系统性能指标
   - P95/P99 延迟：< 50ms / 100ms
   - QPS：应对预期流量
   - 错误率：< 0.1%
   - CPU/内存使用率

2. 混排质量指标
   - 多样性分数（Shannon Entropy）：> 0.65
   - 广告主覆盖率：> 60%
   - NDCG@20（排序质量）：> 0.8
   - 与精排的相似度：> 0.7（避免过度修改）

3. 商业指标
   - RPM（千次展示收益）：环比不下降 > 1%
   - CTR：环比不下降 > 0.5%
   - 转化率（CVR）：环比不下降 > 0.5%

4. 用户体验指标
   - DAU：环比增长 > 0%
   - 7/30 日留存率：环比增长 > 1%
   - 广告跳过率：< 30%（反感指标）
   - 广告投诉率：< 1%

【告警设置】

Critical（立即通知 PM/Tech Lead）：
- P99 延迟 > 200ms
- 错误率 > 1%
- RPM 下降 > 5%
- 多样性分数 < 0.55

Warning（监控，晨会讨论）：
- P95 延迟 > 100ms
- RPM 下降 1-5%
- DAU 下降 > 2%
- 广告投诉率 > 1%

【监控实现】

```python
from prometheus_client import Counter, Gauge, Histogram
import logging

# 性能指标
mixing_latency = Histogram('mixing_latency_ms', 'Mixing latency')
mixing_errors = Counter('mixing_errors_total', 'Total mixing errors')

# 质量指标
diversity_score = Gauge('diversity_score', 'Feed diversity')
advertiser_coverage = Gauge('advertiser_coverage', 'Advertiser coverage ratio')

# 商业指标
rpm = Gauge('rpm_per_thousand', 'RPM')
ctr = Gauge('ctr', 'Click through rate')

@timer(mixing_latency)
def mix_feed(user_id, candidates):
    try:
        result = mixing_engine.run(user_id, candidates)
        return result
    except Exception as e:
        mixing_errors.inc()
        logging.error(f"Mixing error: {e}")
        raise

def collect_metrics():
    """每分钟收集一次指标"""
    while True:
        diversity = compute_diversity_metrics()
        diversity_score.set(diversity)
        
        coverage = compute_advertiser_coverage()
        advertiser_coverage.set(coverage)
        
        rpm_val = compute_rpm()
        rpm.set(rpm_val)
        
        time.sleep(60)
```

【告警规则】
```
告警名称：MixingLatencyHigh
条件：histogram_quantile(0.99, mixing_latency_ms) > 200
持续时间：5m
触发动作：发送钉钉/Slack 通知
```
```

---

### 题 47：如何设计混排系统的"灰度上线"流程？

**答案框架**：
```
【灰度上线的 5 阶段】

阶段 1：白名单流量（1-2 天）
- 内部员工 + 核心合作方
- 用途：快速发现功能 bug、性能问题
- 流量：< 0.01%
- 回滚条件：任何 critical bug

阶段 2：小流量（3-5 天）
- 1% 的真实用户流量
- 用途：验证指标方向是否正确
- 指标监控：RPM、CTR、DAU、留存率
- 回滚条件：RPM 下降 > 5%，CTR 下降 > 2%

阶段 3：中流量（5-7 天）
- 10% 的真实用户
- 用途：积累更多数据，降低统计波动
- 指标监控：更多维度（分时段、分地区）
- 回滚条件：同阶段 2，但阈值更严格

阶段 4：大流量（7-14 天）
- 50% 的真实用户
- 用途：长期稳定性验证，包括周末效果
- 指标监控：长期指标（7 日留存、14 日留存）
- 回滚条件：任何不可解释的异常

阶段 5：全量（2+ 周后）
- 100% 流量
- 用途：全面上线，持续监控
- 维护：持续 A/B 测试，迭代优化

【关键决策点】

每个阶段结束时，需要决策：
✓ Continue：指标向好，继续下一阶段
△ Hold：指标平稳但不明确，多观察几天
✗ Rollback：指标明确变差，立即回滚

【自动回滚机制】

```python
class AutoRollbackManager:
    def __init__(self, alert_threshold):
        self.threshold = alert_threshold
        self.baseline_metrics = load_baseline()
    
    def check_and_rollback(self):
        """每 30 分钟检查一次"""
        current_metrics = compute_metrics_last_30m()
        
        # 计算相对变化
        rpm_change = (current_metrics['rpm'] - self.baseline_metrics['rpm']) / self.baseline_metrics['rpm']
        
        if rpm_change < -0.05:  # RPM 下降 > 5%
            self.trigger_rollback('RPM 下降 > 5%')
            return
        
        dau_change = (current_metrics['dau'] - self.baseline_metrics['dau']) / self.baseline_metrics['dau']
        
        if dau_change < -0.02:  # DAU 下降 > 2%
            self.trigger_rollback('DAU 下降 > 2%')
            return
        
        logging.info(f"Metrics OK: RPM {rpm_change:+.2%}, DAU {dau_change:+.2%}")
    
    def trigger_rollback(self, reason):
        logging.critical(f"Auto-rollback triggered: {reason}")
        
        # 1. 立即切换回旧版本
        switch_to_previous_version()
        
        # 2. 发送告警
        send_alert(f"混排系统自动回滚: {reason}")
        
        # 3. 记录事件
        log_rollback_event(reason, timestamp=now())
```

【事后分析】
- 如果不是预期的改进，需要分析原因
- 可能是：数据 bug、特征不对、权重设置有问题
- 修复后，进行下一轮测试
```

---

### 题 48：混排系统如何处理"突发流量"和"服务降级"？

**答案框架**：
```
【处理策略】

Level 1：流量轻微增加（+20%）
- 缓存命中率优化
- 特征预计算，减少计算

Level 2：流量中等增加（+50%）
- 简化混排算法（早停 k=10 而非 20）
- 降低多样性计算精度（采样而非遍历）

Level 3：流量大幅增加（+100%）
- 关闭多样性约束，直接用精排结果
- 缓存预热的 Feed 排序（离线计算）

Level 4：服务即将崩溃（+200%）
- 全部用缓存或兜底方案
- 返回"最近 24 小时最热的 Feed"

【自动降级实现】

```python
class AdaptiveRankingEngine:
    def __init__(self):
        self.latency_threshold = 100  # ms
        self.current_load = 0  # 0-100%
    
    def process_request(self, user_id, candidates):
        start_time = time.time()
        
        # 根据当前负载，选择不同的混排策略
        if self.current_load < 50:
            # 正常情况
            result = self.full_mixing(candidates)
        elif self.current_load < 80:
            # 高负载，简化混排
            result = self.simplified_mixing(candidates)
        else:
            # 超高负载，用缓存或兜底
            result = self.cached_or_fallback(user_id)
        
        elapsed = (time.time() - start_time) * 1000
        
        # 如果实际延迟超过阈值，下次自动降级
        if elapsed > self.latency_threshold * 1.2:
            self.adaptive_degrade()
        
        return result
    
    def full_mixing(self, candidates):
        """完整的混排（多样性计算）"""
        scores = self.compute_scores(candidates)
        return self.greedy_select_with_diversity(scores)
    
    def simplified_mixing(self, candidates):
        """简化的混排（只计算多样性，不优化）"""
        scores = self.compute_scores(candidates)
        # 只考虑前 10 个候选的多样性
        return self.quick_select(scores, top_k=10)
    
    def cached_or_fallback(self, user_id):
        """返回缓存的结果"""
        cached = self.get_cached_feed(user_id)
        if cached:
            return cached
        else:
            # 兜底：返回最热的 Feed
            return self.get_hot_feed()
    
    def adaptive_degrade(self):
        """记录降级，可能触发自动扩容"""
        self.current_load += 5
        logging.warning(f"System overloaded, degrading. Load: {self.current_load}%")
```

【监控和恢复】
- 实时监控 P99 延迟
- 如果恢复正常（P99 < 80ms），自动升级到完整混排
- 记录每次降级的原因和持续时间
```

---

### 题 49：如何在不暴露用户隐私的情况下优化混排？

**答案框架**：
```
【隐私保护的混排方案】

方案 1：本地计算（最彻底的隐私保护）
- 用户的浏览历史保存在本地设备
- 混排计算在本地进行
- 只上传最终的"反馈"（点击、点赞）

```swift
// iOS 本地混排计算
class LocalMixingEngine {
    let userLocalStorage = UserDefaults.standard
    
    func mix(candidates: [Ad]) -> [Ad] {
        // 本地读取用户历史
        let userHistory = userLocalStorage.array(forKey: "ad_history")
        
        // 本地计算多样性
        let diversity = computeDiversityLocal(userHistory)
        
        // 本地混排
        let mixed = mixLocal(candidates, diversity: diversity)
        
        // 更新本地历史
        userLocalStorage.set(mixed, forKey: "ad_history")
        
        return mixed
    }
}
```

方案 2：联邦学习（隐私 + 模型优化）
- 用户数据不离开设备
- 模型在设备端训练
- 只上传模型梯度（无法反演个人数据）

方案 3：差分隐私（Differential Privacy）
- 混排算法中加入噪声
- 单个用户的行为无法被推断
- 大规模统计仍然准确

```python
def differentially_private_mixing(candidates, epsilon=1.0):
    """
    epsilon：隐私预算（越小越隐私，越大越准确）
    """
    
    # 计算理想的多样性分数
    ideal_diversity = compute_diversity(candidates)
    
    # 加入 Laplace 噪声
    noise = np.random.laplace(0, 1/epsilon)
    noisy_diversity = ideal_diversity + noise
    
    # 用带噪声的分数进行混排
    result = mix_with_noisy_score(candidates, noisy_diversity)
    
    return result
```

方案 4：TEE（Trusted Execution Environment）
- 使用 Intel SGX 或 ARM TrustZone
- 敏感计算在硬件保护的环境中进行
- 操作系统无法访问用户数据

【权衡】
```
方案      │ 隐私  │ 效果  │ 复杂度 │ 推荐场景
──────────┼────────┼────────┼──────────┼─────────
本地计算  │ 最高  │ 低     │ 高      │ 欧盟 GDPR
联邦学习  │ 很高  │ 中等   │ 很高    │ 大规模平台
差分隐私  │ 高    │ 中等   │ 中等    │ 公共数据
TEE       │ 很高  │ 高     │ 很高    │ 金融/医疗
```

【GDPR 合规建议】
✓ 明确征求用户同意
✓ 只收集必要的数据
✓ 实施数据最小化（用户可选择是否参与个性化）
✓ 用户有权访问/删除自己的数据
✓ 定期进行隐私审计
```

---

### 题 50：设计一个"可扩展的混排架构"，支持未来的算法创新。

**答案框架**：
```
【架构设计原则】
- 模块化：各个部分独立，易于替换
- 可配置：参数和策略可以动态配置
- 可观测：每个步骤有完整的日志和指标
- 易于测试：离线可以快速验证新算法

【系统架构】

```
Request Handler（请求入口）
    ↓
Feature Provider（特征服务）
    ├─ User Embedding Service
    ├─ Ad Feature Service
    └─ Context Feature Service
    ↓
Strategy Router（策略路由）
    ├─ Rule-Based Mixing
    ├─ ML-Based Mixing (DPP, LTR)
    ├─ RL-Based Mixing
    └─ LLM-Based Mixing
    ↓
Result Renderer（结果渲染）
    ├─ Ranking
    ├─ Interleaving
    └─ Caching
    ↓
Feedback Collector（反馈收集）
    └─ User actions → Learning System
```

【代码示例】

```python
# 策略接口（Abstract Base Class）
class MixingStrategy(ABC):
    @abstractmethod
    def mix(self, candidates: List[Ad], context: Dict) -> List[Ad]:
        pass

# 具体策略 1：基于规则
class RuleBasedMixing(MixingStrategy):
    def mix(self, candidates, context):
        # 简单的硬约束混排
        return apply_rules(candidates)

# 具体策略 2：DPP
class DPPMixing(MixingStrategy):
    def mix(self, candidates, context):
        L_matrix = build_L_matrix(candidates)
        return greedy_dpp(L_matrix, k=20)

# 具体策略 3：RL（未来）
class RLMixing(MixingStrategy):
    def mix(self, candidates, context):
        return self.actor_network.select_actions(candidates)

# 混排引擎
class MixingEngine:
    def __init__(self, config):
        self.strategy = self.load_strategy(config['strategy_type'])
        self.config = config
    
    def mix(self, user_id, candidates):
        # 1. 特征提取
        features = self.feature_provider.extract_features(user_id, candidates)
        
        # 2. 策略路由
        strategy = self.router.select_strategy(user_id, features)
        
        # 3. 执行混排
        result = strategy.mix(candidates, features)
        
        # 4. 结果校验
        assert self.validate_result(result)
        
        # 5. 反馈收集
        self.feedback_collector.register(user_id, result)
        
        return result
    
    def load_strategy(self, strategy_type):
        """支持动态加载策略"""
        strategy_registry = {
            'rule_based': RuleBasedMixing,
            'dpp': DPPMixing,
            'rl': RLMixing,
            'llm': LLMMixing,  # 未来
        }
        return strategy_registry[strategy_type]()

# 配置文件（JSON）
config = {
    "strategy_type": "dpp",  # 可以从 "rule_based" 切换到 "rl"
    "params": {
        "diversity_weight": 0.3,
        "ad_ratio": 0.2,
    },
    "monitoring": {
        "metrics": ["rpm", "ctr", "diversity_score"],
        "alert_threshold": 0.05,  # 5% 变化触发告警
    },
}

# 支持 A/B 测试
ab_test_config = {
    "control": {"strategy_type": "rule_based"},
    "treatment": {"strategy_type": "dpp"},
    "traffic_split": 0.5,  # 50-50 分流
}
```

【可扩展性】
✓ 新算法：只需实现 MixingStrategy 接口
✓ 新特征：FeatureProvider 添加新方法
✓ 新指标：MetricsCollector 添加新指标
✓ 新反馈：FeedbackCollector 支持新反馈类型

【未来扩展方向】
1. LLM 混排：add LLMMixing 策略
2. 多模态：add MultimodalFeatureProvider
3. 因果推断：add CausalMixing 策略
4. 个性化：add PersonalizedRouter
```

---

## 总结

**50 张卡片覆盖**：
- 概念题（1-15）：DPP、多目标优化、多样性度量
- 设计题（16-35）：系统设计、混排架构、用户分群
- 优化题（36-45）：指标优化、算法优化、参数调优
- 工程题（46-50）：监控告警、灰度上线、隐私保护、可扩展架构

**面试准备建议**：
1. 将卡片分类，每天重点准备 3-5 个
2. 每个答案都要准备"追问版本"
3. 实战案例要能背下来（DPP、小红书、抖音）
4. 最后的三个工程题是"加分项"

---

**维护者**：Boyu | 2026-03-24 | 50 张卡片
