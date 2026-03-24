# 小红书混排系统架构：从原理到工程落地

> 深度案例研究 | 内容 + 广告双重混排 | 字数：450+ 行

---

## 前言

小红书是一个**内容 + 广告 + 电商**的混合平台。与抖音不同的是：
- 小红书内容（笔记）来自用户（UGC）
- 广告也在 Feed 中展示
- 转化率（CVR）比单纯的品牌广告重要

这导致小红书的混排系统需要同时优化：
1. **内容推荐质量**（什么笔记推给用户）
2. **广告混排质量**（什么广告混在笔记中间）
3. **内容与广告的比例**（ad ratio 多少最优）

---

## Part A：系统架构总览

### A.1 整体 Pipeline

```
用户请求（加载 Feed）
    ↓
【召回层】多路召回 1000+ 候选
    ├─ 内容召回（用户兴趣）：500+ 笔记
    ├─ 广告召回（定向）：100+ 广告
    └─ 运营推荐：50+ 强推内容
    ↓
【粗排】快速预筛选，Top-300
    ├─ 特征工程（GBDT）
    ├─ 新鲜度惩罚（防止陈旧内容）
    └─ 基础多样性检查（话题均衡）
    ↓
【精排】精确排序，Top-50
    ├─ DNN CTR/CVR 预估
    ├─ 内容质量评分
    ├─ 广告 eCPM 计算
    └─ 用户反馈融合（如果有）
    ↓
【混排层】（关键）
    ├─ 内容与广告的交错
    ├─ 话题多样性约束
    ├─ 广告主频率控制
    ├─ 电商转化率优化
    └─ 个性化参数调整
    ↓
【最终输出】Feed 列表（20 条）
    ├─ 内容 16 条
    ├─ 广告 4 条（ad_ratio = 20%）
    └─ 排列顺序已优化
```

### A.2 混排层的具体决策

```
【输入】精排的 Top-50
- 内容：45 条笔记（排序分数已有）
- 广告：5 条广告（bid 已有）

【目标】
1. 选择 16 条最优内容
2. 选择 4 条最优广告
3. 交错排列，使最终 Feed 多样化

【处理流程】

Step 1：内容选择（16 条）
├─ 按内容质量排序（CTR + 收藏率 - 反感率）
├─ 加入多样性约束
│  ├─ 话题分布：单个话题不超过 30%
│  ├─ 创作者分布：同一创作者最多 2 条
│  └─ 创意类型分布：图文、视频、长文交错
└─ 选择 Top-16

Step 2：广告选择（4 条）
├─ 按 eCPM 排序（bid × CVR）
├─ 品类限制
│  ├─ 美妆最多 2 个（用户易疲劳）
│  ├─ 其他品类最多 1 个
│  └─ 保证品类多样性
└─ 选择 Top-4

Step 3：交错混排
├─ 策略：1 广告 + 3 内容，循环
│  位置：1, 4, 8, 12, 16, 19
│  （在位置 1、4、8、12 插入广告）
├─ 或者用列表级优化（见 Part C）
└─ 输出最终排序

Step 4：个性化调整
├─ 新用户：内容比例提高到 85%（ad_ratio = 15%）
├─ VIP 用户：ad_ratio = 10%
├─ 活跃老用户：ad_ratio = 25%（用户容忍度高）
└─ 流失风险用户：ad_ratio = 5%（保护体验）
```

---

## Part B：关键技术模块

### B.1 多目标排序：CTR vs CVR vs 多样性

**小红书的特色**：广告的核心指标不是点击，而是**转化率**。

```python
def ad_ranking_score(ad, context):
    """
    广告排序分数，综合考虑多个目标
    """
    
    # 1. 基础 eCPM（展示价格 × 转化概率）
    ctr_score = predict_ctr(ad, context)
    cvr_score = predict_cvr(ad, context)  # 关键！
    ecpm = ad['bid'] * cvr_score
    
    # 2. 广告质量指标
    quality_score = (
        0.3 * ad['creative_quality'] +  # 创意质量
        0.2 * ad['landing_page_quality'] +  # 落地页质量
        0.3 * ad['brand_reputation'] +  # 品牌信誉
        0.2 * (1.0 if ad['has_coupon'] else 0.5)  # 是否有优惠
    )
    
    # 3. 用户匹配度
    user_affinity = (
        0.5 * similarity(user_profile, ad_brand) +
        0.3 * similarity(user_price_segment, ad_price) +
        0.2 * similarity(user_interests, ad_category)
    )
    
    # 4. 多样性惩罚
    diversity_penalty = 0.0
    if ad_category in recent_categories[-3:]:
        diversity_penalty -= 0.15
    if ad_brand in recent_brands[-3:]:
        diversity_penalty -= 0.2
    
    # 5. 综合排序分数
    final_score = (
        0.5 * normalize(ecpm) +
        0.2 * quality_score +
        0.2 * user_affinity +
        0.1 * diversity_penalty
    )
    
    return final_score
```

### B.2 内容与广告的交错规则

**关键决策**：在哪些位置插入广告？

```
【固定交错模式】（简单，最常见）

Feed 位置：[1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, ...]
内容/广告：[C1, C2, C3, A1, C4, C5, C6, A2, C7, C8, C9, A3, ...]

广告位置：4, 8, 12, ...（每 4 个位置一个广告）

优点：简单，用户习惯
缺点：不考虑内容质量

【动态交错模式】（复杂，但更优）

根据内容质量动态决定广告位置：

如果位置 3 的内容质量很高（点击率 > 10%）
    → 在位置 3 后插入广告（用户注意力高）

如果位置 5 的内容质量低（点击率 < 2%）
    → 推迟插入广告（避免内容-广告竞争）

实现：用强化学习或 ILP（整数规划）优化

示例代码：
```

```python
def dynamic_interleaving(contents, ads, user_context):
    """
    根据内容质量动态决定广告位置
    """
    result = []
    content_idx = 0
    ad_idx = 0
    
    # 为每条内容计算"好的"程度
    content_scores = [predict_ctr(c, user_context) for c in contents]
    
    for position in range(len(contents) + len(ads)):
        # 计算"插入广告"的收益
        if ad_idx < len(ads):
            # 如果当前内容质量低，插广告不会太伤害
            if content_scores[content_idx] < 0.05:
                result.append(ads[ad_idx])
                ad_idx += 1
                continue
        
        # 否则插入内容
        if content_idx < len(contents):
            result.append(contents[content_idx])
            content_idx += 1
    
    return result
```

### B.3 广告主频率控制

```python
class AdvertiserFrequencyControl:
    def __init__(self, user_id, feed_length=20):
        self.user_id = user_id
        self.feed_length = feed_length
        
        # 加载用户最近 7 天看过的广告
        self.recent_ads = load_recent_ads(user_id, days=7)
    
    def filter_ads(self, ad_candidates):
        """
        过滤候选广告，应用频率限制
        """
        filtered = []
        advertiser_count = {}
        
        for ad in ad_candidates:
            advertiser = ad['brand']
            
            # 规则 1：7 天内同一品牌最多展示 3 次
            brand_recent = sum(1 for a in self.recent_ads if a['brand'] == advertiser)
            if brand_recent >= 3:
                continue
            
            # 规则 2：当前 Feed 中同一品牌最多 1 个
            if advertiser_count.get(advertiser, 0) >= 1:
                continue
            
            # 规则 3：同一广告（创意）7 天内最多 1 次
            if any(a['ad_id'] == ad['ad_id'] for a in self.recent_ads):
                continue
            
            # 规则 4：竞争品牌（如口红 A vs 口红 B）不能同时出现
            if self.has_competitor(ad, filtered):
                continue
            
            filtered.append(ad)
            advertiser_count[advertiser] = advertiser_count.get(advertiser, 0) + 1
        
        return filtered
    
    def has_competitor(self, ad, current_ads):
        """检查是否与当前 Feed 中已有的广告竞争"""
        ad_category = ad['category']  # 如 "lip_stick"
        ad_brand = ad['brand']
        
        for current_ad in current_ads:
            # 同品类的不同品牌视为竞争
            if (current_ad['category'] == ad_category and 
                current_ad['brand'] != ad_brand):
                return True
        
        return False
```

---

## Part C：列表级优化

### C.1 从单个排序到列表优化

**问题**：
传统方法对每条内容/广告独立评分，然后排序。
但最优的 Feed 不一定是"最优的逐个排序结果"。

**示例**：
```
独立评分排序：[高分内容, 高分内容, 高分内容, 低分广告, ...]
→ 前 3 个高分内容都是"旅游"话题
→ 用户可能感到单调

列表级优化：[中分旅游内容, 高分内容（电商）, 中分内容（美妆）, 高分广告, ...]
→ 多样性更好
→ 整体用户体验更优
```

### C.2 实现方式：迭代优化

```python
def list_level_optimization(candidates, k=20):
    """
    逐个位置决策，考虑前面已选的内容
    """
    result = []
    remaining = list(candidates)
    
    for position in range(k):
        best_candidate = None
        best_score = -np.inf
        
        for candidate in remaining:
            # 特征 1：候选本身的质量
            individual_quality = predict_quality(candidate)
            
            # 特征 2：与前面内容的"和谐度"
            diversity_bonus = compute_diversity_bonus(candidate, result[-3:])
            
            # 特征 3：与用户兴趣的匹配度
            user_affinity = compute_user_affinity(candidate, user_profile)
            
            # 特征 4：位置偏好（广告倾向于中间位置）
            position_bias = compute_position_bias(candidate, position)
            
            # 综合分数
            list_score = (
                0.5 * individual_quality +
                0.2 * diversity_bonus +
                0.2 * user_affinity +
                0.1 * position_bias
            )
            
            if list_score > best_score:
                best_score = list_score
                best_candidate = candidate
        
        result.append(best_candidate)
        remaining.remove(best_candidate)
    
    return result

def compute_diversity_bonus(candidate, history):
    """
    计算多样性奖励
    - 如果话题与历史不同，加分
    - 如果来自不同创作者，加分
    """
    bonus = 0.0
    
    # 话题多样性
    recent_topics = [item['topic'] for item in history]
    if candidate['topic'] not in recent_topics:
        bonus += 0.1  # 新话题，加 0.1
    else:
        bonus -= 0.05  # 重复话题，减 0.05
    
    # 创作者多样性
    recent_creators = [item['creator_id'] for item in history]
    if candidate['creator_id'] not in recent_creators:
        bonus += 0.05
    
    return bonus
```

### C.3 性能优化：早停 + 缓存

```python
def list_optimization_fast(candidates, k=20, use_cache=True):
    """
    加速版本，适合生产环境
    """
    result = []
    remaining = list(candidates)
    
    # 预计算候选特征，避免重复计算
    feature_cache = {}
    for c in candidates:
        feature_cache[c['id']] = {
            'quality': predict_quality(c),
            'user_affinity': compute_user_affinity(c, user_profile),
            'position_bias': [compute_position_bias(c, p) for p in range(k)]
        }
    
    for position in range(k):
        best_candidate = None
        best_score = -np.inf
        
        # 只考虑前 10 个剩余候选（早停）
        candidates_to_consider = remaining[:min(10, len(remaining))]
        
        for candidate in candidates_to_consider:
            c_id = candidate['id']
            
            # 从缓存读取特征
            quality = feature_cache[c_id]['quality']
            user_affinity = feature_cache[c_id]['user_affinity']
            position_bias = feature_cache[c_id]['position_bias'][position]
            
            # 快速计算多样性（只看最近 3 个）
            diversity_bonus = compute_diversity_bonus(candidate, result[-3:])
            
            list_score = (
                0.5 * quality + 0.2 * diversity_bonus + 
                0.2 * user_affinity + 0.1 * position_bias
            )
            
            if list_score > best_score:
                best_score = list_score
                best_candidate = candidate
        
        result.append(best_candidate)
        remaining.remove(best_candidate)
    
    return result
```

---

## Part D：A/B 测试与效果评估

### D.1 关键指标

```
【内容维度】
- 内容 CTR：用户是否点击笔记（越高越好）
- 内容收藏率：用户是否收藏笔记（质量信号）
- 内容分享率：用户是否分享笔记（传播力）

【广告维度】
- 广告 CTR：用户是否点击广告
- 广告 CVR：用户是否转化（购买、加购、点赞）
- 广告 ROI：广告花费 / 销售额

【用户体验维度】
- 停留时长：用户在 App 中停留多久
- 滑动深度：用户向下滑动到第几条
- 日活 DAU：日活跃用户数
- 7 日留存：7 天后用户是否回来

【平台维度】
- eCPM：千次展示收益（广告侧）
- 日均收入：平台广告收入
- 品牌满意度：广告主投放意愿
- 创作者满意度：笔记被推荐次数
```

### D.2 实战案例（推测）

```
【对照组】：基础精排（无混排优化）

【实验组】：列表级混排优化

【运行时间】：2 周

【结果】

指标                  │ 对照    │ 实验    │ 变化
──────────────────────┼─────────┼─────────┼──────
内容 CTR              │ 2.8%    │ 2.9%    │ +3.6%
广告 CTR              │ 2.0%    │ 1.95%   │ -2.5%
广告 CVR              │ 1.2%    │ 1.35%   │ +12.5%
用户停留时长          │ 8m20s   │ 9m10s   │ +9.0%
用户滑动深度          │ 12.3    │ 13.8    │ +12.2%
eCPM（千次）          │ ¥3.2    │ ¥3.45   │ +7.8%
7 日留存              │ 48%     │ 49.5%   │ +3.1%

【整体评价】
✓ CVR 提升 12.5%（转化更好，品牌增加投放）
✓ eCPM 提升 7.8%（综合收入提升）
✓ 留存提升 3.1%（长期价值）
✓ 停留时长 +9%（用户更沉浸）
→ 全量上线，成为核心混排策略
```

---

## Part E：工程挑战与解决方案

### E.1 实时性要求

```
【需求】
- 加载 Feed 需要 < 300ms
- 其中混排优化需要 < 50ms

【挑战】
- 列表级优化需要 O(K²) 计算
- K=50 时，50² = 2500 次评分
- 如果每次评分 1ms，总耗时 2500ms（太慢）

【解决方案】

1. 特征预计算
   - 在粗排/精排阶段，预计算好每条内容的特征
   - 混排层只做"融合 + 排序"

2. 早停 + 贪心
   - 只考虑前 10 个候选，而不是全部 50 个
   - 贪心选择，而不是遍历所有组合

3. 分布式计算
   - 精排和混排并行运行
   - 精排在 GPU 上计算，混排在 CPU 上计算

4. 缓存
   - 缓存用户最近浏览的广告（避免重复计算频率）
   - 缓存话题分布（避免每次重新计算）

【目标耗时分解】
- 特征提取（缓存）：2ms
- 多样性计算：5ms
- 广告选择：8ms
- 内容选择：10ms
- 交错排序：5ms
- 总耗时：30ms（符合要求）
```

### E.2 A/B 测试的陷阱

```
【陷阱 1】：周期性混淆
现象：周末用户行为与工作日不同
    - 周末用户停留时长自然高 +15%
    - 易误认为是混排优化的效果

解决：
    - A/B 测试运行 2+ 周（包括完整周期）
    - 分层分析（周末 vs 工作日单独对比）

【陷阱 2】：新鲜效应
现象：用户尝试新的 Feed 排序方式时，短期行为改变
    - 第 1 天：点击率 +5%
    - 第 7 天：回到正常水平

解决：
    - 只看第 7-14 天的数据（排除新鲜效应）
    - 或用 holdout 对照组（部分用户永不切换）

【陷阱 3】：广告主竞价变化
现象：投资者看到 CVR 提升，增加投放预算
    - 广告量增加
    - 广告平均质量下降
    - 整体效果反而变差

解决：
    - 固定广告投放池（A/B 期间不变）
    - 控制变量，分离"混排优化"和"广告量变化"
```

---

## 总结

**小红书混排的核心创新**：
1. **内容 + 广告的双重混排**（不仅是广告之间的混排）
2. **CVR 作为核心指标**（相比 CTR，更接近商业价值）
3. **列表级优化**（不是单个排序，而是整体 Feed 优化）
4. **个性化 ad_ratio**（不同用户群体的广告比例不同）

**工程挑战**：
- 实时性：< 50ms 的混排计算
- 多目标权衡：内容体验 vs 广告收入 vs 创作者收益
- 数据闭环：快速迭代，周期从 2 周降至 2 天

**最佳实践**：
✓ 从简单方案开始（固定交错）
✓ 逐步引入列表级优化
✓ 持续 A/B 测试，数据驱动迭代
✓ 保持指标透明，与所有利益相关方沟通

---

**维护者**：Boyu | 2026-03-24 | 字数：480+ 行
