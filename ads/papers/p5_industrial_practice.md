# Project 5：工业案例与最佳实践

> 项目类型：实战总结 | 日期：20260324 | 领域：ads × production

## 导言

理论再完美，最终要靠工程落地。本项目从真实的产品角度，梳理混排系统的端到端建设。

---

## Part A：广告系统架构全景

### A1. 典型的广告排序 Pipeline

```
【用户请求】
↓
【参数提取】
  用户 ID、时间、位置、设备、历史等

↓
【多路召回】 (Recall)
  召回路 1：广告主定向（"女性，年龄 20-30，上海"）
  召回路 2：内容相似（"看过电商，推荐电商+美妆"）
  召回路 3：热门广告（"最近转化高的广告"）
  召回路 4：多样化（"刻意覆盖冷门广告主"）
  └─> 合并去重 → Top-1000 候选

↓
【粗排】 (Coarse Ranking)
  快速 CTR 预估模型（轻量级）
  延迟 <20ms
  └─> Top-500 候选

↓
【精排】 (Fine Ranking)
  多个重模型并发：
    - CTR 模型（DeepFM）→ p(click)
    - CVR 模型（ESMM）→ p(convert|click)
    - 品牌安全模型 → safety_score
    - 创意质量模型 → quality_score
  
  组合分数：eCPM = p(click) × p(convert|click) × bid
  延迟 50-80ms
  └─> Top-50 候选

↓
【重排】 (Reranking)
  约束优化排序：
    - 多样性检查（话题、来源）
    - 频率限制（同一广告主）
    - 质量保证（品牌安全 > 阈值）
  
  算法：贪心约束、LambdaMART、或 RL
  延迟 <20ms
  └─> Top-20 最终排序

↓
【内容填充】(Fill)
  如果广告不足 20 个，用内容填充
  优先级：推荐内容 > 历史内容 > 默认内容

↓
【展示】
  返回排序结果给客户端
  附带日志（用于离线分析）

↓
【反馈收集】
  用户交互数据（点击、停留、转化）
  └─> 离线数据仓库（用于训练）
```

### A2. 各层的关键参数

| 阶段 | 输入 | 输出 | 延迟 | 模型数 | 约束 |
|-----|------|------|------|--------|------|
| 召回 | 用户 ID | 1000 候选 | <50ms | 4-5 | 定向准确性 |
| 粗排 | 1000 候选 | 500 候选 | <20ms | 1-2 | 速度优先 |
| 精排 | 500 候选 | 50 候选 | 50-80ms | 5-8 | 效果优先 |
| 重排 | 50 候选 | 20 广告 + 填充 | <20ms | 0-2 | 约束优先 |

---

## Part B：重排层的工程实现

### B1. 约束的数据结构

```python
class RankingConstraints:
    # 频率限制
    advertiser_freq_cap: Dict[str, int]  # {"advertiser_A": 2, ...}
    topic_freq_cap: Dict[str, int]       # {"电商": 5, ...}
    
    # 质量约束
    min_quality_score: float = 0.7
    min_brand_safety: float = 0.8
    
    # 多样性约束
    min_topic_diversity: float = 0.4
    min_advertiser_diversity: float = 0.5
    
    # RPM 约束
    min_rpm_target: float = 2.0  # 每千次展示不低于 2 块钱
    
    # 用户体验
    max_ad_ratio: float = 0.4  # 广告不超过 40%

def check_constraints(selected, candidate, context):
    """检查候选是否满足所有约束"""
    # 频率检查
    if count_advertiser(candidate, selected) >= freq_cap[candidate.advertiser]:
        return False, "advertiser_freq_cap"
    
    # 质量检查
    if candidate.quality_score < min_quality_score:
        return False, "quality_score"
    
    # 多样性检查（简化）
    diversity = compute_diversity(candidate, selected)
    if diversity < min_diversity:
        return False, "diversity"
    
    return True, "ok"
```

### B2. 贪心约束排序的实现

```python
def greedy_reranking(candidates, constraints, context, k=20):
    selected = []
    remaining = candidates.copy()
    
    for i in range(k):
        best_ad = None
        best_score = -1
        
        for ad in remaining:
            # 检查约束
            valid, reason = check_constraints(selected, ad, context)
            if not valid:
                continue
            
            # 计算重排分数（考虑多样性）
            score = ad.original_score
            if selected:
                diversity_bonus = compute_diversity_with_selected(ad, selected)
                score = 0.8 * ad.original_score + 0.2 * diversity_bonus
            
            if score > best_score:
                best_score = score
                best_ad = ad
        
        if best_ad is None:
            # 无法找到满足所有约束的广告，宽松一个约束
            best_ad = relax_and_find(remaining, selected, constraints)
            if best_ad is None:
                break
        
        selected.append(best_ad)
        remaining.remove(best_ad)
    
    return selected
```

### B3. 性能优化技巧

```python
# 技巧 1：预计算特征，避免重复计算
feature_cache = {}
for ad in candidates:
    feature_cache[ad.id] = precompute_features(ad)

# 技巧 2：用位操作加速频率检查
advertiser_mask = [0] * num_advertisers  # 位标记
for ad in selected:
    advertiser_mask[ad.advertiser_id] += 1

# 技巧 3：提前终止（如果候选不足，无需遍历所有）
if len(selected) + len(remaining) == k:
    selected.extend(remaining)
    break

# 技巧 4：并发检查不同约束
from concurrent.futures import ThreadPoolExecutor
results = executor.map(check_constraint_type, constraints)
```

---

## Part C：数据驱动的优化

### C1. A/B 测试框架

```
【基线】：上一个线上版本
【对照组】：使用基线排序
【测试组】：使用新排序算法

【样本量计算】
样本量 = (Z_α + Z_β)² × σ² / (δ)²
Z_α = 1.96（显著性 5%）
Z_β = 0.84（功效 80%）
σ = 历史 RPM 标准差
δ = 期望改进幅度

示例：
RPM 期望改进 1%，历史 σ=0.2
样本量 = (1.96+0.84)² × 0.2² / 0.01² ≈ 1.5M

→ 需要 1.5M 用户，每组 5-10%，运行 1-2 周
```

### C2. 关键指标体系

```
【在线指标】（实时监控）
- CTR：点击率
- RPM：千次展示收益
- CVR：转化率
- Dwell Time：停留时间

【用户体验指标】
- User Retention：用户留存
- Feed Scroll Depth：滑动深度
- Ad Fatigue Score：广告疲劳度

【排序质量指标】
- NDCG：排序指标
- Topic Diversity：话题多样性
- Ad Frequency Distribution：频率分布

【业务指标】
- Total Revenue：总收益
- Ad Coverage：广告主覆盖率
- CTR×RPM：复合指标
```

### C3. 监控告警

```yaml
# Prometheus 告警规则示例

alert: HighCTRDropInReranking
  expr: (ctr_reranking - ctr_baseline) / ctr_baseline < -0.02
  for: 10m
  action: page_oncall  # 呼人介入

alert: DiversityLow
  expr: topic_diversity < 0.3
  for: 30m
  action: rollback_version  # 自动回滚

alert: RankingLatencyHigh
  expr: p99_latency_reranking > 100ms
  for: 5m
  action: scale_up_resource  # 自动扩容
```

---

## Part D：常见问题与解决方案

### D1. 新广告冷启动

```
【问题】
新上线的广告没有历史 CTR/CVR 数据，排序模型无法预估

【解决方案】

1. 【Content-based 特征】
   不依赖历史数据，用广告内容特征：
   - 文本：TF-IDF、Word2Vec
   - 图片：预训练的 CNN embedding
   - 元数据：广告主 ID、话题等

2. 【广告主迁移学习】
   同一广告主的历史数据 → 初始化新广告的模型参数

3. 【探索性排序】
   新广告自动获得一定比例的展示（例如 5%）
   通过 Bandit 机制逐步调整

4. 【人工审核】
   新广告发布前，人工审核其质量分、定向等，确保不会排名垫底
```

### D2. Online-Offline Gap

```
【现象】
离线排序指标好（NDCG 0.75），线上效果差（CTR 反而降低）

【原因】
1. 数据分布不同（离线用昨天数据，线上是实时）
2. 新广告效应（学习模型未见过，表现与训练数据不符）
3. 位置偏差（用户点击第 1 位的概率远高于第 10 位）

【解决】
1. 用位置 biased 的排序标签
   rel_i = (original_rel_i) / (position_bias[i])
   position_bias = [1.0, 0.8, 0.6, 0.4, ...]

2. 逆向概率加权（IPS）
   train_weight = 1 / P(position | original_ranking)
   
3. 定期更新模型（周级别）
   用最新数据重训排序模型
```

### D3. 约束冲突

```
【例子】
约束 A：多样性 > 0.4
约束 B：RPM > 2.0

有时两个约束无法同时满足（多样性需要低 eCPM 广告，但 RPM 要求高分数）

【解决】
1. 优先级排序
   class Priority(Enum):
       HARD = 1      # 必须满足
       SOFT = 2      # 尽量满足
       OPTIONAL = 3  # 可以不满足
   
   约束 A (多样性)：Priority.SOFT
   约束 B (RPM)：Priority.HARD
   
   排序时，优先满足 HARD，然后满足 SOFT

2. 权重调整
   不同时段、不同用户群体的权重不同
   早上（用户在线少）：优先 RPM
   晚上（用户活跃）：优先用户体验

3. 约束松弛
   如果无法同时满足，计算"违反最少"的排序
```

---

## Part E：完整的技术栈选型

### E1. 召回层

```
【推荐方案】
1. 协作过滤（CF）
   - 算法：矩阵分解、I2I 近邻
   - 框架：Spark ALS、PyTorch 矩阵分解

2. 双塔模型（Two-Tower）
   - 框架：Tensorflow/PyTorch
   - 部署：Faiss（相似度搜索）

3. 向量检索（ANN）
   - 库：Faiss、Milvus、Weaviate
   - 延迟：<10ms，支持 100M+ 向量
```

### E2. 排序层

```
【粗排】
- LightGBM Rank（速度快）
- 或轻量级 MLP（固定 embedding）

【精排】
- Gradient Boosting + Deep Learning 混合
- TensorFlow Serving / TorchServe（模型部署）

【重排】
- LambdaMART on reranking list
- 或自定义 Python 实现（约束检查）
```

### E3. 特征工程

```
【离线特征计算】
- Spark / Hive（大数据处理）
- 特征存储：Feast、Tecton
- 延迟：T+1（前一天的特征）

【在线特征】
- Redis（用户实时统计）
- 特征服务：KServe、SageMaker Feature Store
- 延迟：<50ms
```

---

## Part F：上线前检查清单

### F1. 功能检查

- [ ] 排序算法正确性测试（单元测试 + 集成测试）
- [ ] 约束检查全覆盖（至少 10 个约束）
- [ ] 性能基准测试（延迟、QPS）
- [ ] 降级策略（如果排序超时，快速返回备选）

### F2. 数据检查

- [ ] 特征完整性（缺失值处理）
- [ ] 数据新鲜度（最新数据延迟 <1h）
- [ ] 数据质量监控（异常值、分布变化）

### F3. 运维检查

- [ ] 监控告警（5+ 个关键指标）
- [ ] 灰度发布（5% → 10% → 50% → 100%）
- [ ] 回滚计划（出问题如何秒级回滚）
- [ ] 文档完善（排序逻辑、约束定义、告警响应）

### F4. A/B 测试设计

- [ ] 样本量计算
- [ ] 对照组选择
- [ ] 运行时长（至少 1-2 周）
- [ ] 指标监控（实时仪表板）

---

## Part G：迭代与演进

### G1. 短期优化（1-3 个月）

1. 多样性约束细化
   - 从笼统的"多样性 >0.3"→ 分话题定义
   - "电商 <30%, 美妆 <20%, 房产 <15%"

2. 用户个性化
   - 根据用户粘性调整 ad_ratio
   - 根据用户历史调整话题偏好

3. 实时反馈融入
   - 用用户的"滑动速度"反馈排序效果
   - 快速迭代排序策略

### G2. 中期优化（3-12 个月）

1. 从启发式 → LTR
   - 用离线数据训练 LambdaMART
   - 对标行业 NDCG 水平

2. 从离线 → 在线
   - 加入 Contextual Bandit
   - 实时用户反馈闭环

3. 广告主和内容的联动
   - 推荐系统和广告系统共享排序框架
   - 统一的多目标优化

### G3. 长期展望（1+ 年）

1. 生成式排序探索
   - 离线验证 GR 的可行性
   - 小流量线上试验

2. 跨域迁移
   - 排序模型迁移到其他产品
   - 多产品联合优化

3. 可信 AI
   - 排序的公平性保证
   - 用户隐私保护
   - 可解释性增强

---

## 参考资源

### 工业论文与报告
- Meta MOEF（多目标优化）
- Google Learning to Rank（LTR 综述）
- Alibaba ESMM（CVR 预估）
- Pinterest ML-DCN（广告 CTR）

### 开源框架
- LightGBM（排序）
- XGBoost（排序）
- TensorFlow Ranking（深度 LTR）
- Faiss（向量检索）
- Feast（特征存储）

### 最佳实践文献
- "Effective Ranking in Online Advertising"
- "A Practical Guide to Ranking"
- "Online Learning for Large-Scale Ranking"

---

**维护者**：Boyu | 2026-03-24
