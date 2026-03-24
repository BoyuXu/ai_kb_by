# 广告系统多目标优化：从人工调权到 Pareto 自动化

> 📚 参考文献
> - [Multi-Objective-Ads-Ranking](../../ads/papers/20260316_multi-objective-ads-ranking.md) — 多目标广告排序：MMoE、PLE 与 Pareto 优化
> - [Multi-Objective-Optimization-For-Online-Adverti...](../../ads/papers/20260321_multi-objective-optimization-for-online-advertising-balancing-revenue-and-user-experience.md) — Multi-Objective Optimization for Online Advertising: Bala...
> - [Wukong Towards A Scaling Law For Large-Scale Recom](../../ads/papers/20260323_wukong_towards_a_scaling_law_for_large-scale_recom.md) — Wukong: Towards a Scaling Law for Large-Scale Recommendation
> - [Est-Ctr-Scaling](../../ads/papers/20260316_est-ctr-scaling.md) — EST: Efficient Scaling Laws in CTR Prediction via Unified...
> - [Real-Time-Bidding-Optimization-With-Multi-Agent...](../../ads/papers/20260321_real-time-bidding-optimization-with-multi-agent-deep-reinforcement-learning.md) — Real-Time Bidding Optimization with Multi-Agent Deep Rein...
> - [Esmm-Cvr](../../ads/papers/20260317_esmm-cvr.md) — ESMM：全空间多任务 CVR 预估
> - [Multi Objective Online Advertising](../../ads/papers/20260322_multi_objective_online_advertising.md) — Multi-Objective Optimization for Online Advertising: Bala...
> - [Counterfactual-Learning-For-Unbiased-Ad-Ranking...](../../ads/papers/20260321_counterfactual-learning-for-unbiased-ad-ranking-in-industrial-search-systems.md) — Counterfactual Learning for Unbiased Ad Ranking in Indust...


> 创建：2026-03-24 | 领域：广告系统 | 类型：综合分析
> 来源：Pantheon, GNOLR, Wukong Scaling Law, MMoE/PLE 系列

---

## 🎯 核心洞察（5条）

1. **广告多目标的本质矛盾**：CTR（平台收入）、CVR（广告主 ROI）、用户体验（长期留存）三者天然冲突——高 CTR 可能是标题党，高 CVR 可能是低价促销商品霸占曝光
2. **Pareto 优化取代人工调权**：传统方式是手动设定 `score = w1×CTR + w2×CVR + w3×体验分`，Pantheon 证明可以自动搜索 Pareto 前沿，找到多组最优权重组合
3. **行为序关系是天然约束**：GNOLR 发现"曝光→点击→转化"的概率单调性（P(展示) ≥ P(点击) ≥ P(转化)）可以作为正则化，减少 85% 的概率违序
4. **多任务架构三代演进**：Shared-Bottom → MMoE → PLE，核心趋势是"任务间共享越来越少，隔离越来越多"
5. **Scaling Law 指导资源分配**：Wukong 验证在固定计算预算下，扩大 Embedding Table（稀疏参数）比加深 MLP 的 ROI 更高

---

## 📈 技术演进脉络

```
单目标 CTR 排序（2012-2015）
  → 手动多目标加权 eCPM = pCTR × pCVR × bid（2015-2018）
    → MMoE/ESMM 多任务建模（2018-2020）
      → PLE 渐进分层提取 + 自动搜权（2020-2022）
        → Pareto 优化（Pantheon）+ Scaling Law 指导（2023-2025）
          → LLM 意图理解 + 多目标推理（2025+）
```

**关键转折点**：
- **ESMM（2018）**：解决 CVR 样本选择偏差，通过 CTCVR=CTR×CVR 在全空间建模
- **PLE（2020）**：解决 MMoE 中任务间 gradient 冲突，渐进分层隔离不同任务
- **Pareto 自动化（2023）**：结束"调权靠经验"的时代，算法自动发现最优权重组合

---

## 🔗 跨文献共性规律

| 规律 | 体现论文/系统 | 说明 |
|------|-------------|------|
| 任务冲突需要显式隔离 | MMoE→PLE 演进 | 共享越多冲突越大，gate 机制是平衡手段 |
| 概率单调性是天然正则 | GNOLR, ESMM | 行为漏斗的因果关系可以约束模型 |
| 稀疏参数 > 稠密参数的扩展效率 | Wukong, DLRM | Embedding 扩展的边际收益一直高于 MLP |
| 自动化搜索取代人工调参 | Pantheon, AutoML | 超参数/权重搜索空间太大，人工不可能穷尽 |

---

## 🎓 面试考点（6条）

### Q1: MMoE vs PLE vs Shared-Bottom 的核心区别？
**30秒答案**：Shared-Bottom 所有任务共享同一个底层网络；MMoE 引入多个 Expert + Gate 动态分配；PLE 在 MMoE 基础上增加任务特有 Expert，渐进分层（多层堆叠），显式隔离任务间干扰。
**追问方向**：什么时候 MMoE 就够了？答：任务相关性高（如 CTR 和点赞率）用 MMoE，任务冲突大（如 CTR 和用户停留时长）用 PLE。

### Q2: ESMM 如何解决 CVR 样本选择偏差？
**30秒答案**：CVR 训练只有"点击样本"有标签（转化/未转化），但推理时要预估"全部曝光"的转化率。ESMM 建模 pCTCVR = pCTR × pCVR，在全曝光空间训练 CTR 任务，间接约束 CVR 分支。
**追问方向**：ESMM 的假设是什么？答：假设 CVR 不依赖于"是否被点击"（即 P(conv|click,x) = P(conv|x)），实际中可能不完全成立。

### Q3: 如何设计广告排序的多目标 eCPM 公式？
**30秒答案**：`eCPM = pCTR × pCVR × bid × quality_factor`，quality_factor 包含用户体验分（如广告相关性、着陆页质量）。不同广告类型（品牌/效果）可用不同权重。
**追问方向**：如何处理短期收入和长期用户体验的 trade-off？答：引入用户体验的长期指标（如 30 日 DAU 影响），用 counterfactual 方法评估。

### Q4: Gradient 冲突在多任务学习中怎么处理？
**30秒答案**：GradNorm 动态调整每个任务的 loss 权重使梯度范数均衡；PCGrad 投影冲突梯度到无冲突方向；MGDA 在 Pareto 梯度方向上做优化。
**追问方向**：实际工程中最常用哪种？答：PLE 架构本身缓解了大部分冲突，loss 权重用简单的 uncertainty weighting。

### Q5: Scaling Law 对广告模型的工程意义？
**30秒答案**：Wukong 发现：①模型效果随参数量幂律增长；②Embedding Table 的 scaling 效率远高于 MLP；③给定计算预算，应优先扩大 Embedding Table。
**追问方向**：万亿参数 Embedding 怎么存储？答：分布式 Parameter Server，每台机器存一部分，通过网络通信合并梯度。

### Q6: Pareto 优化在多目标广告中怎么用？
**30秒答案**：Pantheon 将每组权重 (w_CTR, w_CVR, w_体验) 视为一个解，通过进化算法搜索 Pareto 前沿（没有任何解在所有目标上都优于前沿上的解），让业务方在前沿上选择偏好点。
**追问方向**：线上怎么切换权重？答：A/B Test 验证 Pareto 前沿上不同点的线上效果。

---

## 🌐 知识体系连接

- **上游依赖**：CTR/CVR 预估、多任务学习基础、优化理论
- **下游应用**：eCPM 排序、出价策略、广告预算分配
- **相关 synthesis**：std_ads_bidding_landscape.md, std_rec_ranking_evolution.md, std_cross_multi_objective_unified.md
- **相关论文笔记**：ads/02_ads_advanced.md, rec-sys/05_ranking_deep.md
