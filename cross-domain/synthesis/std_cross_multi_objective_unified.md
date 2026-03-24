# 多目标优化在推荐/广告/搜索中的统一框架

> 📚 参考文献
> - [Multi-Objective-Ads-Ranking](../../ads/papers/20260316_multi-objective-ads-ranking.md) — 多目标广告排序：MMoE、PLE 与 Pareto 优化
> - [Pantheon Personalized Multi-Objective Ensemble Sor](../../rec-sys/papers/20260323_pantheon_personalized_multi-objective_ensemble_sor.md) — Pantheon: Personalized Multi-objective Ensemble Sort via ...
> - [Multi-Objective-Optimization-For-Online-Adverti...](../../ads/papers/20260321_multi-objective-optimization-for-online-advertising-balancing-revenue-and-user-experience.md) — Multi-Objective Optimization for Online Advertising: Bala...
> - [Esmm-Cvr](../../ads/papers/20260317_esmm-cvr.md) — ESMM：全空间多任务 CVR 预估
> - [Est-Ctr-Scaling](../../ads/papers/20260316_est-ctr-scaling.md) — EST: Efficient Scaling Laws in CTR Prediction via Unified...
> - [Multi Objective Online Advertising](../../ads/papers/20260322_multi_objective_online_advertising.md) — Multi-Objective Optimization for Online Advertising: Bala...
> - [Mmoe-Multi-Task-Learning](../../rec-sys/papers/20260317_mmoe-multi-task-learning.md) — MMoE：多门控混合专家（Multi-gate Mixture-of-Experts）
> - [Multi-Behavior-Rec-Survey](../../rec-sys/papers/20260319_multi-behavior-rec-survey.md) — Multi-behavior Recommender Systems: A Survey


> 创建：2026-03-24 | 领域：跨域 | 类型：综合分析
> 来源：Pantheon, MMoE, PLE, ESMM, Pareto 前沿, GradNorm 系列

---

## 🎯 核心洞察（5条）

1. **三个领域的多目标本质相同**：推荐（点击+时长+多样性）、广告（CTR+CVR+ROI）、搜索（相关性+多样性+时效性）都是在多个冲突目标间寻找平衡
2. **统一的数学框架是 Pareto 优化**：不存在在所有目标上都最优的单一解，只有 Pareto 前沿上的一组解。业务需求决定在前沿上选哪个点
3. **多任务学习是多目标优化的工程实现**：MMoE/PLE 用共享+专用网络架构让模型同时预估多个目标，是三个领域都通用的基础设施
4. **梯度冲突是多目标学习的核心难题**：不同任务的梯度方向可能相反（提升 CTR 的更新可能降低时长），需要 GradNorm/PCGrad 等方法调解
5. **线上多目标融合公式是业务最终表达**：`final_score = f(pCTR, pCVR, duration, diversity, ...)`，这个公式编码了公司的商业策略

---

## 📈 技术演进脉络

```
单目标排序（2010-2015）
  → 人工加权多目标 score = Σw_i × f_i（2015-2018）
    → 多任务共享网络 MMoE/ESMM（2018-2020）
      → 任务隔离架构 PLE + 梯度调解 GradNorm（2020-2022）
        → Pareto 自动搜索 Pantheon（2023-2024）
          → LLM 辅助多目标推理（2025+）
```

---

## 🔗 跨文献共性规律

| 规律 | 推荐 | 广告 | 搜索 |
|------|------|------|------|
| 主目标 | 点击率+完播率 | CTR×CVR×bid | 相关性 |
| 辅助目标 | 多样性、新颖性 | ROI、预算消耗率 | 多样性、时效性 |
| 多任务架构 | MMoE/PLE | MMoE/ESMM | 多头排序 |
| 融合方式 | 加权求和 | eCPM 公式 | 加权求和+规则 |
| 冲突点 | 点击率 vs 多样性 | 短期收入 vs 长期体验 | 相关性 vs 多样性 |

---

## 🎓 面试考点（5条）

### Q1: 推荐/广告/搜索的多目标有什么共性和差异？
**30秒答案**：共性——都是多个冲突目标的 Pareto 优化，都用多任务学习作为基础架构。差异——广告有显式的商业约束（预算/ROI），推荐更偏用户体验（多样性/新颖性），搜索强调相关性+时效性。
**追问方向**：哪个领域的多目标最复杂？答：广告，因为有硬约束（预算必须花完、ROI 不能低于阈值）。

### Q2: Pareto 前沿在多目标优化中的作用？
**30秒答案**：Pareto 前沿是"没有任何其他解能在不牺牲某个目标的情况下改善另一个目标"的解集。作用：①展示目标间的 trade-off 关系；②让业务决策者在前沿上选择偏好点；③自动化搜索避免人工调权。

### Q3: 梯度冲突怎么检测和处理？
**30秒答案**：检测——计算不同任务梯度的余弦相似度，<0 表示冲突。处理：①GradNorm 动态调整 loss 权重；②PCGrad 投影冲突梯度到非冲突方向；③PLE 架构本身减少共享，降低冲突。

### Q4: 线上多目标融合公式怎么设计？
**30秒答案**：`score = pCTR^a × pCVR^b × duration^c × diversity_bonus`，a/b/c 是超参数。设计原则：①主目标权重最大；②辅助目标用小权重微调；③通过 A/B Test 验证不同权重组合。
**追问方向**：为什么用乘法不用加法？答：乘法对各指标的量级不敏感，且符合"所有指标都重要"的语义（任何一个为 0 则总分为 0）。

### Q5: 多场景/多业务线怎么统一多目标？
**30秒答案**：STAR（阿里）为每个场景维护独立的 Batch Normalization 参数，共享底层网络。不同场景（首页推荐/搜索结果/广告位）有不同的目标权重，但共享特征提取能力。

---

## 🌐 知识体系连接

- **上游依赖**：多任务学习、优化理论、Pareto 优化
- **下游应用**：排序公式设计、业务策略表达、A/B 测试
- **相关 synthesis**：std_ads_multi_objective.md, std_rec_ranking_evolution.md
