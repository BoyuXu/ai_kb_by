# 偏差治理体系：推荐/广告/搜索中的偏差识别与纠正

> 📚 参考文献
> - [Multi-Objective-Ads-Ranking](../../ads/papers/20260316_multi-objective-ads-ranking.md) — 多目标广告排序：MMoE、PLE 与 Pareto 优化
> - [Multi-Objective-Optimization-For-Online-Adverti...](../../ads/papers/20260321_multi-objective-optimization-for-online-advertising-balancing-revenue-and-user-experience.md) — Multi-Objective Optimization for Online Advertising: Bala...
> - [A Generative Re-Ranking Model For List-Level Multi](../../rec-sys/papers/20260323_a_generative_re-ranking_model_for_list-level_multi.md) — A Generative Re-ranking Model for List-level Multi-object...
> - [Multi-Agent Llm Systems Coordination Protocols ...](../../llm-infra/20260323_multi-agent_llm_systems_coordination_protocols_and_.md) — Multi-Agent LLM Systems: Coordination Protocols and Emerg...
> - [Debiased Ctr Uplift Coupon](../../ads/papers/20260313_debiased_ctr_uplift_coupon.md) — Jointly Optimizing Debiased CTR and Uplift for Coupons Ma...
> - [Esmm-Cvr](../../ads/papers/20260317_esmm-cvr.md) — ESMM：全空间多任务 CVR 预估
> - [Multiple-Hypothesis-Bias-Ctr](../../ads/papers/20260319_multiple-hypothesis-bias-ctr.md) — Addressing Multiple Hypothesis Bias in CTR Prediction for...
> - [Est-Ctr-Scaling](../../ads/papers/20260316_est-ctr-scaling.md) — EST: Efficient Scaling Laws in CTR Prediction via Unified...


> 创建：2026-03-24 | 领域：跨域 | 类型：综合分析
> 来源：IPS, Doubly Robust, ESMM, Position Bias Correction, Causal Inference 系列

---

## 🎯 核心洞察（5条）

1. **偏差无处不在且种类繁多**：位置偏差（排名靠前被点击多）、选择偏差（只有被展示的才有标签）、流行度偏差（热门物品被过度推荐）、曝光偏差（模型推什么用户看什么，形成反馈循环）
2. **偏差治理的核心工具是因果推断**：从"相关性"到"因果性"——"用户点击了排名第一的广告"不等于"这个广告最好"，需要因果推断方法分离位置效应和真实偏好
3. **IPS（逆倾向加权）是最基础的去偏方法**：给每个样本乘以 1/P(展示|特征) 的权重，数学上等价于在无偏分布上训练，但方差大
4. **双重稳健（Doubly Robust）是实践最优**：结合 IPS 和回归模型的优点，只要 IPS 或回归模型其一正确，估计就是无偏的
5. **位置偏差是三个领域最普遍的问题**：搜索结果、推荐列表、广告排序都有"靠前=被点"的问题，解决方案高度通用

---

## 📈 技术演进脉络

```
忽略偏差直接训练（~2015）
  → IPS 逆倾向加权（2016-2018）
    → ESMM 全空间建模解决选择偏差（2018）
      → 双重稳健估计 DR（2019-2020）
        → 因果表示学习（2021-2023）
          → LLM 辅助因果发现（2024+）
```

**关键转折点**：
- **IPS 引入（2016）**：首次在推荐系统中系统性处理偏差，从"知道有偏差"到"能纠正偏差"
- **ESMM（2018）**：用全空间建模优雅解决 CVR 的样本选择偏差
- **因果推断工具普及（2021）**：DoWhy/EconML 等库使因果推断从理论走向工程实践

---

## 🔗 跨文献共性规律

| 偏差类型 | 推荐 | 广告 | 搜索 |
|---------|------|------|------|
| 位置偏差 | 推荐列表排名 | 广告位排名 | 搜索结果排名 |
| 选择偏差 | 只有曝光物品有标签 | 只有点击广告有转化标签 | 只有被展示的文档有点击标签 |
| 流行度偏差 | 热门物品霸占曝光 | 头部广告主挤压长尾 | 高权重网站排名靠前 |
| 反馈循环 | 推荐什么用户看什么 | 投放什么数据就偏向什么 | 点击什么排名就更高 |

---

## 🎓 面试考点（6条）

### Q1: 推荐/广告中的位置偏差怎么处理？
**30秒答案**：①训练时：加位置特征但推理时置为默认值（"dropout 位置信息"）；②IPS 加权：`weight = 1/P(click|position)`，位置越靠后权重越大；③双塔去偏：单独建一个位置塔，推理时去掉位置塔的影响。
**追问方向**：位置偏差的 propensity 怎么估计？答：经验公式 P(click|pos=k) ∝ 1/k^α（α≈0.5-1.0），或通过 randomization 实验精确估计。

### Q2: ESMM 解决了什么偏差？
**30秒答案**：样本选择偏差——CVR 训练只有"被点击"的样本有转化标签，但推理时要预估"所有曝光"的转化率。ESMM 建模 pCTCVR = pCTR × pCVR，在全曝光空间训练 CTR 任务，间接约束 CVR 任务。
**追问方向**：还有其他解决选择偏差的方法吗？答：①IPS 加权（给点击样本更高权重）；②全空间负采样（随机选未展示的作负例）。

### Q3: IPS 的方差问题怎么缓解？
**30秒答案**：IPS 的权重 1/P 可能极大（罕见事件 P 很小导致权重爆炸），方差大。缓解：①权重裁剪（cap 最大权重为 M）；②Self-Normalized IPS（权重归一化 Σw_i=1）；③Doubly Robust（结合回归模型减少方差）。
**追问方向**：什么是 Doubly Robust？答：DR = IPS 估计 + 回归模型修正项，只要 IPS 或回归其一正确就无偏。

### Q4: 流行度偏差怎么纠正？
**30秒答案**：①训练去偏：对热门物品的正样本降权（inverse popularity weighting）；②推理调整：`score = model_score × popularity^(-β)`，β>0 抑制热门；③因果方法：Causal Embedding 分离"物品质量"和"流行度效应"。
**追问方向**：去偏太狠会怎样？答：过度抑制热门物品导致推荐"冷门但不好"的内容，用户体验下降。

### Q5: 反馈循环（Feedback Loop）怎么打破？
**30秒答案**：①在线随机化：预留 5-10% 流量做随机推荐/展示，获取无偏数据；②离线去偏训练：用 IPS/DR 方法在有偏数据上训练无偏模型；③多样性注入：重排阶段强制引入非模型推荐的物品。
**追问方向**：为什么不能全部随机化？答：用户体验严重下降，只能用小比例流量做探索。

### Q6: 因果推断在推荐/广告中的应用？
**30秒答案**：①Uplift Modeling：预估"推荐这个物品对用户购买的因果效应"而非相关性；②Treatment Effect Estimation：评估广告投放/促销活动的真实效果；③工具变量/断点回归：利用自然实验估计因果效应。

---

## 🌐 知识体系连接

- **上游依赖**：因果推断理论（Rubin Causal Model, DoWhy）、统计学（IPS, DR）
- **下游应用**：公平推荐、广告效果评估、A/B 测试设计
- **相关 synthesis**：std_ads_multi_objective.md, std_rec_ranking_evolution.md
- **相关论文笔记**：synthesis/20260322_ad_bias_correction_trilogy.md, rec-sys/07_causal_inference.md
