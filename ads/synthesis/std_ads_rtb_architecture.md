# 广告系统 RTB 架构全景：从竞价到展示的 100ms

> 📚 参考文献
> - [Multi-Objective-Optimization-For-Online-Adverti...](../../ads/papers/20260321_multi-objective-optimization-for-online-advertising-balancing-revenue-and-user-experience.md) — Multi-Objective Optimization for Online Advertising: Bala...
> - [Multi-Objective-Ads-Ranking](../../ads/papers/20260316_multi-objective-ads-ranking.md) — 多目标广告排序：MMoE、PLE 与 Pareto 优化
> - [Est-Ctr-Scaling](../../ads/papers/20260316_est-ctr-scaling.md) — EST: Efficient Scaling Laws in CTR Prediction via Unified...
> - [Gsp-Vcg-Auction](../../ads/papers/20260316_gsp-vcg-auction.md) — 广告竞价机制：GSP 与 VCG 详解
> - [Multi Objective Online Advertising](../../ads/papers/20260322_multi_objective_online_advertising.md) — Multi-Objective Optimization for Online Advertising: Bala...
> - [Joint-Value-Estimation-Bidding-Repeated-Fpa](../../ads/papers/20260320_joint-value-estimation-bidding-repeated-fpa.md) — Joint Value Estimation and Bidding in Repeated First-Pric...
> - [Real-Time-Bidding-Optimization-With-Multi-Agent...](../../ads/papers/20260321_real-time-bidding-optimization-with-multi-agent-deep-reinforcement-learning.md) — Real-Time Bidding Optimization with Multi-Agent Deep Rein...
> - [Gsp-Vcg-Auction](../../ads/papers/20260317_gsp-vcg-auction.md) — GSP/VCG 拍卖机制（广告竞价理论）


> 创建：2026-03-24 | 领域：广告系统 | 类型：综合分析
> 来源：RTB 实践系列, DSP/SSP/ADX 架构

---

## 🎯 核心洞察（4条）

1. **RTB 是一场 100ms 的精密战役**：用户打开 App/网页 → SSP 发竞价请求 → ADX 广播给多个 DSP → DSP 10ms 内完成"用户识别+CTR预估+出价" → ADX 选最高价展示
2. **eCPM 是统一排序货币**：不同计费方式（CPM/CPC/CPA/oCPX）通过 eCPM = pCTR × pCVR × bid 统一排序，保证平台收益最大化
3. **广告质量分制衡纯价格竞争**：Google 的 Ad Rank = bid × quality_score，quality_score 包含 CTR 预估、广告相关性、着陆页体验，防止"钱多就能排第一"
4. **从二价拍卖到一价拍卖的迁移改变了出价策略**：GSP/VCG 下诚实出价是最优策略，FPA 下需要 Bid Shading 策略性压低出价

---

## 📈 技术演进脉络

```
展示广告直接售卖（~2005）→ RTB 实时竞价 + 二价拍卖（2008-2019）
  → 一价拍卖 FPA（2019-2024）→ 程序化保量 PG + RTB 混合（2024+）
```

---

## 🎓 面试考点（5条）

### Q1: RTB 的完整流程？
**30秒答案**：用户访问页面 → SSP 发送 bid request（用户ID、页面信息、广告位规格）→ ADX 转发给多个 DSP → 各 DSP 在 10ms 内完成用户匹配+CTR 预估+出价决策 → ADX 选最高 eCPM 的广告展示 → 展示/点击/转化事件回传。

### Q2: GSP vs VCG vs FPA 拍卖机制的区别？
**30秒答案**：GSP（广义二价）：每人付下一名出价，不是 truthful（有策略空间）；VCG：每人付"没有你时其他人的收益差"，是 truthful 但收入低；FPA（一价）：赢者付自己出价，需要 Bid Shading。

### Q3: 为什么从二价迁移到一价？
**30秒答案**：①二价拍卖在多层中间商（ADX/SSP/Header Bidding）下失去 truthful 特性；②一价拍卖规则更透明（出多少付多少）；③Google/Meta 在 2019-2021 完成迁移。

### Q4: 广告定向的技术方案？
**30秒答案**：①人口定向（年龄/性别/地域）；②行为定向（浏览/搜索/购买历史）；③Lookalike 扩量（种子用户 → 相似用户）；④重定向 Retargeting（看过未买的用户）；⑤上下文定向（页面内容匹配）。

### Q5: DSP 的 10ms 出价延迟怎么满足？
**30秒答案**：①特征缓存（Redis 预存用户画像）；②模型轻量化（双塔模型 <1ms 推理）；③预计算（高频用户的出价预先计算好缓存）；④降级策略（超时返回默认出价或放弃竞价）。

---

## 🌐 知识体系连接

- **上游依赖**：拍卖理论、CTR/CVR 预估、用户画像
- **下游应用**：出价策略、预算控制、广告效果归因
- **相关 synthesis**：std_ads_bidding_landscape.md, std_ads_multi_objective.md
