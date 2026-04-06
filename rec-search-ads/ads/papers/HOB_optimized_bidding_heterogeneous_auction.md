# HOB: Holistically Optimized Bidding under Heterogeneous Auction Mechanisms

**ArXiv:** 2510.15238 | **Venue:** KDD 2026 | **Date:** 2025-10

## 核心问题
电商广告平台混合使用一价拍卖（FPA）和二价拍卖（SPA），自动出价系统需在异构拍卖机制下制定最优策略，同时满足 MaxReturn 或 TargetROAS 等多样化广告主目标。

## 两大核心贡献

### 1. FPA 最优出价解（考虑自然流量）
推导出包含 organic traffic 情形下 FPA 的高效最优出价解。

### 2. MCA 边际成本对齐策略（Marginal Cost Alignment）
跨异构拍卖机制保证出价效率的对齐策略，解决 FPA 和 SPA 策略不兼容问题。

## 验证
- 公开数据集离线实验
- 大规模线上 A/B 测试：持续超越现有方法

## 面试考点
- FPA vs SPA 的核心差异和出价策略？
- 广告主不同 KPI（MaxReturn vs TargetROAS）如何影响出价？
- 为什么有机流量（organic traffic）会影响广告出价？
- Marginal Cost 在出价优化中的意义？

**Tags:** #ads #bidding #auction #fpa #spa #auto-bidding #kdd
