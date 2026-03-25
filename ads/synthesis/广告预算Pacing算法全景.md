# 知识卡片 #010：广告预算 Pacing 算法全景

> 📚 本文档已整合至主文件
>
> **主文档**: [AutoBidding 技术演进：从规则出价到强化学习](./AutoBidding技术演进_从规则到RL.md)

本文档之前强调的 Pacing 核心算法（PID 控制、对偶梯度下降、FPA Shading、Wasserstein 自适应）已全部合并至上述主文档的**"深度理论分析"和"工业常见做法"章节**中。

## 导航指引

- **PID Pacing 生活类比** → [PID Pacing 的生活类比](./AutoBidding技术演进_从规则到RL.md#pid-pacing-的生活类比)
- **对偶梯度下降详解** → [对偶梯度下降 Pacing 详解](./AutoBidding技术演进_从规则到RL.md#对偶梯度下降-pacing-详解)
- **FPA Shading 原理** → [第一价格拍卖（FPA）中的 Bid Shading](./AutoBidding技术演进_从规则到RL.md#第一价格拍卖fpa中的-bid-shading)
- **Wasserstein 自适应** → [非平稳环境下的 Wasserstein Pacing](./AutoBidding技术演进_从规则到RL.md#非平稳环境下的-wasserstein-pacing)
- **Pacing 最佳实践** → [预算 Pacing 最佳实践](./AutoBidding技术演进_从规则到RL.md#预算-pacing-最佳实践)
- **面试考点** → [Q12: 预算 Pacing 和广告出价的关系？](./AutoBidding技术演进_从规则到RL.md#q12预算-pacing-和广告出价是什么关系)、[Q13: 多约束 Pacing](./AutoBidding技术演进_从规则到RL.md#q13多约束情况下预算--roas-约束如何设计-pacing)

## 内容迁移说明

| 原章节 | 新位置 | 备注 |
|--------|--------|------|
| 核心公式 | 主文档·深度理论分析 | 整合到统一的数学框架中 |
| PID 控制原理 | 主文档·PID Pacing 详解 | 增加了与对偶的对比 |
| 对偶梯度下降 | 主文档·对偶梯度下降详解 | 详细展开了在线更新规则 |
| FPA Shading | 主文档·第一价格拍卖详解 | 补充了实践估计方法 |
| Wasserstein | 主文档·Wasserstein Pacing | 增加了遗憾界分析 |
| 面试考点 | 主文档·Q12-Q14, Q11 | 重组并补充了细节 |

---

**更新时间**: 2026-03-25  
**整合完成**: MelonEggLearn  
**行数增长**: 原 189 行 → 主文档新增 ~400 行 Pacing 相关内容
