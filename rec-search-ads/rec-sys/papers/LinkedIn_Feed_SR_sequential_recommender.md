# An Industrial-Scale Sequential Recommender for LinkedIn Feed Ranking

**ArXiv:** 2602.12354 | **Date:** 2026-02 | **Org:** LinkedIn

## 核心贡献
Feed-SR（Feed Sequential Recommender）：基于 Transformer 的序列排序模型，替换 LinkedIn Feed 原有的 DCNv2 排序模型，满足严格的生产约束。

## 模型特性
- **架构**：Transformer-based sequential ranking
- **取代**：DCNv2 生产模型
- **作者团队**：23 位作者，LinkedIn 规模部署

## 线上 A/B 测试结果
- Member 时长（time spent）：**+2.10%**
- 成为 LinkedIn Feed 的主要 member experience

## 技术要点
- 解决大规模 sequential 推荐的工程约束（延迟、吞吐量）
- Transformer 架构在工业排序中的实际应用
- 多种训练技术和 serving 优化

## 工业意义
展示了 Transformer-based 序列推荐在超大规模 (LinkedIn) 生产环境中的成功落地，验证了 sequential modeling 对排序的价值。

## 面试考点
- Sequential recommender vs 非序列方法的优势？
- 如何在生产中部署 Transformer 推荐模型？
- LinkedIn Feed 排序的特殊挑战（多样内容类型）？

**Tags:** #rec-sys #sequential-recommendation #transformer #ranking #linkedin #industrial
