# 生成式推荐系统前沿进展综合 - 2026-04-07

## 综合论文
- RASTP (2511.16943) - SID Token 剪枝效率
- GLASS (2602.05663) - 长序列 SID 层级建模
- Cold-Start GenRec (2603.29845) - 冷启动可复现研究
- Reason4Rec (2502.02061) - 深思熟虑的 LLM 推荐
- LinkedIn Feed-SR (2602.12354) - 工业序列推荐
- MTMH (2506.06239) - I2I 多任务多头检索
- SPINRec (2511.18047) - 解释保真度
- Beyond Interleaving - 因果注意力重构

---

## 一、技术演进脉络

```
协同过滤 → 深度学习推荐 → 序列推荐（SASRec, BERT4Rec）
  → 生成式推荐（TIGER, GenRec, NOVA）
    → SID 设计演进：atomic → semantic（RQ-VAE） → hierarchical（GLASS）
    → 效率优化：RASTP（SID Token 剪枝）
    → 冷启动研究：Cold-Start GenRec（可复现性）
  → LLM 增强推荐
    → Reason4Rec（深思熟虑对齐）
  → 工业序列推荐
    → LinkedIn Feed-SR（Transformer 大规模部署）
    → MTMH（I2I 召回的 recall vs relevance 权衡）
```

## 二、核心技术对比

| 方向 | 方法 | 核心创新 | 关键发现 |
|------|------|----------|----------|
| 效率 | RASTP | 语义显著性+注意力中心性剪枝 | 训练时间 -26.7%，性能持平 |
| 长序列 | GLASS | SID-Tier + Semantic Hard Search | TAOBAO-MM, KuaiRec SOTA |
| 冷启动 | Cold-Start | 统一冷启动评估框架 | 标识符设计是关键，非模型规模 |
| LLM 对齐 | Reason4Rec | Deliberative 三阶段框架 | 推理质量和预测准确性双提升 |
| 工业排序 | LinkedIn Feed-SR | Transformer 序列排序 | time spent +2.10% |
| I2I 召回 | MTMH | 多任务+多头 | recall +14.4%, relevance +56.6% |

## 三、核心公式

### SID 层级生成（GLASS）
$$P(\text{item}) = P(\text{SID}_1) \cdot P(\text{SID}_2 | \text{SID}_1) \cdot P(\text{SID}_3 | \text{SID}_1, \text{SID}_2)$$

### RASTP Token 重要性评分
$$\text{Importance}(t) = \alpha \cdot \|\mathbf{h}_t\|_2 + (1-\alpha) \cdot \sum_{l} A_{lt}$$

其中 $\|\mathbf{h}_t\|_2$ 是语义显著性，$A_{lt}$ 是层 $l$ 的累积注意力权重。

### MTMH 多任务损失
$$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{recall}} + \lambda_2 \mathcal{L}_{\text{relevance}}$$

## 四、工业实践经验

1. **SID 设计是决定性的**（Cold-Start 发现）：
   - 文本 SID 改善 item 冷启但损害其他场景
   - 分层语义 SID 更鲁棒
   - 不要期望增大模型规模解决冷启动

2. **长序列效率必须兼顾**（RASTP + GLASS）：
   - SID 带来的序列增长必须通过剪枝平衡
   - 分层建模把全局和细粒度解耦

3. **I2I 召回必须兼顾多样性**（MTMH）：
   - 纯 co-engagement 优化导致过度局部化
   - 显式语义相关性目标促进兴趣发现

## 五、面试高频考点

**Q：生成式推荐 vs 传统推荐的根本区别？**
A：GenRec 将推荐建模为序列生成（next SID prediction），而传统方法是 ranking/retrieval。

**Q：SID 设计的核心原则？**
A：语义保留、唯一性、层级（粗→细）、可泛化到冷启动。

**Q：冷启动问题在生成式推荐中为什么更复杂？**
A：item cold-start 需要新 item 的 SID，user cold-start 缺历史序列；两者需要不同的解决策略。

**Q：为什么 LinkedIn 用 Transformer 而非更复杂的生成式模型做排序？**
A：工业约束：低延迟、可解释性、稳定性；生成式模型在排序场景还不成熟。

**Tags:** #synthesis #rec-sys #generative-recommendation #sid #cold-start #industrial
