# OneSug: The Unified End-to-End Generative Framework for E-commerce Query Suggestion
> 来源：arxiv/2312.xxxxx | 领域：rec-sys | 学习日期：20260326

## 问题定义
电商搜索中的 Query Suggestion（查询建议/联想词）面临挑战：
- 传统方法：基于 n-gram 统计，无法理解语义，建议质量差
- 多阶段流水线：候选生成 → 过滤 → 排序，各阶段独立优化
- 个性化不足：未考虑用户上下文和购物意图
- 长尾查询覆盖不足：低频查询缺乏统计数据

## 核心方法与创新点
**OneSug**：端到端生成式查询建议框架，统一候选生成和排序。

**Prefix-to-Query 生成模型：**
```
P(q_suggestion | prefix, user_context) = Π P(token_t | token_{<t}, prefix, user_ctx)

# 用户上下文：当前 session 的搜索历史、点击序列、购物车
user_ctx = Encoder([q_1, q_2, ..., q_{t-1}, clicked_items])
```

**统一训练目标：**
```
L = L_gen + λ·L_rank
L_gen  = -log P(gold_query | prefix, ctx)     # 生成正确查询
L_rank = ListMLE(scores, [q_pos, q_neg_1, ...]) # 排序正样本高于负样本
```

**个性化 Prefix-Conditioned Decoding：**
- Prefix Tree 约束：生成结果必须是合法查询（在历史查询树中）
- 用户 embedding 注入每层 Transformer（Cross-attention）
- 实时 session 更新：每次点击后更新 user_ctx

**负样本构建：**
- Hard Negative：语义相近但用户未点击的查询
- In-batch Negative：batch 内其他用户的查询作为负例

## 实验结论
- 淘宝搜索 A/B 测试：点击率 +3.2%，搜索 GMV +1.8%
- 查询建议接受率（用户点击建议比例）：+15%
- 长尾查询覆盖（低频 prefix 的 Recall@5）：+22% vs n-gram 方法
- 生成多样性（Distinct-4）：+31% vs 统计方法

## 工程落地要点
1. **Prefix Tree 预计算**：离线构建所有历史查询的 Trie 树，在线解码时约束 token 空间
2. **实时 session 特征**：Kafka 消费用户行为流，低延迟更新 session embedding
3. **缓存策略**：高频 prefix（Top 1M）预计算建议结果并缓存
4. **多样性控制**：beam search + 多样性惩罚（避免建议相似的查询）
5. **安全过滤**：生成后过违禁词、品牌词、敏感词过滤

## 常见考点
**Q1: Query Suggestion 为什么适合生成式框架？**
A: Query 本质是文本序列，生成式模型天然适合；可以建模查询的语义完整性；能处理长尾前缀（无需大量统计数据）；端到端优化消除多阶段 gap。

**Q2: Prefix Tree 约束解码是什么？**
A: 在 beam search 每步解码时，只允许生成当前 Trie 节点的合法子 token（保证生成结果是真实存在的查询）。这避免了 LLM 幻觉生成无意义查询。

**Q3: 如何平衡建议的相关性和多样性？**
A: Diverse Beam Search（分组 beam search，惩罚组内相似序列）+ 最终 MMR（最大边际相关性）重排：选择与用户意图相关但彼此不相似的建议组合。

**Q4: 个性化如何注入生成过程？**
A: ①Prefix 层融合：将用户 embedding 拼接到 prefix tokens ②Cross-attention：每层 Transformer 对用户表示做 cross-attention ③Prompt 注入：将用户偏好描述化为文本 prompt。

**Q5: 如何处理查询建议的实时性（session 内）？**
A: 滑动窗口 session 表示 + 在线更新：每次用户交互后（点击/购买/加购），增量更新 session embedding，下次前缀输入时使用最新 session 状态。
