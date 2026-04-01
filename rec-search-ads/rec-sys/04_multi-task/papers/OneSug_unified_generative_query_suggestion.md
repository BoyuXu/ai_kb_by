# OneSug: Unified Generative Query Suggestion for E-commerce Search
> 来源：工业论文 | 领域：推荐系统/搜索 | 学习日期：20260327

## 问题定义
电商搜索的Query Suggestion（搜索建议/补全）面临挑战：
1. **传统方案局限**：基于历史统计的前缀匹配，只能推荐出现过的query，无法生成新颖建议
2. **个性化不足**：同样的前缀"苹果"，不同用户意图截然不同（手机 vs 水果 vs 电脑）
3. **质量控制难**：生成式方案可能产出语义正确但用户不感兴趣的query
**目标**：构建统一生成式框架，结合个性化信号和质量约束，生成高质量搜索建议。

## 核心方法与创新点

### 1. 统一生成框架
将query suggestion建模为条件文本生成：

$$
P(q | q_{prefix}, u_{hist}, ctx) = \prod_{i=|prefix|+1}^{|q|} P(q_i | q_{<i}, q_{prefix}, u_{hist})
$$

输入：用户输入的前缀 + 用户历史行为 + 会话上下文
输出：完整的搜索query

### 2. Trie约束Beam Search
防止生成幻觉（不在商品库中的query），使用前缀树约束生成：
1. 离线构建候选query的Trie树（来自历史高质量query）
2. 每个解码步只允许选择Trie中合法的下一个字符/token
3. 保留了生成的个性化能力，同时约束了生成空间

$$
\text{NextTokens}(q_{<i}) = \{t : \text{Trie.has}}_{\text{{\text{prefix}}}(q_{<i} + t)\}
$$

### 3. 个性化生成
用户历史搜索序列作为prompt前缀：
```
[用户最近搜索: "运动鞋 Nike Air Max 跑步..."] 当前输入: "Ad" → 
建议: "Adidas跑步鞋", "Adidas超轻跑鞋"
```

### 4. 质量奖励微调
用点击率作为奖励信号，通过RLHF微调生成质量：

$$
r(q) = \text{CTR}(q) \cdot \text{商业价值}(q)
$$

## 实验结论
- **用户点击率**：+8%（相比基于统计的前缀匹配方案）
- **Query覆盖率**：新颖query（历史未出现）占比从5%提升到25%
- **准确率**：Trie约束后，生成query的商品覆盖率>99%（幻觉基本消除）
- 个性化版本比非个性化版本CTR再提升+3%

## 工程落地要点
1. **Trie树大小**：百亿级别的历史query构建的Trie树内存消耗大，需要压缩（DARTS或Hash Trie）
2. **实时性**：用户输入后<50ms内返回建议，需要提前计算KV Cache
3. **Trie更新**：新热搜词需要及时加入Trie树，建议小时级更新
4. **多语言支持**：中文搜索需要character-level Trie，英文可以subword-level
5. **A/B实验**：新方案建议先灰度10%流量，监控搜索成功率和GMV

## 面试考点
Q1: Trie约束Beam Search如何工作？
A: Trie（前缀树）中存储所有合法的候选query。在Beam Search的每个解码步，对当前所有beam路径，查询Trie树获取合法的下一个token集合（字符或subword）。只在合法token上做softmax，屏蔽其他token的logits（设为-inf）。这样保证所有生成结果都在预定义的合法空间内，完全消除幻觉。

Q2: 个性化Query Suggestion和通用Query Suggestion如何平衡？
A: (1)用户历史权重衰减：时间越久远的历史影响越小；(2)个性化置信度：历史行为少的用户降低个性化权重，退化到通用建议；(3)多样性保证：确保Top-5建议中至少有2-3个来自通用排名；(4)实验表明：活跃用户个性化效果显著，低活用户通用方案更好。

Q3: RLHF在Query Suggestion中的挑战？
A: (1)延迟奖励：用户点击某个suggestion后还需要搜索成功（找到想要的商品）才是真正正向奖励，但这个信号有延迟；(2)探索与利用：只优化历史高CTR query会陷入局部最优，需要探索新query；(3)奖励稀疏：大多数展示的suggestion没有被点击，正样本稀缺。
