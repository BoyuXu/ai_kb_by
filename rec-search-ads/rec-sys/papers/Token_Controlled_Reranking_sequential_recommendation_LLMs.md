# Token-Controlled Re-ranking for Sequential Recommendation via LLMs

> 来源：arXiv 2025 | 领域：rec-sys | 学习日期：20260404

## 问题定义

传统重排（Re-ranking）面临两个核心问题：
1. **候选集依赖**：排序质量受限于召回候选的质量
2. **忽略列表级交互**：独立打分，忽略列表内物品间的互补/替代关系

LLM 重排虽能建模列表级语义，但存在 **位置偏差（position bias）** 和 **输出不可控（无法保证物品 ID 合法）** 的问题。

## 核心方法与创新点

**Token-Controlled Re-ranking** 通过 token 约束控制 LLM 重排输出：

1. **Token 约束解码（Constrained Decoding）**：
   - 候选集物品 ID → Token Trie 树
   - Beam Search 时只允许沿 Trie 有效路径生成
   
$$V_{\text{allowed}}(t) = \{v : \exists i \in \mathcal{C}, \text{prefix}(i, t) = \text{generated}_{<t}\}$$

2. **列表感知提示（List-Aware Prompting）**：
   - 一次 forward 生成完整排序序列
   - Prompt 包含物品间关系（互补/替代/多样性约束）
   
3. **顺序生成偏差校正**：
   - 位置 i 的物品生成概率除以其自回归先验概率（消除位置偏差）
   
$$P_{\text{corrected}}(i | \text{pos}=k) = \frac{P_{\text{LLM}}(i | \text{context})}{P_{\text{LM}}(i)}$$

4. **轻量适配**：仅需 LoRA 微调，不重训完整 LLM

## 实验结论

- NDCG@5 vs DPP 重排: **+9.2%**
- Token Constraint 将幻觉率（推荐非候选物品）从 8% 降至 **0%**
- 位置偏差校正提升尾部排位准确性 **+14%**
- 延迟：LoRA 微调后推理 ~80ms（7B 模型）

## 工程落地要点

- Trie 树构建离线完成，在线查询 O(len(ID))
- Candidate set size 建议 ≤ 20（过大影响 LLM 注意力质量）
- LoRA rank=16, alpha=32 足够（微调数据 10k 样本）
- 生产中作为精排后的重排层，不替代精排

## 面试考点

1. **Q**: LLM 重排如何防止幻觉（推荐不在候选集的物品）？  
   **A**: Constrained Decoding + Token Trie：解码时只允许合法 ID 路径，从解码机制上保证输出合法。

2. **Q**: 什么是位置偏差？如何校正？  
   **A**: LLM 倾向输出靠前位置的物品，与实际排序质量无关。校正：用物品语言模型先验概率归一化（去除语言模型自然偏好）。

3. **Q**: 为什么用自回归生成而非打分+排序？  
   **A**: 自回归生成天然建模列表级交互（后生成的物品考虑已生成物品的多样性/互补性），打分+排序是独立的。
