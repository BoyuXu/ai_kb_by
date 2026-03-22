# LURE-RAG：轻量效用驱动的 RAG 高效重排序

> 来源：LURE-RAG: Lightweight Utility-driven Reranking for Efficient RAG | 日期：20260318 | 领域：search

## 问题定义

RAG（Retrieval-Augmented Generation）系统中，检索到的文档在送入 LLM 生成前通常需要重排序，筛选出最有用的 top-K 文档。现有重排序方案存在问题：
1. **Cross-encoder 重排序**：精度高但速度慢，无法处理大量候选文档。
2. **传统相关性重排序**：只考虑查询-文档相关性，不考虑文档对 LLM 最终**生成质量**的"效用"（Utility）。相关但冗余、或相关但超出 LLM 处理范围的文档效用低。
3. **计算预算限制**：RAG 场景延迟敏感，重排序不能成为瓶颈。

## 核心方法与创新点

1. **效用（Utility）定义**：不只看文档与查询的相关性，而是估计"将这个文档加入 LLM 上下文后，对生成正确答案的贡献"：
   - 相关性（Relevance）：文档是否回答了查询。
   - 互补性（Complementarity）：与已选文档的信息互补程度（避免冗余）。
   - 可读性（Usability）：文档格式是否适合 LLM 理解（过于结构化的表格或代码段可能降低效用）。

2. **轻量效用评估器**：训练一个轻量 bi-encoder（而非 cross-encoder），输入为 (query, doc, already_selected_docs_summary) 三元组，预测效用分数。Summary 用固定长度 embedding 表示，不随选文档增加而线性增长。

3. **贪心选择**：按效用分数贪心选择文档，每选一个更新 already_selected_docs_summary，直到达到 token budget 上限。

## 实验结论

- 在 NaturalQuestions、HotpotQA、FEVER 等 RAG 基准上，相比 BM25 重排序，答案 EM 提升约 3-5%，相比 cross-encoder 重排序仅差约 1%，但速度快约 5x。
- 互补性建模的贡献：去掉后 EM 下降约 2%（冗余文档占用 token budget）。
- token budget 约束下（1000 tokens），LURE-RAG 选择的文档集质量明显高于 top-K 截断。

## 工程落地要点

- **token budget 管理**：实际部署需精确计算已选文档的 token 数（不同 LLM tokenizer 结果不同），建议用目标 LLM 的 tokenizer 计算。
- **Summary 更新策略**：每选一个文档后更新 summary 的方式：append embedding（简单但维度增长）或 EMA 更新（稳定维度，推荐）。
- **训练数据构建**：效用标签来自"将文档加入 RAG 后的答案准确率变化"，需要运行 LLM 生成，训练数据成本较高；建议用开源 LLM（Llama）构建。
- **与重排序模型结合**：可以先用 cross-encoder 重排序 top-20，再用 LURE-RAG 在 top-20 中做效用感知的文档集选择。

## 面试考点

**Q: RAG 中文档重排序的目标是什么？和搜索重排序有什么不同？**
- 搜索重排序：最大化单文档的相关性；RAG 重排序：最大化文档集合对 LLM 生成的整体效用（需考虑集合互补性和 token budget）。

**Q: 为什么相关的文档不一定对 RAG 有用？**
- 冗余：多个文档说同一件事，占用 context window；噪声：包含相关词但无关内容的文档干扰 LLM；格式问题：复杂表格/代码段降低 LLM 理解效率。

**Q: RAG 中的 Lost-in-the-Middle 问题是什么？**
- LLM 倾向于更好地利用 context 首尾的文档，中间位置的文档容易被忽略；因此重排序应将最重要的文档放在首尾，而非简单按分数顺序拼接。
