# LLM-Embedder: Leveraging LLMs for Text Embeddings in Retrieval

> 来源：arXiv 2024 | 领域：search | 学习日期：20260404

## 问题定义

传统 BERT-style 双塔模型用于文本检索（Dense Retrieval）存在两个局限：
1. **表达能力有限**：BERT 语义理解不如 GPT-4 等大模型
2. **任务通用性差**：同一 Embedding 难以同时胜任 QA/知识检索/对话检索等多场景

能否利用 LLM（解码器架构）的语义理解能力直接生成文本 Embedding？

## 核心方法与创新点

**LLM-Embedder** 将 LLM 转化为高质量 Embedding 模型：

1. **双向注意力适配（Bidirectional Attention）**：
   - Decoder-only LLM 原本只有因果 Attention（单向）
   - 修改 Attention Mask，允许 Embedding 生成使用全局上下文
   - 最终取最后一层 [EOS] token 的隐状态作为文本表示

$$e_{\text{text}} = h_L(\text{[EOS]}) \text{ with bidirectional attention}$$

2. **多任务对比学习**：
   - 统一框架同时训练多个检索任务（QA、代码检索、文档检索）
   - 任务特定 prompt：`"Represent the {task_type} for searching: {text}"`

3. **负样本采样策略**：
   - Hard Negative Mining：每条正例配 K 个困难负例
   - In-Batch Negatives + Cross-Batch（队列机制）
   
$$\mathcal{L}_{\text{contrastive}} = -\log \frac{e^{s(q,d^+)/\tau}}{\sum_{j} e^{s(q,d_j^-)/\tau}}$$

4. **蒸馏自 Cross-Encoder**：
   - Cross-Encoder 作为 Teacher（精排质量）
   - Bi-Encoder（LLM-Embedder）作为 Student
   - 软标签蒸馏提升召回质量

## 实验结论

- BEIR Benchmark NDCG@10: **66.2** vs BGE-Large-EN 63.5
- MTEB 多任务均分: **62.8**（当时 SOTA）
- 多任务泛化: 同一模型在 14 个数据集上均表现优秀
- 代码检索: 比 CodeBERT **+8.4%** NDCG@10

## 工程落地要点

- 双向 Attention 改造：修改 `attention_mask` 参数（2行代码）
- 推理时 batch 处理（GPU 利用率高于单条推理）
- Embedding 维度：取最后隐层维度（4096 for 7B），可用 PCA 压缩至 1024
- 多任务 prompt：不同检索场景需配置对应 task_type

## 面试考点

1. **Q**: 为什么 Decoder-only LLM 做 Embedding 需要修改 Attention？  
   **A**: 因果 Attention 让每个 token 只看前文，[EOS] 的隐状态仅聚合单向信息。双向 Attention 让 [EOS] 汇聚全句双向语义，质量更高。

2. **Q**: Bi-Encoder 和 Cross-Encoder 的 tradeoff？  
   **A**: Bi-Encoder：离线 Embedding + 在线向量检索（快，O(1)，但质量稍低）；Cross-Encoder：在线 concat 输入精排（慢，O(n)，但精度高）。蒸馏是两者折中。

3. **Q**: 多任务训练如何防止任务间干扰？  
   **A**: Task-specific prompt 区分任务（模型知道当前做什么任务）+ 梯度累积（交替任务 mini-batch）+ 任务权重平衡。
