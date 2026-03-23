# E5-mistral-7b-instruct: Improving Text Embeddings with Large Language Models
> 来源：https://arxiv.org/abs/2401.00368 | 领域：search | 日期：20260323

## 问题定义
微软提出用大型语言模型（Mistral-7B）作为文本嵌入模型骨干，通过指令微调学习通用文本嵌入，在各种文本检索和相似度任务上超越专门的bi-encoder模型。

## 核心方法与创新点
- LLM作为Encoder：用7B参数的Mistral作为embedding模型，提取最后token的表示
- 指令微调：用统一指令格式训练多种嵌入任务（检索、分类、聚类、相似度）
- 合成数据：GPT-4生成多样化的训练数据对，覆盖各种任务类型
- Last token pooling：用EOS token的表示作为句子embedding

## 实验结论
E5-Mistral-7B在MTEB（大规模文本嵌入benchmark）排行榜上取得SOTA，56个任务平均分超越之前最优约2%；检索任务平均NDCG@10约57%；但推理速度约为小型模型的10x慢。

## 工程落地要点
- 7B参数模型推理需要GPU，embedding速度约100-500 token/s
- 生产环境通常蒸馏到较小模型（<1B）保持质量同时提升效率
- 指令格式：使用时需要为查询添加任务指令前缀（"Query: ", "Represent this for retrieval: "）

## 面试考点
1. **Q: 为什么LLM比BERT更适合做embedding模型？** A: LLM预训练数据更多更多样、理解复杂指令能力强、跨任务泛化好
2. **Q: Last token pooling为什么优于CLS token？** A: 因果注意力（causal attention）中，最后token能attend到所有前文；CLS只在MLM模型有效
3. **Q: MTEB是什么？包含哪些任务类型？** A: Massive Text Embedding Benchmark，包含检索、重排、分类、聚类、STS等56个任务
4. **Q: 如何将7B embedding模型蒸馏到小模型？** A: 对比蒸馏（对齐embedding空间）、逐层蒸馏、任务特定微调
5. **Q: 指令微调的embedding模型在搜索中如何使用？** A: 查询加指令前缀，文档不加（或加不同前缀），保证两侧语义空间对齐
