# SPLADE-v3: Advancing Sparse Retrieval with Deep Language Models

> 来源：arxiv | 日期：20260322 | 领域：搜索系统

## 问题定义

BM25 等传统稀疏检索方法依赖字面词汇匹配，无法处理同义词（"跑鞋" vs "运动鞋"）和语义理解。SPLADE 系列通过深度语言模型生成语义感知的稀疏向量，结合稀疏检索的高效性和密集检索的语义能力。

## 核心方法与创新点

- **SPLADE 核心原理**：
  - 使用 BERT MLM head，对 query 和 document 生成词汇表维度的稀疏权重向量
  - 稀疏性通过 FLOPS 正则化（log(1+ReLU(w))）实现，强制大部分词权重为 0
  - 检索时做稀疏向量内积（等同于倒排索引查找），速度接近 BM25
- **v3 改进**：
  - **更强基座模型**：从 BERT-base 升级到 DeBERTa-v3 / E5-mistral（更好的上下文理解）
  - **Distillation 增强**：用 Dense Retrieval 模型（如 E5-large）作为教师，蒸馏知识到稀疏模型
  - **文档扩展**：使用 doc2query（T5 生成潜在 query）扩充文档词汇表，提升召回
  - **量化压缩**：权重量化为 int8，索引大小减少 4×，检索速度提升 2×
- **FLOPS 控制**：通过调整正则化系数 λ，控制平均非零权重数（3-10 个词），平衡效果和效率

## 实验结论

- BEIR 基准（多领域检索）：SPLADE-v3 平均 NDCG@10 = 0.546，超过 BM25 (0.427) 和 DPR (0.468)
- MS MARCO Dev：MRR@10 = 0.398（接近 top Dense 模型，但延迟低 3-5×）
- 文档扩展对长尾词汇查询贡献最大：+7.2% NDCG（vs 无扩展）
- 蒸馏后：vs BM25 的语义查询 NDCG 提升 28%，精确匹配查询持平

## 工程落地要点

- **索引构建**：可直接复用 Elasticsearch/OpenSearch 等倒排索引基础设施，无需部署向量数据库
- **权重存储**：使用 int8 量化权重 + 稀疏压缩（仅存储非零项），内存是 Dense HNSW 的 20%
- **批量 Encoding**：文档 Encoding 可完全离线，query Encoding 需在线（<10ms，BERT inference）
- **混合系统集成**：SPLADE 可作为 BM25 的直接替换，无需改变检索基础架构

## 面试考点

1. **Q：SPLADE 稀疏向量和 BM25 TF-IDF 向量有什么本质区别？**
   A：BM25 的稀疏向量只包含文档中实际出现的词（词汇匹配）；SPLADE 向量通过 MLM 扩展到语义相关的词（如文档含"跑步鞋"，SPLADE 向量中"运动鞋"也有非零权重）

2. **Q：FLOPS 正则化是什么？为什么有效？**
   A：FLOPS 正则化 = sum(log(1+ReLU(w_i)))²，对每个词的激活值施加对数惩罚，使模型学会只在最重要的词上分配权重；有效是因为梯度在大权重处小（鼓励稀疏）

3. **Q：SPLADE 在电商搜索中相比 BM25 的核心优势场景？**
   A：同义词匹配（用户用不同词描述同一商品）；属性推断（搜索"防水手表"，文档含"IP68"也能匹配）；跨语言场景（借助多语言 BERT）
