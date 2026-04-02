# ReasonIR: Training Dense Retrievers for Reasoning-Intensive Tasks
> 来源：arXiv:2504.20595 | 领域：search | 学习日期：20260330

## 问题定义
标准 dense retrieval（如 DPR）在简单事实检索上表现良好，但在需要多步推理的复杂任务（数学推理、逻辑问题、多跳问答）上表现差。原因是这类任务的 query 和 document 相关性不是字面匹配，而是"推理路径相关"。ReasonIR 专门训练面向推理密集型任务的检索器。

## 核心方法与创新点
1. **Reasoning-Aware Hard Negative Mining**：用 LLM 生成"表面相关但推理不相关"的难负样本（superficially similar but logically unrelated），训练检索器区分推理相关性。
2. **Synthetic Reasoning Data Generation**：用 GPT-4 为每个检索任务生成 (query, reasoning_path, supporting_document) 三元组，reasoning_path 连接 query 和 document 的逻辑链。
3. **Progressive Training**：
   - Stage 1：一般检索训练（MS-MARCO、Natural Questions）
   - Stage 2：推理检索精调（合成推理数据 + 真实推理数据集）
4. **Bi-encoder + Cross-encoder 联合**：Bi-encoder 做初步检索（高效），Cross-encoder 重排时输入 reasoning path 做精细对齐。
5. **评估基准扩展**：构建 ReasonIR-Bench，覆盖数学推理（MATH）、科学推理（SciQ）、逻辑推理（LogiQA）的检索场景。

## 实验结论
- BEIR 通用检索：NDCG@10 +2.3%（对比 E5-Mistral baseline）
- ReasonIR-Bench 推理检索：NDCG@10 +11.6%（大幅领先，推理场景专属优势）
- 端到端 RAG 准确率：在 MATH 数学推理任务 +8.4%（检索质量提升带动最终准确率）

## 工程落地要点
- Reasoning-aware 训练数据生成成本高（需 GPT-4），建议针对业务场景的推理类型定向生成
- 推理检索中 query 往往更复杂（长问题），需要更长的 max_seq_len（512 → 1024）
- Cross-encoder 重排输入 reasoning path 会大幅增加 token 数，需做 path 压缩
- 适合知识库问答（FAQ 检索）、法律/医疗专业问答（需要逻辑推理的检索场景）

## 常见考点
- Q: 什么是 Hard Negative Mining，为什么重要？
  - A: 随机负样本太容易（检索器轻松区分）→ 梯度小，学习效率低。Hard negative 与正样本相似但不相关（容易混淆），训练器被迫学习细粒度区分，AUC 显著提升
- Q: 推理密集型检索和普通检索的核心差异？
  - A: 普通：query 和 document 词汇/语义重叠高（TF-IDF/dense 都有效）；推理密集型：相关性基于逻辑推导，需要模型理解"因果/数学/逻辑"关系
- Q: Bi-encoder 和 Cross-encoder 的速度/精度权衡？
  - A: Bi-encoder：query 和 document 独立编码，ANN 检索，O(1) 推理，适合召回；Cross-encoder：query×document 联合编码，O(n) 重排，精度高但慢，适合精排
