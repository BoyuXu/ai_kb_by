# Graph Retrieval-Augmented LLM for Conversational Recommendation Systems (G-CRS)
> 来源：arXiv:2503.06430 | 领域：rec-sys | 学习日期：20260419

## 问题定义
对话式推荐系统（CRS）需要在多轮对话中理解用户需求并推荐物品。现有方法要么需要大量任务特定训练数据，要么LLM缺乏推荐领域知识。

## 核心方法与创新点
1. **Training-Free框架**：无需任务特定训练，利用ICL能力
2. **两阶段Retrieve-and-Recommend**：
   - Stage 1: GNN-based图推理器识别候选物品
   - Stage 2: Personalized PageRank探索发现潜在物品和相似用户交互
3. **结构化Prompt**：将检索到的上下文转化为结构化prompt，供LLM推理
4. **Knowledge Graph增强**：利用物品-用户-属性知识图谱提供推荐依据

## 面试考点
- Q: LLM在推荐系统中的主要挑战？
  - A: ①缺乏用户行为数据访问；②实时性不足；③幻觉问题（推荐不存在的物品）；④冷启动场景知识不足
