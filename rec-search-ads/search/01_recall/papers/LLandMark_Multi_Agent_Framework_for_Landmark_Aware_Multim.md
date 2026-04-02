# LLandMark: Multi-Agent Framework for Landmark-Aware Multimodal Interactive Video Retrieval

> 来源：arxiv | 日期：20260316 | 领域：search

## 问题定义

视频检索（Video Retrieval）面临的挑战：
1. **语义鸿沟**：用户文字查询（"sunset over mountains"）与视频帧的视觉内容之间存在巨大语义差距。
2. **时序信息**：视频不同于图片，关键信息分散在不同时间段（"landmark moment"），需要精准定位。
3. **交互式检索**：用户往往需要多轮对话式交互来精炼查询，单轮检索难以满足。
4. **多模态融合**：文字、图像、音频、字幕等多模态信息需要联合利用。

LLandMark 提出多智能体框架，每个 agent 负责不同模态或检索子任务。

## 核心方法与创新点

1. **Landmark Detection Agent（关键帧识别）**：
   - 用 Vision-Language Model（如 InternVL/LLaVA）对视频帧序列打分，识别"语义关键帧"（landmark frames）。
   - 关键帧标准：信息密度高、与查询主题相关、时间上均匀分布。

2. **Multimodal Query Agent（多模态查询扩展）**：
   - 接收用户查询，用 LLM 扩展为多个视角的子查询（视觉描述、动作描述、场景描述）。
   - 用 CLIP/BLIP 将文字子查询编码为多模态向量。

3. **Retrieval Agent（检索与排序）**：
   - 对关键帧 embedding 建立 FAISS 索引。
   - 用多子查询的 embedding 检索，融合多个得分（max/avg pooling）。

4. **Interaction Agent（交互式精炼）**：
   - 分析用户对初始检索结果的反馈（正面/负面示例）。
   - 用 Relevance Feedback（RF）更新查询向量（Rocchio 算法变体）。
   - 支持多轮交互，逐步逼近用户真实意图。

## 实验结论

- ActivityNet-Captions 数据集：Recall@1 +8.3%，相比 CLIP4Clip 等基线。
- 多轮交互后（3 轮）：Recall@1 较单轮 +15%，验证交互式检索价值。
- Landmark 帧选择：选 10% 关键帧 vs 全帧检索，速度快 5x，精度仅损失 1.2%。

## 工程落地要点

- 关键帧提取可离线完成，建立索引后实时检索无需重复处理视频。
- FAISS 索引类型：视频库 <100M 帧用 IVF-HNSW，更大规模用 IVF-PQ（有损压缩但内存可控）。
- 多智能体协作需要消息传递协议，建议用 LangGraph 或 AutoGen 框架实现。
- 用户反馈收集：隐式反馈（点击/停留）>显式反馈（标注好/坏），工程上更易收集。

## 常见考点

- Q: CLIP 是什么？在视频检索中怎么用？
  A: CLIP（Contrastive Language-Image Pre-Training）用对比学习训练文字和图片在同一空间的 embedding。视频检索中：将每帧编码为图片 embedding，用户 query 编码为文字 embedding，计算余弦相似度进行检索。可扩展：视频 embedding = 多帧 embedding 的时序聚合。

- Q: Rocchio 算法是什么？
  A: 信息检索中的经典相关性反馈算法：新查询向量 = α×原始查询 + β×Σ相关文档向量 - γ×Σ不相关文档向量。用相关结果"拉近"查询，用不相关结果"推远"，迭代精炼查询。

- Q: 多智能体框架在检索系统中的优势？
  A: (1) 模块化：各 Agent 专注子任务，易于独立优化和替换；(2) 并行执行：多个 Agent 可并行工作（如同时处理不同模态）；(3) 迭代精炼：Agent 可以多轮协作改进结果；(4) 可解释性：每个 Agent 的输出可追溯，便于调试。
