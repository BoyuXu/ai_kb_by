# Unlocking Multimodal Document Intelligence: Visual Document Retrieval Survey
> 来源：https://arxiv.org/abs/2502.16700 | 日期：20260319

## 问题定义
企业文档（PDF、演示文稿、报告）包含丰富的图文混合信息，传统文本检索忽略了表格、图表、版式等视觉信息。Visual Document Retrieval（VDR）旨在直接对文档图像进行检索，无需OCR+文本提取中间步骤，保留完整的视觉语义。本文系统综述该领域的方法、数据集和挑战。

## 核心方法与创新点
1. **VDR三大范式**：
   - 文本中心：OCR → 文本检索（BiEncoder/ColBERT）
   - 视觉中心：文档图像直接编码（ViT/CLIP）
   - 多模态融合：文本+视觉联合编码（LayoutLM、Donut、ColPali）
2. **ColPali创新**：用PaliGemma视觉语言模型直接对文档页面生成patch级别embedding，Multi-Vector检索，无需OCR
3. **文档版式理解**：位置编码融入视觉特征，捕捉表格行列、标题层级等版式信息
4. **数据集综述**：DocVQA、ViDoRe、ArXivQA等基准的特点和适用场景
5. **挑战**：多页文档理解、跨语言文档、手写内容、低质量扫描件

## 实验结论
- ColPali在ViDoRe基准上显著优于传统OCR+检索流水线，NDCG@5提升约15%
- 视觉模型对包含图表/表格的文档优势最大，纯文字文档与传统方法持平
- 多模态融合方法在各类文档类型上最均衡

## 工程落地要点
- 企业知识库RAG场景优先考虑VDR，避免OCR错误累积
- 文档page级别索引，每页生成embedding，查询时返回最相关页面
- 推理成本高于纯文本，建议异步离线索引，在线查询只做向量检索
- 多页文档需要跨页上下文融合策略

## 面试考点
**Q: ColPali相比传统文档检索的核心优势？**
A: 直接用视觉语言模型（VLM）对文档页面图像生成多向量表示，保留版式、图表、表格等视觉信息，无需OCR中间步骤，对复杂格式文档鲁棒性强。

**Q: 什么是Multi-Vector检索（ColBERT范式）？**
A: 查询和文档都生成token级别的多个向量，相似度计算用MaxSim（查询token与文档token的最大相似度之和），比单向量更精确，比全交叉注意力更高效。

**Q: RAG系统中文档检索质量如何评估？**
A: 检索阶段：Recall@K、NDCG@K、MRR；端到端：EM（精确匹配）、F1（词级别）、RAGAS框架（忠实度、相关性、上下文精确率/召回率）。
