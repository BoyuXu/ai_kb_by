# Unlocking Multimodal Document Intelligence: Visual Document Retrieval Survey
> 来源：https://arxiv.org/abs/2502.16700 | 日期：20260319

## 问题定义
传统文档检索只处理纯文本，但现实中大量文档包含图表、表格、图像等视觉元素（PDF报告、学术论文、商业文档）。视觉文档检索（Visual Document Retrieval）旨在理解文档的视觉布局和内容，回答需要结合文字和图像理解的查询。

## 核心方法与创新点
1. **文档表示方法**：
   - **OCR-based**：提取文字→文本嵌入，丢失视觉布局信息
   - **Layout-aware**：LayoutLM等，将文字位置信息编码进嵌入
   - **Vision-only**：直接将文档页面作为图像，用视觉模型编码（如ColPali）
   - **Multimodal**：VLM（视觉语言模型）同时理解文字和视觉内容

2. **ColPali框架**（核心创新）：
   - 将文档页面划分为patch（图像块）
   - 每个patch独立编码，支持细粒度匹配
   - Late interaction（类似ColBERT）：查询token与文档patch的MaxSim匹配
   - 完全基于视觉，无需OCR预处理

3. **评估基准**：
   - DocVQA：文档视觉问答
   - ViDoRe：视觉文档检索基准
   - 覆盖财务报告、学术论文、医疗文档等多场景

## 实验结论
- ColPali相比纯OCR方法在包含复杂图表的文档上NDCG@5提升约20-30%
- 视觉模型在理解表格结构上比纯文本强约15%
- 但在纯文本文档上，OCR+文本嵌入仍有竞争力

## 工程落地要点
1. **索引成本**：视觉嵌入维度高（>1000），存储成本比文本嵌入高5-10倍
2. **推理延迟**：VLM推理比文本模型慢5-10倍，需GPU加速
3. **混合策略**：对纯文本文档用文本检索，对含图表文档用视觉检索
4. **PDF处理**：需要PDF渲染工具（如pdf2image）将页面转为图像

## 面试考点
Q1: 为什么PDF文档检索需要视觉理解而不能只用OCR？
> OCR只提取文字，丢失了(1)视觉布局（哪些文字是标题/图注/表格）；(2)图表中的信息（折线图趋势、饼图比例）；(3)文字-图像关系（某图表说明的是哪段文字）。视觉理解可以同时捕获文字内容和视觉结构，更好地理解文档语义。

Q2: ColBERT的Late Interaction机制是什么？
> 传统双塔：query和doc各编码为单个向量，点积计算相似度；ColBERT：query编码为token向量序列，doc也编码为token向量序列，相似度 = Σ(query每个token与所有doc tokens的最大相似度)，即MaxSim操作。Late Interaction比单向量更细粒度，比交互式模型更快。

Q3: RAG系统中如何处理包含图表的文档？
> 多模态RAG方案：(1) 图表转文本：用VLM将图表转述为文字描述，然后文本检索；(2) 多模态嵌入：用CLIP等多模态模型同时嵌入文本和图像；(3) 混合索引：文本和图像分别建索引，检索时融合结果；(4) ColPali：直接基于视觉的文档检索，不需要OCR。
