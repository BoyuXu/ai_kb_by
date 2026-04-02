# Unlocking Multimodal Document Intelligence: Visual Document Retrieval Survey

> 来源：arXiv | 日期：20260317

## 问题定义

现实世界的文档（PDF、PPT、网页、扫描件）包含**文字、表格、图表、图像**等多模态内容，纯文本检索系统无法理解视觉元素（如柱状图的趋势、表格的结构关系、图片的语义）。

**Visual Document Retrieval（VDR）**：直接对文档页面图像进行向量表征和检索，无需 OCR 或复杂文本提取，端到端处理多模态文档。

## 核心方法与创新点（综述梳理）

### 方法分类

1. **Text-Centric 方法（传统）**
   - OCR → 文本提取 → 文本检索
   - 缺陷：OCR 错误传播、视觉元素丢失（图表、表格结构）

2. **Embedding-based VDR**
   - **ColPali**（2024 Google DeepMind）：基于 PaliGemma 视觉语言模型，对文档页面图像直接生成 patch-level 向量，用 Late-Interaction（MaxSim）检索
   - **DSE**（Document Screenshot Embedding）：文档截图 → CLIP/ViT 编码 → 单向量
   - **PDFEmbed**：PDF 渲染为图像后 Vision Transformer 编码

3. **Cross-modal Alignment**
   - 训练时构建 (Query, Document Image) 对，对比学习对齐文本查询和文档图像向量空间
   - 负样本：同一文档集中的其他页面、其他文档

4. **Multi-granularity Representation**
   - Page-level：整页图像编码为单/多向量
   - Patch-level：图像分 patch（16×16 或 32×32）各自编码（ColPali 方式），细粒度 MaxSim

### 评测基准

- **DocVQA**、**InfoVQA**：视觉问答
- **DUDE**：文档理解评测
- **ViDoRe**（Visual Document Retrieval Benchmark）：专门的视觉文档检索评测

## 实验结论（综述观察）

- ColPali 在 ViDoRe 基准上 nDCG@5 = 0.811，大幅超越传统 OCR + BM25（0.512）
- Patch-level Late-Interaction 相比 Page-level 单向量提升约 8~12%（在含有复杂表格/图表的文档上）
- 多模态文档检索在工业文档（财报、说明书）场景相对提升最显著

## 工程落地要点

1. **渲染成本**：PDF/PPT 渲染为图像需要额外计算，建议离线预处理
2. **存储**：Patch-level 向量存储量约为单向量的 256x（32×32 patch），需要向量压缩（PQ）
3. **GPU 推理**：Vision Transformer 编码比文本 BERT 慢约 5~10x，批量预处理更重要
4. **OCR 互补**：视觉检索 + 文本检索 RRF 融合通常优于单一方法

## 常见考点

- **Q: 视觉文档检索 vs OCR + 文本检索的优势？**
  A: 视觉检索保留文档的布局、表格结构、图表视觉信息；OCR 会损失视觉元素且存在错误传播。对含大量图表、复杂表格的文档（财报、技术手册），视觉检索显著更优；对纯文字文档，OCR 方法仍有优势（计算成本低）。

- **Q: ColPali 的核心创新是什么？**
  A: ColPali 将 ColBERT 的 Late-Interaction 思想扩展到视觉领域：文档图像分割为 patch，每个 patch 生成一个向量，查询文本也生成 token 向量；相关性 = MaxSim(Query tokens, Document patches)，可以精确定位文档中与查询最相关的区域。

- **Q: 多模态文档检索的主要挑战？**
  A: 1) 跨模态对齐（文本查询 vs 图像文档）；2) 计算效率（Vision Transformer 推理慢）；3) 存储（patch-level 向量庞大）；4) 长文档（多页 PDF 需要页面级检索 + 页面内定位）。
