# 多模态搜索：图文音视频的统一检索

> 来源：技术综述 | 日期：20260316 | 领域：search

## 问题定义

用户的信息需求越来越多模态：
- 用图片搜索相似商品（以图搜图）
- 用文字描述搜索视频片段（text-to-video）
- 用语音查询搜索文档（speech-to-text-to-search）
- 跨模态对齐（"show me images of the text I'm reading about"）

传统单模态搜索系统无法处理这些需求，多模态搜索需要在**统一向量空间**中对齐不同模态。

## 核心方法与创新点

### 1. CLIP 系统（Contrastive Language-Image Pre-Training）

**训练目标**：将 4 亿图文对的图片 embedding 和文字 embedding 对齐到同一空间。

```python
# 对比学习目标
logits = image_features @ text_features.T  # [N, N] 矩阵
labels = torch.arange(N)  # 对角线为正例
loss = (CE(logits, labels) + CE(logits.T, labels)) / 2
```

**应用**：
- Text → Image 检索：用文字 embedding 搜索图片库
- Image → Text 检索：用图片 embedding 搜索文本描述
- Zero-shot 图片分类：将类别名转为文字 embedding，最近邻分类

### 2. 统一多模态 Embedding 模型

**ImageBind（Meta）**：将 6 种模态（图、文、音频、深度、热成像、IMU）映射到同一空间：
- 以图片-文字对为锚点（CLIP），其他模态通过图片桥接对齐。
- 一个 embedding 空间支持任意跨模态检索。

**E5-V / LLaVA-Emb**：用多模态 LLM 生成统一 embedding：
- 输入：任意模态（图/文/图+文）
- 输出：统一 embedding，任意模态间可计算相似度。

### 3. 视频检索的特殊处理

**时序池化策略**：
```python
# 方案1: 均值池化（适合均匀内容）
video_emb = mean([CLIP(frame) for frame in sample_frames])

# 方案2: 最大池化（适合关键帧检索）
video_emb = max([CLIP(frame) for frame in frames], dim=0)

# 方案3: 时序 Transformer（保留时序信息）
frame_embs = [CLIP(frame) for frame in frames]
video_emb = TemporalTransformer(frame_embs)
```

**片段级检索（Video Grounding）**：不只返回视频，返回具体时间戳 [start, end]。

### 4. 商品多模态搜索（电商场景）

- 图文联合 embedding：商品主图 + 标题 + 属性融合为统一向量。
- 用户 query：文字/图片/图片+文字描述（如"这件衣服，要蓝色的"）。
- 多模态融合：Late Fusion（分别检索再合并）vs Early Fusion（联合 embedding）。

## 实验结论

- CLIP 在 MS-COCO I2T/T2I 检索：Recall@1 分别 58.4%/37.8%，远超传统方法。
- ImageBind 跨模态检索（音频→图片）：Top-1 accuracy 50.3%（zero-shot）。
- 视频检索（MSRVTT T2V）：CLIP4Clip 相比早期方法 +15% R@1。

## 工程落地要点

- 电商图搜：图片 embedding 用 ViT-L/14 @ 336px CLIP，维度 768；建 FAISS HNSW 索引。
- 批量编码图片：用 TorchScript + GPU 批处理，单 A100 可达 1000 QPS（图片→embedding）。
- 多模态融合权重：图片权重通常高于文字（视觉信息量更大），经验 0.6:0.4。
- 部署注意：CLIP 图片 encoder 约 307M 参数（ViT-L），推理需 GPU；文字 encoder 较小可以 CPU 推理。

## 面试考点

- Q: CLIP 为什么能做零样本分类？
  A: CLIP 将图片和文字映射到同一空间，类别名（如"dog", "cat"）可以作为文字 embedding。对于新图片，计算其与所有类别文字 embedding 的相似度，取最高的类别即为预测结果。无需针对该分类任务训练，故称"零样本"。

- Q: 多模态搜索中的语义对齐挑战有哪些？
  A: (1) 模态差距（Modality Gap）：不同模态的原始特征分布差异大，对齐需要大量数据；(2) 粒度不匹配：文字描述可能很抽象，图片很具体；(3) 多义性：同一文字描述对应多种视觉内容；(4) 数据集偏差：训练数据的图文对质量参差不齐（噪声标签）。

- Q: Late Fusion 和 Early Fusion 各自的优缺点？
  A: Late Fusion：各模态独立检索，结果融合。优点：灵活，可用不同索引；缺点：丢失模态间交互信息。Early Fusion：先将多模态输入合并，生成统一 embedding 再检索。优点：捕捉模态交互（"红色的+这张图片的款式"）；缺点：架构复杂，需要重新构建索引当任何模态更新时。
