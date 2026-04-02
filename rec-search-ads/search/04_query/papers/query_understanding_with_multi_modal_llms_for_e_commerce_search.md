# Query Understanding with Multi-Modal LLMs for E-Commerce Search
> 来源：arxiv/2402.xxxxx | 领域：search | 学习日期：20260326

## 问题定义
电商搜索中的 Query Understanding（查询理解）挑战：
- 用户查询歧义性高："苹果手机" vs "苹果手机壳"，意图截然不同
- 图文混合查询：用户拍照搜索（"拍照搜同款"），需多模态理解
- 长尾查询：低频专业词汇（型号/品牌/规格）理解困难
- 意图识别：购买意图 vs 咨询意图 vs 比较意图

## 核心方法与创新点
**MM-QU（Multi-Modal Query Understanding with LLMs）**

**多模态查询理解框架：**
```python
# 输入：文本查询 + 可选图片
query_text = "这款手机壳多少钱"
query_image = upload_photo  # 可选

# 视觉编码
if query_image:
    visual_features = CLIP_encoder(query_image)
    query_repr = cross_modal_fusion(text_emb, visual_features)
else:
    query_repr = text_encoder(query_text)

# LLM 推理意图
intent = LLM.classify(
    prompt=f"用户查询：{query_text}\n图片描述：{image_caption}\n识别购买意图、查询类型（导航/信息/交易）、核心词",
    output_schema={"intent": str, "core_keyword": str, "query_type": str}
)
```

**查询扩展（Query Expansion）：**
```
原始查询 → LLM 扩展 → 多路检索融合
"华为手机充电器" → LLM → ["华为超级充电器", "65W充电头", "Type-C充电器", "PD充电器"]
```

**实体识别与规范化：**
```
"iPhne 15 pro 手机壳" → 纠错 → "iPhone 15 Pro 手机壳"
                       → 实体 → [品牌:Apple, 型号:iPhone 15 Pro, 品类:手机壳]
                       → 结构化查询
```

## 实验结论
- 某头部电商搜索系统（2023）：
  - 搜索点击率 +3.7%（多模态查询场景 +8.2%）
  - 搜索无结果率 -24%（查询扩展覆盖更多商品）
  - 查询意图识别准确率：92.3%（vs 规则+分类器 84.1%）
- 多模态图文搜索场景：商品匹配精度 +15.3%

## 工程落地要点
1. **延迟预算**：Query Understanding 必须 <20ms（总延迟 200ms 中的一部分）
2. **LLM 调用策略**：简单查询用规则/轻量模型，复杂/多模态查询才调 LLM
3. **图片预处理**：CLIP 编码离线缓存，相似图片复用
4. **意图分布监控**：实时监控各意图分类的分布，异常时报警
5. **多路检索融合**：扩展后的多个查询并行检索，RRF（Reciprocal Rank Fusion）融合

## 常见考点
**Q1: 电商搜索的 Query Understanding 包含哪些子任务？**
A: ①意图识别（购买/咨询/导航）②实体识别（品牌/型号/品类/属性）③查询扩展（同义词/关联词）④纠错（错别字/OCR错误）⑤多模态理解（图文混合）⑥个性化（基于用户偏好重写查询）。

**Q2: 为什么电商搜索需要多模态理解？**
A: 用户自然行为：拍照搜索（"这个东西叫什么/哪里买"）比文字描述更直接；商品视觉特征（颜色/款式/材质）很难用文字精确描述；多模态 = 更完整的用户意图。

**Q3: 查询扩展的主要挑战？**
A: ①过扩展：扩展词偏离原意，引入不相关结果 ②方向不对："苹果" 扩展为 "苹果汁" 而非 "iPhone" ③用户意图不明确时扩展效果差。解决：基于用户历史行为做个性化扩展，用 LLM 推理扩展意图一致性。

**Q4: LLM 的查询理解与传统 NER/意图分类模型相比优劣？**
A: LLM 优势：泛化强（处理罕见查询）、多任务统一、理解上下文（session）。传统优势：延迟低（<5ms vs 50-500ms）、成本低、可控性强。实践：传统模型处理高频简单查询，LLM 处理复杂/长尾查询（分流策略）。

**Q5: 多模态搜索中图文对齐如何实现？**
A: CLIP 模型：对比学习对齐图片和文本 embedding 空间，使相同语义的图文 embedding 相近。实现：图片 → CLIP_visual_encoder → visual_emb；文本 → CLIP_text_encoder → text_emb；相似度 = cosine(visual_emb, text_emb)。
