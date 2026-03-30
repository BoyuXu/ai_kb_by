# ELEC: Efficient Large Language Model-Empowered Click-Through Rate Prediction
> 来源：arXiv:2509.07594 | 领域：ads | 学习日期：20260330

## 问题定义
将 LLM 引入 CTR 预估面临的核心矛盾：LLM 语义理解能力强但推理慢（>100ms），传统 CTR 模型快（<5ms）但语义理解弱。ELEC 提出高效融合方案：用 LLM 离线生成增强特征（特征工厂），传统 CTR 模型在线消费这些特征，兼顾语义质量和推理效率。

## 核心方法与创新点
1. **LLM 特征工厂（Offline LLM）**：LLM 离线对用户/物品/广告做语义分析，生成：(a) 语义标签（用户兴趣标签、物品品类语义）；(b) 跨模态描述（图像→文本）；(c) 用户画像 summary。
2. **特征压缩**：LLM 生成的高维语义 embedding（4096 dim）通过 distillation 压缩到 64 dim，与 ID embedding 拼接。
3. **在线轻量模型**：在线 CTR 模型只做 embedding lookup + MLP forward，LLM 特征作为预计算特征，不增加在线延迟。
4. **增量更新**：新物品/用户特征定期批量更新（每小时），旧特征缓存复用。
5. **多粒度 LLM 增强**：物品粒度（item-level）+ 用户粒度（user-level）+ 交互粒度（user×item），各级别特征分层注入。

## 实验结论
- 某信息流广告：CTR AUC +0.92%，相比 DIN baseline
- 冷启动广告（<100 曝光）：AUC +2.8%（LLM 语义特征弥补 ID 特征稀疏）
- 在线延迟仅增加 0.3ms（特征预计算，在线零 LLM 开销）

## 工程落地要点
- LLM 批量特征生成需要高吞吐推理（vLLM + 连续批处理），成本控制关键
- 特征缓存（Feature Store）需要物品 ID → LLM 特征的快速查找（Redis/Aerospike）
- LLM 特征过期策略：时效性强的物品（新品）频繁更新，长尾物品可缓存更久
- 压缩 embedding 的信息损失需监控（定期对比 LLM full embedding vs 压缩版 AUC 差）

## 面试考点
- Q: LLM 在线 vs 离线做 CTR 特征增强，如何选择？
  - A: 在线：特征最新，延迟高（不适合实时系统）；离线：特征可能过时，延迟友好，工程复杂度低。实际大多用离线+定期更新
- Q: 特征压缩（4096→64）的方法有哪些？
  - A: PCA（线性）、AutoEncoder（非线性）、Knowledge Distillation（保持语义对齐）、Product Quantization（向量量化）
- Q: CTR 模型中 LLM 特征和 ID 特征如何融合？
  - A: ① Concatenation + MLP；② Cross-attention（LLM 特征作为 key/value，ID 特征 query）；③ Feature Gating（用 ID 特征动态加权 LLM 特征）
