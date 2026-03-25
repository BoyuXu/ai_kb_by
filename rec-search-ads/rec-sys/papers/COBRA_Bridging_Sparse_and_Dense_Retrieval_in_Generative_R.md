# COBRA: Bridging Sparse and Dense Retrieval in Generative Recommendation
> 来源：https://arxiv.org/search/?query=COBRA+sparse+dense+retrieval+generative+recommendation&searchtype=all | 领域：rec-sys | 日期：20260323

## 问题定义
生成式推荐系统在召回阶段面临稀疏（lexical）和稠密（semantic）检索的选择困境。COBRA提出一个统一框架，融合两种检索范式的优势用于生成式推荐。

## 核心方法与创新点
- 双路编码：同时学习稀疏（BM25-like）和稠密（embedding）表示
- 级联生成：先用稀疏检索粗筛，再用稠密特征精排生成
- 对比学习桥接：用对比学习让稀疏和稠密表示空间对齐
- 自适应融合：根据query类型动态调整稀疏/稠密权重

## 实验结论
COBRA在电商推荐benchmark上Recall@50提升约8%，相比纯稠密或纯稀疏检索均有显著改进，尤其在长尾物品召回上优势明显。

## 工程落地要点
- 需要维护两套索引（倒排索引+向量索引），存储成本翻倍
- 融合权重可以通过cross-encoder或简单的learned scalar实现
- 长尾物品通常稀疏检索效果更好，热门物品稠密检索更优

## 面试考点
1. **Q: 稀疏检索和稠密检索各自的优缺点？** A: 稀疏：精确匹配强，可解释，但语义理解弱；稠密：语义理解好，但OOV问题，需要ANN索引
2. **Q: 如何在工业系统中融合稀疏和稠密检索？** A: RRF（Reciprocal Rank Fusion）、learned hybrid scorer、级联召回
3. **Q: 生成式推荐为何需要同时考虑稀疏和稠密？** A: 生成token需要同时覆盖精确匹配（品牌/型号）和语义相似（风格/类目）
4. **Q: 长尾物品召回的挑战和解法？** A: 稠密模型训练数据少，效果差；可用稀疏检索补充，或用内容特征增强embedding
5. **Q: COBRA的对比学习如何实现稀疏-稠密对齐？** A: 用同一物品的稀疏表示和稠密表示作为正样本对，拉近距离
