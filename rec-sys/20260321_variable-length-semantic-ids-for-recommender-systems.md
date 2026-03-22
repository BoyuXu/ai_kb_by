# Variable-Length Semantic IDs for Recommender Systems

> 来源：arxiv | 日期：20260321 | 领域：rec-sys

## 问题定义

现有 Semantic ID 方案（如 RQ-VAE）为所有物品分配固定长度的 token 序列（如每个物品 4 个 token）。这忽视了物品间信息复杂度的差异：

- 一个简单商品（如"普通白T恤"）和一个复杂商品（如"带3D打印装饰的限量版联名运动鞋"）需要不同的表达粒度
- 固定长度导致简单物品 token 浪费（padding），复杂物品表达不足（截断）
- 序列长度固定使得生成式推荐的 beam search 在不同层级终止条件相同，无法适应物品复杂度

本文提出**变长 Semantic ID**，让不同物品根据语义复杂度自适应分配不同数量的 token。

## 核心方法与创新点

1. **自适应量化深度（Adaptive Quantization Depth）**：
   - 基于 RQ-VAE 框架，在每个量化层后引入**停止标准（Stop Criterion）**
   - 若当前层的重建误差已低于阈值 ε，则停止量化，分配 `[EOS]` token
   - 物品 embedding 信息量高 → 需要更多量化层 → 更长 Semantic ID
   
2. **变长感知的生成式检索**：
   - Beam Search 时不再按固定步数停止，而是遇到 `[EOS]` token 时终止
   - 前缀树（Trie）结构支持变长路径，剪枝仍有效
   - 模型学习在合适的层生成 `[EOS]`，隐含地学习物品复杂度

3. **熵控制目标**：
   - 优化目标引入信息论约束：期望 ID 长度的熵最大化（覆盖更多物品）和重建质量的 ELBO
   ```
   Loss = Reconstruction_Loss + λ × E[ID_Length_Entropy]
   ```

4. **统计发现**：
   - 长尾冷门物品通常 ID 较短（信息少），热门复杂物品 ID 较长
   - 平均 ID 长度从固定 4 降至 2.8，序列总长度减少 ~30%

## 实验结论

- 相比固定长度 4 的 Semantic ID：Recall@10 **+4.3%**，NDCG@10 **+3.9%**
- 序列总 token 数减少 **~30%**，直接降低生成式检索的推理计算量
- 长尾物品（交互 <10 次）的召回率提升 **+8.1%**（因为短 ID 更容易被模型"生成"出来）
- 热门物品（交互 >1000 次）的精度提升 **+2.2%**（更长 ID 表达更丰富细粒度特征）

## 工程落地要点

1. **Trie 索引更新**：物品的 Semantic ID 长度可能随时更新（embedding 漂移），需要版本化管理 Trie，避免索引不一致导致线上空结果
2. **批处理填充（Padding）**：变长序列批处理时需要 padding 到 batch 内最长 ID，极端不均匀时效率低下；可按长度分桶（Bucketing）组 batch
3. **EOS token 阈值**：停止阈值 ε 是关键超参，过小 → ID 过长，过大 → 物品区分度下降；建议在验证集上用网格搜索确定
4. **监控 ID 长度分布**：线上需要监控 ID 长度分布，若发现所有物品 ID 趋同（退化为相同长度），说明 Stop Criterion 失效

## 面试考点

- Q: RQ-VAE 是什么？在 Semantic ID 生成中如何使用？
  A: Residual Quantization VAE，在标准 VQ-VAE 基础上增加残差量化：第一层量化后，对量化误差（残差）再做一次 VQ，如此叠加多层。每层量化对应 Semantic ID 的一个 token，多层组合形成层次化的物品编码，浅层 token 代表粗粒度语义，深层 token 代表细粒度特征。

- Q: 变长 ID 相比固定长度 ID 有什么工程上的额外挑战？
  A: (1) Trie 索引需要支持变长路径终止；(2) 批处理需要 padding/masking，效率降低；(3) 生成时停止条件从"固定步数"变为"遇 EOS"，需要修改 beam search 逻辑；(4) 版本管理更复杂（长度变了等于 ID 变了）。

- Q: 为什么长尾物品适合用更短的 Semantic ID？
  A: 长尾物品交互少，其 embedding 本身就不太准确，信息量有限。更短的 ID 减少了量化误差的累积，同时在 beam search 时更容易被采样到（路径短，分支少），有效提升长尾召回率。
