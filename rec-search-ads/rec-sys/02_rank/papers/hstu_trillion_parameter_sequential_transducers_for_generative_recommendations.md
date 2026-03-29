# HSTU: Trillion-Parameter Sequential Transducers for Generative Recommendations
> 来源：arxiv/2402.17152 | 领域：rec-sys | 学习日期：20260326

## 问题定义
工业级推荐系统面临超大规模（万亿级别参数）序列建模的挑战：
- 用户行为序列极长（数千到数万条历史）
- 普通 Transformer 注意力复杂度 O(n²) 无法处理
- 训练和推理延迟都需严格控制
- Meta 广告推荐场景需统一检索与排序

## 核心方法与创新点
**HSTU (Hierarchical Sequential Transducer Unit)**：

架构核心：将序列推荐视为生成式序列到序列问题，借鉴 RNN Transducer 思路。

**分层时序建模：**
```
H_t = HSTU(x_1, x_2, ..., x_t)
y_t = Softmax(H_t · E^T)   # E为item embedding矩阵
```

**关键创新：**
1. **线性注意力近似**：将 O(n²) 降为 O(n)，支持百万级序列
2. **分层结构**：局部窗口 attention + 全局稀疏 attention
3. **因果掩码**：保证自回归生成性质
4. **多任务头**：同时预测点击、购买、停留时长

**万亿参数来源：**
- 用户/物品 embedding table（数十亿实体 × embedding_dim）
- 参数量 ∝ |users| × d + |items| × d ≈ 万亿级

## 实验结论
- Meta 广告平台线上 A/B：CTR +1.5%，CVR +2.1%
- 离线 NDCG@10：相比 DIN/DIEN 提升约 8%
- 延迟：比同参数 Transformer 快 3.2x（线性注意力优化）
- 规模扩展：1T 参数模型 vs 10B 模型有显著提升（scaling law 在推荐领域成立）

## 工程落地要点
1. **分布式 Embedding**：使用参数服务器或 ZeRO 切分万亿 embedding table
2. **序列截断策略**：动态截断，高活跃用户保留更长序列
3. **流式更新**：实时消费用户行为日志更新序列缓存
4. **两阶段推理**：检索阶段用轻量 HSTU，排序阶段用完整模型
5. **量化部署**：INT8 量化 embedding，FP16 计算层

## 面试考点
**Q1: HSTU 如何解决长序列 O(n²) 问题？**
A: 采用线性注意力近似（Linear Attention），通过核函数分解将 QK^T·V 变为 K^T·V 先行计算，复杂度降为 O(n·d²)，支持万级序列建模。

**Q2: 万亿参数推荐模型如何训练？**
A: 模型并行（Embedding 切分到多机）+ 数据并行（批次切分）+ 流水线并行（层切分），配合混合精度和梯度压缩。

**Q3: HSTU 与 BERT4Rec/SASRec 的核心区别？**
A: HSTU 是自回归生成式（因果掩码），而 BERT4Rec 用双向 mask；HSTU 有分层结构处理超长序列；HSTU 规模远大于传统序列模型。

**Q4: 推荐系统 Scaling Law 是否成立？**
A: HSTU 论文表明推荐系统同样存在 Scaling Law：更多参数（尤其是 Embedding 和序列模型）持续带来指标提升，但需配合充足的交互数据。

**Q5: 如何处理冷启动用户（序列极短）？**
A: 对短序列用户做 padding + 位置编码区分；可引入用户画像特征补充；部分场景用基于 item 内容的 embedding 初始化。
