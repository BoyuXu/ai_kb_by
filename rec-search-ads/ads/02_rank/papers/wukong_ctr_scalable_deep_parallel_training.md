# Wukong CTR: Scalable Deep CTR Prediction via Massive Parallel Training

> 来源：arxiv | 领域：ads | 学习日期：20260328

## 问题定义

随着广告系统规模增长，CTR 模型的训练面临两个核心挑战：
1. **模型容量瓶颈**：单机训练限制了模型深度和宽度的扩展
2. **训练效率**：数十亿样本需要快速迭代训练，传统方法无法满足

Wukong（悟空）是 Kuaishou 提出的大规模 CTR 预估框架，核心目标：在超大规模并行训练下保持模型质量，同时实现极高的训练吞吐量。

## 核心方法与创新点

### 架构设计：Stacked 交叉模块

Wukong 采用 L 层堆叠的交叉单元，每个单元包含：

$$
h^{(l)} = LN\left(FFN\left(LN\left(Cross(h^{(l-1)})\right)\right)\right)
$$

- **Cross 模块**：类似 DCN V2 的矩阵交叉
- **FFN**：Position-wise 前馈网络
- **LayerNorm**：稳定深层训练

类似 Transformer 的"Pre-LN"结构，使深层网络（10+ 层）稳定训练成为可能。

### 大规模并行训练优化

1. **Embedding 并行**：ID 特征 embedding 表分布式存储（Parameter Server）
2. **模型并行**：深层 dense 网络按层切分到多个 GPU
3. **数据并行**：多机多卡同步训练，gradient all-reduce
4. **混合精度训练**：FP16 forward，FP32 gradient accumulation

### Feature Grouping

将特征按语义分组（用户行为、物品属性、上下文等），不同组的特征在交叉之前先内部交互，再跨组交叉，减少计算复杂度。

### Online Learning 适配

支持增量训练，每天新数据快速 finetune，保持模型时效性。

## 实验结论

- 在快手广告系统（千亿级样本）上部署，NE/AUC 显著优于 DLRM、DCN V2
- 支持 100B+ 参数规模的 CTR 模型训练
- 训练吞吐量比原有系统提升 3-5x
- 深层架构（12层）比浅层（3层）AUC 提升约 0.2%

## 工程落地要点

1. **Pre-LN vs Post-LN**：深层 CTR 模型必须用 Pre-LN，Post-LN 在 10 层以上容易梯度消失
2. **Embedding 分片策略**：高频 ID 特征的 embedding 放本地 cache，低频走 PS 远程访问
3. **梯度压缩**：all-reduce 时用 PowerSGD 或 gradient compression 减少通信量
4. **AUC 一致性**：大 batch size 训练可能影响 AUC，需要调整学习率和 warmup 策略
5. **特征时效性**：实时特征（如近 1 小时点击率）需要单独的流式更新路径
6. **监控指标**：除了 NE/AUC，还需监控各特征的 embedding gradient norm，防止某些特征过拟合

## 面试考点

**Q：为什么深层 CTR 模型（10+ 层）需要用 Pre-LN 而不是 Post-LN？**
A：Post-LN 在深层时梯度传播路径长，LayerNorm 在残差路径上会干扰梯度流，导致梯度消失；Pre-LN 将 LayerNorm 放在子层输入前，残差连接保持干净的梯度通道，训练更稳定，适合 10+ 层深度。

**Q：CTR 模型的大规模并行训练中，Embedding 和 Dense 部分为什么要用不同的并行策略？**
A：Embedding 是稀疏访问（每个样本只激活少数 ID），适合用 Parameter Server 分布式存储，避免数据并行时复制大量参数；Dense 网络是全量计算，适合数据并行 + 模型并行组合，利用 GPU 计算效率。

**Q：如何保证大批量训练时 AUC 不下降？**
A：1) Linear scaling rule 调整学习率（batch size 翻倍，lr 也翻倍）；2) 加长 warmup 周期；3) 使用 LARS/LAMB 优化器对不同层自适应调整学习率；4) 监控 loss landscape，必要时降低 batch size。
