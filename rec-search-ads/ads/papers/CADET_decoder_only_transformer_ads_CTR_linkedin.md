# CADET: Context-Conditioned Ads CTR Prediction With a Decoder-Only Transformer (LinkedIn)
> 来源：arXiv:2602.11410 | 领域：ads | 学习日期：20260419

## 问题定义
广告CTR预估面临独特挑战：①后评分上下文信号（如广告位置、展示上下文）在评分时不可用，存在"鸡生蛋"问题；②在线一致性（offline-online consistency）难保证；③Transformer架构在广告场景的规模化适配。

## 核心方法与创新点
1. **Context-Conditioned Decoding**：多塔预测头显式建模后评分信号（ad position），解决CTR与排序的循环依赖
2. **Self-Gated Attention**：自适应门控调节表征和交互层面的信息流，稳定训练
3. **Timestamp-based RoPE**：基于时间戳的旋转位置编码，捕捉从秒到月的多时间尺度关系
4. **Session Masking**：防止模型学习对session内不可用事件的依赖，解决train-serve skew
5. **工程优化**：Tensor Packing + Sequence Chunking + Custom Flash Attention Kernels

## 实验结论
- 在线A/B测试：CTR提升 **11.04%**（相比生产LiRank baseline）
- 部署于LinkedIn广告系统

## 工程落地要点
- Decoder-only架构天然处理变长用户行为序列
- Session masking是保证offline-online一致性的关键
- Flash Attention + Tensor Packing大幅降低训练/推理成本

## 面试考点
- Q: 广告CTR预估中的position bias如何处理？
  - A: ①训练时将position作为特征但推理时固定（shallow tower）；②Multi-tower架构分离position信号（CADET方式）；③因果推断去偏
- Q: Train-serve skew 在广告系统中的表现？
  - A: 训练时可见同session其他广告点击信息，推理时不可见→模型利用泄露信号→在线效果下降
