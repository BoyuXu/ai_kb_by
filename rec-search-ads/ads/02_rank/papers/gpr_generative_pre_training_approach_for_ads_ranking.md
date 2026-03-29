# GPR: A Generative Pre-training Approach for Ads Ranking
> 来源：arxiv/2312.xxxxx | 领域：ads | 学习日期：20260326

## 问题定义
广告排序模型的预训练面临：
- 有监督数据不足：高质量标注数据（精确 CVR/ROI）稀缺
- 跨广告主迁移困难：不同广告主的特征空间差异大
- 冷启动：新广告主/新广告缺乏历史数据
- 特征表示质量：ID-based 特征无语义，难以迁移

## 核心方法与创新点
**GPR for Ads（Generative Pre-training for Ranking）**：自监督预训练 + 排序微调。

**预训练任务（无监督/自监督）：**
```
Task 1: 掩码广告特征预测（MAE-style）
  Input:  [ad_feat_1, [MASK], ad_feat_3, user_feat_1, ...]
  Target: 预测 [MASK] 位置的特征值
  L_mae = ||predicted_feat - actual_feat||²

Task 2: 用户行为序列预测
  Input:  [clicked_ad_1, clicked_ad_2, ...]
  Target: 预测下一个会点击的广告
  L_seq = -log P(next_ad | history)

Task 3: 对比学习（广告语义对齐）
  Positive: 同一广告的不同 augmentation（dropout/masking）
  Negative: 不同广告的表示
  L_cl = InfoNCE(aug1, aug2)
```

**微调阶段（CTR/CVR 预测）：**
```python
# 使用预训练表示初始化
ranking_model = FinetuneHead(pretrained_encoder, task="CTR")
L_finetune = BCE(predicted_ctr, actual_click)
```

## 实验结论
- 某广告平台实验：
  - 冷启动广告（<1000次曝光）CTR AUC +0.012（vs 从头训练）
  - 全量广告 CTR AUC +0.004
  - 模型收敛速度：微调仅需 10% 的有监督数据达到同等效果
- 跨平台迁移：预训练模型迁移到新平台，AUC +0.008（vs 新平台从头训练）

## 工程落地要点
1. **预训练数据**：使用全量曝光日志（不需要转化标签）构建自监督任务
2. **特征掩码比例**：MAE 任务掩码率 40-60%（广告特征比 NLP token 少，需更高掩码率）
3. **预训练与微调对齐**：确保预训练和微调的特征 Schema 一致
4. **增量预训练**：定期（每周）用新数据做增量预训练，保持表示新鲜
5. **知识蒸馏**：大型预训练模型 → 蒸馏到线上小模型

## 面试考点
**Q1: 广告 CTR 模型为什么需要预训练？**
A: 解决数据效率问题：有转化标签的数据稀缺（购买率 <0.1%），而曝光/点击数据海量。预训练在大量无标签数据上学习特征表示，微调时用少量有标签数据就能达到好效果，类似 NLP 的 BERT 范式。

**Q2: 掩码特征预测（MAE）如何适配广告场景？**
A: 广告特征是结构化（非连续文本），需要：①将每个 field 视为一个 token ②掩码整个 field（而非 token 级） ③预测目标是类别分布（softmax）或数值（regression）。重建损失作为预训练信号。

**Q3: 对比学习在广告预训练中如何增强样本？**
A: 广告 Augmentation 策略：①特征 Dropout：随机删除一些非关键特征 ②特征值扰动：连续特征加小噪声 ③时间切片：同一广告不同时间段的特征视为正样本对。关键是增强后仍保持语义不变。

**Q4: GPR 预训练后微调时是否需要 Freeze 某些层？**
A: 通常采用 Gradual Unfreezing：先只训练 Head 层，再逐步解冻底层。广告特征变化快，底层需要更新以适应当前特征分布。实践中 Freeze 前2层，训练后6层+Head。

**Q5: 如何评估预训练模型的表示质量？**
A: ①线性探针（Linear Probing）：固定预训练 encoder，只训练线性分类器，AUC 反映表示质量 ②Few-shot 评估：用 N 个有标签样本微调，看收敛速度 ③语义聚类：同类广告的表示是否聚集（t-SNE 可视化）。
