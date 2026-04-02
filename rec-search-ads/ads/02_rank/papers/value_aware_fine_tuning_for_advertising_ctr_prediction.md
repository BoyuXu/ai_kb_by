# Value-Aware Fine-Tuning for Advertising CTR Prediction
> 来源：arxiv/2310.xxxxx | 领域：ads | 学习日期：20260326

## 问题定义
广告 CTR 预估模型的训练目标与业务价值目标存在偏差：
- 标准 CTR 模型：最小化 BCE(预测CTR, 实际点击)
- 业务目标：最大化 RPM（每千次展示收入）或 GMV
- 高价值广告（高出价/高转化）和低价值广告对 CTR 预测精度要求不同
- 全量样本均等权重，导致模型对长尾低价值广告过拟合

## 核心方法与创新点
**Value-Aware Fine-Tuning (VAFT)**：以广告价值为权重进行差异化微调。

**核心思想：价值加权的 CTR 损失：**
```python
# 标准 BCE
L_standard = -Σ_i [y_i·log(p_i) + (1-y_i)·log(1-p_i)]

# Value-Aware 加权
value_weight_i = f(bid_i, predicted_cvr_i, quality_score_i)
L_vaft = -Σ_i value_weight_i · [y_i·log(p_i) + (1-y_i)·log(1-p_i)]
```

**价值权重计算：**
```python
def compute_value_weight(ad):
    ecpm = ad.bid × ad.predicted_cvr × 1000   # 预计千次展示收入
    quality = ad.quality_score                   # 广告质量分
    value = ecpm × quality
    # Clipping 防止极端权重
    weight = clip(value / mean_value, 0.1, 10.0)
    return weight
```

**两阶段训练：**
```
Stage 1: 全量数据预训练（标准 BCE）→ 得到基础 CTR 模型
Stage 2: 高价值广告子集微调（VAFT）→ 精调高价值广告的预测精度
```

**Curriculum Learning：**
- 先训练简单（高价值/明确点击）样本
- 逐步引入困难（低价值/模糊）样本

## 实验结论
- 某头部广告平台 A/B 测试：
  - RPM +2.3%，高价值广告 CTR AUC +0.004
  - 低价值广告 CTR AUC 略降（-0.001，可接受）
- 整体 AUC 几乎持平（+0.001），但业务指标显著提升
- Calibration（校准误差）：高出价广告段校准误差 -32%

## 工程落地要点
1. **价值权重计算**：需要在线出价信息 + 历史 CVR 预估，权重每天离线更新
2. **权重 Clipping**：防止单条样本权重过大导致梯度爆炸（上限设 10x 均值）
3. **分层评估**：按广告价值分桶评估 AUC，不只看整体 AUC
4. **与 oCPX 结合**：VAFT 的价值权重与 oCPX 出价类型（oCPC/oCPM）对齐
5. **AB 实验设计**：业务指标（RPM/GMV）作为主要评估，AUC 作为参考

## 常见考点
**Q1: 为什么传统 CTR 模型优化 AUC 但业务 RPM 提升有限？**
A: AUC 是排序指标，衡量正负样本区分度，不区分样本价值。对 1 元广告和 1000 元广告的预测准确度贡献相同。VAFT 让高价值广告（对 RPM 贡献大）的预测更准确，从而直接提升 RPM。

**Q2: 价值加权如何影响梯度更新？**
A: 高价值样本梯度权重大 → 模型更倾向于正确预测高价值样本的点击 → 以牺牲部分低价值样本精度为代价换取高价值广告的精准度。这是业务价值与统计精度的 tradeoff。

**Q3: VAFT 与 FocalLoss 有什么区别？**
A: FocalLoss 关注"困难样本"（预测不准的样本）；VAFT 关注"高价值样本"。两者都是重加权，但目标不同：FocalLoss 提升整体精度，VAFT 提升业务价值样本精度。可以结合：高价值×困难样本权重最高。

**Q4: 两阶段训练（预训练+微调）相比直接 VAFT 训练的优势？**
A: 直接 VAFT 训练可能导致模型忘记低价值广告的分布（过拟合高价值）；两阶段先学全局分布，再精调高价值区域，既保证基础校准，又提升价值精度。

**Q5: 如何设计 VAFT 的 AB 实验？**
A: 主要指标：RPM、Revenue；次要指标：整体 CTR AUC、高价值广告 CTR AUC；保护指标：低价值广告 CTR 不显著下降、用户体验（CTR 不暴跌）；实验周期：至少2周（确保包含不同出价周期）。
