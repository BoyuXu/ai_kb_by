# 广告 CTR/CVR 预估与校准：从模型到上线的全链路

> 创建：2026-03-24 | 领域：广告系统 | 类型：综合分析
> 来源：ESMM, GNOLR, Calibration 实践, 延迟转化处理系列

---

## 🎯 核心洞察（5条）

1. **广告 CTR 预估的核心不是 AUC 而是校准性**：排序只需要相对顺序正确（AUC），但出价需要绝对概率准确（pCTR=5% 的广告要付的钱和 pCTR=10% 的不同）
2. **CVR 预估的三大难题**：样本选择偏差（只有点击才有转化标签）、延迟转化（点击后 7 天才转化怎么处理？）、正样本极稀疏（转化率通常 <3%）
3. **ESMM 是 CVR 去偏的标准方案**：通过 CTCVR=CTR×CVR 在全曝光空间建模，CVR 分支隐式获得全空间约束
4. **校准（Calibration）是模型上线前的必要步骤**：Platt Scaling（sigmoid 映射）或 Isotonic Regression（保序回归），确保预测概率与实际发生率一致
5. **在线学习是广告模型的工程特色**：广告数据分布变化快（促销/节日），需要小时级/天级模型更新，FTRL（Follow The Regularized Leader）是在线更新的标准算法

---

## 📈 技术演进脉络

```
LR + 手工特征（2010-2014）→ FM/FFM（2014-2016）→ Wide&Deep/DeepFM（2016-2018）
  → DIN/序列建模（2018-2020）→ ESMM 多任务（2018+）→ LLM 增强 CTR（2024+）
校准：无校准 → Platt Scaling → Isotonic Regression → Field-aware Calibration
更新：离线训练 → 天级增量 → FTRL 实时更新 → 增量 + 全量混合
```

---

## 🎓 面试考点（6条）

### Q1: 广告 CTR 为什么需要校准？
**30秒答案**：出价公式 bid = target_CPA × pCVR，如果 pCVR 预估偏高（过校准），实际出价就偏高，广告主亏损；偏低则投不出去。所以广告模型对概率值的绝对准确性要求远高于推荐。

### Q2: Platt Scaling vs Isotonic Regression？
**30秒答案**：Platt Scaling 用 sigmoid 做全局线性校准 `p_calibrated = sigmoid(a×logit + b)`，参数少适合数据量小；Isotonic Regression 用分段常数做非参数校准，更灵活但需要更多数据。

### Q3: 延迟转化怎么处理？
**30秒答案**：①等待窗口法：点击后等 7 天再确认转化标签（简单但浪费数据）；②Elapsed-Time Model：将"点击后经过多久"作为特征，预测最终转化概率；③Fake Negative Calibration：短期内标记为负样本，事后回补正样本。

### Q4: FTRL 在线学习的核心思想？
**30秒答案**：FTRL 在每个样本到达时更新模型参数，带 L1 正则化产生稀疏解（自动特征选择）。核心公式考虑了历史梯度累积（类似 Adagrad）和正则化。适合高维稀疏特征（十亿级特征维度）。

### Q5: 广告和推荐的 CTR 模型有什么关键差异？
**30秒答案**：①校准要求不同（广告严格 vs 推荐宽松）；②更新频率不同（广告天级/小时级 vs 推荐周级）；③样本构建不同（广告有竞价日志 vs 推荐只有曝光日志）；④特征不同（广告有出价/预算/广告主特征）。

### Q6: CVR 正样本极少怎么办？
**30秒答案**：①ESMM 多任务缓解（CTR 任务有大量样本辅助）；②过采样正样本 + 欠采样负样本 + 样本权重修正；③Focal Loss 聚焦难分样本；④数据增广（类似转化事件作为弱正样本）。

---

## 🌐 知识体系连接

- **上游依赖**：CTR/CVR 预估模型、校准理论、在线学习
- **下游应用**：出价策略、eCPM 计算、ROI 优化
- **相关 synthesis**：std_ads_bidding_landscape.md, std_rec_ranking_evolution.md
