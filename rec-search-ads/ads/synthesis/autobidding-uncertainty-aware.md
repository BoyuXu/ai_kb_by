# Synthesis: 不确定性感知的自动出价系统

**生成日期：** 2026-04-16
**涵盖论文：** DenoiseBid (2603.01825), RobustBid (2510.08788), BAT Benchmark (2505.08485), Auto-bidding Survey (2408.07685), CADET (2602.11410)

---

## 1. 技术演进路径

```
规则出价 → 确定性优化出价 → 不确定性感知出价 → 端到端 Transformer 出价
(eCPM排序)   (线性规划/PID)    (DenoiseBid/RobustBid)    (CADET)
```

## 2. 核心问题：CTR/CVR 噪声对出价的影响

**影响链：**
```
CTR/CVR 模型预测不准确
    → bid = f(pCTR, pCVR, advertiser_target) 出价偏离
    → 赢得错误的拍卖 / 错过有价值的拍卖
    → 广告主 ROI 下降 / 平台收入损失
```

## 3. 两大解决方案对比

| 维度 | DenoiseBid (贝叶斯去噪) | RobustBid (鲁棒优化) |
|------|------------------------|---------------------|
| **核心思想** | 用后验期望替代噪声点估计 | 在最坏情况下保证出价质量 |
| **数学框架** | 贝叶斯推断 | 鲁棒优化 |
| **噪声建模** | 先验分布 + 似然函数 → 后验 | 有界扰动集 |
| **优化目标** | E[utility \| prediction, prior] | min_{noise} max utility |
| **适用场景** | 噪声分布可估计 | 噪声分布未知但有界 |
| **计算开销** | 需要先验估计（历史数据） | 需要求解 minimax 问题 |

**DenoiseBid 核心公式：**
```
bid_denoised = f(E[CTR | ĈTR, π_CTR], E[CVR | ĈVR, π_CVR])
其中 π 为从历史数据恢复的先验分布
```

**RobustBid 核心公式：**
```
bid_robust = argmax_b min_{δ ∈ Δ} utility(b, CTR + δ_CTR, CVR + δ_CVR)
其中 Δ 为扰动集合
```

## 4. CADET: 从特征交叉到 Transformer 端到端

**传统 DLRM 架构：**
```
特征 → Embedding → 特征交叉（DCN/DeepFM） → CTR
```

**CADET 架构（LinkedIn）：**
```
特征序列 → Decoder-Only Transformer → 条件化注意力 → CTR
```

**CADET 解决的三大广告特有挑战：**
1. 后评分上下文信号（如广告位置）的因果处理
2. 离线训练与在线服务的一致性
3. 工业规模的推理效率

## 5. 工业实践要点

### 5.1 自动出价系统架构
```
广告主设定目标（CPA/ROAS）
    → Autobidder（出价优化器）
        → CTR/CVR 预估模型（噪声来源）
        → 不确定性量化（DenoiseBid/RobustBid）
        → 最终出价
    → 拍卖引擎（GSP/VCG/第一价格）
    → 展示 → 点击/转化反馈
```

### 5.2 预算节奏控制（BAT Benchmark 相关）
- **PID 控制器：** 基于预算消耗偏差实时调整出价乘数
- **MPC 方法：** 预测未来流量模式，优化出价序列
- **在线学习：** UCB/Thompson Sampling 探索最优出价策略

### 5.3 延迟转化处理
- **等待窗口法：** 等待 T 天后再用数据训练（简单但数据延迟大）
- **重要性采样：** 对已观测转化进行权重修正
- **生存分析：** 建模转化时间分布，预测最终转化概率

## 6. 面试考点总结

1. **CTR/CVR 预估噪声如何影响自动出价？**
   - 噪声导致 eCPM 排序错误，赢得低价值/错过高价值拍卖

2. **DenoiseBid vs RobustBid 如何选择？**
   - 噪声分布可学习 → DenoiseBid；噪声不可知但有界 → RobustBid

3. **第一价格拍卖对 autobidding 的挑战？**
   - 不再有 GSP 的 next-price 保护，出价策略需要估计对手行为

4. **如何评估 autobidding 算法？**
   - 离线：BAT Benchmark + 反事实评估
   - 在线：广告主 ROI + 平台收入 + 预算消耗均匀性

5. **CADET 相比 DLRM 的核心优势？**
   - 自动特征交叉（无需手工设计）、序列化处理上下文信号、统一的端到端优化

---

*本 synthesis 文档由 MelonEgg 每日学习自动生成，覆盖 2026-04-16 ads 领域 5 篇核心论文*
