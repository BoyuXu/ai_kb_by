# 量化策略开发框架：从想法到实盘

> 策略开发不是写个回测跑出漂亮曲线就完了。真正的难点在于：如何确保回测结果在实盘中依然有效？

## 1. 策略分类

### 1.1 按持有周期

| 类型 | 持仓时间 | 换手率 | 典型策略 | 技术要求 |
|------|---------|--------|---------|---------|
| 高频 HFT | 秒-分钟 | 极高 | 做市、统计套利 | C++、低延迟、co-location |
| 中高频 | 分钟-天 | 高 | 日内动量、事件驱动 | Python + 实时数据 |
| 中频 | 天-周 | 中 | 因子选股、技术指标 | Python + 日频数据 |
| 低频 | 周-月 | 低 | 基本面选股、宏观策略 | 基本面分析 + 季度数据 |

**面试tip**：大多数量化面试考的是中频策略（因子选股）。高频面试会单独考 C++ 和系统设计。

### 1.2 按策略逻辑

#### 动量策略（Momentum）
- **核心假设**：趋势会持续，赢家继续赢
- **实现**：买入过去 N 月涨幅最大的股票，卖出最差的
- **风险**：动量崩溃（momentum crash），在市场反转时亏损巨大

```python
# 经典横截面动量
def momentum_signal(returns, lookback=252, skip=21):
    """过去12个月收益，跳过最近1个月"""
    return returns.rolling(lookback).sum() - returns.rolling(skip).sum()
```

#### 均值回归（Mean Reversion）
- **核心假设**：价格偏离均值后会回归
- **实现**：买入超跌股，卖出超涨股
- **常用指标**：Bollinger Bands、RSI、z-score

#### 统计套利（Statistical Arbitrage）
- **核心假设**：相关资产价差回归
- **配对交易**：找协整的股票对，价差偏离时交易
- **Ornstein-Uhlenbeck 过程**建模价差

```python
# 配对交易：协整检验
from statsmodels.tsa.stattools import coint
score, pvalue, _ = coint(stock_a_prices, stock_b_prices)
if pvalue < 0.05:
    # 这对股票可以做配对交易
    spread = stock_a_prices - hedge_ratio * stock_b_prices
    z_score = (spread - spread.mean()) / spread.std()
    # z_score > 2: 做空 spread; z_score < -2: 做多 spread
```

## 2. 回测框架

### 2.1 事件驱动 vs 向量化

| 特性 | 向量化回测 | 事件驱动回测 |
|------|-----------|-------------|
| 速度 | 极快（矩阵运算） | 慢（逐 tick/bar 遍历） |
| 真实度 | 低（难模拟细节） | 高（接近实盘逻辑） |
| 适用场景 | 快速验证想法 | 精确评估、接近实盘 |
| 代表工具 | pandas/numpy | zipline/backtrader/vnpy |

**建议**：先用向量化快速筛选想法，有戏的再用事件驱动精确回测。

### 2.2 回测中的常见偏差（Bias）

这是面试重点中的重点：

| 偏差 | 含义 | 如何避免 |
|------|------|---------|
| **Look-ahead bias** | 用了未来信息 | 严格按发布日期对齐数据（如财报发布日 vs 报告期） |
| **Survivorship bias** | 只用了存活股票 | 使用包含退市股票的全样本数据 |
| **Overfitting** | 对历史数据过拟合 | 样本外测试、交叉验证、经济直觉 |
| **Selection bias** | 只展示跑得好的策略 | 记录所有尝试过的策略（策略墓地） |
| **Transaction cost bias** | 忽略交易成本 | 建模真实的滑点和冲击成本 |

### 2.3 A 股特有陷阱

- **涨跌停**：涨停板买不进、跌停板卖不出，回测中必须处理
- **ST/停牌**：长期停牌的股票会扭曲收益计算
- **IPO 新股**：上市初期交易行为异常，通常排除前 N 天
- **指数调整效应**：入选沪深 300 的股票短期跑赢（被动资金买入）

## 3. 过拟合检测与防范

### 3.1 Deflated Sharpe Ratio（DSR）

López de Prado 提出：如果你尝试了 K 个策略，选了 Sharpe 最高的那个，真实 Sharpe 应该打折。

$$DSR = P\left[\hat{SR}^* > 0 \mid K, V[\hat{SR}], skew, kurt\right]$$

**直觉**：你试了 1000 个策略，最好的 Sharpe = 2.0。但如果只试了 10 个，最好的 Sharpe = 1.5，后者反而更可信。

### 3.2 Walk-Forward Analysis

1. 在训练窗口训练模型/参数
2. 在紧接着的测试窗口验证
3. 滚动前进，重复

```
|--train--|--test--|
     |--train--|--test--|
          |--train--|--test--|
```

**比固定 train/test split 更真实**，因为模拟了实际中定期重新训练的过程。

### 3.3 Purged K-Fold CV

金融时间序列不能用普通 K-Fold（数据有自相关）。

- **Purge**：在 train 和 test 之间留 gap，避免信息泄露
- **Embargo**：test 之后的一段数据也不能用于 train

这是 AFML 的核心贡献之一，面试必考。

## 4. 风险度量

### 4.1 核心指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **Sharpe Ratio** | $(R_p - R_f) / \sigma_p$ | 单位风险收益，>1 良好，>2 优秀 |
| **Max Drawdown** | 最大峰谷回撤 | 衡量极端损失 |
| **Calmar Ratio** | 年化收益 / MaxDD | 结合收益和回撤 |
| **VaR** | 在置信水平下的最大损失 | 如 95% VaR = 在 95% 情况下损失不超过 X |
| **CVaR (ES)** | VaR 之外的平均损失 | 比 VaR 更保守，考虑尾部风险 |
| **Sortino Ratio** | 超额收益 / 下行波动率 | 只惩罚下行波动，比 Sharpe 更合理 |

### 4.2 Sharpe Ratio 的坑

面试高频题：Sharpe Ratio 有什么问题？

1. **假设收益正态分布**：金融收益有厚尾（fat tail），Sharpe 低估了尾部风险
2. **时间依赖**：年化 Sharpe = 日 Sharpe × √252，但这假设收益 i.i.d.
3. **可以被操纵**：写 put option 可以人为提高 Sharpe（卖尾部风险换日常收益）
4. **不区分上行和下行波动**：涨很多也被惩罚

## 5. 仓位管理

### 5.1 Kelly 准则

$$f^* = \frac{p \cdot b - q}{b} = \frac{edge}{odds}$$

- p: 胜率, q=1-p: 败率, b: 盈亏比
- 实际操作中通常用 **半 Kelly**（Half Kelly）降低波动

### 5.2 风险平价（Risk Parity）

$$w_i \propto \frac{1}{\sigma_i}$$

让每个资产对组合的风险贡献相等。桥水 All Weather 策略的核心思想。

### 5.3 均值方差优化（Mean-Variance / Markowitz）

$$\max_w \quad w'\mu - \frac{\lambda}{2} w'\Sigma w$$

问题：对输入参数极度敏感（estimation error amplification）。实操中通常需要：
- 收缩估计（shrinkage）
- Black-Litterman 模型（融合先验和观点）
- 约束优化（上下限、换手率约束）

## 6. 面试常见问题

**Q: 描述一个完整的策略开发流程。**
→ 假设形成 → 数据准备 → 信号构建 → 回测 → 风险分析 → 参数敏感性 → 纸上交易 → 实盘

**Q: 你的策略 Sharpe 2.5，为什么我不应该直接用它？**
→ 检查：样本外表现？考虑交易成本了吗？多少个策略里选出来的（multiple testing）？数据是否有 look-ahead bias？

**Q: 动量和均值回归矛盾吗？**
→ 不矛盾。它们在不同时间尺度和不同资产类别上有效。可以组合使用。

**Q: 如何衡量回测过拟合程度？**
→ DSR、WF 分析、参数稳定性（参数微调后策略是否剧变）、多市场验证。

---

## 交叉引用

- 特征工程/因子构建 → [[quant-finance/synthesis/factor_investing.md]]
- ML 模型选择 → [[quant-finance/synthesis/ml_in_quant.md]]
- 序列建模 → [[concepts/sequence_modeling_evolution.md]]
- 过拟合防范方法论在搜广推中也有对应 → 推荐系统离线评估

---

## 相关概念

- [[concepts/multi_objective_optimization|多目标优化]]
