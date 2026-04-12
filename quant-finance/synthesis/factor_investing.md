# 因子投资：从 CAPM 到现代多因子体系

> 因子投资是量化投资的基石。理解因子，就理解了量化选股的底层逻辑。

## 1. 因子投资的演进

### 1.1 CAPM：单因子时代（1964）

Sharpe 的 CAPM 告诉我们：股票的超额收益只由一个因子解释——**市场因子（β）**。

$$R_i - R_f = \beta_i (R_m - R_f) + \epsilon_i$$

- $R_i$：股票 i 的收益
- $R_f$：无风险利率
- $\beta_i$：股票对市场的敏感度
- $\epsilon_i$：特异性收益（CAPM 认为这是噪声）

**问题**：实证发现很多「异象」（anomalies）无法用 CAPM 解释——小市值股票长期跑赢大盘、低 PE 股票跑赢高 PE 股票。

### 1.2 Fama-French 三因子（1993）

Fama 和 French 提出：除了市场因子，还需要两个因子才能解释股票收益。

$$R_i - R_f = \alpha_i + \beta_i^{MKT} \cdot MKT + \beta_i^{SMB} \cdot SMB + \beta_i^{HML} \cdot HML + \epsilon_i$$

| 因子 | 含义 | 构建方式 |
|------|------|---------|
| MKT | 市场因子 | 市场组合收益 - 无风险利率 |
| SMB | 规模因子 | Small minus Big（小盘 - 大盘） |
| HML | 价值因子 | High minus Low（高 B/M - 低 B/M） |

**直觉**：小公司风险大→要求更高回报；便宜的（高 B/M）公司可能有麻烦→要求更高回报。

### 1.3 五因子模型（2015）

Fama-French 加入两个新因子：

| 因子 | 含义 | 构建 |
|------|------|------|
| RMW | 盈利因子 | Robust minus Weak（高盈利 - 低盈利） |
| CMA | 投资因子 | Conservative minus Aggressive（低投资 - 高投资） |

**争议**：加入 RMW 和 CMA 后，HML 变得不显著了。价值因子是否已死？这在学界和业界都有持续争论。

### 1.4 动量因子（Carhart, 1997）

Jegadeesh & Titman 发现：**过去 12 个月涨得好的股票，未来还会涨**。

$$MOM = Winners_{12m} - Losers_{12m}$$

动量因子是最强的异象之一，但也最容易崩溃（2009 年动量崩溃）。

## 2. Alpha 因子 vs 风险因子

这是面试必考的区分：

| 维度 | Alpha 因子 | 风险因子 |
|------|-----------|---------|
| **本质** | 市场定价错误（mispricing） | 承担风险的补偿 |
| **持续性** | 会随着被发现而衰减 | 长期存在 |
| **例子** | 事件驱动、另类数据 | 市场、规模、价值 |
| **收益来源** | 信息优势 | 风险溢价 |

**实际操作**：大多数因子处于灰色地带——很难严格区分是 alpha 还是 risk premium。面试时要能讨论这个模糊性。

## 3. 因子构建实战

### 3.1 价量因子（最常用）

```python
# 动量因子
momentum_12m = stock_returns.rolling(252).sum() - stock_returns.rolling(21).sum()
# 过去12个月收益，排除最近1个月（避免反转效应）

# 波动率因子（低波动异象）
volatility = stock_returns.rolling(60).std()

# 换手率因子
turnover = volume / shares_outstanding
turnover_factor = turnover.rolling(20).mean()
```

### 3.2 基本面因子

```python
# EP（Earnings/Price，PE的倒数）
ep = earnings_ttm / market_cap

# ROE
roe = net_income_ttm / book_equity

# 营收增长率
revenue_growth = (revenue_ttm - revenue_ttm.shift(4)) / revenue_ttm.shift(4)
```

### 3.3 另类因子

- **分析师预期修正**：consensus EPS 上调 → 正信号
- **新闻情绪**：NLP 分析新闻/公告情感
- **卫星数据**：停车场车辆数→零售业绩（在中国较少用）

## 4. 因子测试方法论

### 4.1 IC 分析（Information Coefficient）

IC = 因子值与未来收益的截面相关系数（通常用 Rank IC/Spearman）

| IC 水平 | 评价 |
|---------|------|
| \|IC\| > 0.05 | 有效因子 |
| \|IC\| > 0.1 | 优秀因子 |
| ICIR > 0.5 | 稳定有效 |

$$ICIR = \frac{mean(IC)}{std(IC)}$$

ICIR 比单看 IC 更重要——一个 IC=0.03 但 ICIR=0.8 的因子，比 IC=0.08 但 ICIR=0.3 的因子更可用。

### 4.2 分组回测

1. 每个换仓日，按因子值将股票分成 N 组（通常 5 或 10 组）
2. 做多 Top 组，做空 Bottom 组（或只看 Top 组 vs 基准）
3. 观察：多空收益、单调性、换手率、最大回撤

**关键指标**：
- 多空组合年化收益率和 Sharpe
- Top 组 vs Bottom 组的收益差是否单调递增
- 换手率（太高的因子交易成本会吃掉收益）

### 4.3 回归分析

$$R_{i,t+1} = \alpha + \beta \cdot Factor_{i,t} + Controls + \epsilon_{i,t}$$

Fama-MacBeth 回归：每期做截面回归，然后对时序取均值。这是学术标准方法。

## 5. Barra 风险模型

Barra 是业界最主流的风险模型框架（MSCI 旗下）。

### 5.1 核心思想

$$R_i = \sum_k X_{ik} \cdot f_k + \epsilon_i$$

- $X_{ik}$：股票 i 在因子 k 上的暴露（exposure/loading）
- $f_k$：因子收益率（factor return）
- $\epsilon_i$：特异性收益

### 5.2 Barra CNE6 因子体系（A 股常用）

| 类别 | 因子 |
|------|------|
| 风格因子 | Beta, Momentum, Size, Earnings Yield, Volatility, Growth, Value, Leverage, Liquidity, Dividend Yield |
| 行业因子 | 中信一级行业（约 30 个） |

### 5.3 风险归因

组合风险分解为：
$$\sigma^2_p = X'_p \cdot \Sigma_f \cdot X_p + \Delta^2_p$$

- 第一项：因子风险（系统性风险）
- 第二项：特异性风险（个股独有）

**面试高频题**：你的组合 Sharpe 很高，但 Barra 归因显示 90% 的收益来自行业暴露，怎么看？
→ 说明不是真 alpha，是行业 beta 赚的钱，需要行业中性化。

## 6. 因子组合

### 6.1 等权组合
最简单：每个因子打分标准化后直接加总。

### 6.2 IC 加权
$$w_k = \frac{IC_k}{\sum IC}$$

用历史 IC 的均值（或衰减加权均值）作为权重。

### 6.3 优化组合
最大化 IR（Information Ratio）= 预期超额收益 / 跟踪误差：

$$\max_w \frac{w' \cdot \alpha}{\sqrt{w' \Sigma w}} \quad s.t. \quad constraints$$

约束通常包括：行业中性、风格中性、换手率上限、个股权重上限。

## 7. 面试常见问题

**Q: 你最熟悉的因子是什么？详细说说构建和测试过程。**
→ 选一个你真正做过的因子，从数据→构建→测试→组合完整说。

**Q: 如何判断一个因子是否过拟合？**
→ 样本外测试、Deflated Sharpe Ratio、因子经济学直觉、多市场验证。

**Q: 动量因子和反转因子矛盾吗？**
→ 不矛盾。时间尺度不同：短期（1周）反转、中期（3-12月）动量、长期（3-5年）反转。

**Q: 因子拥挤（crowding）怎么衡量？**
→ 因子估值水平、因子组合的 short interest、配对相关性升高。

---

## 交叉引用

- 特征工程方法论 → [[rec-search-ads/]] 搜广推特征工程
- Embedding 表征 → [[concepts/embedding_everywhere.md]]
- 多目标优化 → [[concepts/multi_objective_optimization.md]]（收益-风险帕累托）
- [[quant-finance/synthesis/ml_in_quant.md]] — ML 在因子选股中的应用
- [[quant-finance/synthesis/strategy_development.md]] — 因子策略的回测与实战
