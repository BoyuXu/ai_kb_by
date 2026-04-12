# 量化风控度量全景

> 没有风控的策略不是策略，是赌博。本文梳理量化投资中所有核心风险度量指标，每个指标都给出直觉、公式和 Python 一行代码。

## 1. 收益类指标

### 1.1 累积收益（Cumulative Return）

最直观的指标：从开始到结束赚了多少。

$$R_{cum} = \frac{P_{end}}{P_{start}} - 1$$

```python
cum_return = (1 + daily_returns).cumprod()[-1] - 1
```

### 1.2 年化收益率（Annualized Return）

将不同时间长度的收益标准化到"年"的尺度进行比较。

$$R_{ann} = (1 + R_{cum})^{252/N} - 1$$

其中 N 是交易日数，252 是年交易日数。

```python
ann_return = (1 + cum_return) ** (252 / len(daily_returns)) - 1
```

### 1.3 对数收益 vs 简单收益

- **简单收益**：$r_t = P_t / P_{t-1} - 1$，直觉好，但多期不能直接相加
- **对数收益**：$r_t = \ln(P_t / P_{t-1})$，多期可以相加，统计性质更好
- **实践**：单期用简单收益，多期统计分析用对数收益

## 2. 风险类指标

### 2.1 波动率（Volatility）

收益率的标准差，衡量收益的不确定性。

$$\sigma = \text{std}(r_t) \times \sqrt{252}$$

```python
vol = daily_returns.std() * np.sqrt(252)
```

**注意**：波动率对上涨和下跌一视同仁——但投资者真正怕的是下跌。

### 2.2 最大回撤（Maximum Drawdown）

从历史最高点到最低点的最大跌幅。**投资者最能直观感受的风险指标**。

$$MDD = \max_{t} \left( \frac{\text{Peak}_t - \text{Trough}_t}{\text{Peak}_t} \right)$$

```python
cummax = (1 + daily_returns).cumprod().cummax()
drawdown = (1 + daily_returns).cumprod() / cummax - 1
max_drawdown = drawdown.min()
```

**经验法则**：最大回撤超过 30%，大部分投资者会赎回。

### 2.3 下行波动率（Downside Deviation）

只计算负收益的波动，比普通波动率更合理。

$$\sigma_d = \sqrt{\frac{1}{N}\sum_{t} \min(r_t - \text{MAR}, 0)^2}$$

MAR（Minimum Acceptable Return）通常取 0 或无风险利率。

```python
downside = daily_returns[daily_returns < 0].std() * np.sqrt(252)
```

### 2.4 最大回撤持续时间（Max Drawdown Duration）

从最高点跌下去到恢复到新高，经历了多长时间。

**直觉**：一个策略最大回撤 20% 但 3 个月恢复，vs 最大回撤 15% 但 2 年没恢复——后者可能更令人痛苦。

## 3. 风险调整收益指标

### 3.1 Sharpe Ratio

**最广泛使用的风险调整指标**。每承担一单位风险，获得多少超额收益。

$$\text{Sharpe} = \frac{R_p - R_f}{\sigma_p}$$

```python
sharpe = (ann_return - risk_free) / vol
```

| Sharpe | 评价 |
|--------|------|
| < 0 | 不如存银行 |
| 0 - 1.0 | 一般 |
| 1.0 - 2.0 | 优秀 |
| 2.0 - 3.0 | 非常优秀（对冲基金目标） |
| > 3.0 | 怀疑过拟合或数据问题 |

### 3.2 Sortino Ratio

用下行波动率替换总波动率，只惩罚"坏的"波动。

$$\text{Sortino} = \frac{R_p - R_f}{\sigma_d}$$

```python
sortino = (ann_return - risk_free) / downside
```

**何时用 Sortino**：策略收益分布不对称时（如卖期权策略，正收益多但偶有大亏损）。

### 3.3 Calmar Ratio

年化收益除以最大回撤。衡量"收益和最大痛苦的比值"。

$$\text{Calmar} = \frac{R_{ann}}{|MDD|}$$

```python
calmar = ann_return / abs(max_drawdown)
```

**实践**：Calmar > 1.0 是不错的策略，> 3.0 非常优秀。

### 3.4 Information Ratio

相对于基准的超额收益 / 跟踪误差。衡量主动管理能力。

$$\text{IR} = \frac{R_p - R_b}{\sigma(R_p - R_b)}$$

```python
excess = daily_returns - benchmark_returns
ir = excess.mean() / excess.std() * np.sqrt(252)
```

**IR vs Sharpe**：Sharpe 衡量绝对表现，IR 衡量相对基准的表现。

## 4. 尾部风险指标

### 4.1 VaR（Value at Risk）

在给定置信水平下，最大可能亏损。

$$\text{VaR}_{95\%} = -\text{Quantile}(r_t, 0.05)$$

```python
var_95 = -np.percentile(daily_returns, 5)
```

**直觉**：95% VaR = 3% 意味着"95% 的交易日，你的亏损不超过 3%"。

**致命缺陷**：VaR 不告诉你"超过 VaR 的那 5% 会亏多少"——可能是 3.1%，也可能是 30%。

### 4.2 CVaR / Expected Shortfall（条件VaR）

超过 VaR 阈值后的平均亏损。修补了 VaR 的致命缺陷。

$$\text{CVaR}_{95\%} = -E[r_t \mid r_t < -\text{VaR}_{95\%}]$$

```python
cvar_95 = -daily_returns[daily_returns < -var_95].mean()
```

**重要性质**：CVaR 是一致性风险度量（coherent risk measure），满足次可加性；VaR 不满足。

### 4.3 尾部比率（Tail Ratio）

右尾（大赚）和左尾（大亏）的比值。

$$\text{Tail Ratio} = \frac{\text{Quantile}(r_t, 0.95)}{|\text{Quantile}(r_t, 0.05)|}$$

```python
tail_ratio = np.percentile(daily_returns, 95) / abs(np.percentile(daily_returns, 5))
```

**直觉**：尾部比率 > 1 说明"大赚比大亏的幅度更大"——策略有正偏度。

## 5. 指标选择决策树

```
你要评估什么？
├── 绝对收益表现 → Sharpe Ratio
├── 下行风险敏感 → Sortino Ratio
├── 最大亏损承受力 → Calmar Ratio / Max Drawdown
├── 相对基准表现 → Information Ratio
├── 尾部风险管理 → CVaR（不要只看 VaR）
└── 策略对比排序 → 多指标综合（Sharpe + MaxDD + Calmar）
```

## 6. 常见陷阱

### 6.1 Sharpe 的局限性

- **假设正态分布**：现实中收益分布有厚尾和偏度
- **卖期权策略的 Sharpe 陷阱**：平时稳定赚小钱（高 Sharpe），偶尔巨亏（黑天鹅）
- **时间尺度敏感**：日 Sharpe × √252 ≠ 真实年化 Sharpe（自相关时）

### 6.2 回测 Sharpe 虚高

- 回测中的 Sharpe 通常比实盘高 0.5-1.0
- 原因：交易成本低估、滑点忽略、幸存者偏差、前视偏差

### 6.3 最大回撤的采样偏差

- 回测期越长，最大回撤越大——这是数学必然
- 不能因为"这个策略最大回撤小"就认为安全——可能只是回测时间不够长

## 相关文档

- [[synthesis/risk_and_execution.md]] — 风险管理与交易执行
- [[concepts/factor_investing_framework.md]] — 因子投资体系
- [[interview/strategy_design.md]] — 策略设计面试题
