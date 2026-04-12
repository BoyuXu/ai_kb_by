# 执行与风控 — 风险度量、仓位管理与交易执行

> 收益是不确定的，但风险是可以度量和管理的。一个好的量化策略，核心不在于赚多少，而在于每单位风险赚多少。

## 1. 风险度量全家福

### 1.1 VaR (Value at Risk)

**定义**：在给定置信水平（如95%/99%）和持有期（如1天/10天）下，投资组合的最大预期损失。

三种计算方法：

| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| 参数法 | 假设收益正态分布，VaR = μ - z_α × σ | 计算快 | 正态假设不符合现实（肥尾） |
| 历史模拟法 | 用历史收益分布的分位数 | 不需要分布假设 | 依赖历史数据长度，尾部样本少 |
| 蒙特卡洛法 | 模拟大量路径取分位数 | 灵活，可处理非线性头寸 | 计算量大 |

> **面试常问**：VaR的缺陷是什么？
> - VaR 不满足**次可加性**（subadditivity）：两个组合合并后的 VaR 可能大于各自 VaR 之和，这违反了分散化降低风险的直觉
> - VaR 只告诉你"不超过的最大损失"，但不告诉你**超过之后会亏多少**
> - VaR 在尾部不连续，优化时可能不稳定

### 1.2 CVaR (Expected Shortfall)

**定义**：在损失超过 VaR 的条件下，损失的期望值。即"最坏的 α% 情况下，平均亏多少"。

$$CVaR_\alpha = E[L | L > VaR_\alpha]$$

**CVaR 为什么更好**：
- 满足**一致性风险度量**(coherent risk measure)的四条公理：单调性、次可加性、正齐次性、平移不变性
- 能捕捉**尾部风险**的严重程度
- 在优化问题中是凸函数，便于求解

### 1.3 Sharpe Ratio

$$SR = \frac{R_p - R_f}{\sigma_p}$$

- R_p：组合收益率，R_f：无风险利率，σ_p：组合收益标准差
- **无风险利率选择**：中国常用 SHIBOR 或国债收益率，美国用 T-bill
- **年化**：日 Sharpe × √252（假设日收益独立同分布）
- **经验值**：SR > 1 算不错，SR > 2 很优秀，SR > 3 可能有问题（过拟合或数据泄露）

### 1.4 Sortino Ratio

$$Sortino = \frac{R_p - R_f}{\sigma_{down}}$$

与 Sharpe 的区别：分母只用**下行标准差**（低于目标收益的部分），不惩罚正向波动。对于收益分布不对称的策略更合理。

### 1.5 Maximum Drawdown (最大回撤)

$$MDD = \max_{t \in [0,T]} \left( \frac{\text{peak}_t - \text{value}_t}{\text{peak}_t} \right)$$

- 衡量从历史最高点到最低点的最大跌幅
- **直观易懂**，投资者最关心的指标之一
- 缺点：只捕捉一次最差事件，不反映整体风险分布

### 1.6 Calmar Ratio

$$Calmar = \frac{\text{年化收益率}}{|\text{最大回撤}|}$$

- 衡量每承受一单位回撤能获得多少年化收益
- **经验值**：Calmar > 1 可接受，> 2 优秀

### 1.7 Information Ratio

$$IR = \frac{R_p - R_b}{\sigma_{R_p - R_b}}$$

- R_b：基准收益率，分母是**跟踪误差**（tracking error）
- 衡量主动管理能力：每承受一单位跟踪误差，获得多少超额收益
- 基金经理考核的核心指标

### 1.8 Python 实现：一站式风险指标

```python
import numpy as np
import pandas as pd
from scipy import stats

def compute_risk_metrics(returns: pd.Series, rf: float = 0.0,
                         confidence: float = 0.95) -> dict:
    """
    计算全套风险指标

    Args:
        returns: 日收益率序列
        rf: 日无风险利率（默认0）
        confidence: VaR/CVaR 的置信水平
    """
    excess = returns - rf

    # --- VaR & CVaR (历史模拟法) ---
    var = -np.percentile(returns, (1 - confidence) * 100)
    cvar = -returns[returns <= -var].mean()

    # --- Sharpe (年化) ---
    sharpe = excess.mean() / excess.std() * np.sqrt(252)

    # --- Sortino ---
    downside = excess[excess < 0]
    downside_std = np.sqrt((downside ** 2).mean())
    sortino = excess.mean() / downside_std * np.sqrt(252) if downside_std > 0 else np.inf

    # --- Max Drawdown ---
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = -drawdown.min()

    # --- Calmar ---
    annual_return = (1 + returns.mean()) ** 252 - 1
    calmar = annual_return / max_drawdown if max_drawdown > 0 else np.inf

    return {
        'annual_return': annual_return,
        'annual_volatility': returns.std() * np.sqrt(252),
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar,
        f'VaR_{confidence}': var,
        f'CVaR_{confidence}': cvar,
    }
```

参见 [[concepts/risk_measures.md]] 了解风险度量的公理化体系。

---

## 2. 滑点与市场冲击模型

### 2.1 滑点来源

| 来源 | 描述 | 量级 |
|------|------|------|
| 延迟滑点 | 信号产生到下单的时间差 | 取决于系统架构，毫秒到秒级 |
| 波动滑点 | 下单瞬间价格已变动 | 与波动率正相关 |
| 流动性滑点 | 订单量大于最优档口，吃掉多档 | 与订单大小/盘口深度相关 |
| 冲击滑点 | 自身交易推动价格不利方向移动 | 大单最严重 |

### 2.2 线性冲击模型

$$\Delta P = \eta \times \frac{V}{ADV}$$

- V：交易量，ADV：日均成交量，η：冲击系数（经验值约 0.1-0.5）
- 适用于小订单，简单直观
- 缺点：大订单时高估冲击（实际冲击增速递减）

### 2.3 平方根冲击模型 (Almgren-Chriss)

$$\Delta P = \eta \times \sigma \times \sqrt{\frac{V}{ADV}}$$

- σ：日波动率
- 平方根关系更符合实证：冲击随交易量增加，但增速递减
- **Almgren-Chriss 最优执行框架**：在冲击成本和时间风险之间权衡，求解最优执行轨迹

### 2.4 Python 实现

```python
def linear_impact(volume, adv, eta=0.3):
    """线性冲击模型：冲击比例 = eta * V/ADV"""
    return eta * volume / adv

def sqrt_impact(volume, adv, daily_vol, eta=0.3):
    """平方根冲击模型 (Almgren-Chriss)"""
    return eta * daily_vol * np.sqrt(volume / adv)

def estimate_execution_cost(price, volume, adv, daily_vol,
                            model='sqrt', eta=0.3):
    """估算执行成本（含冲击）"""
    if model == 'linear':
        impact = linear_impact(volume, adv, eta)
    else:
        impact = sqrt_impact(volume, adv, daily_vol, eta)

    cost = price * volume * impact  # 冲击导致的额外成本
    return {
        'impact_bps': impact * 10000,  # 基点
        'total_cost': cost,
        'cost_per_share': cost / volume
    }
```

### 2.5 与搜广推的跨领域类比

搜广推中的竞价/出价策略与交易执行有深层相似性：

| 量化交易 | 搜广推竞价 | 共同本质 |
|----------|-----------|---------|
| 市场冲击（自身交易推动价格） | 竞价环境（自身出价影响竞争格局） | 自身行为改变环境 |
| 最优执行（拆单、TWAP/VWAP） | 预算分配（pacing，分时段投放） | 资源在时间维度上的最优分配 |
| 滑点控制 | ROI 约束 | 成本控制 |
| Almgren-Chriss 最优轨迹 | 在线学习出价策略 | 动态优化框架 |

这种类比在面试中展现**跨领域迁移能力**，是加分项。

---

## 3. 仓位管理

### 3.1 Kelly 准则

**最优下注比例**（使长期财富增长率最大化）：

$$f^* = \frac{bp - q}{b}$$

- b：赔率（盈利/亏损比），p：胜率，q = 1-p：败率
- 对于连续收益分布：$f^* = \frac{\mu}{\sigma^2}$（均值/方差）

**半 Kelly（Half-Kelly）**：
- 实践中用 f*/2，原因：
  - Kelly 假设已知真实概率，实际中是估计值
  - 满仓 Kelly 波动极大，心理压力大
  - 半 Kelly 牺牲约 25% 长期增长率，但降低约 50% 波动

> **面试常问**：Kelly 准则的推导思路？
> 对数财富的期望最大化：$E[\ln(1 + f \cdot X)]$ 对 f 求导令其为 0。连续情况下，二阶泰勒展开得到 $f^* \approx \mu/\sigma^2$。

### 3.2 风险平价 (Risk Parity)

**核心思想**：每个资产对组合总风险的**贡献**相等，而非权重相等。

简化版（波动率倒数加权）：

$$w_i = \frac{1/\sigma_i}{\sum_j 1/\sigma_j}$$

- 波动率大的资产分配少，波动率小的分配多
- 经典案例：桥水 All Weather 基金
- 缺点：未考虑相关性（完整版需要考虑协方差矩阵）

### 3.3 均值方差优化 (Markowitz)

$$\min_w \quad w^T \Sigma w \quad \text{s.t.} \quad w^T \mu = \mu_{target}, \quad \sum w_i = 1$$

**有效前沿**：所有最优风险-收益组合构成的曲线。

**实践中的问题**：
- 协方差矩阵估计误差大，导致权重极端（某些资产权重为负或极大）
- 对输入参数（均值、方差）极度敏感，被称为"error maximizer"
- 缓解方法：加正则化约束、缩减估计(shrinkage)、Black-Litterman

### 3.4 Black-Litterman 模型

**核心思想**：将**市场均衡隐含收益**作为先验，结合投资者的**主观观点**，用贝叶斯方法得到后验收益估计。

$$E[R] = [(\tau\Sigma)^{-1} + P^T\Omega^{-1}P]^{-1} [(\tau\Sigma)^{-1}\Pi + P^T\Omega^{-1}Q]$$

- Π：市场均衡隐含超额收益（从市值权重反推）
- P：观点矩阵，Q：观点收益向量，Ω：观点不确定性
- 优势：避免了 Markowitz 的极端权重问题，输出更稳定

### 3.5 Python 实现

```python
def kelly_fraction(win_rate, win_loss_ratio, half=True):
    """
    Kelly 准则计算最优仓位比例

    Args:
        win_rate: 胜率
        win_loss_ratio: 盈亏比 (avg_win / avg_loss)
        half: 是否使用半Kelly（推荐True）
    """
    b = win_loss_ratio
    p = win_rate
    q = 1 - p
    f = (b * p - q) / b
    f = max(f, 0)  # 不做负期望的赌注
    return f / 2 if half else f


def risk_parity_weights(cov_matrix: np.ndarray) -> np.ndarray:
    """
    风险平价权重（简化版：波动率倒数加权）
    完整版需要迭代求解使每个资产的风险贡献相等
    """
    vols = np.sqrt(np.diag(cov_matrix))
    inv_vols = 1.0 / vols
    weights = inv_vols / inv_vols.sum()
    return weights


def risk_parity_full(cov_matrix: np.ndarray, budget: np.ndarray = None,
                     max_iter: int = 1000, tol: float = 1e-8) -> np.ndarray:
    """
    完整风险平价（考虑相关性），迭代求解
    使得每个资产的边际风险贡献(MRC)相等
    """
    n = cov_matrix.shape[0]
    if budget is None:
        budget = np.ones(n) / n  # 等风险预算

    w = np.ones(n) / n  # 初始等权
    for _ in range(max_iter):
        sigma_p = np.sqrt(w @ cov_matrix @ w)
        mrc = (cov_matrix @ w) / sigma_p  # 边际风险贡献
        rc = w * mrc  # 风险贡献

        # 目标：rc_i / sum(rc) = budget_i
        w_new = budget / mrc
        w_new = w_new / w_new.sum()

        if np.max(np.abs(w_new - w)) < tol:
            break
        w = w_new

    return w
```

参见 [[concepts/portfolio_optimization.md]] 了解组合优化的完整理论。

---

## 4. 止损策略与尾部风险

### 4.1 止损策略对比

| 类型 | 规则 | 优点 | 缺点 |
|------|------|------|------|
| 固定止损 | 亏损达到固定比例（如-5%）就平仓 | 简单明确 | 不考虑波动率，震荡市频繁触发 |
| 追踪止损 | 从最高点回撤固定比例（如-3%）平仓 | 保护利润 | 趋势末端可能卖在回调底部 |
| 波动率止损 | 止损线 = 入场价 - k × ATR | 自适应波动环境 | 需要调参，ATR 参数敏感 |
| 时间止损 | 持仓超过 N 天未盈利则平仓 | 释放资金占用 | 可能错过后续行情 |

**ATR 止损示例**：
```python
def atr_stop_loss(entry_price, atr, multiplier=2.0, direction='long'):
    """基于 ATR 的波动率止损"""
    if direction == 'long':
        return entry_price - multiplier * atr
    else:
        return entry_price + multiplier * atr
```

### 4.2 尾部风险管理

**肥尾分布**：金融收益的尾部远比正态分布厚。实证表明，3σ 以上的事件出现频率是正态分布预测的数倍。

关键认知：
- **正态分布下**：日跌幅超过 3σ 的概率约 0.13%（约3年一次）
- **实际市场**：这种事件几乎每年都有，且幅度可能远超 3σ
- 策略必须在**肥尾假设**下做风控，不能用正态分布计算止损

### 4.3 压力测试

| 类型 | 方法 | 示例 |
|------|------|------|
| 历史情景 | 用历史极端事件的市场数据重演 | 2008金融危机、2015股灾、2020疫情 |
| 假设情景 | 假设极端但可能的宏观冲击 | 利率暴涨200bp、汇率贬值10% |
| 反向压力测试 | 反推"什么情况下会爆仓" | 从最大可承受亏损反推所需市场变动 |

```python
def stress_test_portfolio(weights, returns_df, scenarios):
    """
    对组合进行历史情景压力测试

    Args:
        weights: 资产权重
        returns_df: 历史收益率 DataFrame
        scenarios: dict，{情景名: (start_date, end_date)}
    """
    results = {}
    for name, (start, end) in scenarios.items():
        scenario_returns = returns_df.loc[start:end]
        portfolio_returns = (scenario_returns * weights).sum(axis=1)
        cum_return = (1 + portfolio_returns).prod() - 1
        max_dd = compute_max_drawdown(portfolio_returns)
        results[name] = {
            'cumulative_return': cum_return,
            'max_drawdown': max_dd,
            'worst_day': portfolio_returns.min(),
            'duration_days': len(scenario_returns)
        }
    return pd.DataFrame(results).T

def compute_max_drawdown(returns):
    cum = (1 + returns).cumprod()
    return -(cum / cum.cummax() - 1).min()
```

---

## 5. 面试高频考点

### Q1: VaR vs CVaR 的区别和优劣？

**答题框架**：

| 维度 | VaR | CVaR |
|------|-----|------|
| 含义 | 最大损失的阈值 | 超过阈值后的平均损失 |
| 数学性质 | 不满足次可加性 | 满足一致性风险度量公理 |
| 尾部信息 | 只有一个分位点，忽略尾部分布形状 | 捕捉尾部平均损失程度 |
| 优化性质 | 非凸，不利于优化 | 凸函数，可以高效优化 |
| 实务使用 | 监管要求（Basel III），但逐渐被 CVaR 补充 | 更适合内部风控 |

**加分回答**：用一个反例说明 VaR 不满足次可加性——两个独立的极端亏损头寸，分别的 VaR 可能都为 0，但合并后 VaR 不为 0。

### Q2: Kelly 准则的推导和实际应用注意事项？

**推导**（离散情况）：
1. 假设每轮以比例 f 下注，胜率 p，赔率 b
2. 经过 n 轮后，财富 $W_n = W_0 \times (1+bf)^{np} \times (1-f)^{nq}$
3. 最大化对数增长率 $G(f) = p\ln(1+bf) + q\ln(1-f)$
4. 对 f 求导令为 0：$f^* = \frac{bp-q}{b}$

**实际注意事项**：
- 真实胜率和赔率是**估计值**，有误差 → 用半 Kelly
- 多资产同时下注时需要考虑**相关性**
- Kelly 优化的是**长期几何增长率**，短期波动可能很大
- 不适合有**杠杆约束**或**流动性约束**的场景

### Q3: 如何衡量一个策略的风险？

**答题思路**（多维度评估）：

```
1. 收益维度：年化收益率、Alpha
2. 波动维度：年化波动率、下行波动率
3. 风险调整：Sharpe、Sortino、Calmar
4. 极端风险：VaR、CVaR、最大回撤、回撤持续时间
5. 相对指标：IR、跟踪误差、Beta
6. 压力测试：历史情景、假设情景
7. 风险分解：因子暴露、行业集中度
```

**关键提醒**：单一指标都有盲区，必须多指标综合评估。Sharpe 高但回撤大说明有尾部风险，Calmar 高但 Sharpe 低说明收益来源可能不稳定。

> **面试常问**：如果只能看一个指标衡量策略好坏，你选哪个？
> 最稳妥的回答是 **Sharpe Ratio**（风险调整后收益），但要补充它的局限性：不区分上行/下行波动，不反映尾部风险。如果策略有明显肥尾特征，应该同时看 **CVaR** 和 **最大回撤**。

---

## 交叉引用

- [[concepts/risk_measures.md]] — 风险度量公理化体系
- [[quant-finance/synthesis/strategy_development.md]] — 策略开发框架
- [[quant-finance/synthesis/factor_investing.md]] — 因子投资与风险因子分解
- [[quant-finance/synthesis/ml_in_quant.md]] — ML模型中的风险控制
