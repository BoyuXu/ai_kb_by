# 量化数学基础 — 概率统计、随机过程与时间序列

> 量化金融的数学工具箱。重点在直觉理解和实际应用，不追求数学严谨性。
> 关联：[[factor_investing]] | [[ml_in_quant]] | [[strategy_development]]

---

## 1. 概率统计基础

### 1.1 关键分布及金融应用

| 分布 | 核心特征 | 金融应用场景 |
|------|----------|-------------|
| **正态分布** N(μ,σ²) | 对称钟形，由均值和方差完全确定 | 收益率建模（短期近似）、VaR计算、因子收益分布 |
| **对数正态分布** | ln(X)~N(μ,σ²)，X>0 | 股价建模（GBM假设下股价服从对数正态）|
| **t分布** | 比正态分布厚尾，自由度越小尾越厚 | 小样本假设检验、因子IC显著性检验 |
| **卡方分布** χ²(k) | k个标准正态平方和 | 方差检验、协方差矩阵估计的检验 |
| **F分布** | 两个卡方分布之比 | 多因子回归的联合显著性检验（F检验）|

> **面试常问**：为什么金融收益率不是严格正态分布？
> 答：真实收益率有**厚尾**（极端事件比正态预测的多）和**偏度**（涨跌不对称）。正态只是一阶近似，实务中常用t分布或做尾部修正。

### 1.2 假设检验：因子显著性

核心逻辑：**先假设因子没用（H₀），看数据有多大概率推翻这个假设。**

- **t检验**：单个因子的IC均值是否显著不为零
  - t统计量 = IC均值 / (IC标准差 / √n)
  - |t| > 2 大致对应 p < 0.05（95%置信度）
  - 量化中常用标准：|t| > 2.0，IC_IR = IC均值/IC标准差 > 0.5

- **F检验**：多个因子联合是否显著（多因子回归中）
  - 比较有因子模型 vs 无因子模型的解释力提升是否显著

### 1.3 贝叶斯推断：先验到后验

**直觉**：不是非黑即白地判断"因子有效/无效"，而是不断用新数据更新信念。

$$P(\theta | D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}$$

- 后验 ∝ 似然 × 先验
- **贝叶斯收缩估计**：把样本估计值向先验均值"收缩"，减少过拟合
  - 协方差矩阵收缩：Ledoit-Wolf 方法，把样本协方差矩阵向结构化先验（如对角矩阵）收缩
  - 因子收益预测：把因子历史IC向零收缩，避免过度相信噪声大的因子

### 1.4 Python代码：因子IC的t检验

```python
import numpy as np
from scipy import stats

def factor_ic_ttest(ic_series: np.ndarray) -> dict:
    """
    对因子IC序列做单样本t检验
    H0: IC均值 = 0（因子无预测能力）
    """
    n = len(ic_series)
    ic_mean = np.mean(ic_series)
    ic_std = np.std(ic_series, ddof=1)

    # t统计量
    t_stat = ic_mean / (ic_std / np.sqrt(n))
    # 双尾p值
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    # IC_IR (年化信息比率，假设月频)
    ic_ir = ic_mean / ic_std * np.sqrt(12)

    return {
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        't_stat': t_stat,
        'p_value': p_value,
        'ic_ir': ic_ir,
        'significant': abs(t_stat) > 2.0
    }

# 示例
np.random.seed(42)
ic_values = np.random.normal(0.03, 0.15, size=60)  # 60个月的IC
result = factor_ic_ttest(ic_values)
print(f"IC均值={result['ic_mean']:.4f}, t={result['t_stat']:.2f}, p={result['p_value']:.4f}")
```

---

## 2. 随机过程

### 2.1 布朗运动 (Wiener Process)

**直觉**：喝醉的人在一维直线上走路。每一步方向完全随机，步长服从正态分布。

数学定义 — W(t) 满足：
1. W(0) = 0
2. 增量独立：W(t) - W(s) 与 W(s) - W(u) 独立（u < s < t）
3. 增量正态：W(t) - W(s) ~ N(0, t-s)
4. 路径连续但处处不可微（"无限锯齿"）

**关键性质**：
- E[W(t)] = 0（期望不动）
- Var[W(t)] = t（方差随时间线性增长 → 不确定性越来越大）
- W(t) 可以取负值 → 不适合直接建模股价

### 2.2 几何布朗运动 (GBM)

**为什么股价要用GBM而不是BM？**

布朗运动可以取负值，但股价不能为负。解决方案：让**对数价格**做布朗运动。

$$dS = \mu S \, dt + \sigma S \, dW$$

等价地：$d(\ln S) = (\mu - \frac{\sigma^2}{2}) dt + \sigma \, dW$

这意味着：
- 股价 S(t) 服从对数正态分布 → **保证 S > 0**
- 对数收益率 ln(S(t)/S(0)) 服从正态分布 → 方便统计建模
- μ 是漂移率（期望收益），σ 是波动率

> **面试常问**：布朗运动和几何布朗运动的区别？
> - BM：加性噪声，dX = μdt + σdW，可以为负
> - GBM：乘性噪声，dS = μSdt + σSdW，始终为正
> - BM描述绝对变化，GBM描述百分比变化
> - 股价用GBM因为：(1)非负 (2)对数收益率正态 (3)收益率与价格水平成比例

### 2.3 Ito引理

**直觉**：微积分中链式法则的随机版本。普通微积分 df = f'dx，但随机微积分中 (dW)² = dt（不能忽略二阶项）。

设 f(S, t) 是 S 和 t 的函数，S 满足 dS = μSdt + σSdW，则：

$$df = \left(\frac{\partial f}{\partial t} + \mu S \frac{\partial f}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 f}{\partial S^2}\right)dt + \sigma S \frac{\partial f}{\partial S} \, dW$$

**经典应用**：令 f = ln(S)，代入Ito引理：
- ∂f/∂S = 1/S，∂²f/∂S² = -1/S²，∂f/∂t = 0
- d(ln S) = (μ - σ²/2)dt + σdW

这就是为什么GBM的解析解中漂移项是 μ - σ²/2 而不是 μ（Ito修正项）。

### 2.4 Python代码：模拟GBM路径

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths=5):
    """
    模拟几何布朗运动路径
    S0: 初始价格
    mu: 年化漂移率
    sigma: 年化波动率
    T: 时间长度（年）
    """
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)

    # 生成随机增量
    dW = np.random.normal(0, np.sqrt(dt), size=(n_paths, n_steps))

    # 对数价格递推（精确离散化，非欧拉近似）
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * dW
    log_S = np.zeros((n_paths, n_steps + 1))
    log_S[:, 0] = np.log(S0)
    log_S[:, 1:] = np.log(S0) + np.cumsum(log_returns, axis=1)

    S = np.exp(log_S)
    return t, S

# 模拟
t, paths = simulate_gbm(S0=100, mu=0.08, sigma=0.2, T=1, n_steps=252, n_paths=10)
for i in range(paths.shape[0]):
    plt.plot(t, paths[i], alpha=0.6)
plt.title('GBM模拟路径 (μ=8%, σ=20%)')
plt.xlabel('时间 (年)')
plt.ylabel('价格')
plt.show()
```

---

## 3. 时间序列分析

### 3.1 平稳性

**为什么重要**：非平稳序列做回归会产生**伪回归**（spurious regression）——两个毫无关系的随机游走回归也能得到很高的R²。

**平稳性定义**（弱平稳）：
- 均值恒定：E[X(t)] = μ（不随时间变化）
- 自协方差只依赖时间差：Cov(X(t), X(t+h)) = γ(h)

**ADF检验**（Augmented Dickey-Fuller）：
- H₀：序列有单位根（非平稳）
- p < 0.05 → 拒绝H₀ → 序列平稳
- 股价通常非平稳，收益率通常平稳

### 3.2 ACF/PACF

- **ACF**（自相关函数）：X(t) 和 X(t-k) 的相关性（包含中间变量的间接影响）
- **PACF**（偏自相关函数）：X(t) 和 X(t-k) 的相关性（去掉中间变量影响后）

判断模型阶数：
| 模型 | ACF 特征 | PACF 特征 |
|------|----------|-----------|
| AR(p) | 拖尾衰减 | p阶后截尾 |
| MA(q) | q阶后截尾 | 拖尾衰减 |
| ARMA(p,q) | 拖尾 | 拖尾 |

### 3.3 ARIMA

**思路**：先差分让序列平稳，再用 AR+MA 建模。

ARIMA(p, d, q)：
- d：差分阶数（通常1次差分就够）
- p：AR阶数（当前值 = 过去p个值的线性组合）
- q：MA阶数（当前值 = 过去q个噪声的线性组合）

### 3.4 协整 (Cointegration)

**直觉**：两只狗被同一根绳子拴着散步。各自的路径是随机游走（非平稳），但两者之间的距离是平稳的。

$$Y_t = \beta X_t + \epsilon_t, \quad \epsilon_t \sim \text{平稳}$$

**配对交易应用**：
1. 找到协整的股票对（如可口可乐和百事可乐）
2. 计算价差（spread）= Y - βX
3. 价差偏离均值 → 开仓；回归均值 → 平仓

> **面试常问**：相关性和协整的区别？
> - 相关性：短期收益率的同步性（可以突然消失）
> - 协整：长期价格水平的均衡关系（更稳定）
> - 两只股票可以高相关但不协整，也可以协整但短期相关性不高

### 3.5 GARCH — 波动率聚集

**现象**：大波动后面往往跟着大波动，小波动跟着小波动（volatility clustering）。

GARCH(1,1)模型：

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

- ω：长期方差的基线
- α：昨天冲击（收益率平方）对今天波动率的影响
- β：昨天波动率的持续性
- α + β < 1 保证平稳；α + β 接近 1 表示波动率高度持续

**应用**：期权定价中的波动率预测、风险管理中的VaR计算。

### 3.6 Python代码：ARIMA + GARCH

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from arch import arch_model

def time_series_analysis(returns: pd.Series):
    """时间序列分析流水线：ADF检验 → ARIMA → GARCH"""

    # Step 1: ADF平稳性检验
    adf_stat, p_value, _, _, _, _ = adfuller(returns.dropna())
    print(f"ADF统计量: {adf_stat:.4f}, p值: {p_value:.4f}")
    print(f"结论: {'平稳' if p_value < 0.05 else '非平稳，需差分'}")

    # Step 2: ARIMA 拟合均值方程
    model_arima = ARIMA(returns.dropna(), order=(1, 0, 1))  # AR(1)+MA(1)
    result_arima = model_arima.fit()
    print(f"\nARIMA(1,0,1) AIC: {result_arima.aic:.2f}")

    # Step 3: GARCH 拟合波动率
    residuals = result_arima.resid
    model_garch = arch_model(residuals, vol='Garch', p=1, q=1)
    result_garch = model_garch.fit(disp='off')
    print(f"\nGARCH(1,1) 参数:")
    print(f"  omega={result_garch.params['omega']:.6f}")
    print(f"  alpha={result_garch.params['alpha[1]']:.4f}")
    print(f"  beta={result_garch.params['beta[1]']:.4f}")

    # 波动率预测
    forecast = result_garch.forecast(horizon=5)
    print(f"\n未来5天波动率预测: {np.sqrt(forecast.variance.values[-1]) * np.sqrt(252)}")

    return result_arima, result_garch

# 示例
np.random.seed(42)
returns = pd.Series(np.random.normal(0.0005, 0.02, 500))
time_series_analysis(returns)
```

### 3.7 与搜广推序列建模的异同

| 维度 | 量化时间序列 | 搜广推序列建模 |
|------|-------------|---------------|
| **数据** | 价格/收益率，连续值 | 用户行为序列，离散事件 |
| **平稳性** | 核心假设，必须检验 | 通常不要求（用户兴趣本身就在漂移）|
| **模型** | ARIMA/GARCH（统计模型为主）| Transformer/GRU（深度学习为主）|
| **目标** | 预测未来值或波动率 | 预测下一个点击/购买 |
| **因果性** | 严格要求不能用未来信息 | 同样严格（但更容易犯错）|
| **共同点** | 都关注时序依赖、都需要处理噪声、都面临过拟合风险 | |

---

## 4. 蒙特卡洛模拟

### 4.1 基本原理

**直觉**：用大量随机实验来近似理论期望值。掷骰子掷一万次，平均值就很接近3.5。

$$E[f(X)] \approx \frac{1}{N}\sum_{i=1}^{N} f(X_i)$$

误差以 O(1/√N) 的速度下降 → 精度翻倍需要4倍样本量。

### 4.2 期权定价：蒙特卡洛方法

**欧式看涨期权**：到期日payoff = max(S(T) - K, 0)

步骤：
1. 模拟 N 条GBM路径，得到 N 个 S(T)
2. 计算每条路径的payoff
3. 对payoff取均值并折现：C = e^{-rT} × mean(payoff)

### 4.3 VaR计算

- **历史模拟法**：直接用历史收益率排序，取分位数
- **蒙特卡洛法**：假设收益率分布（如GBM），模拟大量路径，取分位数
- 蒙特卡洛法更灵活，可以处理复杂的非线性组合（如含期权的组合）

### 4.4 Python代码：蒙特卡洛期权定价

```python
import numpy as np

def monte_carlo_option_price(S0, K, r, sigma, T, n_sims=100000, option_type='call'):
    """
    蒙特卡洛法计算欧式期权价格
    S0: 当前价格, K: 行权价, r: 无风险利率
    sigma: 波动率, T: 到期时间(年)
    """
    # 模拟到期日价格（直接用解析解，不需要逐步模拟）
    Z = np.random.standard_normal(n_sims)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # 计算payoff
    if option_type == 'call':
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)

    # 折现
    price = np.exp(-r * T) * np.mean(payoff)

    # 标准误（衡量模拟精度）
    se = np.exp(-r * T) * np.std(payoff) / np.sqrt(n_sims)

    return {'price': price, 'std_error': se, '95%_CI': (price - 1.96*se, price + 1.96*se)}

# 示例：S0=100, K=105, r=5%, σ=20%, T=1年
result = monte_carlo_option_price(100, 105, 0.05, 0.2, 1.0)
print(f"看涨期权价格: {result['price']:.4f} ± {result['std_error']:.4f}")
print(f"95%置信区间: ({result['95%_CI'][0]:.4f}, {result['95%_CI'][1]:.4f})")

# 对比 Black-Scholes 解析解
from scipy.stats import norm
d1 = (np.log(100/105) + (0.05 + 0.5*0.04)*1) / (0.2*1)
d2 = d1 - 0.2
bs_price = 100*norm.cdf(d1) - 105*np.exp(-0.05)*norm.cdf(d2)
print(f"BS解析解: {bs_price:.4f}")
```

---

## 5. PCA/SVD在因子分析中的应用

### 5.1 主成分分析提取市场因子

**动机**：几百个因子之间高度相关（多重共线性），直接回归不稳定。

**PCA思路**：
1. 对因子收益率矩阵（T×N，T个时间点，N个因子）做PCA
2. 前几个主成分 = 市场中最主要的驱动力（通常PC1≈市场因子，PC2≈价值/成长）
3. 用主成分替代原始因子做回归 → 消除共线性

### 5.2 降维处理多重共线性

**协方差矩阵估计**中的应用：
- 500只股票 → 500×500协方差矩阵 → 需要估计125,250个参数
- 样本协方差矩阵在 T < N 时甚至不可逆
- PCA降维：只保留前K个主成分，剩余视为噪声

### 5.3 Python代码：因子矩阵PCA

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def factor_pca_analysis(factor_returns: np.ndarray, factor_names: list, n_components=5):
    """
    对因子收益率矩阵做PCA分析
    factor_returns: shape (T, N)，T个时间点，N个因子
    """
    pca = PCA(n_components=n_components)
    pca.fit(factor_returns)

    # 解释方差比例
    print("各主成分解释方差比例:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {ratio:.2%}")
    print(f"  前{n_components}个主成分累计: {sum(pca.explained_variance_ratio_):.2%}")

    # 主成分载荷（哪些因子贡献最大）
    print(f"\nPC1 载荷 Top 5:")
    loadings = pca.components_[0]
    top_idx = np.argsort(np.abs(loadings))[::-1][:5]
    for idx in top_idx:
        print(f"  {factor_names[idx]}: {loadings[idx]:.4f}")

    # 转换为主成分
    pc_returns = pca.transform(factor_returns)  # shape (T, n_components)

    return pca, pc_returns

# 示例
np.random.seed(42)
n_days, n_factors = 252, 20
factor_names = [f'Factor_{i+1}' for i in range(n_factors)]

# 构造有结构的因子收益（前3个共享市场因子）
market = np.random.normal(0, 0.01, (n_days, 1))
factor_returns = np.random.normal(0, 0.005, (n_days, n_factors))
factor_returns[:, :5] += market  # 前5个因子与市场高度相关

pca, pc_returns = factor_pca_analysis(factor_returns, factor_names)
```

### 5.4 与搜广推中的降维方法对比

| 维度 | 量化因子PCA | 搜广推Embedding降维 |
|------|-----------|-------------------|
| **输入** | 因子收益率矩阵（连续值）| 用户/物品ID（离散高维）|
| **方法** | PCA/SVD（线性）| Embedding层（非线性）|
| **目标** | 提取潜在风险因子、消除共线性 | 学习语义表示、捕获交互 |
| **可解释性** | 高（主成分可对应经济含义）| 低（Embedding维度无明确含义）|
| **在线更新** | 通常离线批量重算 | 支持在线学习 |
| **共同思想** | 都是把高维空间映射到低维空间，保留核心信息，去除噪声 | |

> 关联阅读：[[embedding_everywhere]] 中的降维思想、[[sequence_modeling_evolution]] 中的时序建模

---

## 附录：面试高频问题速查

1. **布朗运动 vs 几何布朗运动** → 2.2节
2. **为什么需要Ito引理？** → 随机微积分中 (dW)² = dt 不能忽略，多出 σ²/2 修正项
3. **平稳性为什么重要？** → 非平稳做回归是伪回归（3.1节）
4. **相关性 vs 协整** → 短期同步性 vs 长期均衡关系（3.4节）
5. **蒙特卡洛精度怎么提高？** → N翻4倍精度才翻倍；或用方差缩减技术（对偶变量、控制变量）
6. **PCA提取的第一个主成分通常是什么？** → 市场因子（所有股票的共同运动方向）
