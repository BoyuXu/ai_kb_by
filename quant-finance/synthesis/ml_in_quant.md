# 机器学习在量化投资中的应用

> ML 在量化中的应用与搜广推有大量交叉，但金融数据的低信噪比和非平稳性带来了独特挑战。搜广推工程师转量化，最大的坑不是模型，是数据和评估。

## 1. 搜广推 vs 量化：ML 应用对比

先建立直觉——搜广推做 ML 和量化做 ML，哪些相同，哪些不同：

| 维度 | 搜广推 | 量化 |
|------|--------|------|
| **信噪比** | 高（点击/转化信号明确） | 极低（收益率噪声大） |
| **数据量** | 海量（日均亿级样本） | 有限（A 股 ~5000 只，日频数据量小） |
| **标签** | 明确（点击/购买） | 模糊（未来收益？涨跌？） |
| **特征** | 用户画像+物品属性+上下文 | 因子（价量/基本面/另类） |
| **评估** | A/B 测试（在线） | 回测（离线，且容易过拟合） |
| **时效性** | 实时推荐 | 日频/周频换仓 |
| **分布变化** | 用户兴趣漂移 | 市场 regime 切换（更剧烈） |
| **模型复杂度** | 可以很深（DNN/Transformer） | 越简单越好（过拟合风险大） |

**核心差异**：搜广推的 ML 是「有充足数据去拟合复杂模式」，量化的 ML 是「在极低信噪比下提取微弱信号，同时不被噪声骗」。

## 2. 特征工程（因子构建）

### 2.1 搜广推经验可直接复用的部分 🔄

| 技术 | 搜广推用法 | 量化用法 |
|------|-----------|---------|
| **时序特征** | 用户最近 N 次行为统计 | 最近 N 日收益率/波动率/换手率统计 |
| **交叉特征** | user×item | 股票×行业、股票×宏观 |
| **Embedding** | 物品/用户 Embedding | 股票 Embedding（基于持仓/关联图） |
| **特征选择** | 信息增益/置换重要性 | IC/IR/MDI/MDA（AFML方法） |
| **特征标准化** | z-score/min-max | 截面 z-score（每天在全部股票中标准化） |

### 2.2 量化特有的特征工程

**截面标准化**是量化特征工程的核心：

```python
# 搜广推：直接用原始特征或全局标准化
feature_normalized = (feature - global_mean) / global_std

# 量化：必须做截面标准化（每天在所有股票中标准化）
def cross_sectional_normalize(df):
    """每个交易日，在所有股票中做 z-score"""
    return df.groupby('date').transform(lambda x: (x - x.mean()) / x.std())

# 为什么？因为我们关心的是相对排名，不是绝对值
# PE=20 在银行股里很贵，在科技股里很便宜
```

**去极值**（winsorize）：金融数据极端值多，通常在 ±3σ 或 ±5% 分位数处截断。

**行业/市值中性化**：去除行业和市值的影响，提取纯因子信号。

```python
# 因子中性化：回归掉行业和市值的影响
import statsmodels.api as sm
def neutralize(factor, industry_dummies, log_market_cap):
    X = pd.concat([industry_dummies, log_market_cap], axis=1)
    X = sm.add_constant(X)
    residuals = sm.OLS(factor, X).fit().resid
    return residuals  # 这就是中性化后的因子
```

## 3. 标签构建

这是量化 ML 最关键的设计决策之一，搜广推里没有对应概念。

### 3.1 简单标签

```python
# 未来 N 天收益率
label = returns.shift(-N)  # 前移 N 天

# 二分类：涨/跌
label = (returns.shift(-5) > 0).astype(int)
```

**问题**：收益率噪声大、分布不稳定。

### 3.2 Triple Barrier Method（AFML 核心）

López de Prado 提出的标签方法，解决了固定时间窗口标签的问题：

```
上界 barrier（止盈）─────────────────── +τ
                       /\    价格路径
                      /  \  /
起始价格 ─────────────/────\/─────────
                                    时间 barrier（超时）
下界 barrier（止损）─────────────────── -τ
```

- 价格先碰上界 → 标签 = 1（做多获利）
- 价格先碰下界 → 标签 = -1（做多亏损）
- 超时都没碰到 → 标签 = 0（无明显方向）

**优点**：标签质量高，自然处理了持仓时间和止损逻辑。

### 3.3 Meta-labeling

先用一级模型判断方向（多/空），再用二级模型决定下注大小（sizing）。

一级模型可以很简单（规则、因子），二级模型用 ML。这比直接用 ML 预测收益率更稳健。

## 4. 模型选择

### 4.1 树模型是量化 ML 的主力 🔄

```
搜广推：   LR → FM → DeepFM → DIN → Transformer
量化：     LR → Ridge → RF → XGBoost/LightGBM → (谨慎的) NN
```

**为什么量化偏爱树模型？**

1. **天然处理非线性和交互**：不需要手动构造交叉特征
2. **对噪声鲁棒**：树的分裂天然有正则化效果
3. **可解释性**：特征重要性直接可得
4. **数据量要求低**：适合金融小数据场景

```python
import lightgbm as lgb

# 量化选股模型（与搜广推 CTR 模型结构非常相似！）
params = {
    'objective': 'regression',  # 或 'binary' 做涨跌分类
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,  # 类似 dropout，防过拟合
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'min_data_in_leaf': 50,  # 金融数据少，这个要调大
}
model = lgb.train(params, train_set, valid_sets=[valid_set],
                  num_boost_round=500, callbacks=[lgb.early_stopping(50)])
```

### 4.2 深度学习：谨慎使用

搜广推里 DNN 是标配，但量化里要非常谨慎：

| DL 方法 | 适用场景 | 风险 |
|---------|---------|------|
| MLP | 因子非线性组合 | 容易过拟合，需要强正则化 |
| LSTM/GRU | 时序建模 | 学术上不太 work |
| Transformer | 长序列时序 | PatchTST 有一定效果 |
| GNN | 股票关系建模 | 研究前沿，实战少 |
| TabNet | 表格数据 | 可解释性好于 MLP |

**经验法则**：如果你的因子库不超过 100 个、股票池不超过 5000 只、数据不超过 20 年，Tree > NN。

### 4.3 线性模型仍然有用

```python
# Ridge 回归：简单但有效
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# 好处：可解释、稳定、不容易过拟合
# 坏处：无法捕捉非线性
```

**面试常问**：为什么不用更复杂的模型？
→ 金融数据信噪比低，复杂模型更容易拟合噪声而非信号。Occam's razor 在量化中格外重要。

## 5. 交叉验证：金融数据的特殊处理

### 5.1 为什么不能用普通 K-Fold？

```
普通 K-Fold（搜广推可用）：
[1][2][3][4][5]  → fold 3 是 test，其余是 train
                    问题：fold 2 和 fold 4 的数据与 fold 3 高度相关（自相关）
                    → 信息泄露 → 过拟合

金融 Purged K-Fold：
[1][gap][3][gap][5]  → fold 3 是 test
   ↑purge   ↑embargo
```

### 5.2 实现

```python
from sklearn.model_selection import TimeSeriesSplit

# 基础时序 split（sklearn 自带）
tscv = TimeSeriesSplit(n_splits=5)

# Purged K-Fold（需要自己实现或用 AFML 库）
class PurgedKFold:
    def __init__(self, n_splits, purge_days=5, embargo_pct=0.01):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_pct = embargo_pct

    def split(self, X, y, groups=None):
        # 在 train/test 边界留出 purge_days 的间隔
        # test 之后 embargo_pct 的数据不进入下一折的 train
        ...
```

## 6. 特征重要性（Feature Importance）

### 6.1 搜广推可复用方法 🔄

| 方法 | 说明 | 量化注意事项 |
|------|------|-------------|
| MDI（Mean Decrease Impurity） | 树模型内置 | 有偏，偏向连续/高基数特征 |
| MDA（Mean Decrease Accuracy） | 置换特征后看准确率下降 | AFML 推荐，更可靠 |
| SHAP | Shapley 值分解 | 计算慢但最准确 |
| 单因子 IC 测试 | 量化独有 | 最直接 |

### 6.2 Clustered Feature Importance（AFML）

因子之间有共线性 → 单个因子的重要性被分散到一组相关因子上。

解决方案：先聚类相关因子，再在聚类层面做置换检验。

## 7. 实战 Pipeline

```python
# 量化 ML 选股 pipeline（端到端）

# Step 1: 数据准备
raw_factors = load_factors()  # 加载因子数据
prices = load_prices()         # 加载价格数据

# Step 2: 因子预处理
factors = (raw_factors
    .pipe(winsorize, limits=(0.01, 0.01))    # 去极值
    .pipe(cross_sectional_normalize)          # 截面标准化
    .pipe(neutralize, ['industry', 'log_cap']) # 中性化
    .pipe(fillna_by_industry_median)          # 缺失值填充
)

# Step 3: 标签构建
labels = compute_forward_returns(prices, horizon=5)  # 未来5日收益
# 或用 triple barrier method

# Step 4: 训练（Purged Walk-Forward）
predictions = []
for train_idx, test_idx in purged_walk_forward(factors, purge=5):
    model = lgb.train(params, factors.iloc[train_idx], labels.iloc[train_idx])
    pred = model.predict(factors.iloc[test_idx])
    predictions.append(pred)

# Step 5: 构建组合
scores = pd.concat(predictions)
portfolio = scores.groupby('date').apply(
    lambda x: x.nlargest(50)  # 每天选 top 50
)

# Step 6: 回测评估
backtest_result = run_backtest(portfolio, prices, tc=0.003)
print(f"Sharpe: {backtest_result.sharpe:.2f}")
print(f"MaxDD: {backtest_result.max_drawdown:.2%}")
print(f"Annual Return: {backtest_result.annual_return:.2%}")
```

## 8. 面试常见问题

**Q: 搜广推的 ML 经验如何迁移到量化？**
→ 特征工程方法论、树模型调参、Embedding 思想可直接迁移。但需要重新理解标签构建、交叉验证、过拟合在金融中的特殊性。

**Q: 为什么量化很少用深度学习？**
→ 数据量小、信噪比低、分布非平稳。DL 需要大数据才能发挥优势，金融数据不满足这个前提。但在高频和另类数据（NLP/图像）场景，DL 有优势。

**Q: 如何处理金融数据的非平稳性？**
→ 分数阶差分（fractional differentiation，AFML Ch.5）：在保留记忆性的同时实现平稳化。比直接取收益率（一阶差分）保留更多信息。

**Q: 你的模型 IC=0.05，这个好吗？**
→ 在量化里这已经不错了。关键看 ICIR（稳定性），以及多因子组合后的效果。单因子 IC=0.05 如果很稳定（ICIR>0.5），比 IC=0.1 但波动大要好。

**Q: 如何防止回测过拟合？**
→ 参见 [[strategy_development.md]] 中的过拟合检测部分。核心：Purged CV、DSR、经济学直觉检验、多市场验证。

---

## 交叉引用

- 搜广推 ML 基础 → [[rec-search-ads/]] 推荐系统 ML pipeline
- 特征工程 → [[quant-finance/synthesis/factor_investing.md]] 因子构建
- 回测框架 → [[quant-finance/synthesis/strategy_development.md]] 策略开发
- Embedding 技术 → [[concepts/embedding_everywhere.md]]
- 序列建模 → [[concepts/sequence_modeling_evolution.md]]
- 多目标优化 → [[concepts/multi_objective_optimization.md]]

---

## 相关概念

- [[concepts/attention_in_recsys|Attention 在搜广推中的演进]]
