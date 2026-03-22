# A股选股系统 ML 预测模块设计报告

> 整理时间：2026-03-16 | MelonEggLearn
> 资料来源：Microsoft Qlib 源码、ML for Trading (2nd Ed.)、华泰AlphaNet报告、业界实践

---

## 一、特征体系设计

### 1.1 Qlib Alpha158 特征体系（推荐生产使用）

Alpha158 是 Qlib 内置的 158 维手工特征集，设计平衡了信息量与可解释性。

#### K线形态特征（9个）

| 特征名 | 公式 | 含义 |
|--------|------|------|
| KMID | `($close-$open)/$open` | 实体相对涨幅（阳/阴线强度）|
| KLEN | `($high-$low)/$open` | K线长度（波动幅度）|
| KMID2 | `($close-$open)/($high-$low+ε)` | 实体在影线中的位置 |
| KUP | `($high-max($open,$close))/$open` | 上影线长度 |
| KUP2 | `($high-max($open,$close))/($high-$low+ε)` | 上影线占比 |
| KLOW | `(min($open,$close)-$low)/$open` | 下影线长度 |
| KLOW2 | `(min($open,$close)-$low)/($high-$low+ε)` | 下影线占比 |
| KSFT | `(2*$close-$high-$low)/$open` | 收盘偏移量（多空力量对比）|
| KSFT2 | `(2*$close-$high-$low)/($high-$low+ε)` | 收盘偏移比 |

#### 价格水平特征（4个，t=0当日）

```
OPEN0 = $open/$close   # 当日开盘/收盘比
HIGH0 = $high/$close   # 当日最高/收盘比
LOW0  = $low/$close    # 当日最低/收盘比
VWAP0 = $vwap/$close   # 当日成交均价/收盘比
```

#### 滚动统计特征（windows=[5, 10, 20, 30, 60]，共 ~145 个）

| 特征组 | 公式（以window=d为例）| 含义 |
|--------|---------------------|------|
| **ROC** | `Ref($close, d)/$close` | d日价格动量（Rate of Change）|
| **MA** | `Mean($close, d)/$close` | d日均价/收盘比（均线偏离）|
| **STD** | `Std($close, d)/$close` | d日收盘价标准差（波动率）|
| **BETA** | `Slope($close, d)/$close` | d日价格线性趋势斜率 |
| **RSQR** | `Rsquare($close, d)` | d日线性拟合R²（趋势强度）|
| **RESI** | `Resi($close, d)/$close` | d日线性拟合残差（非线性成分）|
| **MAX** | `Max($high, d)/$close` | d日最高价/收盘比 |
| **MIN** | `Min($low, d)/$close` | d日最低价/收盘比 |
| **QTLU** | `Quantile($close, d, 0.8)/$close` | d日80分位数（阻力位）|
| **QTLD** | `Quantile($close, d, 0.2)/$close` | d日20分位数（支撑位）|
| **RANK** | `Rank($close, d)` | 当前收盘在d日中的分位排名 |
| **RSV** | `($close-Min($low,d))/(Max($high,d)-Min($low,d)+ε)` | KDJ随机指标分子（超买超卖）|
| **IMAX** | `IdxMax($high, d)/d` | 最高价出现位置（距今天数/d）|
| **IMIN** | `IdxMin($low, d)/d` | 最低价出现位置（距今天数/d）|
| **IMXD** | `(IdxMax($high,d)-IdxMin($low,d))/d` | 高点领先低点天数（动量方向）|
| **CORR** | `Corr($close, Log($volume+1), d)` | 价格与成交量对数相关性 |
| **CORD** | `Corr($close/Ref($close,1), Log($volume/Ref($volume,1)+1), d)` | 日收益率与成交量变化相关性 |
| **CNTP** | `Mean($close>Ref($close,1), d)` | d日上涨天数占比 |
| **CNTN** | `Mean($close<Ref($close,1), d)` | d日下跌天数占比 |
| **CNTD** | `CNTP - CNTN` | 多空天数差 |
| **SUMP** | `Sum(max($close-Ref($close,1), 0), d)/(Sum(abs($close-Ref($close,1)), d)+ε)` | RSI分子成分（上涨力量）|
| **SUMN** | `Sum(max(Ref($close,1)-$close, 0), d)/(Sum(abs($close-Ref($close,1)), d)+ε)` | RSI分母成分（下跌力量）|
| **SUMD** | `SUMP - SUMN` | RSI信号（多空力量差）|
| **VMA** | `Mean($volume, d)/($volume+ε)` | 成交量均值（量比分母）|
| **VSTD** | `Std($volume, d)/($volume+ε)` | 成交量标准差 |
| **WVMA** | `Std(abs($close/Ref($close,1)-1)*$volume, d)/Mean(abs($close/Ref($close,1)-1)*$volume, d)+ε` | 量价加权波动率 |
| **VSUMP/VSUMN/VSUMD** | 类似 SUMP/SUMN/SUMD，基于成交量 | 成交量多空分布 |

**总计：Alpha158 ≈ 9（kbar）+ 4（price）+ 145（rolling）= 158 个特征**

---

### 1.2 Qlib Alpha360 特征体系（时序模型使用）

Alpha360 提供原始价格序列，适合 LSTM/Transformer 直接建模：

```
特征数 = 6 类 × 60 天 = 360 维

每类特征（以 CLOSE 为例）：
  CLOSE59 = Ref($close, 59)/$close   # 59天前收盘/今日收盘
  CLOSE58 = Ref($close, 58)/$close
  ...
  CLOSE1  = Ref($close, 1)/$close    # 昨收/今收（动量）
  CLOSE0  = $close/$close = 1        # 今日（归一化基准）

6类：CLOSE, OPEN, HIGH, LOW, VWAP, VOLUME（均以当日值归一化）
```

**设计要点：**
- 全部用当日收盘价归一化，消除绝对价格量纲
- VOLUME 用 `Ref($volume, d)/($volume+1e-12)` 归一化
- 标签：`Ref($close, -2)/Ref($close, -1) - 1`（**次日收益**，非当日，避免未来泄露）

---

### 1.3 A股增强特征（超越 Qlib 标准特征）

| 类别 | 特征 | 公式/说明 | 数据来源 |
|------|------|-----------|---------|
| **资金流向** | 主力净流入比 | `主力净流入/成交额` | 东方财富/同花顺 |
| **北向资金** | 沪深港通净买入占比 | `北向净买入/流通市值` | 沪深交易所 |
| **融资融券** | 融资余额变化率 | `融资余额_t/融资余额_t-5 - 1` | 交易所 |
| **龙虎榜** | 机构席位净买入 | 龙虎榜机构净额/成交额 | 交易所 |
| **基本面** | PE/PB分位 | 行业内PE百分位（截面标准化）| Wind/同花顺 |
| **财务质量** | ROE-TTM | 近4季度净利润/平均净资产 | 财报 |
| **技术信号** | 量价异动 | `当日成交量/20日均量 > 2` 且涨幅>3%为1 | 计算 |

---

### 1.4 标签定义

```python
# 主流标签：N日后收益率（截面排名预测）
label_5d  = "Ref($close, -6)/Ref($close, -1) - 1"   # 5日收益
label_10d = "Ref($close, -11)/Ref($close, -1) - 1"  # 10日收益
label_20d = "Ref($close, -21)/Ref($close, -1) - 1"  # 20日收益

# A股注意：Ref($close, -2) 对应 T+2 的收盘价
# 因为 T+1 是买入日，T+1 收盘才能卖出（T+1制度）
# 实际可操作标签：Ref($close, -2)/Ref($close, -1) - 1
```

---

## 二、模型结构推荐

### 2.1 主方案：LightGBM（截面排名预测）

**推荐理由：** 训练快（分钟级）、可解释性强、对异常值鲁棒、天然支持 LambdaRank。

#### 基础配置

```python
import lightgbm as lgb

params = {
    # 核心参数
    "objective": "lambdarank",   # 排序学习目标
    "metric": "ndcg",
    "ndcg_eval_at": [5, 10, 20],  # 评估 Top-5/10/20

    # 树结构
    "num_leaves": 64,             # 叶子节点数，防过拟合用 31-128
    "max_depth": 6,
    "min_child_samples": 50,      # 叶节点最小样本数（A股 ~4500 只，不宜过大）

    # 学习率
    "learning_rate": 0.05,
    "n_estimators": 500,

    # 正则化
    "reg_alpha": 0.1,             # L1
    "reg_lambda": 1.0,            # L2
    "min_split_gain": 0.01,

    # 采样（提升泛化、加速训练）
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,      # Feature fraction

    # 金融数据特有
    "max_bin": 255,
    "verbose": -1,
    "n_jobs": -1,
}

# 截面排名：每天所有股票作为一个 query group
groups = df.groupby("trade_date")["code"].count().values
model = lgb.train(params, train_data, valid_sets=[val_data],
                  group=groups,
                  early_stopping_rounds=50)
```

#### 特征重要度与IC筛选

```python
# 训练后筛选低IC特征
feature_importance = pd.Series(
    model.feature_importance(importance_type="gain"),
    index=feature_names
).sort_values(ascending=False)

# 保留 IC > 0.02 且 IR > 0.5 的特征
def calc_ic(factor_df, ret_df, window=20):
    """计算因子 IC：每日截面 rank 相关系数"""
    ic_series = []
    for date in factor_df.index.get_level_values("date").unique():
        f = factor_df.xs(date, level="date")
        r = ret_df.xs(date, level="date")
        ic = f.corrwith(r, method="spearman")  # RankIC
        ic_series.append(ic)
    ic_df = pd.DataFrame(ic_series)
    return ic_df.mean(), ic_df.std(), ic_df.mean()/ic_df.std()  # IC, IC_std, ICIR
```

---

### 2.2 补充方案A：LSTM（时序建模）

适用于 Alpha360 输入（60天 × 6维特征序列）：

```python
import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,         # (batch, seq_len, features)
            dropout=dropout,
            bidirectional=False       # 股票预测不用双向（未来数据泄露）
        )
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, 60, 6)  → 60天的6维特征
        out, (hn, cn) = self.lstm(x)
        out = self.dropout(out[:, -1, :])   # 取最后时间步
        out = self.bn(out)
        return self.fc(out).squeeze(-1)     # (batch_size,)

# 训练配置
# Loss: MSELoss 或 ListNet（排序损失）
# Optimizer: Adam(lr=1e-3, weight_decay=1e-5)
# Scheduler: CosineAnnealingLR
# Batch size: 512（每批包含多个时间截面的股票）
# Input shape: (N_stocks × N_dates, 60, 6)
```

---

### 2.3 补充方案B：AlphaNet（华泰证券 2020）

AlphaNet 是将时序特征矩阵视为"图像"，用卷积挖掘价格-成交量交叉特征：

#### 核心架构
```
输入层：(T天 × F个特征)矩阵，例如 (30×9)
    ↓
特征提取层：
  - 跨时间卷积（stride=10）→ 捕获中期趋势
  - 特征内核：计算任意两个特征在时间窗内的统计关系
    * CS_STD(f_i, window)  - 特征在时间窗内标准差
    * CS_ZSCORE(f_i, f_j, window) - 归一化差异
    * CS_RETURN(f_i, window) - 时间窗收益率
    * CS_DECAYLINEAR(f_i, window) - 线性衰减加权均值
    * CS_CORR(f_i, f_j, window) - 两特征相关系数
    ↓
BatchNorm → 全连接 → Dropout(0.5) → 输出（预测收益率）
```

#### AlphaNetV3 实现要点（扩展）
```python
# 华泰原始特征：9个价量因子 × 30天
# 特征间交叉：C(9,2) = 36 对两两相关 + 9 个自身统计 = 45 种组合
# 使用 stride=10 的卷积提取 3 个时间窗口
# 总特征数：45 × 3 × (时间子窗口特征) ≈ 几百维

# AlphaNetV4 改进点（Wu 2024）：
# - 用 Transformer 替代 LSTM 层
# - 引入多头注意力捕获特征间关系
# - 数据增强：时间序列裁剪、高斯噪声
```

---

### 2.4 补充方案C：Transformer（Qlib-Localformer）

```python
# Qlib 内置 Transformer 配置参考
model_config = {
    "class": "Localformer",
    "module_path": "qlib.contrib.model.pytorch_localformer_ts",
    "kwargs": {
        "d_feat": 158,          # Alpha158 特征维度
        "d_model": 256,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.5,
        "n_epochs": 200,
        "lr": 1e-4,
        "early_stop": 20,
        "batch_size": 2048,
        "loss": "mse",
        "device": "cuda:0"
    }
}
# Localformer: 局部注意力（减少计算复杂度）+ 全局注意力交替
# 优势：捕获特征间非线性交互，适合 Alpha360
```

---

### 2.5 模型选型建议

| 模型 | 适用场景 | 优势 | 劣势 | 推荐级别 |
|------|---------|------|------|---------|
| **LightGBM** | 生产首选 | 快、稳、可解释 | 无时序建模能力 | ⭐⭐⭐⭐⭐ |
| **LSTM** | 高频/序列特征 | 天然处理时序 | 慢、过拟合风险 | ⭐⭐⭐⭐ |
| **AlphaNet** | 价量交叉特征 | 自动特征工程 | 依赖特征设计 | ⭐⭐⭐⭐ |
| **Transformer** | 复杂交互 | 全局注意力 | 样本量要求高 | ⭐⭐⭐ |
| **LightGBM Ensemble** | 竞赛/高精度 | 多样性集成 | 部署复杂 | ⭐⭐⭐⭐ |

**实战建议：LightGBM（Alpha158）为基础 + LSTM（Alpha360）辅助，输出取加权平均。**

---

## 三、训练流程（Walk-Forward Validation）

### 3.1 Walk-Forward 时间设计

```
训练窗口示意（月度重训，滑动窗口）：

时间 →  2020-01  2020-06  2020-12  2021-06  2021-12
         |                |
         |← 训练窗口 A ─→|← 验证 →|
                  |                |
                  |← 训练窗口 B ─→|← 验证 →|
                           |                |
                           |← 训练窗口 C ─→|← 验证 →|

推荐配置：
  - 训练窗口：24个月（≈480天）
  - 验证窗口：3个月（≈60天）  
  - 测试窗口：1个月（实际预测）
  - 滑动步长：1个月
```

#### Qlib 配置示例
```yaml
# qlib workflow config
task:
  dataset:
    class: DatasetH
    kwargs:
      handler:
        class: Alpha158
        kwargs:
          instruments: csi300
          start_time: "2015-01-01"
          end_time: "2023-12-31"
          fit_start_time: "2015-01-01"
          fit_end_time: "2017-12-31"
      segments:
        train: ["2015-01-01", "2019-12-31"]
        valid: ["2020-01-01", "2020-12-31"]
        test:  ["2021-01-01", "2021-12-31"]

  rolling:
    step: 20      # 每20个交易日重训一次
    rtype: expanding  # expanding window（越来越多数据）
    # 或 rtype: rolling（固定长度滑窗）
```

### 3.2 数据泄露防护要点

| 风险点 | 说明 | 解决方案 |
|--------|------|---------|
| **标签泄露** | 用 T+1 数据计算 T 日标签 | 标签用 `Ref($close, -2)/Ref($close, -1)-1`，确保 T 日收盘后才能计算 |
| **特征归一化泄露** | 用全集均值/方差标准化 | ZScoreNorm 只用训练集 fit，apply 到验证/测试集 |
| **因子IC计算泄露** | 用未来数据选因子 | 因子选择只在训练集内计算 IC |
| **超参数泄露** | 在测试集上调参 | 超参数调优只用验证集，测试集最多用一次 |
| **财务数据泄露** | 财报发布时间 ≠ 报告期 | 使用 Point-in-Time 数据库，如 Qlib 的 PIT Database |

```python
# Qlib 的 Point-in-Time 数据库配置（防财报数据泄露）
from qlib.data.dataset.handler import DataHandlerLP
# PIT: 只使用到当时实际可知的财务数据
# 例如：2020Q2 财报在 2020-08 发布，则 2020-07 只能用 Q1 数据
```

### 3.3 特征预处理流水线

```python
# Qlib 标准处理器链
learn_processors = [
    {"class": "DropnaLabel"},                              # 删除无标签样本
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},  # 截面标准化标签
]

infer_processors = [
    {"class": "ProcessInf"},      # 处理无穷值（inf → clip）
    {"class": "ZScoreNorm"},      # 时序Z-Score归一化
    {"class": "Fillna"},          # 用0填充NaN
]

# 截面标准化（CS = Cross-Section）：
# 每日对所有股票做 Z-Score 归一化
# z_i = (x_i - mean(X_t)) / std(X_t)
# 消除不同日期之间的均值漂移，让模型专注截面差异
```

---

## 四、评估指标

### 4.1 IC / ICIR（最重要）

```python
def calc_ic_metrics(pred_df, label_df):
    """
    pred_df: DataFrame, index=(date, stock), values=预测分数
    label_df: 同结构，values=实际收益率
    """
    daily_ic = []
    for date in pred_df.index.get_level_values("date").unique():
        pred = pred_df.xs(date, level="date").squeeze()
        label = label_df.xs(date, level="date").squeeze()
        # Rank IC（Spearman）更稳健，推荐
        ic = pred.rank().corr(label.rank())
        daily_ic.append(ic)
    
    ic_series = pd.Series(daily_ic)
    IC   = ic_series.mean()      # 目标：> 0.03（弱信号）/ > 0.05（强信号）
    ICIR = IC / ic_series.std()  # 目标：> 0.5（可用）/ > 1.0（优秀）
    
    print(f"IC: {IC:.4f}")
    print(f"ICIR: {ICIR:.4f}")
    print(f"IC > 0: {(ic_series > 0).mean():.2%}")  # 正IC胜率，目标 > 55%
    return IC, ICIR, ic_series
```

**参考基准（A股，日频）：**
- IC > 0.02：弱信号，需谨慎
- IC 0.03~0.05：可用信号
- IC > 0.05：强信号（较难维持）
- ICIR > 0.5：信号稳定可用
- ICIR > 1.0：优秀

### 4.2 分层回测（5层）

```python
def layered_backtest(pred_df, ret_df, n_layers=5):
    """
    每日将股票按预测分数分5层，统计各层收益
    """
    results = []
    for date in pred_df.index.unique():
        pred = pred_df.loc[date].squeeze()
        ret = ret_df.loc[date].squeeze()
        
        # 按预测分数分5组
        labels = pd.qcut(pred, n_layers, labels=range(1, n_layers+1))
        for layer in range(1, n_layers+1):
            mask = labels == layer
            avg_ret = ret[mask].mean()
            results.append({"date": date, "layer": layer, "return": avg_ret})
    
    df = pd.DataFrame(results)
    layer_ret = df.groupby("layer")["return"].mean()
    
    # 关键指标：第5层（最高预测）vs 第1层（最低预测）的收益差
    # 理想情况：层收益单调递增，5层-1层差值越大越好
    print(layer_ret)
    print(f"Top-Bottom Spread: {layer_ret[5] - layer_ret[1]:.4%}/day")
    return layer_ret
```

### 4.3 策略回测指标

```python
# 选取 Top-N 股票构建多头组合（等权重）
def calc_portfolio_metrics(returns, benchmark_returns=None):
    """
    returns: pd.Series, 组合日收益率序列
    """
    # 年化收益
    annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
    
    # 年化波动率
    annual_vol = returns.std() * np.sqrt(252)
    
    # 夏普比率（假设无风险利率 2.5%/年）
    risk_free = 0.025 / 252
    sharpe = (returns.mean() - risk_free) / returns.std() * np.sqrt(252)
    
    # 最大回撤
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # 信息比率（相对基准）
    if benchmark_returns is not None:
        excess = returns - benchmark_returns
        ir = excess.mean() / excess.std() * np.sqrt(252)
    
    print(f"年化收益: {annual_return:.2%}")
    print(f"年化波动: {annual_vol:.2%}")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"最大回撤: {max_drawdown:.2%}")
    
    return {"annual_return": annual_return, "sharpe": sharpe, "max_dd": max_drawdown}
```

| 指标 | 说明 | A股参考基准 |
|------|------|-----------|
| **IC** | 截面 Rank 相关系数 | > 0.03 可用 |
| **ICIR** | IC均值/IC标准差 | > 0.5 |
| **IC正胜率** | IC > 0 的天数比例 | > 55% |
| **分层收益单调性** | 5层收益依次递增 | 5-4-3-2-1层应递减 |
| **年化超额收益** | 相对沪深300 | > 10% |
| **Sharpe Ratio** | 年化超额/年化跟踪误差 | > 1.0 |
| **最大回撤** | 净值历史最大亏损 | < -30%（择时辅助）|
| **换手率** | 每月平均换手 | 20-50%（降低交易成本）|

---

## 五、A股特殊注意事项

### 5.1 T+1 交易制度影响

```
A股机制：T日买入 → T+1日才能卖出
         T日卖出 → 当日资金可再用（融资融券除外）

影响特征设计：
- 标签必须是 T+2 收盘价的收益率（T+1日开盘就可以买，T+2日收盘可以卖）
- qlib 默认标签：Ref($close, -2)/Ref($close, -1) - 1
- 含义：T+1日收盘价 / T日收盘价 - 1（即持有1天的收益）

注意 qlib 的时间偏移约定：
  - Ref($close, -1) = 下一个交易日收盘价
  - Ref($close, -2) = 下下个交易日收盘价
```

### 5.2 涨跌停处理

```python
# A股：普通股票涨跌停 ±10%，科创板/创业板 ±20%
# 特殊股票（ST）涨跌停 ±5%

def handle_limit_up_down(df):
    """涨跌停处理：预测日如果已知股票即将涨停/跌停，需过滤"""
    
    # 方法1：事后过滤（回测用）
    # 如果下一日涨停（最高价=收盘价且涨幅接近10%），该样本不可交易
    df["is_limit_up"] = (df["high"] == df["close"]) & (df["pct_chg"] >= 9.9)
    df["is_limit_down"] = (df["low"] == df["close"]) & (df["pct_chg"] <= -9.9)
    
    # 方法2：训练时删除涨跌停样本的标签（无法交易，无需学习）
    valid_mask = ~(df["next_day_is_limit_up"] | df["next_day_is_limit_down"])
    return df[valid_mask]

# 方法3：Qlib 过滤器
filter_pipe = [
    {"class": "NameDFilter",
     "module_path": "qlib.data.filter",
     "kwargs": {"name_rule_re": "^((?!ST).)*$"}}  # 过滤ST股
]
```

### 5.3 北向资金特征

```python
# 北向资金（陆股通）对A股有显著影响
# 特征：
# 1. 单日北向净流入
# 2. 近5/10日北向累计净流入
# 3. 个股北向持股比例变化
# 4. 大盘北向净流入（市场情绪）

# 注意：北向资金数据每日收盘后公布，不存在未来泄露问题
# 频率：日频，节假日无数据，需前向填充

northbound_feature = """
NB_NET1 = 当日北向净买入/流通市值
NB_NET5 = Sum(北向净买入, 5)/流通市值
NB_HOLD = 北向持股/流通股本  # 持股比例
NB_CHNG = NB_HOLD_t - NB_HOLD_t-1  # 持股变化
"""
```

### 5.4 财报因子防泄露（Point-in-Time）

```python
# 问题：财报于季度结束后约30-60天发布
# 例：2020Q2（4-6月）财报，实际在2020年8月底发布
# 错误做法：2020-07-01 就使用 Q2 的 ROE（此时报告未发布）

# 正确做法：使用 Qlib PIT Database
import qlib
from qlib.data import D

# PIT（Point-in-Time）因子：使用实际可知时间的数据
# qlib pit 配置
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", dataset_cache=None)

# 财务因子延迟处理
# 季报：发布时间 = 季度末后 60天（保守估计）
# 年报：发布时间 = 年末后 90天
factor_config = {
    "roe_ttm": {
        "formula": "Net_Profit_TTM / Avg_Net_Assets",
        "delay_days": 60,   # 季报延迟60天使用
    }
}
```

### 5.5 节假日 & 交易日历

```python
import pandas_market_calendars as mcal
import exchange_calendars as xcals

# A股交易日历（正确处理春节等长假）
cal = xcals.get_calendar("XSHG")  # 上交所
trading_days = cal.sessions_in_range("2020-01-01", "2023-12-31")

# 注意：长假前后的流动性变化
# 春节前：通常流动性收缩，模型预测可靠性下降
# 节后首日：跳空gap大，策略容易滑点
# 建议：节假日前后1-2天降低仓位或不交易
```

---

## 六、参考文献列表

### 框架 & 工具

1. **Microsoft Qlib - 量化投资AI平台**
   - GitHub: https://github.com/microsoft/qlib
   - 论文: [Qlib: An AI-oriented Quantitative Investment Platform](https://arxiv.org/abs/2009.11189)
   - Alpha158/Alpha360 源码: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py

2. **ML for Trading (Stefan Jansen, 2nd Edition)**
   - GitHub: https://github.com/stefan-jansen/machine-learning-for-trading
   - 覆盖：特征工程、因子评估、LightGBM选股、RNN/CNN

3. **Qlib 数据框架文档**
   - https://qlib.readthedocs.io/en/latest/component/data.html

### 模型论文

4. **AlphaNet (华泰证券研究所, 2020)**
   - 原始报告：《基于深度学习的Alpha因子挖掘框架》
   - arxiv 扩展版：Alphanetv4: Alpha Mining Model (Wu 2024) - arXiv:2411.xxxxx
   - 核心思想：将价量时序矩阵做特征间统计交叉，CNN + LSTM 建模

5. **LightGBM 原始论文**
   - [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)

6. **LambdaRank 排序学习**
   - [Learning to Rank: From Pairwise Approach to Listwise Approach](https://icml.cc/2007/papers/139.pdf)

7. **Localformer (Qlib Transformer)**
   - GitHub: https://github.com/microsoft/qlib/pull/508

### A股特殊主题

8. **北向资金与A股收益预测**
   - 多篇中文量化研究（国泰君安、中信证券等券商研究报告）

9. **Qlib Point-in-Time Database（防财务数据泄露）**
   - https://github.com/microsoft/qlib/pull/343
   - 文档: https://qlib.readthedocs.io/en/latest/component/pit.html

10. **Walk-Forward Validation 参考**
    - "Advances in Financial Machine Learning" - Marcos Lopez de Prado, Chapter 7
    - 核心：Purged Cross-Validation，时序数据的正确验证方法

### 因子评估标准

11. **IC/ICIR 评估体系**
    - 行业标准来源：各大量化机构研报（国泰君安、华泰、招商证券量化组）
    - IC > 0.03 为可用信号（A股经验值）

---

## 附录：快速上手代码

### Qlib 完整工作流示例

```python
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord

# 初始化
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

# 模型配置（LightGBM）
model_config = {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
        "loss": "mse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "num_boost_round": 200,
        "early_stopping_rounds": 50,
        "verbose_eval": 20,
    },
}

# 数据集配置
dataset_config = {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "instruments": "csi300",
                "start_time": "2017-01-01",
                "end_time": "2022-12-31",
            },
        },
        "segments": {
            "train": ("2017-01-01", "2020-12-31"),
            "valid": ("2021-01-01", "2021-12-31"),
            "test":  ("2022-01-01", "2022-12-31"),
        },
    },
}

# 运行实验
with R.start(experiment_name="lgbm_alpha158_csi300"):
    model = init_instance_by_config(model_config)
    dataset = init_instance_by_config(dataset_config)
    
    model.fit(dataset)
    
    # 记录信号
    sr = SignalRecord(model, dataset, R)
    sr.generate()
    
    # 组合分析
    par = PortAnaRecord(R, {"benchmark": "SH000300"})
    par.generate()
```

---

*报告版本：v1.0 | 2026-03-16 | MelonEggLearn*
*下一步建议：① 接入 akshare 实际数据验证特征质量 ② 在 CSI300 成分股上跑 Walk-Forward 基准实验 ③ 北向资金特征增强*
