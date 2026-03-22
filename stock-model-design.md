# 股票预测 ML/Transformer 模型设计思路

> 学习目标：为 A 股量化选股（B1/B2 信号）引入 ML/Transformer 模型
> 整理时间：2026-03-19
> 来源：Microsoft Qlib 文档、arXiv 论文、工业界实践

---

## 一、模型结构设计

### 1.1 工业界主流架构对比

| 架构 | 代表模型 | 适用场景 | Qlib IC 参考 |
|------|---------|---------|------------|
| LightGBM/XGBoost | Qlib 基线 LGBModel | 截面因子预测，可解释性强 | IC ~0.045 |
| LSTM | Qlib LSTM 基线 | 时序特征建模 | IC ~0.040 |
| Attention LSTM | Qlib AttentionLSTM | 长短期依赖 + 注意力 | IC ~0.053 |
| Transformer | Qlib Transformer | 全局依赖建模 | IC ~0.051 |
| TRA (Temporal Routing Adaptor) | KDD 2021, Microsoft | 多交易模式混合预测 | IC ~0.059 ↑ |
| HIST | arXiv 2110.13716 | 概念共享信息 + 隐藏概念图 | IC 超过 Transformer |
| FactorVAE | 生成式因子模型 | 因子解耦、分布建模 | 理论框架 |
| AlphaNet | 招商证券研报 | CNN 自动挖掘时序特征 | 实盘有效 |

**来源：Qlib 文档 (qlib.readthedocs.io/component/model) + arXiv:2106.12950 (TRA, KDD 2021) + arXiv:2110.13716 (HIST)**

---

### 1.2 截面预测 vs 时序预测

**截面预测（Cross-sectional）**：
- 每个交易日，对所有股票打分排序 → 取 Top-N 持有
- 标准输入：每只股票过去 N 日的特征向量
- 优势：直接对应选股逻辑（B1/B2 就是截面排名）
- 代表：LightGBM + Alpha158 特征

**时序预测（Time-series）**：
- 对单只股票的未来收益率做预测
- 输入：OHLCV 序列（通常 20-60 日）
- 适合：捕捉个股历史形态（N 型结构等）
- 代表：LSTM、Transformer

**推荐（与当前系统结合）**：
> 采用**截面预测为主**，输入特征包含时序窗口（过去 20 日特征矩阵），模型用 LightGBM 或 TabNet，输出排名分数替代/辅助 B1/B2 规则。

---

### 1.3 TRA：多交易模式路由（推荐重点参考）

> 论文：*Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport*（KDD 2021，Microsoft Qlib 团队）

**核心思想**：
- 股市存在多种交易模式（趋势、震荡、反转等），单一模型难以全覆盖
- TRA 包含：**一组独立预测器**（每个学习一种模式）+ **路由器**（将样本分发到合适的预测器）
- 用 Optimal Transport（OT）解决路由器训练的监督信号问题

**效果**：在 Attention LSTM 基础上 IC 从 0.053 → 0.059（+11%），Transformer 从 0.051 → 0.056

**与 B1/B2 系统结合点**：
- B1（趋势突破）、B2（回调做多）本质就是两种不同模式 → TRA 的多预测器天然对应这种结构
- 可以用 TRA 分别训练"趋势买"和"回调买"两个子模型

**代码**：https://github.com/microsoft/qlib/tree/main/examples/benchmarks/TRA

---

### 1.4 HIST：概念共享信息图模型

> 论文：arXiv:2110.13716，*A Graph-based Framework for Stock Trend Forecasting via Mining Concept-Oriented Shared Information*

**核心思想**：
- 同行业/概念板块股票价格高度相关（如半导体板块联动）
- HIST 同时利用：预定义概念（行业/板块）+ 隐藏概念（数据驱动发现的关联）
- 对 A 股板块效应显著，适合中国市场

**与当前系统结合点**：可在现有特征基础上加入行业归属作为 graph 节点，提升板块联动捕捉能力。

---

### 1.5 AlphaNet：CNN 自动挖掘时序 Alpha

**核心思想**（来源：招商证券研报《深度学习与股票选择》）：
- 输入：过去 10/20/30 日的 OHLCV + 技术因子矩阵（形状 T×F）
- 用 1D-CNN 在时间轴上自动挖掘特征交互
- 无需人工设计复杂公式，网络自动学习"N 型结构"类时序模式
- 实盘表现优于传统因子选股（Rank IC > 0.06）

---

### 1.6 实用建议：从简单到复杂的路线图

```
Phase 1（1-2 周）：LightGBM + 现有 90 个特征 → 截面排名
Phase 2（2-4 周）：加时序窗口特征 → TabNet 或 LSTM
Phase 3（1-2 月）：TRA 多模式路由（B1/B2 对应两个子模型）
Phase 4（可选）：HIST 图模型（加行业关联）
```

---

## 二、特征工程（Feature Engineering）

### 2.1 当前系统特征概览

当前 `features/calculator.py`（867 行）已包含约 90 个特征：
- **趋势类**：MA5/10/20/60, EMA10/20, 白线（EMA of EMA 10,10）, 黄线, 趋势斜率
- **动量类**：MACD(12,26,9)
- **震荡类**：KDJ(9,3,3)
- **量价类**：量比、缩量标志、成交额排名
- **形态类**：N 型结构、前低保护
- **波动类**：ATR14、近 20 日高低点

这是一套高质量的规则特征，可以直接作为 ML 模型输入。

---

### 2.2 Qlib Alpha158 vs Alpha360

**Alpha158**（158 个特征）：
- 基于 OHLCV 的 158 个公式化 alpha
- 典型特征：`ROC5 = Close/Ref(Close,5)-1`（5 日动量），`VSTD10`（10 日成交量标准差）
- 包含：动量、反转、换手率、波动率、量价关系 5 大类
- 适合 LightGBM/MLP，Qlib 默认 baseline

**Alpha360**（360 个特征）：
- 过去 60 日每日的 OHLCV 归一化值（6×60=360）
- 时序矩阵形式，专为 LSTM/Transformer 设计
- 来源：Qlib 文档 data handler

**推荐**：
> 当前系统特征 + Alpha158 的补充因子（特别是量价关系、反转因子）效果最佳。

---

### 2.3 特征归一化方式（关键！）

**截面标准化（Cross-sectional Z-score）**：
```python
# 每个交易日对所有股票做标准化
features[day] = (features[day] - mean(features[day])) / std(features[day])
```
- ✅ 消除市场整体涨跌影响，学习相对强弱
- ✅ 适合排名预测（B1/B2 是选相对强的股票）
- ✅ Qlib Alpha158 使用此方法

**时序标准化（Time-series Z-score）**：
```python
# 对单只股票历史窗口做标准化
features[stock] = (features[stock] - mean(last_N)) / std(last_N)
```
- ✅ 适合捕捉个股自身的极端状态
- 适合：LSTM 时序输入

**实践建议**：
> 截面模型（LightGBM）用截面标准化；时序模型（LSTM/Transformer）用时序标准化后再做截面排名。

---

### 2.4 高频 vs 日频特征

| 维度 | 日频特征 | 分钟级特征 |
|------|---------|-----------|
| 数据成本 | 低（免费） | 高（需 Level 2）|
| 模型复杂度 | 低 | 高 |
| 噪声 | 低 | 高 |
| A 股适用性 | ✅ 高 | ⚠️ 需清洗 |
| 推荐场景 | B1/B2 日线信号 | 日内择时（暂不适用）|

**结论**：当前 B1/B2 系统为日线，**继续使用日频特征**。后续如需引入尾盘集合竞价特征可考虑分钟级。

---

### 2.5 新特征建议（补充当前系统）

1. **资金流向因子**：主力净流入/流出（akshare 可获取）
2. **北向资金持仓变动**：近 5 日北向净买入比例
3. **行业相对强弱**：个股涨幅 - 行业平均涨幅（HIST 的截面信号）
4. **龙虎榜特征**：是否出现在龙虎榜（事件驱动因子）
5. **财报因子**：PE/PB/ROE（低频，季度更新）

---

## 三、Label 设计

### 3.1 常见 Label 方案对比

| Label 类型 | 公式 | 优点 | 缺点 |
|-----------|------|------|------|
| 未来 N 日绝对收益率 | `Ret_N = Close(t+N)/Close(t) - 1` | 直观 | 受市场整体涨跌干扰 |
| 未来 N 日超额收益率 | `Ret_N - BenchmarkRet_N` | 消除 Beta | 需要基准指数数据 |
| 截面排名（Rank） | `rank(Ret_N)` in cross-section | 对异常值鲁棒 | 损失幅度信息 |
| 二元分类 | `1 if Ret_N > threshold else 0` | 简单，易部署 | 阈值敏感 |

**工业界主流选择**：
> **未来 5 日超额收益率**（相对中证 500 或全市场）+ 截面排名作为训练目标

---

### 3.2 Rank IC vs Pearson IC

**Pearson IC**：预测值与真实收益率的相关系数
- 对极端值（ST 股暴跌等）敏感
- Qlib 默认评估指标

**Rank IC（Spearman IC）**：排名相关系数
- ✅ 对异常值不敏感（更稳健）
- ✅ A 股有大量涨跌停股，Rank IC 更可靠
- 工业界更常用

**实践建议**：
> 训练时 loss 用 MSE（Pearson 方向），评估时**同时报告 Pearson IC 和 Rank IC**，以 Rank IC > 0.04 为有效因子门槛。

---

### 3.3 避免未来函数（Look-ahead Bias）

**常见错误场景**：

```python
# ❌ 错误：用当日收盘价计算特征，又用当日涨跌幅作为Label
# 等于用"已知答案"训练
feature = close[t] / close[t-1]  # OK
label = close[t+1] / close[t] - 1  # ✅ 正确

# ❌ 错误：标准化时用了未来数据
mean_val = features.mean()  # 如果包含未来，就是 look-ahead bias
```

**防范规则**：
1. **特征截止日**：所有特征只使用 T 日（含）之前的数据
2. **Label 起始日**：使用 T+1 开盘价或 T+2 收盘价（规避 T 日收盘效应）
3. **标准化边界**：截面标准化只在已发生的交易日截面内做
4. **Qlib 的实现**：`Ref($close, -2)/Ref($close, -1) - 1`（即 T+2 相对 T+1 的收益，规避 T+1 开盘买入的滑点）

**来源**：Qlib 文档 alpha.html 中的 label 示例

---

### 3.4 持有期与 Multi-task Label 设计

**单一持有期（简单方案）**：
- B1 信号：持有 5 日 → `label = Ret_5`
- B2 信号：持有 10 日 → `label = Ret_10`

**Multi-task 多持有期**：
```python
labels = {
    'ret_3': Ret_3,   # 短线
    'ret_5': Ret_5,   # 中短线（B1 对应）
    'ret_10': Ret_10, # 中线（B2 对应）
}
# 损失：loss = w1*MSE(ret_3) + w2*MSE(ret_5) + w3*MSE(ret_10)
```

**优势**：
- 共享底层特征表示，提升泛化性
- 强迫模型学习对多时间尺度都有预测力的特征
- 可以针对 B1 取 `ret_5` 的输出，针对 B2 取 `ret_10` 的输出

**推荐**：
> 采用 **Multi-task：ret_5 + ret_10**，与 B1/B2 两个信号天然对应。

---

## 四、与当前 B1/B2 系统的结合建议

### 4.1 最小侵入式引入 ML（Phase 1 推荐）

```python
# 当前逻辑（规则）
b1_signal = white_line_up AND kdj_golden_cross AND volume_breakout

# 改进：ML 辅助评分（不替换，而是加权）
ml_score = lgbm_model.predict(features_90)  # 0~1 得分
final_score = 0.6 * rule_signal + 0.4 * ml_score
```

- 风险低，可以 A/B 测试
- 规则保持可解释性，ML 提供连续分数

### 4.2 特征矩阵构建

```python
# 为每只股票构建特征矩阵（用于 LightGBM）
# 输入：当日的 90 个特征值（已计算好的 calculator.py 输出）
# 截面标准化后直接喂给 LightGBM

X = features_df.xs(today, level='trade_date')  # 当日截面
X_norm = (X - X.mean()) / X.std()

# Label：T+5 超额收益率（回测期内）
y = excess_return_5d
```

### 4.3 模型评估体系

| 指标 | 目标值 | 说明 |
|------|--------|------|
| Rank IC | > 0.04 | 年化 IC 均值 |
| IC IR（IC/ICSTD）| > 0.5 | IC 稳定性 |
| 多空年化收益 | > 20% | Top 20% - Bottom 20% |
| 最大回撤 | < 30% | 风控 |

### 4.4 数据量估算

- A 股上市公司：~5000 支
- 每日截面样本：~5000 个
- 10 年数据：约 2500 个交易日
- 总样本量：**1250 万行**
- LightGBM 完全可以处理，Transformer 需要 GPU

---

## 五、关键结论（3 条最重要）

1. **TRA（多模式路由）与 B1/B2 双信号天然契合**：B1 趋势突破 + B2 回调做多 = 两种交易模式，TRA 的多预测器架构可以分别学习，IC 提升约 11%（0.053→0.059）。
   
2. **Label 用截面超额收益率 + Rank IC 评估**：A 股个股波动大（T+1 涨跌停常见），用相对收益率作为 label、Rank IC 作为评估指标比绝对收益率/Pearson IC 更稳健，工业界标准做法。

3. **Phase 1 最快落地路径**：直接把 calculator.py 的 90 个特征 + 截面标准化 → LightGBM，label 用 T+6/T+1 的超额收益率，一周内可得到第一个 ML baseline，Rank IC 预期 0.03~0.05。

---

*参考资料：*
- *Microsoft Qlib 文档：https://qlib.readthedocs.io*
- *TRA 论文：arXiv:2106.12950（KDD 2021）*
- *HIST 论文：arXiv:2110.13716*
- *Qlib Alpha158/360 特征：github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py*
- *AlphaNet：招商证券研报《深度学习与股票选择》*
