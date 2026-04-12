# 金融量化算法工程师 — 知识库导航

> 面向量化算法工程师面试的系统性知识库。与搜广推知识库并行，共享 ML/DL 基础。

## 📊 知识地图

| 领域 | 文件 | 难度 | 一句话 |
|------|------|------|--------|
| **因子投资** | [synthesis/factor_investing.md](synthesis/factor_investing.md) | ⭐⭐ | Alpha/风险因子、因子构建测试、Barra 模型 |
| **策略开发框架** | [synthesis/strategy_development.md](synthesis/strategy_development.md) | ⭐⭐⭐ | 动量/均值回归/套利、回测框架、过拟合陷阱 |
| **ML 在量化中的应用** | [synthesis/ml_in_quant.md](synthesis/ml_in_quant.md) | ⭐⭐⭐ | 特征工程、模型选择、时序预测、与搜广推交叉 |
| 量化基础 | — | ⭐ | 金融市场、交易机制、订单类型、A股/美股差异 |
| 数学基础 | — | ⭐⭐ | 概率统计、随机过程、时间序列、蒙特卡洛 |
| 执行与风控 | — | ⭐⭐ | VaR/CVaR/Sharpe/MaxDD、滑点、仓位管理 |
| 高频与微观结构 | — | ⭐⭐⭐⭐ | LOB、做市策略、tick 数据处理 |
| DL 量化前沿 | — | ⭐⭐⭐⭐ | Transformer 时序、GNN 选股、生成模型 |
| 编程技能 | — | ⭐⭐ | Python/SQL/C++、分布式计算 |
| 面试题库 | [interview/](interview/) | ⭐⭐⭐ | 概率题、估算题、quant brain teasers |

## 🔗 与搜广推交叉知识

以下知识点可直接复用搜广推知识库：
- **ML 基础**：XGBoost/LightGBM → 同样是量化选股主力模型，参见 [[rec-search-ads/rec-sys/synthesis/]]
- **特征工程**：时序特征、交叉特征 → 因子构建本质就是特征工程
- **Embedding**：参见 [[concepts/embedding_everywhere.md]]，量化中用于股票/基金表征
- **序列建模**：参见 [[concepts/sequence_modeling_evolution.md]]，行情预测=序列预测
- **多目标优化**：参见 [[concepts/multi_objective_optimization.md]]，收益-风险帕累托
- **Attention 机制**：参见 [[concepts/attention_in_recsys.md]]，Transformer 在金融时序中的应用

## 📚 学习路径

详见 [LEARNING_PATH.md](LEARNING_PATH.md) — 10 周从零到面试。

## 📖 概念页

| 概念 | 文件 | 说明 |
|------|------|------|
| 因子投资体系 | [concepts/factor_investing_framework.md](concepts/factor_investing_framework.md) | 从 CAPM 到多因子，因子的前世今生 |
| 量化风控度量 | [concepts/risk_measures.md](concepts/risk_measures.md) | VaR/CVaR/Sharpe/MaxDD 全景 |
| 时间序列分析 | [concepts/time_series_in_quant.md](concepts/time_series_in_quant.md) | ARIMA→GARCH→DL，金融时序专题 |
