# 金融量化算法工程师 — 知识库导航

> 面向量化算法工程师面试的系统性知识库。与搜广推知识库并行，共享 ML/DL 基础。

## 📊 知识地图

| 领域 | 文件 | 难度 | 一句话 |
|------|------|------|--------|
| **因子投资** | [synthesis/factor_investing.md](synthesis/factor_investing.md) | ⭐⭐ | Alpha/风险因子、因子构建测试、Barra 模型 |
| **策略开发框架** | [synthesis/strategy_development.md](synthesis/strategy_development.md) | ⭐⭐⭐ | 动量/均值回归/套利、回测框架、过拟合陷阱 |
| **ML 在量化中的应用** | [synthesis/ml_in_quant.md](synthesis/ml_in_quant.md) | ⭐⭐⭐ | 特征工程、模型选择、时序预测、与搜广推交叉 |
| **量化基础** | [synthesis/quant_fundamentals.md](synthesis/quant_fundamentals.md) | ⭐ | 金融市场结构、A股交易机制、订单类型、交易费用、A股vs美股 |
| **数学基础** | [synthesis/math_foundations.md](synthesis/math_foundations.md) | ⭐⭐ | 概率统计、布朗运动/GBM/Itô、ARIMA/GARCH、蒙特卡洛、PCA |
| **执行与风控** | [synthesis/risk_and_execution.md](synthesis/risk_and_execution.md) | ⭐⭐ | VaR/CVaR/Sharpe/MaxDD、滑点模型、Kelly/风险平价、止损 |
| **高频与微观结构** | [synthesis/hft_microstructure.md](synthesis/hft_microstructure.md) | ⭐⭐⭐⭐ | LOB结构、做市策略、tick数据处理、A股高频限制 |
| **DL 量化前沿** | [synthesis/dl_quant_frontier.md](synthesis/dl_quant_frontier.md) | ⭐⭐⭐⭐ | Transformer时序(TFT/PatchTST)、GNN选股、RL组合优化、DL落地难点 |
| **编程与系统设计** | [synthesis/coding_system_design.md](synthesis/coding_system_design.md) | ⭐⭐ | Python量化栈、向量化vs事件驱动回测、SQL、因子平台架构、C++低延迟 |
| **Brain Teasers** | [synthesis/brain_teasers_and_puzzles.md](synthesis/brain_teasers_and_puzzles.md) | ⭐⭐⭐ | 20道概率题、10道Fermi估算、5道金融直觉题 |
| **面试题库** | [interview/](interview/) | ⭐⭐⭐ | 概率面试题、策略设计题、面试准备计划 |

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
