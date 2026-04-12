# 量化算法工程师面试 — 学习路径

> 10 周系统准备。标注 🔄 的内容可复用搜广推知识库。

---

## Phase 1：基础夯实（Week 1-2）

### 1.1 金融市场基础（Week 1）
- [ ] 股票/期货/期权基本概念，A 股 T+1、涨跌停、集合竞价
- [ ] 订单类型：限价单/市价单/IOC/FOK/冰山单
- [ ] 交易费用：佣金/印花税/滑点/冲击成本
- [ ] A 股 vs 美股：交易规则、数据获取、监管差异

**推荐资源：**
- 《证券投资学》（前 3 章够用）
- Investopedia（查概念比看书快）

### 1.2 数学基础（Week 2）
- [ ] 概率统计复习：分布、假设检验、贝叶斯 🔄
- [ ] 随机过程：布朗运动、几何布朗运动、Itô 引理
- [ ] 时间序列：平稳性、ACF/PACF、ARIMA、协整
- [ ] 蒙特卡洛模拟：期权定价、VaR 计算
- [ ] 线性代数复习：PCA/SVD 在因子分析中的应用 🔄

**推荐资源：**
- 《Stochastic Calculus for Finance》Shreve（Vol I 够面试用）
- 《时间序列分析及应用》Shumway & Stoffer（前 5 章）
- MIT 18.S096 Topics in Mathematics with Applications in Finance（OCW 免费）

---

## Phase 2：核心技能（Week 3-6）

### 2.1 因子投资（Week 3-4）
- [ ] CAPM → Fama-French 三因子 → 五因子 → 多因子体系演进
- [ ] Alpha 因子构建：价量因子、基本面因子、另类因子
- [ ] 因子测试：IC/IR/分组回测/turnover
- [ ] 因子组合：等权/IC加权/优化组合
- [ ] Barra 风险模型：纯因子组合、风险归因
- [ ] 因子衰减与拥挤度
- [ ] **实战**：用 Tushare/AKShare 拉数据，构建 3-5 个因子并回测

**推荐资源：**
- 《Quantitative Equity Portfolio Management》Chincarini & Kim
- 《Expected Returns》Ilmanen（因子投资圣经）
- Barra Risk Model Handbook（USE4/CNE6 文档）

### 2.2 策略开发（Week 5-6）
- [ ] 经典策略：动量/反转、均值回归、统计套利、配对交易
- [ ] 回测框架：事件驱动 vs 向量化回测、look-ahead bias、survivor bias
- [ ] 过拟合陷阱：多重检验、Deflated Sharpe Ratio、WF/Purged CV
- [ ] 交易成本建模：线性/非线性冲击模型
- [ ] 仓位管理：Kelly 准则、风险平价、均值方差优化
- [ ] **实战**：实现一个完整的动量+因子选股策略（含回测报告）

**推荐资源：**
- 《Advances in Financial Machine Learning》Marcos López de Prado（AFML，必读）
- 《Algorithmic Trading》Ernest Chan
- 《Active Portfolio Management》Grinold & Kahn（基本法）

---

## Phase 3：ML/DL 进阶（Week 7-8）

### 3.1 ML 在量化中的应用（Week 7）🔄 大量复用
- [ ] 特征工程：因子 = 特征，时序特征、横截面特征 🔄
- [ ] 模型选择：XGBoost/LightGBM 选股 🔄（搜广推同款）
- [ ] 时间序列交叉验证：Purged K-Fold、Embargo
- [ ] 标签构建：收益率标签、分类标签、triple barrier method
- [ ] 样本权重：uniqueness、decay
- [ ] 集成方法：bagging/boosting 在金融中的注意事项 🔄

**推荐资源：**
- AFML Ch. 3-8（标签/权重/CV/特征重要性）
- 《Machine Learning for Asset Managers》López de Prado

### 3.2 DL 量化前沿（Week 8）🔄 部分复用
- [ ] Transformer 时序预测：Temporal Fusion Transformer、PatchTST 🔄
- [ ] GNN 选股：股票关系图建模、知识图谱+GNN
- [ ] 强化学习：portfolio optimization as RL
- [ ] 生成模型：市场模拟、合成数据增强
- [ ] **注意**：金融中 DL 落地难度远大于搜广推，过拟合是核心问题

**推荐资源：**
- 《Deep Learning for Finance》(Springer)
- FinRL 开源框架（RL 入门）
- 各量化顶会论文：ICAIF、KDD Finance Workshop

---

## Phase 4：面试冲刺（Week 9-10）

### 4.1 编程与系统设计（Week 9）
- [ ] Python 数据处理：Pandas 高级操作、向量化计算 🔄
- [ ] SQL：窗口函数、时序查询、大表优化 🔄
- [ ] C++ 基础（高频方向）：内存管理、低延迟编程
- [ ] 系统设计：实时行情系统、回测引擎架构、因子平台
- [ ] **实战**：LeetCode 量化相关题（滑动窗口、时序处理）

### 4.2 Quant Brain Teasers + 面试题（Week 10）
- [ ] 概率题：掷骰子/抽牌/随机游走/条件概率
- [ ] 估算题：Fermi estimation（交易量/市值/换手率相关）
- [ ] 金融直觉题：Black-Scholes 直觉、Greeks 含义、套利原理
- [ ] 行为面试：为什么做量化、项目经历、团队协作
- [ ] **模拟面试**：每天 2 道 brain teaser + 1 道策略设计

**推荐资源：**
- 《Heard on the Street》Timothy Crack（经典 brain teaser）
- 《A Practical Guide to Quantitative Finance Interviews》Xinfeng Zhou（绿皮书）
- 《Fifty Challenging Problems in Probability》Mosteller
- QuantNet / Glassdoor 面经

---

## 🔄 搜广推可复用知识清单

| 量化知识点 | 搜广推对应 | 复用程度 |
|-----------|-----------|---------|
| XGBoost/LightGBM 选股 | CTR 预估模型 | ⭐⭐⭐⭐⭐ 直接复用 |
| 特征工程/因子构建 | 特征工程 | ⭐⭐⭐⭐ 方法论相同，领域知识不同 |
| Transformer 时序 | 序列建模 | ⭐⭐⭐ 架构复用，loss/数据不同 |
| Embedding 股票表征 | 物品 Embedding | ⭐⭐⭐ 思路复用 |
| 多目标（收益-风险）| 多目标优化 | ⭐⭐⭐ 帕累托框架复用 |
| 在线学习 | 实时推荐 | ⭐⭐ 概念相似，实现差异大 |
| A/B 测试 | 在线实验 | ⭐⭐ 金融用回测替代 |
| 冷启动 | 用户/物品冷启动 | ⭐ 量化无直接对应 |

---

## 推荐书单（按优先级排序）

1. 🔴 **必读**：AFML (López de Prado) — 量化 ML 圣经
2. 🔴 **必读**：绿皮书 (Xinfeng Zhou) — 面试题集
3. 🟠 **强烈推荐**：Active Portfolio Management (Grinold & Kahn) — 因子投资理论基础
4. 🟠 **强烈推荐**：Algorithmic Trading (Ernest Chan) — 策略开发实战
5. 🟡 **推荐**：Heard on the Street — Brain teasers
6. 🟡 **推荐**：Stochastic Calculus for Finance Vol I — 数学基础
7. 🟢 **选读**：Expected Returns (Ilmanen) — 因子投资大全
8. 🟢 **选读**：Quantitative Equity Portfolio Management — 组合管理
