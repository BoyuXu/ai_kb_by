# 广告系统论文笔记 — 2026-04-16

## 1. Uncertainty Quantification of Click and Conversion Estimates for the Autobidding

**来源：** https://arxiv.org/abs/2603.01825 (Mar 2026)
**领域：** 自动出价 × 不确定性量化
**核心问题：** CTR/CVR 预估模型的不确定性直接影响 autobidding 算法性能，噪声预估导致出价偏离最优

**核心方法 DenoiseBid：**
- 贝叶斯自动出价方法：用后验期望替代噪声 CTR/CVR 点估计
- 从历史数据恢复先验分布，结合模型预测计算后验
- 核心公式：`bid = f(E[CTR|prediction, prior], E[CVR|prediction, prior])`

**实验验证：** 在 4 个数据集上（合成噪声 + 经验噪声），DenoiseBid 一致性提升出价效率

**与 RobustBid 对比：**
- DenoiseBid：贝叶斯去噪，替换点估计为后验期望
- RobustBid：鲁棒优化，考虑最坏情况下的出价稳定性

**面试考点：** CTR/CVR 噪声对出价的影响路径、贝叶斯去噪 vs 鲁棒优化的适用场景、在线实验设计

---

## 2. Robust Autobidding for Noisy Conversion Prediction Models (RobustBid)

**来源：** https://arxiv.org/abs/2510.08788 (Oct 2025)
**领域：** 自动出价 × 鲁棒优化
**核心问题：** CTR/CVR 模型预测不确定性直接影响广告主收入和出价策略

**核心方法 RobustBid：**
- 鲁棒优化框架：在 CTR/CVR 估计存在扰动的情况下，防止大幅出价误差
- 利用先进鲁棒优化技术，在模型不确定性下保证出价质量下界
- 适用于各种拍卖机制（GSP/VCG/第一价格拍卖）

**关键创新：**
- 将 CTR/CVR 噪声建模为有界扰动集
- 优化最坏情况下的广告主效用
- 理论保证：在扰动范围内的出价偏差上界

**面试考点：** 鲁棒优化与随机优化的区别、不确定性集合的构造方法、自动出价中的约束优化

---

## 3. CADET: Context-Conditioned Ads CTR Prediction With a Decoder-Only Transformer

**来源：** https://arxiv.org/abs/2602.11410 (LinkedIn, Feb 2026)
**领域：** 广告 CTR × Transformer 架构
**核心问题：** 传统 DLRM 依赖显式特征交叉，生成式推荐在内容推荐表现好但迁移到广告 CTR 面临独特挑战

**CADET 三大挑战与解决方案：**
1. **后评分上下文信号：** 广告拍卖后才确定的信号（如竞价排名位置），CADET 通过条件化注意力机制处理
2. **离线-在线一致性：** 训练时用历史数据，服务时需实时推理，CADET 确保特征处理一致
3. **工业规模扩展：** 处理 LinkedIn 量级的广告请求，优化推理效率

**架构特点：** 纯 decoder-only Transformer 替代传统特征交叉网络，端到端处理广告 CTR 预估

**面试考点：** DLRM vs Transformer 架构在广告 CTR 的对比、上下文信号的因果性处理、大规模在线服务的延迟优化

---

## 4. BAT: Benchmark for Auto-bidding Task

**来源：** https://arxiv.org/abs/2505.08485 (May 2025)
**领域：** 自动出价基准测试
**核心问题：** 自动出价算法缺乏全面的数据集和标准化基准

**基准设计：**
- 覆盖两种最主流拍卖格式
- 实现强基线，解决两个 RTB 问题域：
  1. **预算节奏均匀性（Budget Pacing）：** 在投放周期内均匀消耗预算
  2. **CPC 约束优化：** 在点击成本约束下最大化转化

**价值：** 为自动出价研究提供公平可比的评估环境，加速算法迭代

**面试考点：** 预算节奏控制算法（PID/MPC）、CPC 约束下的出价策略、在线学习 vs 离线优化

---

## 5. AIE: Auction Information Enhanced Framework for CTR Prediction in Online Advertising

**来源：** https://arxiv.org/abs/2408.07907 (Aug 2024)
**领域：** 拍卖信息 × CTR 预估
**核心问题：** 复杂在线竞价过程给 CTR 优化带来困难，拍卖信息利用不足且存在拍卖偏差

**AIE 框架两大模块：**
1. **自适应市场价格辅助模块（Adaptive Market-price Auxiliary Module）：** 构建辅助任务利用市场价格信息，配合动态网络自适应学习
2. **出价校准模块（Bid Calibration Module）：** 通过逼近目标分布来缓解拍卖偏差

**核心洞察：** 拍卖机制产生的选择偏差（只有赢得拍卖的广告才有点击数据）需要显式建模和校正

**面试考点：** 拍卖偏差（auction bias）的产生机制、市场价格信号的利用方式、选择偏差纠正方法

---

## 6. LDACP: Long-Delayed Ad Conversions Prediction Model for Bidding Strategy

**来源：** https://arxiv.org/abs/2411.16095 (Nov 2024)
**领域：** 延迟转化预估 × 出价策略
**核心问题：** 广告转化存在长延迟（用户点击后数天甚至数周才转化），实时出价需要准确预估 pCTCVR

**技术要点：**
- 排序模型预估 pCTCVR = pCTR × pCVR
- 将广告主 CPA 出价转换为展示出价
- 处理延迟标签（delayed feedback）的建模方法

**核心挑战：**
- 标签延迟导致训练数据中正样本不完整
- 需要区分"尚未转化"和"不会转化"

**面试考点：** 延迟反馈的处理方法（等待窗口/重要性采样/生存分析）、pCTCVR 分解建模、延迟转化对出价策略的影响

---

## 7. Generative Click-through Rate Prediction with Applications to Search Advertising

**来源：** https://arxiv.org/abs/2507.11246 (2025)
**领域：** 生成式 CTR × 搜索广告
**核心创新：** 将 CTR 预估从判别式范式转向生成式范式

**方法论转变：**
- **判别式（传统）：** 输入特征 → 特征交叉网络 → CTR 概率
- **生成式（新范式）：** 输入上下文 → 生成模型 → 直接生成点击/不点击决策

**在搜索广告中的应用：** query-ad 匹配、广告相关性评估、点击概率生成

**面试考点：** 生成式 vs 判别式 CTR 的本质区别、搜索广告的特殊性、生成式模型的校准问题

---

## 8. Auto-bidding and Auctions in Online Advertising: A Survey

**来源：** https://arxiv.org/abs/2408.07685 (Google, Aug 2024)
**领域：** 自动出价 × 拍卖机制综述
**核心定位：** Google 出品的全面综述，覆盖自动出价和在线广告拍卖的理论与实践

**核心主题：**
- **拍卖机制设计：** GSP/VCG/第一价格拍卖的理论性质
- **自动出价算法：** 从规则基出价到基于 RL 的智能出价
- **预算管理：** 预算分配、节奏控制、ROI 约束
- **多目标优化：** 平衡点击/转化/品牌曝光等多重目标
- **博弈论视角：** 多广告主同时使用 autobidder 的均衡分析

**面试考点：** GSP vs VCG 的激励兼容性、自动出价中的 regret bound、多广告主博弈均衡

---

## 9. CTR Prediction on Alibaba's Taobao Advertising Dataset

**来源：** https://arxiv.org/abs/2511.21963 (Nov 2025)
**领域：** CTR 工程实践
**核心内容：** 在阿里巴巴淘宝广告数据集上系统性对比传统和深度学习 CTR 模型

**关键 CPC 机制：** 广告展示顺序 = eCPM = pCTR × bid

**实验对比模型谱系：** LR → GBDT → FM → DeepFM → DIN → DIEN → DCN → AutoInt

**面试考点：** CPC 竞价机制、eCPM 排序原理、模型迭代的工程经验

---

## 10. Improved Online Learning Algorithms for CTR Prediction in Ad Auctions

**来源：** https://arxiv.org/abs/2403.00845 (Mar 2024)
**领域：** 在线学习 × CTR × 拍卖
**核心问题：** 卖方需要在线学习每个广告候选的 CTR，同时通过 PPC 拍卖最大化收入

**技术要点：**
- 将 CTR 学习与收入最大化统一建模
- 在线学习算法的 regret 分析
- 探索-利用权衡在广告拍卖中的应用

**面试考点：** Bandit 算法在广告中的应用、收入最大化的 regret bound、探索策略设计

---

**今日 ads 总结：** 10 篇论文聚焦自动出价与 CTR 预估两大核心方向。关键趋势：(1) 自动出价从确定性优化向不确定性感知（DenoiseBid/RobustBid）演进；(2) CTR 架构从显式特征交叉向 Transformer 端到端（CADET）迁移；(3) 拍卖信息的显式建模（AIE）日益重要。
