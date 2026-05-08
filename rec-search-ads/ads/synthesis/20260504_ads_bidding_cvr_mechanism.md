# 广告竞价、CVR预估与机制设计前沿综合（10篇论文）

**生成日期：** 2026-05-08
**涵盖论文：** AllSERP (2605.04949), BAR (2508.05206), RobustBid (2510.08788), DHEN (2504.08169), Taobao CTR Benchmark (2511.21963), MIAA (2401.01656), Collapsed&Entangled (2403.00793), Ad Auction Realism (2307.11732), User Response MDP (2302.08108), SS-CVR (2401.16432), Auto-bidding Survey (2408.07685)
**交叉引用：** [[autobidding-uncertainty-aware]], [[ads_autobidding_moe_2024_2026]], [[20260504_generative_ads_auction_mechanism]], [[multi_objective_optimization]], [[embedding_everywhere]]

---

## 技术全景：三大主线

```
主线1: CVR/CTR预估增强
  数据稀疏 → 自监督预训练(SS-CVR) → 层级集成(DHEN) → 传统vs深度基准(Taobao CTR)

主线2: 竞价策略优化
  确定性出价 → 鲁棒出价(RobustBid) → 竞价感知召回(BAR) → 长期收益MDP

主线3: 拍卖机制设计
  分离拍卖+分配 → 整合机制(MIAA) → 真实拍卖建模 → 用户响应感知(MDP)
```

---

## Part 1: CVR/CTR 预估增强

### 1.1 DHEN: Deep Hierarchical Ensemble Network (2504.08169, WWW 2025)

- **Problem**: DHEN 在 CTR 预估取得成功，但在转化广告场景（off-site conversion: purchase/add-to-cart/sign-up）的 CVR 预估效果不明
- **Method**: 层级集成多种特征交叉模块（MLP, DCN, Transformer 等），系统性探索深度/宽度/超参的最优配置
- **Innovation**:
  - 首次系统回答 DHEN 在 CVR 场景的模块选择问题
  - 提出 depth-width trade-off 指导原则：CVR 信号更稀疏，需要更宽而非更深的集成
  - 超参数选择的工业经验（learning rate scheduling, embedding dimension 等）
- **Results**: Meta 广告平台部署，CVR 预估 offline NE 显著提升
- **Keywords**: ensemble learning, feature crossing, CVR prediction, DHEN, Meta

**核心架构：**
$$\text{DHEN}(x) = \text{Fusion}\left(\text{MLP}(x), \text{DCN}(x), \text{Transformer}(x), \ldots\right)$$

每层可选不同的 feature crossing module，层间通过 residual connection 或 concatenation 融合。

### 1.2 SS-CVR: Self-Supervised Pre-Training for CVR (2401.16432, Yahoo)

- **Problem**: CVR 训练数据极度稀疏（转化率通常 <1%），非点击归因转化不能直接加入训练集（破坏校准）
- **Method**: 自监督预训练 auto-encoder，训练集包含所有转化事件（含非点击归因），提取特征表示注入主 CVR 模型
- **Innovation**:
  - 巧妙绕过校准问题：pre-trained features 作为输入而非直接预测，不影响主模型的标签空间
  - 针对表格数据的 SSL loss 设计（非图像/文本的 contrastive learning）
  - 工业级实时推理约束下的神经网络集成方案
- **Results**: Yahoo 广告拍卖系统上线，在严格延迟约束下提升 CVR 预估准确度
- **Keywords**: self-supervised learning, CVR prediction, auto-encoder, data sparsity, calibration

**关键思路：**
```
所有转化事件 ──→ AutoEncoder 预训练 ──→ 特征提取器 ──→ 特征向量
                                                           ↓
点击归因转化 ──→ 主 CVR 模型 ←── 拼接 ←── SSL 特征 + 原始特征
                     ↓
              校准的 CVR 预测
```

### 1.3 Taobao CTR Benchmark (2511.21963)

- **Problem**: 在阿里妈妈淘宝广告数据集上系统对比传统 ML 和深度学习 CTR 模型
- **Method**: LR/LightGBM（静态特征）vs 深度模型（行为序列编码 + 静态特征融合），22天 x 数亿交互数据
- **Innovation**:
  - 用户行为序列编码显著优于纯静态特征模型
  - LightGBM 在静态特征场景仍具竞争力
  - 提供可复现的工业级 benchmark 流程
- **Results**: 深度模型（行为序列）> LightGBM > LR，但深度模型需要精心的序列特征工程
- **Keywords**: CTR prediction, Taobao, benchmark, behavior sequence, LightGBM

### 1.4 Collapsed & Entangled World (2403.00793, Tencent)

- **Problem**: 广告推荐中 embedding 表示的两大退化：维度坍塌（dimensional collapse）和兴趣纠缠（interest entanglement）
- **Method**:
  - 维度坍塌：embedding 只利用少数维度，信息丢失 → 正则化/归一化策略
  - 兴趣纠缠：多任务/多场景下用户兴趣混杂 → 解耦表示学习
  - 三个分析工具：特征相关性分析、坍塌程度度量、纠缠度量
- **Innovation**:
  - 首次系统定义广告推荐中 embedding 退化的两类问题
  - 工业级解决方案：训练技巧（优化器选择、去偏技术）
  - 可复用的诊断工具集
- **Results**: 腾讯广告系统部署
- **Keywords**: dimensional collapse, interest entanglement, embedding, multi-task, Tencent

---

## Part 2: 竞价策略优化

### 2.1 RobustBid (2510.08788)

- **Problem**: CTR/CVR 预测噪声导致自动出价偏离最优，传统方法假设预测准确
- **Method**: 鲁棒优化框架，在 CTR/CVR 预测的有界扰动集合内求解最坏情况下最优出价
- **Innovation**:
  - 推导出鲁棒优化问题的解析解，运行时高效
  - 无需知道噪声分布，只需噪声有界假设
- **Results**: 在 synthetic/iPinYou/BAT 数据集上，大扰动下 conversion volume 更大、CPC 更低
- **Keywords**: robust optimization, autobidding, uncertainty, analytical solution

**核心公式：**
$$b^* = \arg\max_b \min_{\delta \in \Delta} \text{utility}(b, \hat{p}_{\text{CTR}} + \delta_1, \hat{p}_{\text{CVR}} + \delta_2)$$

详见 [[autobidding-uncertainty-aware]] 中与 DenoiseBid 的对比。

### 2.2 BAR: Bidding-Aware Retrieval (2508.05206)

- **Problem**: 广告系统召回阶段不考虑出价信号，导致与排序阶段（eCPM = pCTR x Bid）不一致；自动出价时代加剧了这种不一致
- **Method**: Bidding-Aware Modeling + 异步近线推理
  - 单调性约束学习：保证 bid 越高 → 召回分越高
  - 多任务蒸馏：从排序模型蒸馏 CTR/CVR 知识到召回模型
  - 异步近线推理：实时更新广告 bid embedding
- **Innovation**:
  - 首个将 bid 信号引入召回阶段的系统方案
  - 单调性约束确保经济学合理性（出价单调→得分单调）
  - 解决了召回阶段无法获取实时 bid 的工程挑战
- **Results**: 在线广告平台部署，提升召回-排序一致性，总体 revenue 提升
- **Keywords**: retrieval, bidding-aware, monotonicity constraint, knowledge distillation, multi-stage consistency

**关键架构：**
```
传统召回: score = f(user, ad)  ← 不含 bid
BAR 召回: score = g(user, ad, bid)  ← bid-aware，且 ∂g/∂bid ≥ 0 (单调性)
```

### 2.3 Auto-bidding Survey (2408.07685, Google)

- **Problem**: 系统综述自动出价领域的研究进展
- **核心内容**:
  - **出价算法**: 线性规划、PID 控制、RL、生成式模型
  - **均衡分析**: 自动出价下 GSP/VCG/第一价格拍卖的均衡性质
  - **机制设计**: 面对自动出价代理的最优拍卖设计
  - **关键转变**: 从手动 keyword-level bidding → 目标级自动出价（CPA/ROAS target）
- **Keywords**: autobidding, auction design, equilibrium, survey, Google

---

## Part 3: 拍卖机制设计与用户建模

### 3.1 MIAA: 整合拍卖与分配 (2401.01656, SIGIR 2024)

- **Problem**: Feed 流广告中拍卖（ranking + pricing）和分配（display position）分离导致：(1) 拍卖不考虑展示位置外部性；(2) 分配阶段无法保持激励相容
- **Method**: Deep Automated Mechanism Design — 联合优化 ranking、payment、display position
  - 神经网络参数化机制，端到端学习
  - 约束：激励相容（IC）+ 个体理性（IR）
- **Innovation**:
  - 首个同时决定广告排序+价格+展示位置的端到端机制
  - 通过可微约束保证 IC/IR 经济学性质
  - 解决位置外部性：广告 CTR 依赖于实际展示位置和上下文
- **Results**: 在 Feed 流广告场景提升 revenue 同时维持广告主激励
- **Keywords**: mechanism design, auction, allocation, incentive compatibility, externalities, SIGIR

**核心问题：**
$$\text{Position externality}: \text{CTR}(ad_i) = f(\text{ad\_feature}_i, \text{position}_j, \text{context})$$

传统拍卖假设 CTR 与位置无关 (separability)，MIAA 放松此假设。

### 3.2 User Response MDP (2302.08108, WWW 2024, Google)

- **Problem**: 传统拍卖优化单次收益，忽略用户对广告质量的长期响应（低质量广告 → 用户流失 → 长期收益下降）
- **Method**: MDP 建模用户状态（CTR 随广告质量变化），优化长期折扣收益
  - 状态：用户当前 CTR 水平
  - 动作：展示哪些广告
  - 转移：展示低质量广告 → CTR 下降
- **Innovation**:
  - 最优机制 = 带修正虚拟价值的 Myerson 拍卖
  - 修正虚拟价值同时考虑：价值分布、当前用户状态、未来影响
  - 理论证明：短视最优（忽略用户响应）严格劣于 MDP 最优
- **Results**: 理论框架 + 数值模拟验证长期优化的价值
- **Keywords**: MDP, long-term revenue, user response, Myerson auction, modified virtual value

**核心公式：**
$$V(s) = \max_{\pi} \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \mid s_0 = s, \pi\right]$$

其中 $s_t$ 为用户状态（CTR水平），$R$ 为拍卖收益，$\gamma$ 为折扣因子。

最优机制中的修正虚拟价值：
$$\tilde{\phi}(v, s) = \phi(v) + \underbrace{\gamma \cdot \Delta V(s, a)}_{\text{future impact}}$$

### 3.3 Ad Auction Realism (2307.11732)

- **Problem**: 现有广告拍卖模型过于简化，忽略真实系统的关键特征
- **Method**: 提出更现实的学习模型，融入四个真实特征：
  1. 不同 query 的广告位有不同价值/CTR
  2. 竞争者集合随拍卖变化且不可观测
  3. 广告主只收到部分聚合反馈
  4. 广告主用 bandit 算法学习出价策略
- **Innovation**:
  - 对手建模为 adversarial bandit agent（比 Nash 均衡更现实）
  - 独立于拍卖机制细节的通用分析框架
  - 可解释拍卖平台聚合反馈设计的影响
- **Results**: 理论分析 + 实验验证，揭示信息聚合程度对均衡和收入的影响
- **Keywords**: auction realism, adversarial bandit, information design, feedback aggregation

### 3.4 AllSERP / AdSERP Dataset (2605.04949 / 2507.08003)

- **Problem**: 搜索广告研究缺乏包含用户真实注意力数据的公开数据集
- **Method**: 47 名被试在 Google SERP 上的 2,776 个交易性查询，同时记录鼠标轨迹和眼动数据
- **Innovation**:
  - 首个同时包含眼动+鼠标+HTML+截图+广告边界框的 SERP 数据集
  - 基于眼动的客观注意力 ground truth（vs 传统鼠标代理/自报告）
  - 每元素级别的标注和丰富
- **Results**: 公开数据集，支持广告位注意力建模、点击预测、广告位置效应研究
- **Keywords**: SERP, eye tracking, mouse movement, dataset, search ads, attention

---

## 技术演进脉络总结

### CVR 预估演进
```
特征工程 + LR/GBDT (Taobao Benchmark)
    ↓ 深度特征交叉
DCN / DeepFM / Transformer
    ↓ 层级集成
DHEN (多模块集成, Meta)
    ↓ 数据增强
自监督预训练 (SS-CVR, Yahoo)
    ↓ 表示质量
反坍塌 + 解纠缠 (Tencent)
```

### 竞价策略演进
```
手动出价 (keyword-level)
    ↓ 自动化
确定性自动出价 (LP/PID)
    ↓ 不确定性
RobustBid (鲁棒优化)
    ↓ 全链路
BAR (召回阶段也引入 bid)
    ↓ 长期视角
User Response MDP (长期收益优化)
```

### 机制设计演进
```
独立拍卖 (GSP/VCG)
    ↓ 外部性
MIAA (拍卖+分配整合)
    ↓ 动态
User Response MDP (用户状态演化)
    ↓ 现实建模
Adversarial Bandit (Ad Auction Realism)
```

---

## 工业实践要点

### 1. CVR 数据稀疏的三层解法
| 层次 | 方法 | 代表工作 |
|------|------|---------|
| 数据层 | 自监督预训练扩充信号 | SS-CVR (Yahoo) |
| 模型层 | 多模块集成提升容量 | DHEN (Meta) |
| 表示层 | 反坍塌+解纠缠 | Collapsed World (Tencent) |

### 2. 召回-排序一致性
- BAR 的核心洞察：自动出价时代，bid 是 eCPM 的关键组成，召回阶段忽略 bid 导致大量高价值广告被错误过滤
- 工程方案：异步近线推理更新 bid embedding，避免实时计算瓶颈

### 3. 拍卖机制的现实约束
- IC/IR 约束通过可微松弛实现端到端优化 (MIAA)
- 用户长期响应建模需要 MDP 框架，单次最优 ≠ 长期最优
- 信息设计（反馈聚合程度）影响广告主学习速度和平台收入

---

## 面试考点 Q&A

### Q1: CVR 预估为什么比 CTR 更难？有哪些数据增强思路？
**A**: CVR 更难的原因：(1) 转化事件极度稀疏（<1% of clicks）；(2) 归因窗口长（延迟反馈）；(3) off-site 转化信号不完整。数据增强思路包括：
- **ESMM 样本空间校正**：利用 impression→click→conversion 全空间建模
- **自监督预训练（SS-CVR）**：用 auto-encoder 在全转化数据上预训练特征提取器，不破坏主模型校准
- **多模块集成（DHEN）**：通过更大模型容量从稀疏数据中提取更多信号

### Q2: RobustBid 和 DenoiseBid 分别适用什么场景？
**A**: RobustBid 适用于噪声分布未知但有界的场景，采用 minimax 鲁棒优化，有解析解，运行时高效；DenoiseBid 适用于噪声分布可从历史数据估计的场景，采用贝叶斯后验期望修正。实践中两者可互补——DenoiseBid 在数据充足时更精确，RobustBid 在冷启动或分布偏移时更稳健。详见 [[autobidding-uncertainty-aware]]。

### Q3: 为什么需要将 bid 信号引入召回阶段？如何保证经济学合理性？
**A**: 自动出价时代，广告的 eCPM = pCTR x Bid，召回只看 pCTR 会过滤高出价但 pCTR 中等的广告（可能是高价值广告主）。BAR 通过单调性约束 $\partial g / \partial \text{bid} \geq 0$ 保证出价越高、召回分越高，符合经济学直觉。工程上通过多任务蒸馏将排序知识注入召回模型。

### Q4: MIAA 如何解决广告位置外部性问题？与传统 GSP 有什么区别？
**A**: 传统 GSP 假设 CTR 与位置可分离（separability），即 $\text{CTR}_{i,j} = \alpha_j \cdot \beta_i$。MIAA 放松此假设，用神经网络联合建模 (ad, position, context) → CTR，同时通过可微约束保证 IC/IR。区别：GSP 先排序再分配，MIAA 同时决定排序+位置+价格。

### Q5: 为什么要用 MDP 建模广告拍卖？与 contextual bandit 有什么区别？
**A**: Contextual bandit 假设动作不影响未来状态，但广告展示会影响用户体验（低质量广告 → 用户 CTR 下降 → 长期收入减少）。MDP 通过状态转移捕获这种长期效应。最优策略相当于 Myerson 拍卖加上一个 future impact 修正项 $\gamma \cdot \Delta V(s,a)$，平衡当前收益与未来影响。

### Q6: Embedding 维度坍塌是什么？如何诊断和缓解？
**A**: 维度坍塌指 embedding 空间中只有少数维度携带信息，大量维度退化。诊断方法：计算 embedding 矩阵的奇异值分布，若衰减过快说明坍塌。缓解方法：(1) 正交正则化；(2) 特征归一化；(3) 对比学习促进均匀分布。Tencent 的工作系统性地将此问题与兴趣纠缠一起解决。

### Q7: DHEN 中不同 feature crossing 模块各自的优势是什么？
**A**: MLP 擅长隐式高阶交叉；DCN 擅长显式有界阶交叉（计算高效）；Transformer 擅长序列感知的动态交叉。DHEN 的实践经验：CVR 场景（数据更稀疏）需要更宽的集成（更多并行模块）而非更深的堆叠，避免过拟合。

---

*本 synthesis 文档由 MelonEgg 每日学习任务生成，覆盖 2024-2025 广告竞价/CVR预估/机制设计领域 10 篇核心论文*
*交叉引用：[[autobidding-uncertainty-aware]] | [[ads_autobidding_moe_2024_2026]] | [[20260504_generative_ads_auction_mechanism]] | [[multi_objective_optimization]] | [[Embedding无处不在]]*
