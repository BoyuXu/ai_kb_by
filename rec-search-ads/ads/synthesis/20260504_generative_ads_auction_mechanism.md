# 生成式广告与拍卖机制设计前沿 (2024-2026)

> 10篇论文综合：生成式出价/CTR + 拍卖机制演进 + 延迟转化与工业CTR
> 日期：2026-05-04 | 关联：[[ads_autobidding_moe_2024_2026]] [[广告竞价与CTR预估前沿进展]] [[20260420_ads_ctr_foundation_and_evaluation]] [[20260421_llm_auction_and_delayed_feedback]]

---

## 一、生成式方法在广告中的崛起

### 1.1 GRAD: 生成式大规模预训练出价模型 (Meituan, KDD 2026)

**问题**：传统自动出价基于 MDP + RL，面临 distribution shift、动作空间探索不足、约束满足困难（CPM/ROI）。

**核心方案 — GRAD (Generative Reward-driven Ad-bidding with MoE)**：

将出价建模为条件轨迹生成问题，而非逐步决策：

```
轨迹 τ = (s₁, a₁, r₁, ..., sₜ, aₜ, rₜ)
生成目标: P(aₜ | s₁:ₜ, a₁:ₜ₋₁, R_target, C_constraint)
```

**两大核心模块**：

1. **Action-MoE (动作混合专家)**：多个专家网络各负责不同出价策略分布，Gating Network 根据广告主目标/场景选专家组合，增强动作空间探索多样性
2. **Value Estimator of Causal Transformer (VECT)**：因果 Transformer 预估每个轨迹的约束满足度与长期回报，指导生成过程满足 CPM/ROI 约束

**关键公式 — 条件生成**：

$$a_t = \text{MoE}(\text{Expert}_1(s_t), ..., \text{Expert}_K(s_t); \text{Gate}(s_t, R_{target}))$$

$$\mathcal{L}_{VECT} = \mathbb{E}_\tau \left[ \sum_t \left( V_\theta(s_t) - \hat{R}_t \right)^2 + \lambda \cdot \max(0, C_{pred} - C_{budget}) \right]$$

**工业落地**：美团多个营销场景部署，GMV +2.18%，ROI +10.68%。

**与传统 RL 出价对比**：

| 维度 | MDP-based RL | GRAD 生成式 |
|------|-------------|------------|
| 建模粒度 | 逐步决策 | 全轨迹生成 |
| 约束处理 | Lagrangian / 硬约束 | VECT 内嵌约束感知 |
| 探索能力 | epsilon-greedy / entropy | MoE 多专家多样性 |
| Distribution shift | 严重 | 条件生成 + 离线-在线对齐 |

### 1.2 GenCTR: 生成式CTR预估 (搜索广告)

**问题**：判别式 CTR 模型表征能力有瓶颈，难以建模用户行为序列中的隐含意图。

**核心方案 — GenCTR 两阶段训练**：

**Stage 1: 生成式预训练（Next-Item Prediction）**

在用户行为序列上做自回归预训练：

$$P(x_{t+1} | x_1, ..., x_t, c) = \text{Decoder}(\text{Enc}(x_{1:t}), c)$$

其中 $c$ 为品类条件信号。引入两项关键技术：
- **Conditional Self-Condition Decoder**：解码时自条件化，避免 exposure bias
- **Conditional Negative Sampling**：按品类条件采负样本，比随机负采样更精准

**Stage 2: 判别式微调（CTR Fine-tuning）**

将预训练好的生成模型嵌入判别式 CTR 框架：
- **Parameter Sharing**：生成模型的 Encoder 参数共享给 CTR 模型
- **Model Integration**：生成模型输出的 hidden state 作为额外特征注入 CTR 网络

$$\hat{y}_{CTR} = \sigma(W \cdot [\text{DNN}(x_{static}); \text{GenEncoder}(x_{seq})] + b)$$

**工业落地**：部署在头部电商搜索广告，服务数亿用户。同时开源了包含预训练和微调数据的公开数据集。

**核心洞察**：生成式预训练 → 判别式微调，这一 "GPT for CTR" 范式让序列行为建模从特征工程升级到表征学习。

---

## 二、拍卖机制设计演进：从 GSP 到生成式拍卖

### 2.1 技术演进脉络

```
GSP (2nd-price generalization)
  → VCG (socially optimal but revenue suboptimal)
    → Myerson Auction (revenue-optimal single-item)
      → Deep AMD (neural network mechanism, 2024)
        → CGA (generative auction with externalities, 2024)
          → BundleNet (joint advertising optimal, 2025)
            → IBPA (information bundling, 2026)
              → Robust MD (anonymous info, 2026)
```

### 2.2 CGA: 上下文生成式拍卖 (KDD 2025)

**核心问题**：传统拍卖假设各位置 CTR 独立，忽略排列级外部性（permutation-level externalities）——广告的点击率不仅取决于自身和位置，还取决于周围广告的排列组合。

**Generator-Evaluator 范式**：

1. **Generator（自回归生成器）**：逐位置生成广告分配序列

$$P(\text{slot}_k = \text{ad}_i | \text{slot}_{1:k-1}) = \text{AutoRegressive}(\text{context}_{1:k-1}, \text{ad}_i)$$

2. **Evaluator（评估器）**：建模排列内交互，输出排列级收益预估

$$\text{Rev}(\pi) = \sum_{k} \text{CTR}_k(\pi) \cdot \text{bid}_k \cdot \text{quality}_k$$

**支付规则的端到端学习**：

将激励兼容 (IC) 约束转化为 ex-post regret 最小化：

$$\text{Regret}_i = \max_{b'_i} [u_i(b'_i, b_{-i}) - u_i(b_i, b_{-i})]$$

$$\mathcal{L}_{payment} = \mathbb{E}[\text{Regret}^2]$$

Regret 可微分，支持梯度优化支付函数。

**结果**：离线和在线均显著提升平台收入和 CTR。

### 2.3 MIAA: 深度自动化机制设计 (Meituan, SIGIR 2024)

**核心问题**：工业 Feed 流中，广告拍卖和位置分配分两个阶段，导致：
- 拍卖时不考虑展示位置和上下文外部性
- 分配时破坏激励兼容性

**解决方案 — 一体化机制 MIAA**：

同时决定排序（ranking）、支付（payment）、展示位置（display position）：

$$(\text{rank}, \text{pos}, \text{pay}) = \text{MIAA}(\text{bids}, \text{context}, \text{organic\_items})$$

- **List-wise 外部性建模**：将完整分配结果（广告+有机内容）作为输入，建模全局外部性
- **IC 约束**：通过 RegretNet 思路确保真实出价是广告主最优策略
- **IR 约束**：广告主支付不超过其出价

**工业落地**：部署在美团零售 Feed，平台收入和 GMV 均显著提升。

### 2.4 BundleNet: 联合广告最优拍卖 (ICML 2025)

**新场景 — 联合广告 (Joint Advertising)**：一个广告位展示两个广告主的 bundle（如品牌+商家），而非传统的单广告主分配。

**理论贡献**：
- 单槽位场景：推导出最优联合广告机制的解析解
- 多槽位场景：提出 BundleNet 神经网络逼近最优机制

**BundleNet 设计**：

$$(\text{alloc}, \text{pay}) = \text{BundleNet}(\{(b_{i1}, b_{i2})\}_{i=1}^N)$$

以 bundle 为单位建模，而非单个广告主，确保：
- 近似 IC：truthful bidding 近似最优
- 近似 IR：非负效用
- Revenue maximization：逼近理论最优

### 2.5 IBPA: 信息捆绑位置拍卖 (2026)

**核心洞察**：平台掌握的定向信息（人群标签、上下文）是一把双刃剑——披露提高广告相关性但降低竞争强度。

**IBPA 机制**：

- 将库存类型视为平台私有信息
- 广告主提交多维出价：对每种可能的库存类型分别出价

$$\text{bid}_i = (b_{i,1}, b_{i,2}, ..., b_{i,T}) \quad \text{(T 种库存类型)}$$

- 平台根据实现的库存类型，比较广告主边际收益进行分配

**理论保证**：IBPA 在任意广告主估值分布和任意信息披露策略下都 dominates GSP。

**实证结果**（零售媒体平台数据）：

| 指标 | IBPA vs GSP 提升 |
|------|-----------------|
| 平台收入 | +68% |
| 分配率 | +19pp |
| 广告主福利 | +29% |
| 总福利 | +54% |

### 2.6 Robust MD: 匿名信息下的鲁棒机制设计 (2026)

**实际问题**：拍卖数据往往是内生删失的、匿名的——卖方只能观察到聚合的顺序统计量，而非完整出价信息。

**核心结果 — 简单机制的鲁棒最优性**：

| 观察到的统计量 | 鲁棒最优机制 |
|--------------|-------------|
| 最高价分布 | Posted Pricing（定价销售） |
| 最低价分布 | Myerson Auction（针对唯一一致 i.i.d. 分布） |
| 中间顺序统计量 | 带最优保留价的二价拍卖 |

$$\text{Rev}^* = \max_{\text{mechanism}} \min_{F \in \mathcal{F}(S)} \mathbb{E}_F[\text{Revenue}]$$

其中 $\mathcal{F}(S)$ 为与观察到的统计量 $S$ 一致的所有乘积分布族。

**实践意义**：为广告平台在信息不完整时选择拍卖机制提供理论指导——不需要完美的出价分布估计，简单机制已经是鲁棒最优。

### 2.7 拍卖机制设计小结

**五篇论文的统一视角**：

| 论文 | 核心创新 | IC保证 | 外部性 | 场景 |
|------|---------|--------|--------|------|
| CGA | 自回归生成分配 | Regret最小化 | 排列级 | 多槽位 |
| MIAA | 拍卖+分配一体化 | RegretNet | 全局(含有机内容) | Feed流 |
| BundleNet | Bundle级最优机制 | 近似IC | Bundle间 | 联合广告 |
| IBPA | 信息捆绑多维出价 | 理论IC | 信息外部性 | 搜索+展示 |
| Robust MD | 匿名统计量鲁棒 | 理论最优 | N/A | 一般拍卖 |

---

## 三、延迟转化与工业 CTR 预估

### 3.1 LDACP: 长延迟转化预测 (Kuaishou, WWW 2025)

**问题**：CPA 出价场景中，广告转化有长延迟（数小时到数天）。实时追踪的转化数会严重低估真实值，导致 CPA 高估 → 出价过于保守。

**解决方案 — 双模块 + MoE 融合**：

**模块 1: BCMS (Bucket Classification Method Sub-module)**

将转化数预测转为分桶分类问题：
- 对转化数做分桶离散化
- 将 one-hot 硬标签转为非归一化软标签
- 同时优化 CE Loss + MSE Loss

$$\mathcal{L}_{BCMS} = \alpha \cdot \text{CE}(p_{bucket}, y_{soft}) + \beta \cdot \text{MSE}(\hat{n}_{conv}, n_{conv})$$

**模块 2: VRMP (Valley Regression Method Part)**

学习 PCOC（预测转化成本）的回归模型，从另一个角度预估转化数。

**融合**：MoE 结构动态加权两个模块的输出：

$$\hat{n}_{final} = g_1 \cdot \hat{n}_{BCMS} + g_2 \cdot \hat{n}_{VRMP}, \quad (g_1, g_2) = \text{Gating}(\text{context})$$

**在线效果**：5天 A/B 测试，达标率提升 2.29%。

### 3.2 AIE: 拍卖信息增强 CTR 预估 (RecSys 2024)

**核心洞察**：CTR 模型通常只用用户/广告/上下文特征，忽略了拍卖过程本身产生的信号（市场价格、竞争强度等）。

**两个问题**：
1. **拍卖信号利用不足**：市场价格等后验信号包含广告竞争力信息
2. **拍卖偏差 (Auction Bias)**：训练数据来自拍卖胜出的广告，存在选择偏差

**两个即插即用模块**：

**AM2 (Adaptive Market-price Auxiliary Module)**：
- 将市场价格作为辅助信号注入 CTR 模型
- 自适应地决定价格信号的融合权重

$$h_{aug} = h_{CTR} + \alpha(h_{CTR}) \cdot \text{Enc}(p_{market})$$

**BCM (Bid Calibration Module)**：
- 校正由拍卖机制引入的出价相关偏差
- 使 CTR 预估不受广告主出价策略影响

**关键贡献**：首次揭示并系统性解决"拍卖偏差"问题，为 CTR 模型引入拍卖视角。

### 3.3 淘宝广告数据集 CTR 预估实践

**数据集**：阿里巴巴通过天池平台发布的淘宝广告数据集：
- 2660万条用户-广告交互记录
- 7.04亿条用户行为日志（点击/加购/收藏/购买）
- 时间跨度：2017年4月16日-5月13日

**方法论对比**：

| 方法层级 | 模型 | AUC | 特点 |
|---------|------|-----|------|
| 静态特征 | LR | baseline | 快速、可解释 |
| 静态特征 | LightGBM | baseline+ | 树模型特征交叉 |
| +行为序列 | DNN | 0.645 | 行为 embedding 融合 |
| +行为序列 | Transformer | 0.687 | 序列注意力建模 |

**核心结论**：
- 引入用户行为序列建模，AUC 提升 6.64%
- Transformer 对长序列行为的建模优势明显
- 证实了 [[序列建模演进|sequence_modeling_evolution]] 在广告 CTR 中的价值

### 3.4 延迟转化与工业CTR小结

```
传统 CTR Pipeline:
  用户特征 + 广告特征 + 上下文特征 → CTR Model → pCTR

增强维度（本节论文贡献）:
  + 拍卖信号 (AIE: 市场价格、竞争强度)
  + 行为序列 (Taobao: Transformer序列建模)
  + 延迟转化校正 (LDACP: 出价策略实时调整)
```

---

## 四、技术交叉与统一视角

### 4.1 生成式范式的两个方向

| 方向 | 代表 | 生成目标 | 模型 |
|------|------|---------|------|
| 生成式出价 | GRAD | 出价动作轨迹 | Causal Transformer + MoE |
| 生成式CTR | GenCTR | 行为序列Next-Item | 条件自回归解码器 |
| 生成式拍卖 | CGA | 广告分配序列 | 自回归生成+评估器 |

统一点：**都用自回归/条件生成替代传统的 "预测+排序+分配" 分离流程**，实现端到端优化。

### 4.2 机制设计中的 IC 保证演进

```
理论IC（解析解）        → Myerson / IBPA / Robust MD
近似IC（神经网络学习）  → RegretNet → MIAA / CGA / BundleNet
```

核心公式（RegretNet 思路）：

$$\mathcal{L} = -\text{Revenue} + \lambda \cdot \text{Regret}^2$$

其中 $\text{Regret} = \max_{b'} u(b') - u(b^{true})$ 衡量偷报的获益。

### 4.3 外部性建模的层次

```
无外部性     → 独立CTR假设 (传统GSP)
集合级       → 候选集内交互 (传统Deep方法)
排列级       → CGA (位置+顺序交互)
全局级       → MIAA (广告+有机内容)
信息级       → IBPA (定向信息的竞争外部性)
```

---

## 五、面试 Q&A

### Q1: 什么是拍卖中的外部性？为什么传统GSP处理不好？

**A**: 外部性指一个广告的效果受其他广告影响。GSP假设各位置CTR独立（separability），但实际上广告的CTR取决于：(1) 位置；(2) 上下文中其他广告的内容和排列。CGA提出排列级外部性，用自回归模型逐位置生成分配，捕捉位置间依赖；MIAA进一步考虑有机内容的影响，建模全局外部性。

### Q2: GRAD为什么比传统RL出价更好？

**A**: 三个核心优势：(1) 全轨迹生成避免逐步error累积（distribution shift）；(2) MoE多专家探索更丰富的动作空间；(3) VECT内嵌约束感知，不需要额外Lagrangian调参。工业效果：GMV +2.18%, ROI +10.68%。

### Q3: GenCTR的"生成式预训练→判别式微调"和NLP的GPT→Fine-tune有什么异同？

**A**: 相同点：都用自回归预训练学通用表征，再微调到下游任务。不同点：(1) GenCTR的预训练是品类条件化的Next-Item预测，不是纯语言建模；(2) 引入Conditional Negative Sampling解决推荐场景的负样本问题；(3) 微调阶段是参数共享+模型集成双机制融合，而非简单加分类头。

### Q4: 延迟转化为什么不能用简单的折扣因子？

**A**: 因为转化延迟不是均匀的——不同广告类型、不同时段的延迟分布差异巨大。LDACP用分桶分类（BCMS）捕捉延迟的多模态分布，用回归（VRMP）从PCOC角度互补，MoE动态融合。简单折扣因子假设延迟分布稳定，无法应对分布漂移。

### Q5: IBPA为什么能比GSP收入高68%？

**A**: GSP直接披露库存类型让广告主针对性出价，降低了竞争强度。IBPA将库存类型保持为平台私有信息，要求广告主对所有可能的库存类型分别出价。这样：(1) 竞争信息更充分；(2) 平台可以比较边际收益做最优分配；(3) 广告主无法通过信息优势降低出价。本质是信息不对称的博弈论应用。

### Q6: 拍卖偏差(Auction Bias)是什么？如何解决？

**A**: CTR模型训练数据只包含拍卖胜出的广告，这些广告的出价通常较高，导致模型隐式学到"高出价→高CTR"的虚假关联。AIE的BCM模块通过校准出价信号的影响来消除这种偏差，使CTR预估回归到真实的用户兴趣信号。

### Q7: 如何在信息不完整时设计鲁棒的拍卖机制？

**A**: Robust MD论文表明：当卖方只能观察到匿名的顺序统计量时，简单机制已经是minimax最优——看到最高价分布用posted pricing，看到最低价分布用Myerson，看到中间分布用带保留价的二价拍卖。不需要复杂的参数估计，这给工业实践提供了"不确定就用简单机制"的理论支撑。

---

## 六、参考论文

1. GRAD: Generative Large-Scale Pre-trained Models for Automated Ad Bidding Optimization. Lei et al., Meituan. arXiv:2508.02002
2. GenCTR: Generative Click-through Rate Prediction with Applications to Search Advertising. Kong et al. arXiv:2507.11246
3. CGA: Contextual Generative Auction with Permutation-level Externalities. Shi et al. arXiv:2412.11544, KDD 2025
4. IBPA: Targeting Information in Ad Auction Mechanisms. Tunuguntla et al. arXiv:2601.09541
5. BundleNet: Optimal Auction Design in the Joint Advertising. Li et al. arXiv:2507.07418, ICML 2025
6. MIAA: Deep Automated Mechanism Design for Integrating Ad Auction and Allocation in Feed. Li et al., Meituan. arXiv:2401.01656, SIGIR 2024
7. LDACP: Long-Delayed Ad Conversions Prediction Model for Bidding Strategy. Cui et al., Kuaishou. arXiv:2411.16095, WWW 2025
8. AIE: Auction Information Enhanced Framework for CTR Prediction in Online Advertising. Yang et al. arXiv:2408.07907, RecSys 2024
9. Robust Mechanism Design with Anonymous Information. Tang & Wang. arXiv:2602.20429
10. CTR Prediction on Alibaba's Taobao Advertising Dataset Using Traditional and Deep Learning Models. arXiv:2511.21963
