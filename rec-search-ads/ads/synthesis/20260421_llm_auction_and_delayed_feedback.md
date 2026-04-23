# LLM 原生广告拍卖 + 延迟反馈建模前沿（2023-2026）

> 综合总结 | 领域：ads | 学习日期：20260421
> 10 篇论文覆盖两大主题：LLM 原生广告系统 & 延迟反馈/校准

---

## 参考文献

| # | 简称 | 论文 | 会议/年份 |
|---|------|------|-----------|
| 1 | LLM-Auction | LLM-Auction: Generative Auction towards LLM-Native Advertising | arXiv 2512.10551, 2025 |
| 2 | Ad-Insertion | Ad Insertion in LLM-Generated Responses | arXiv 2601.19435, 2026 |
| 3 | AIE | AIE: Auction Information Enhanced Framework for CTR Prediction | RecSys 2024 |
| 4 | BAR | Bidding-Aware Retrieval for Multi-Stage Consistency in Online Advertising | arXiv 2508.05206, 2025 |
| 5 | CASCADE | Modeling Cascaded Delay Feedback for Online Net CVR Prediction | WWW 2026 |
| 6 | READER | Delayed Feedback Modeling for Post-Click GMV Prediction | WWW 2026 |
| 7 | CFR-DF | Uplift Modeling with Delayed Feedback: Identifiability and Algorithms | AAAI 2025 |
| 8 | CalCal | Calibration-then-Calculation: Variance Reduced Metric Framework in Deep CTR | arXiv 2401.16692, 2024 |
| 9 | UCB-Auction | Improved Online Learning Algorithms for CTR Prediction in Ad Auctions | ICML 2023 |
| 10 | ECAD | Entire Space Cascade Delayed Feedback Modeling for Effective Conversion | CIKM 2023 |

---

## 一、技术演进全景

```
传统广告拍卖（GSP/VCG + 静态关键词竞价）
  │
  ├─ LLM 原生广告 ──────────────────────────────────────────────┐
  │   ├─ Ad-Insertion (2026): 解耦生成与插入 + Genre 竞价 + VCG    │
  │   ├─ LLM-Auction (2025): 耦合生成与拍卖 + IRPO 偏好对齐       │
  │   └─ 对比: 解耦 vs 耦合两条路线                               │
  │                                                               │
  ├─ 拍卖信息增强 CTR ──────────────────────────────────────────┐│
  │   ├─ AIE (2024): AM2 市场价辅助 + BCM 出价校准                 │
  │   ├─ BAR (2025): 检索阶段引入出价信号 → 多阶段一致性           │
  │   └─ UCB-Auction (2023): 在线学习 CTR + 拍卖机制 O(√T) regret │
  │                                                               │
  ├─ 延迟反馈建模 ─────────────────────────────────────────────┐│
  │   ├─ ECAD (2023): CVR+RFR 级联 + 全空间建模 (Xianyu)          │
  │   ├─ CASCADE/TESLA (2026): NetCVR + 级联去偏 + 延迟感知排序损失│
  │   ├─ READER (2026): GMV 连续目标 + 复购路由 + 动态校准         │
  │   └─ CFR-DF (AAAI 2025): Uplift + 延迟反馈可辨识性 + 因果框架 │
  │                                                               │
  └─ 评估校准 ─────────────────────────────────────────────────┐│
      └─ CalCal (2024): Calibrated Log Loss → 降低评估方差         │
```

---

## 二、主题 A：LLM 原生广告系统

### A1. 核心问题：从广告位到 LLM 输出分布

传统广告拍卖对象是离散的广告位（slot），但 LLM 对话场景中不存在固定 slot，拍卖对象变成了 **LLM 输出 token 分布**。这带来三个新挑战：
1. **外部性建模**：广告嵌入上下文后对用户体验的影响无法预测
2. **计算效率**：每次拍卖不能多次调用 LLM
3. **隐私与合规**：用户意图在对话中高度敏感

### A2. 两条路线对比

| 维度 | Ad-Insertion (解耦) | LLM-Auction (耦合) |
|------|-------------------|-------------------|
| **架构** | 先生成回复，再插入广告 | 拍卖与生成一体化，LLM 直接输出含广告内容 |
| **竞价单位** | Genre（语义类簇）代理竞价 | 直接对 LLM 输出分布竞价 |
| **外部性处理** | 忽略（假设插入后影响可控） | IRPO 让 LLM 内在学习外部性 |
| **推理成本** | 1 次 LLM 推理 + 轻量插入 | 1 次 LLM 推理（已对齐） |
| **激励相容** | 近似 DSIC（VCG on genres） | 学习逼近 IC |
| **隐私保护** | Genre 代理隔离用户意图 | 需额外机制保护 |
| **评估方法** | LLM-as-Judge（$\rho \approx 0.66$ vs 人类） | 社会福利 + 收入指标 |

**关键公式 — LLM-Auction IRPO**：

交替优化奖励模型 $r_\phi$ 和 LLM 策略 $\pi_\theta$：
$$\max_\theta \mathbb{E}_{x \sim \mathcal{D}} \left[ \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} [r_\phi(x, y)] - \beta \cdot \text{KL}[\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)] \right]$$

其中 $r_\phi$ 同时编码用户满意度和广告主价值，$\beta$ 控制偏离参考策略的程度。

**关键公式 — Ad-Insertion VCG on Genres**：

广告主 $i$ 对 genre $g$ 出价 $b_{i,g}$，分配与支付：
$$\text{winner} = \arg\max_i b_{i,g} \cdot q_i, \quad p_i = \frac{b_{-i}^{(2)} \cdot q_{-i}^{(2)}}{q_i}$$

其中 $q_i$ 是 LLM-as-Judge 评估的上下文相关性分数。

### A3. 拍卖信息增强检索与预估

**AIE（RecSys 2024）**— 发现现有 CTR 模型未充分利用拍卖后验信息：

- **AM2（Adaptive Market-price Auxiliary Module）**：用市场价（二价）作辅助特征，训练时用真实市场价，推理时用预估市场价
- **BCM（Bid Calibration Module）**：校准出价偏差对 CTR 预估的影响
- 核心洞察：拍卖过程本身引入数据偏差（高出价 → 更多曝光 → CTR 偏高），AIE 显式建模这种 auction bias

**BAR（arXiv 2025, Alibaba）**— 将出价信号前移到检索阶段：

- 传统广告检索只看 user-ad 相关性，忽略商业价值
- BAR 检索评分 = 用户兴趣 + 广告出价价值
- **Task-Attentive Refinement**：解耦用户兴趣信号和商业价值信号的特征交互
- 效果：平台收入 +4.32%，正运营广告曝光 +22.2%（Alibaba 全量部署）

**UCB-Auction（ICML 2023）**— 在线学习 CTR + 拍卖设计：

- 卖方需要在线学习每个广告的 CTR，同时设计激励相容机制
- 基于 UCB 的在线机制，worst-case $O(\sqrt{T})$ regret
- 当最优广告有明显 gap 时，实现 negative regret（比全知 oracle 更好，因为可以利用竞争信息）

---

## 三、主题 B：延迟反馈与校准

### B1. 延迟反馈问题全景

广告转化链路：`曝光 → 点击 → 转化 → (复购) → (退款)`

每个阶段都有延迟：
- **点击→转化延迟**：小时到天级，经典 DFM 问题
- **转化→退款延迟**：天到周级，ECAD/CASCADE 新关注
- **点击→GMV 延迟**：可能多次购买，READER 新关注

### B2. ECAD → CASCADE/TESLA 演进

| 维度 | ECAD (CIKM 2023) | CASCADE/TESLA (WWW 2026) |
|------|-------------------|--------------------------|
| **目标** | ECVR = CVR × (1-RFR) | NetCVR = CVR × (1-RFR)（同义） |
| **建模方式** | CVR 塔 + RFR 塔，共享底层 | CVR-RFR 级联架构 + 阶段去偏 |
| **延迟处理** | 辅助任务调度利用转化/退款时间 | 延迟时间感知排序损失（delay-time-aware ranking loss） |
| **全空间** | 是（解决 SSB + 数据稀疏） | 是 |
| **数据集** | 闭源（Xianyu） | 开源 CASCADE benchmark（淘宝） |
| **改进幅度** | Xianyu 线上显著提升 | RI-AUC +12.41%，RI-PRAUC +14.94% |

**ECAD 核心架构**：

$$\text{ECVR} = P(\text{convert}|x) \times P(\text{not refund}|\text{convert}, x)$$

三个共享底层 → CVR delayed tower + RFR delayed tower：
- CVR tower：建模 $P(\text{convert}, t_c | x)$，用生存分析处理延迟
- RFR tower：建模 $P(\text{refund}, t_r | \text{convert}, x)$，处理级联延迟

**TESLA 三大创新**：

1. **CVR-RFR 级联架构**：CVR 预估结果作为 RFR 输入特征
2. **阶段去偏（Stage-wise Debiasing）**：分别对 CVR 和 RFR 的延迟反馈进行去偏
3. **延迟时间感知排序损失**：
$$\mathcal{L}_{\text{rank}} = \sum_{(i,j)} \max(0, \tau_{ij} - (s_i - s_j)) \cdot w(d_i, d_j)$$

其中 $w(d_i, d_j)$ 是基于延迟时间 $d$ 的权重函数，延迟越短的正样本权重越高。

### B3. READER：GMV 预估中的延迟反馈

GMV（Gross Merchandise Volume）预估与 CVR 的关键区别：
- CVR 是二分类，GMV 是 **连续回归**
- 一次点击可能带来 **多次购买**（复购），标签是累积的
- 延迟窗口内标签永远是 under-estimation

**READER 架构**：

```
点击特征 → 共享底层 → 复购路由器（Router）
                          ├─ 单次购买专家（Expert-Single）
                          └─ 复购专家（Expert-Repeat）
                      → 加权聚合 → GMV 预估
                      → 动态校准模块（纠正 under-estimation）
```

**关键创新**：
1. **RepurchasE-Aware Dual-branch**：路由器预测复购概率，选择性激活不同专家参数
2. **动态标签校准**：根据样本观察窗口长度，动态调整回归目标
3. 开源 **TRACE benchmark**：包含完整交易序列，支持在线流式延迟反馈建模

### B4. CFR-DF：Uplift + 延迟反馈

**问题**：Uplift modeling 中，treatment 效果可能需要时间显现。Day 1 转化 vs Day 14 转化反映不同的因果敏感度。

**CFR-DF 框架**：
- 建立延迟反馈下 uplift 效果的 **可辨识性条件**（identifiability conditions）
- 联合建模潜在响应时间 $T^*(0), T^*(1)$ 和潜在结果 $Y^*(0), Y^*(1)$
- 基于 Counterfactual Regression 框架，扩展为处理截断观测

$$\tau(x) = \mathbb{E}[Y^*(1) - Y^*(0) | X=x, T^*(1) \leq \Delta, T^*(0) \leq \Delta]$$

其中 $\Delta$ 是观测窗口，CFR-DF 同时建模 $P(T^* \leq \Delta | X, W)$ 和 $\mathbb{E}[Y^* | X, W, T^* \leq \Delta]$。

**实践意义**：在广告增量效果评估中，不能简单忽略延迟转化——CFR-DF 提供了理论和算法基础。

### B5. CalCal：评估指标的方差降低

**问题**：深度 CTR 模型训练有随机性（初始化、数据 shuffle、SGD 噪声），同一 pipeline 多次训练结果方差大，导致 A/B 测试和离线评估不可靠。

**解决方案 — Calibrated Log Loss**：

$$\mathcal{L}_{\text{CalLogLoss}} = \text{LogLoss}(\text{Calibrate}(\hat{y}), y)$$

先对模型预测做校准（消除系统性偏差），再计算 Log Loss。

**核心洞察**：
- 标准 Log Loss = 校准误差 + 判别误差
- 不同随机种子主要影响校准误差，判别能力相对稳定
- 先校准再计算，去掉校准噪声，评估方差显著降低
- 实验：Calibrated Log Loss 在检测优势模型时准确率显著优于原始 Log Loss

---

## 四、核心对比表

### 延迟反馈方法对比

| 方法 | 目标 | 延迟类型 | 核心技巧 | 开源数据 |
|------|------|---------|---------|---------|
| DFM (经典) | CVR | 单阶段 | EM + 生存分析 | Criteo |
| ECAD (2023) | ECVR | 级联（转化+退款） | 全空间 + 辅助任务调度 | 无 |
| TESLA (2026) | NetCVR | 级联（转化+退款） | 阶段去偏 + 延迟感知排序 | CASCADE |
| READER (2026) | GMV | 累积（多次购买） | 复购路由 + 动态校准 | TRACE |
| CFR-DF (2025) | Uplift | 因果（treatment延迟） | 可辨识性 + 联合建模 | - |

### LLM 广告方法对比

| 方法 | 路线 | 拍卖机制 | 推理成本 | 激励相容 |
|------|------|---------|---------|---------|
| Ad-Insertion | 解耦 | VCG on genres | 1x LLM + 轻量 | 近似 DSIC |
| LLM-Auction | 耦合 | 学习型（IRPO） | 1x LLM | 学习逼近 |
| UCB-Auction | 传统+在线学习 | UCB 机制 | 无 LLM | 渐近 IC |
| AIE | 增强现有 | 不改机制，增强 CTR | 0（插件式） | N/A |
| BAR | 检索前移 | 不改机制，前移出价 | 0（检索阶段） | N/A |

---

## 五、面试考点

### Q1: LLM 原生广告与传统搜索广告的本质区别？

**答**：三个层面的根本变化：
1. **拍卖对象**：从离散 slot 变为 LLM 输出分布——没有固定广告位，广告嵌入自然语言回复
2. **外部性**：传统 slot 间外部性有限，LLM 场景中广告影响整段回复的用户体验
3. **意图表达**：用户意图从关键词变为对话上下文，更丰富但也更敏感

**解耦 vs 耦合**：Ad-Insertion 用 genre 代理隐私友好但忽略外部性；LLM-Auction 用 IRPO 直接对齐但需额外隐私保护。工业界可能先走解耦路线（低风险），再逐步走向耦合（高收益）。

### Q2: 为什么需要"级联"延迟反馈建模？

**答**：
- 传统延迟反馈只考虑 click → conversion 一个延迟
- 电商场景中 conversion → refund 是第二个延迟，且两个延迟 **方向相反**：conversion 是正信号，refund 是负信号
- 简单做 NetCVR = CVR - RFR 会因为两个延迟窗口不一致导致严重偏差
- ECAD/TESLA 的核心是分别建模两个阶段的延迟，再级联组合

### Q3: GMV 预估的延迟反馈与 CVR 有何本质不同？

**答**：
1. **标签类型**：CVR 是 0/1，GMV 是连续值——不能用分类方法
2. **标签累积性**：一次点击可能触发多次购买，观测窗口越长 GMV 越高——标签永远是 under-estimation
3. **分布差异**：不同用户的复购模式不同，READER 用路由器区分单次 vs 复购用户

### Q4: Calibrated Log Loss 的实践意义？

**答**：在工业场景中，CTR 模型的离线评估经常遇到"每次训练结果不一样"的问题。CalCal 的洞察是：随机性主要影响校准（calibration），而非判别（discrimination）。先做后处理校准再算 Loss，可以显著降低评估方差，让 A/B 测试决策更可靠。

### Q5: BAR 如何解决检索-排序阶段不一致？

**答**：传统广告系统中，检索按 user-ad 相关性选候选，排序按 eCPM（CTR × bid）排序——检索阶段完全忽略出价，导致高价值广告可能在检索阶段被淘汰。BAR 在检索评分中引入出价信号，用 Task-Attentive Refinement 解耦兴趣和商业价值的特征交互，实现多阶段目标一致。

---

## 六、交叉引用

- [[09_延迟转化预估处理方案]] — 经典 DFM 方法综述
- [[10_模型校准方案全景]] — 校准方法全景（CalCal 补充评估视角）
- [[05_竞价与预算优化]] — GSP/VCG 基础 + AutoBidding
- [[03_LLM驱动广告系统]] — LLM 广告系统其他工作
- [[广告竞价与CTR预估前沿进展]] — GenAuction/BundleNet 等
- [[concepts/attention_in_recsys]] — BAR Task-Attentive Refinement
- [[concepts/generative_recsys]] — LLM-Auction 生成式拍卖
