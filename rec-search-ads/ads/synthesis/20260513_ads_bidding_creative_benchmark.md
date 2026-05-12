# 广告竞价、创意生成、CTR预估与基准评测前沿 (9篇论文)

**生成日期：** 2026-05-13
**涵盖论文：** BiCB (2508.06069), AuctionNet (2412.10798), UniMVT (2602.12972), ABA (2502.02920), SUIN (2604.23810), CTR-Driven Ad Text (2507.20227), TencentGR (2604.04976), CTR Survey (2202.10462), Online Learning CTR Auctions (2403.00845)
**交叉引用：** [[ads_autobidding_moe_2024_2026]], [[20260504_ads_bidding_cvr_mechanism]], [[20260504_generative_ads_auction_mechanism]], [[multi_objective_optimization]], [[embedding_everywhere]], [[attention_in_recsys]], [[generative_recsys]], [[sequence_modeling_evolution]]

---

## 技术全景：四大主线

```
主线1: 智能竞价与预算优化
  PID/LP → 轻量化流量预测竞价(BiCB) → 多渠道组合Bandit(ABA) → 在线学习最优拍卖(UCB Auctions)

主线2: CTR预估模型演进
  浅层模型(LR/FM) → 深层交互(DeepFM/DCN) → 序列建模(DIN/SIM) → 相似用户增强(SUIN) → 因果去偏(UniMVT)

主线3: 广告创意与生成式推荐
  人工撰写 → LLM生成+RAG → CTR反馈偏好优化(DPO) → 全模态生成式推荐(TencentGR)

主线4: 评测基准与理论
  离线数据集 → 大规模拍卖仿真(AuctionNet) → CTR预估综述框架 → 在线学习遗憾界理论
```

---

## Part 1: 智能竞价与预算优化

### 1.1 BiCB: 轻量化直播广告自动竞价 (2508.06069, KDD 2025)

- **Problem**: 直播广告对实时性要求极高（秒级控制），未来流量未知。现有 PID/LP/RL 方法要么不考虑全时段流量，要么计算复杂度过高。
- **Method**: Binary Constrained Bidding (BiCB)，核心思想：
  1. 流量预测模型估计全时段消费请求分布
  2. 将竞价决策简化为二值约束优化：对每个请求决定"出价/不出价"
  3. 在 CPC 等约束下最大化 GMV
- **Key Innovation**: 将连续竞价问题离散化为二值决策，配合流量预测实现秒级轻量决策，避免 RL 的高计算开销
- **Core Formula**:
  $$\max \sum_{i=1}^{N} x_i \cdot v_i \quad \text{s.t.} \quad \frac{\sum x_i \cdot c_i}{\sum x_i} \leq \text{CPC}_{target}, \quad x_i \in \{0, 1\}$$
  其中 $v_i$ 为预估 GMV，$c_i$ 为点击成本
- **Results**: 在快手直播广告系统部署，GMV 提升显著
- **Keywords**: auto-bidding, live advertising, traffic prediction, binary optimization

### 1.2 ABA: 多渠道广告预算自适应优化 (2502.02920, AAMAS 2025)

- **Problem**: 跨渠道广告预算分配面临非平稳环境（市场动态变化），传统 MAB 方法适应性不足
- **Method**: 增强型组合 Bandit 策略：
  1. 修改均值函数（Modified Mean Function）适应非平稳回报
  2. 定向探索（Targeted Exploration）减少无效试探
  3. 变点检测（Change-Point Detection, CPD）自动识别市场变化点
- **Key Innovation**: 首个模拟非平稳多渠道广告场景的仿真环境 + CPD增强的组合 Bandit
- **Core Formula**:
  $$\text{Regret}(T) = \sum_{t=1}^{T} \left[ f(\mathbf{x}^*_t) - f(\mathbf{x}_t) \right]$$
  其中 $\mathbf{x}_t$ 为渠道预算分配向量，$f$ 为组合回报函数
- **Results**: 在非平稳环境下显著优于标准 CUCB/Thompson Sampling
- **Keywords**: combinatorial bandits, budget allocation, change-point detection, non-stationary

### 1.3 在线学习 CTR 拍卖算法 (2403.00845, ICML 2024)

- **Problem**: 卖方需在拍卖中在线学习各广告候选的 CTR，按 PPC 方式收费，需最大化收入
- **Method**: 分两种广告主行为建模：
  1. **Myopic（短视型）**: 基于 UCB 的在线机制，worst-case $O(\sqrt{T})$ regret，静态场景可达负 regret
  2. **Non-Myopic（策略型）**: 广告主会策略性出价影响机制学习，设计反操控算法，静态场景实现负 regret
- **Key Innovation**: 首次在非短视广告主设定下实现负 regret（prior work 为 $O(T^{2/3})$）
- **Core Formula**:
  $$\text{UCB}_i(t) = \hat{\mu}_i(t) + \sqrt{\frac{2\ln t}{N_i(t)}} \quad \text{(CTR上置信界)}$$
  收入最大化: $\text{Rev}_t = \text{CTR}_{winner} \cdot \text{price}_{GSP}$
- **Keywords**: online learning, UCB, regret bounds, GSP auction, strategic bidding

---

## Part 2: CTR 预估模型演进

### 2.1 SUIN: 相似用户增强兴趣网络 (2604.23810, 2026)

- **Problem**: 用户行为序列稀疏导致 CTR 预估精度不足，现有方法仅依赖目标用户自身行为
- **Method**: Similar Users-augmented Interest Network (SUIN):
  1. Sequence Encoder 编码行为序列 embedding
  2. 从用户检索池中检索相似用户（基于行为 embedding 相似度）
  3. 将相似用户行为序列按相似度降序拼接到目标用户序列
  4. User-Specific Target-Aware Position Encoding: 标识行为来源用户 + 相对位置
  5. User-Aware Target Attention: 联合建模 item-item 和 user-user 相关性，过滤噪声
- **Key Innovation**: 利用"近邻用户"行为序列补充稀疏用户画像，并通过 user-aware attention 控制噪声
- **Core Formula**:
  $$\text{Attn}(q, K, V) = \text{softmax}\left(\frac{qK^T}{\sqrt{d}} + \mathbf{M}_{user}\right) V$$
  其中 $\mathbf{M}_{user}$ 为 user-user 相关性 mask
- **Results**: 在短序列和长序列 benchmark 上均显著优于 DIN/SIM/HSTU 等 SOTA
- **Keywords**: CTR prediction, similar users, behavior augmentation, target attention, sequence modeling

### 2.2 CTR 预估综述 (2202.10462, IPM 2022)

- **Problem**: CTR 预估模型发展迅速，缺乏系统性分类框架
- **Method**: 系统综述分类:
  1. **浅层模型**: LR, FM, FFM — 特征交叉的线性/二阶近似
  2. **深度模型**: Wide&Deep, DeepFM, DCN-V2 — 自动高阶特征交互
  3. **序列模型**: DIN, DIEN, SIM — 用户行为序列建模
  4. **图模型**: GNN-based CTR — 利用用户-物品交互图
  5. **注意力增强**: Multi-head self-attention, Transformer-based
- **Key Taxonomy**:
  ```
  CTR Models
  ├── Feature Interaction: FM → DeepFM → DCN-V2 → AutoInt
  ├── User Behavior: DIN → DIEN → SIM → HSTU
  ├── Multi-Task: Shared-Bottom → MMoE → PLE
  └── Calibration: Platt Scaling / Isotonic / Neural Cal
  ```
- **Keywords**: CTR prediction, survey, feature interaction, deep learning, online advertising

### 2.3 UniMVT: 因果去偏 CTR + Uplift 联合优化 (2602.12972, 2026)

- **Problem**: 营销干预（优惠券）引入 confounding bias，传统 CTR 模型误校准 base CTR，扭曲排序和计费
- **Method**: Unified Multi-Valued Treatment Network (UniMVT):
  1. 因果解耦: 将特征分为 confounding factors 和 treatment-sensitive representations
  2. 辅助任务: 干预强度估计（treatment propensity estimation）
  3. 单位 Uplift 目标: 标准化干预效果，量化每单位成本的增量转化
- **Key Innovation**: 统一因果框架同时实现去偏 CTR 预测（系统校准）和精确 Uplift 估计（激励分配）
- **Core Formula**:
  $$\text{CTR}_{debiased}(x) = P(Y=1 | X=x, \text{do}(T=0)) \quad \text{(去除干预效应)}$$
  $$\text{Uplift}(x, t) = E[Y | X=x, T=t] - E[Y | X=x, T=0] \quad \text{(增量效果)}$$
- **Results**: 合成+工业数据集上预测精度和校准双优，线上 A/B 测试业务指标显著提升
- **Keywords**: causal inference, debiased CTR, uplift modeling, coupon marketing, multi-valued treatment

---

## Part 3: 广告创意与生成式推荐

### 3.1 CTR-Driven Ad Text Generation (2507.20227, 2025)

- **Problem**: LLM 生成的广告文案效率高但不保证 CTR 优于人工文案，生成质量与在线效果存在 gap
- **Method**: 两阶段框架:
  1. **多样性采样**: One-shot ICL + RAG 检索高 CTR 样例 + CoT 推理生成多样候选
  2. **CTR 驱动偏好优化**: 在线投放收集 CTR 反馈 → 构建 (winner, loser) 偏好对 → 按 CTR gain 和置信度加权 → DPO 训练
- **Key Innovation**: 将在线 CTR 反馈闭环引入 LLM 文案生成，用加权 DPO 实现 CTR 导向的持续优化
- **Core Formula**:
  $$\mathcal{L}_{DPO} = -\mathbb{E}\left[w_{ij} \cdot \log \sigma\left(\beta \log \frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)} - \beta \log \frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)}\right)\right]$$
  其中 $w_{ij} = \text{CTR}_{gain} \cdot \text{confidence}$ 为 CTR 增益加权
- **Results**: 线上 A/B 测试中 CTR 显著提升
- **Keywords**: ad text generation, LLM, DPO, CTR feedback, RAG, preference optimization

### 3.2 Tencent Advertising Algorithm Challenge 2025: 全模态生成式推荐 (2604.04976, 2026)

- **Problem**: 缺乏大规模、真实、全模态（协同 ID + 视觉 + 文本）的生成式推荐公开基准
- **Method**: 构建两个基准数据集:
  1. **TencentGR-1M**: 100万级交互，包含广告协同 ID + 多模态 embedding
  2. **TencentGR-10M**: 1000万级交互，工业规模
  3. 任务: 给定用户全模态广告交互历史，预测下一个点击/转化广告
- **Key Innovation**: 首个工业级全模态生成式广告推荐基准；多模态 embedding 由 SOTA 模型抽取
- **Benchmark Results**: 提供多种 baseline（SASRec, GRU4Rec, TIGER 等）在两个数据集上的对比
- **Keywords**: generative recommendation, multi-modal, benchmark, Tencent Ads, collaborative ID

---

## Part 4: 评测基准与理论

### 4.1 AuctionNet: 大规模拍卖决策基准 (2412.10798, NeurIPS 2024)

- **Problem**: 广告拍卖中竞价决策算法缺乏统一、大规模、可复现的评测环境
- **Method**: 三模块架构:
  1. **广告机会生成**: 深度生成网络模拟真实数据分布（保护隐私）
  2. **竞价模块**: 48 个多样化自动竞价 Agent（不同算法训练）
  3. **拍卖模块**: 基于 GSP 的可定制拍卖机制
  4. 数据规模: 1000万广告机会 + 5亿拍卖记录
- **Key Innovation**: 首个工业级拍卖仿真基准，已用于 NeurIPS 2024 竞赛（1500 队参赛）
- **Impact**: 为竞价算法、拍卖机制设计、多智能体博弈提供标准化评测平台
- **Keywords**: benchmark, ad auction, GSP, auto-bidding, multi-agent, simulation

---

## 核心公式速查

| 方法 | 核心公式 | 含义 |
|------|----------|------|
| BiCB | $\max \sum x_i v_i$ s.t. CPC 约束, $x_i \in \{0,1\}$ | 二值竞价最大化 GMV |
| ABA-CPD | $\hat{\mu}_{post} = \frac{1}{t-\tau} \sum_{s=\tau+1}^{t} r_s$ | 变点后均值重估 |
| UCB Auction | $\text{UCB}_i = \hat{\mu}_i + \sqrt{2\ln t / N_i}$ | CTR 上置信界探索 |
| SUIN | $\text{Attn}(q,K,V) = \text{softmax}(qK^T/\sqrt{d} + M_{user})V$ | 用户感知注意力 |
| UniMVT | $\text{Uplift} = E[Y|X,T=t] - E[Y|X,T=0]$ | 因果增量效果 |
| CTR-DPO | $\mathcal{L} = -E[w_{ij} \log\sigma(\beta\Delta\log\pi)]$ | CTR加权偏好优化 |

---

## 工业实践要点

### 直播广告竞价 (BiCB)
- 秒级决策要求排除重型 RL → 二值化+流量预测是工业捷径
- 流量预测准确度直接决定竞价效果，需要高频更新

### 多渠道预算分配 (ABA)
- 非平稳环境是真实广告的常态 → CPD 自适应是必要组件
- 组合 Bandit 比独立 Bandit 更适合渠道间存在关联的场景

### 去偏 CTR + Uplift (UniMVT)
- 优惠券干预会系统性抬高 CTR → 不去偏导致 eCPM 排序失真
- 因果框架让 CTR 预估和 Uplift 估计共享特征提取器，减少模型数量

### LLM 广告创意 (CTR-DPO)
- RAG + CoT 生成多样候选是第一步
- 关键是闭环: 在线 CTR 反馈 → 加权 DPO → 模型迭代
- 置信度加权避免低曝光样本的噪声偏好对污染训练

### 基准评测 (AuctionNet / TencentGR)
- AuctionNet 解决了竞价算法缺乏可复现评测的痛点
- TencentGR 填补全模态生成式广告推荐基准空白

---

## 面试考点 Q&A

### Q1: 直播广告竞价为什么不直接用 RL？BiCB 的核心思想是什么？

**A**: 直播广告要求秒级实时决策，RL (如 DQN/PPO) 推理延迟和模型复杂度太高。BiCB 的核心思想是将连续竞价问题离散化为二值决策（出价/不出价），配合流量预测模型估计全时段请求分布，将问题转化为 0-1 背包类约束优化，求解效率极高。关键公式: $\max \sum x_i v_i$ s.t. CPC 约束。

### Q2: 在非平稳广告环境中，为什么标准 UCB/Thompson Sampling 效果差？ABA 如何改进？

**A**: 标准 Bandit 算法假设回报分布平稳，但广告环境中用户行为、竞争格局、季节性等因素导致回报分布持续变化。ABA 的三个改进:
1. **Modified Mean**: 用衰减窗口替代全历史均值，更关注近期回报
2. **Targeted Exploration**: 根据渠道特性定向探索，减少无效试探
3. **Change-Point Detection**: CUSUM 检验自动识别分布突变点，检测到变点后重置统计量

### Q3: UniMVT 为什么要同时做去偏 CTR 和 Uplift？能否分开做？

**A**: 分开做有两个问题:
1. **模型冗余**: 需要维护两套模型（CTR 模型 + Uplift 模型），特征提取重复
2. **一致性问题**: 独立训练的 CTR 和 Uplift 模型可能在相同用户上给出矛盾的信号

UniMVT 用因果框架统一: 先做因果解耦将特征分为 confounding 和 treatment-sensitive 两类，CTR 预测只用 confounding 特征（去偏），Uplift 同时用两类特征计算增量效果。共享 backbone 保证一致性。

### Q4: CTR-Driven Ad Text Generation 中，为什么不直接用 CTR 作为 reward 做 RLHF，而是用加权 DPO？

**A**: 三个原因:
1. **稀疏反馈**: 广告文案 CTR 需要足够曝光才有统计意义，RLHF 需要逐 token reward 更难获取
2. **稳定性**: DPO 直接在偏好对上优化，不需要训练 reward model，训练更稳定
3. **加权机制**: 不同偏好对的可信度不同（高曝光 pair vs 低曝光 pair），加权 DPO ($w_{ij} = \text{CTR}_{gain} \cdot \text{confidence}$) 可以自然地处理这种异质性

### Q5: AuctionNet 相比直接用离线日志数据做评测有什么优势？

**A**: 离线日志评测有三大问题:
1. **反事实偏差**: 日志只记录了执行策略下的结果，无法评估其他策略
2. **多智能体交互**: 竞价算法的效果依赖对手策略，日志数据固化了历史对手行为
3. **机制可定制性**: 无法在日志数据上测试不同拍卖机制

AuctionNet 通过仿真环境解决: 48 个多样化 Agent 并行竞价，可模拟不同对手组合；支持 GSP/VCG 等多种拍卖机制切换；深度生成网络保证数据分布接近真实同时保护隐私。

### Q6: SUIN 的 User-Aware Target Attention 和标准 Target Attention (DIN) 有什么区别？

**A**: DIN 的 Target Attention 只建模 target item 与用户历史 item 的相关性:
$$\alpha_i = \text{MLP}(e_{target}, e_i)$$

SUIN 在此基础上增加了两个维度:
1. **User-User 相关性**: 相似用户的行为权重应低于目标用户自身行为，通过 $M_{user}$ mask 矩阵建模
2. **跨用户位置编码**: 标识每个行为来自哪个用户及其与 target item 的相对位置

这确保了引入相似用户行为时不会引入过多噪声。

### Q7: 生成式推荐 (TencentGR) 和传统推荐的核心区别是什么？

**A**: 核心区别在于推荐建模范式:
- **传统**: 判别式，对候选集中每个 item 打分 → 排序 → 选 Top-K
- **生成式**: 自回归生成 item ID（语义 ID），不需要显式候选集

TencentGR 的独特价值在于全模态: 除协同 ID 外还融合视觉和文本模态 embedding，使生成式推荐能利用广告素材内容信息。这在广告场景特别重要，因为广告创意素材对 CTR 影响很大。

### Q8: 在线学习 CTR 拍卖中，为什么 non-myopic 广告主比 myopic 广告主更难处理？

**A**: Myopic 广告主每轮只最大化当轮效用，其出价真实反映估值。但 non-myopic 广告主会策略性操控:
- **低报**: 早期故意低价出价，让平台低估其 CTR，从而在后期获得更低价格
- **探索操控**: 影响平台的 UCB 估计，使平台对竞争对手的 CTR 估计不准

论文的突破是设计了对策略性行为鲁棒的机制，在静态估值+gap 条件下实现负 regret，而 prior work 只能做到 $O(T^{2/3})$。

---

## 技术演进时间线

```
2022 ─── CTR Survey (2202.10462): 系统分类 FM→DeepFM→DIN→GNN
  │
2024 ─── Online Learning CTR Auctions (2403.00845): UCB + 策略性广告主理论
  │       AuctionNet (2412.10798, NeurIPS): 大规模拍卖仿真基准
  │
2025 ─── BiCB (2508.06069, KDD): 直播广告轻量竞价
  │       ABA (2502.02920, AAMAS): 多渠道 CPD Bandit
  │       CTR-Driven Ad Text (2507.20227): LLM 创意 + CTR DPO
  │
2026 ─── SUIN (2604.23810): 相似用户增强 CTR
          UniMVT (2602.12972): 因果去偏 CTR + Uplift
          TencentGR (2604.04976): 全模态生成式广告推荐基准
```
