# 广告系统论文笔记 — 2026-05-19

> 来源：MelonEgg 每日学习（automated daily-cron）
> 范围：ads 领域 9 篇

---

## 1. Lightweight Auto-bidding based on Traffic Prediction in Live Advertising (BiCB)

**来源：** https://arxiv.org/abs/2508.06069 （KDD 2025，作者来自阿里）
**领域：** Auto-bidding × 直播电商广告

**业务背景：** 直播广告与传统竞价广告差异 — 单场直播窗口短（几小时）、流量波动剧烈、anchor 维度的预算和 ROI 约束更紧

**BiCB（Binary Constrained Bidding）：**
- 数学上推导 optimal bidding 闭式解
- 配合统计方法做"未来直播窗口内剩余流量"的估计
- 计算复杂度极低，可在线分钟级求解，结果逼近 PID / RL 类重型方法

**贡献：** 补全了 auto-bidding 形式化中关于 upper / lower bound 约束的描述；给出理论分析（regret 与 bid envelope 的关系）

**面试考点：** Auto-bidding 的 LP 等价形式、PID 与 RL 在 bidding 中的取舍、直播场景流量预测的特殊性

---

## 2. AuctionNet: Benchmark for Decision-Making in Large-Scale Ad Auctions

**来源：** https://arxiv.org/abs/2412.10798 （NeurIPS 2024 Datasets & Benchmarks）
**领域：** Bidding × 仿真 benchmark

**规模：** 1000 万 ad opportunities × 48 种 auto-bidding agents × 5 亿 auction 记录

**三模块设计：**
1. **Opportunity Generation：** 用 deep generative model 仿真真实流量分布，避免直接暴露敏感数据
2. **Bidding Module：** 内置 LP / RL / Generative 多种 baseline agent
3. **Auction Module：** 以 GSP 为基础，可插拔自定义 mechanism（VCG、FPSB 等）

**应用：** NeurIPS 2024 比赛使用 AuctionNet 评估 1500 个团队、约 1 万次提交，验证了 benchmark 的稳定性

**意义：** Auto-bidding 领域终于有了"ImageNet 级"公开 benchmark，可对比 LP / RL / Diffusion-Bid 等不同范式

**面试考点：** GSP vs VCG 在工业中的取舍、auto-bidding 比赛常见 baseline、仿真器中如何避免 sim-to-real gap

---

## 3. UniMVT: Jointly Optimizing Debiased CTR and Uplift for Coupons Marketing

**来源：** https://arxiv.org/abs/2602.12972 （Kuaishou + BIT，Feb 2026）
**领域：** Uplift Modeling × Marketing × Causal Inference

**核心问题：** 发券类干预会引入严重的 confounding bias：曾经发过券的用户群体本身就更"想买"，简单 CTR 模型会高估 base CTR

**UniMVT 框架：**
- 把 treatment（券面额、券类型）建模为 multi-valued treatment
- 用 disentangled representation 把 confounder 与 treatment-sensitive 表征分离
- Full-space counterfactual inference：同时重建 debiased base CTR + intensity-response curve
- 辅助任务：intensity estimation 估 treatment propensity（multi-valued IPW）
- Unit uplift objective：把干预效应按强度归一

**双目标同时达成：**
- 系统校准准（debiased CTR）
- 发券分配准（precise uplift）

**A/B 验证：** Kuaishou 线上发券业务 GMV 与 ROI 双提升

**面试考点：** S/T/X-learner 与 DML 的区别、IPW 在多值 treatment 下的扩展、ESCM2 等 entire-space 方法的本质

---

## 4. Adaptive Budget Optimization for Multichannel Advertising via Combinatorial Bandits (ABA)

**来源：** https://arxiv.org/abs/2502.02920 （AAMAS 2025，Sony AI）
**领域：** 跨渠道预算分配 × Combinatorial Bandit

**痛点：** 多渠道（Search / Social / Display / Video）联合分预算时，传统 MAB 在非平稳市场（季节性、突发事件）下适应慢

**ABA 三项贡献：**
1. **模拟环境：** 基于 logged real-world 数据的多渠道竞价仿真器
2. **算法：** 饱和均值函数 + 带 change-point detection 的定向探索机制
3. **理论 + 实证：** 多个真实 campaign 上 regret 显著低于 baseline，reward 更高

**关键 trick：** Saturation function 把 reward-budget 关系建成 concave 单调饱和形态（更符合广告投放经验），change-point detector 在市场突变时触发探索

**面试考点：** Combinatorial MAB 的工程化、Saturation modeling 与 Hill function、Non-stationary bandit 的适配方案（discounted UCB、SW-UCB 等）

---

## 5. SUIN: Similar Users-Augmented Interest Network

**来源：** https://arxiv.org/abs/2604.23810
**领域：** CTR × 行为序列稀疏性

**问题：** 长尾用户的历史行为很稀疏，DIN / SIM 等模型在这些用户上掉点严重

**SUIN 方法：**
- 用 sequence encoder 把 target user 的行为编码成 user embedding
- 在 user retrieval pool 里检索 top-k similar users
- 把 similar users 的行为按相似度降序拼接到 target user 行为后，构成 augmented sequence
- **User-specific target-aware position encoding：** 标识每条行为属于哪个 source user + 该行为相对 target item 的位置
- **User-aware target attention：** 同时做 item-item 与 user-user attention，避免相似用户行为"污染"

**实验：** 短期 + 长期序列 benchmark 全面优于 DIN / DIEN / SIM

**面试考点：** Look-alike 与 SUIN 的区别（一个用于召回扩展、一个用于排序增强）、稀疏用户的行为补全策略、attention 中的 user-id 注入方式

---

## 6. CTR-Driven Ad Text Generation via Online Feedback Preference Optimization

**来源：** https://arxiv.org/abs/2507.20227 （Alibaba 淘宝 + HIT，Jul 2025）
**领域：** Ad Creative × LLM × DPO

**两阶段框架：**
1. **多样化候选生成：** one-shot in-context learning + RAG，把过往高 CTR 的 ad text 作为 exemplar；用 CoT 推理生成 diverse candidates
2. **基于在线反馈的 preference optimization：**
   - 用真实曝光数据计算每对候选文案的 CTR gain 与置信度
   - 把 CTR gain × confidence 作为 preference pair 权重做 DPO

**亮点：** 不依赖人工标注 reward 模型，直接用线上 CTR 反馈闭环 fine-tune；offline / online 双指标都显著提升

**与 RELATE / RLHF 对比：** RELATE 用 RL 学 reward model，本方法直接用 DPO 减少训练复杂度

**面试考点：** DPO vs PPO 的优劣、广告文案生成的 cold-start、CTR gain 的置信度估计（PSM / Bootstrap）

---

## 7. Tencent Advertising Algorithm Challenge 2025: All-Modality Generative Recommendation

**来源：** https://arxiv.org/abs/2604.04976 （腾讯广告团队，Apr 2026）
**领域：** 工业 Benchmark × Generative Recommendation

**数据集：**
- **TencentGR-1M：** 100 万用户序列，每个序列最多 100 个交互 item，含曝光 / 点击信号
- **TencentGR-10M：** 1000 万用户，细分到 click 与 conversion 两种事件

**内容特点：** 真实脱敏的腾讯广告日志，含 collaborative ID + 多模态（文本 + 图像）表征

**比赛规模：** 8440 名参赛者，约 30 个国家 / 地区

**论文内容：** 任务定义、数据构造流程、特征 schema、baseline generative recommendation model、评估协议、top solutions 总结

**意义：** 工业级"全模态"生成推荐 benchmark 公开，可与 KuaiSAR / Amazon-Reviews / MovieLens 形成不同 scale 的对比

**面试考点：** Generative Rec 的 tokenization 范式（VQ / RQ-VAE / Semantic ID）、多模态 item embedding 的工业落地、广告 vs 内容推荐的 metric 差异

---

## 8. CTR Prediction in Online Advertising: A Literature Review

**来源：** https://arxiv.org/abs/2202.10462
**领域：** CTR Survey

**结构化分类：**
- **特征工程：** Cross / Cross-Net、AutoInt、FiBiNET
- **结构演化：** LR → FM / FFM → Wide & Deep → DeepFM / xDeepFM / DCN-V2 → AutoInt → MaskNet
- **行为建模：** DIN / DIEN / DSIN / BST / SIM / ETA / SDIM
- **多任务 / 多目标：** ESMM / MMoE / PLE / AITM
- **校准 / 去偏：** Isotonic / Platt / Heckman / IPW / CausalE

**数据趋势：** CTR 论文 2007 起指数增长，2016–2020 达峰；近年焦点从单点 model 优化转向 system 级（feature store、超长序列、生成式）

**面试考点：** CTR 模型演进时间线、FM/FFM/Wide&Deep 的关键差异、ESMM 解决的 SSB / DS 问题

---

## 9. Improved Online Learning Algorithms for CTR Prediction in Ad Auctions

**来源：** https://arxiv.org/abs/2403.00845 （ICML 2023）
**领域：** Ad Auction × Online Learning × Regret Bound

**问题设定：** Seller（平台）通过 PPC 方式向 advertiser 收费；需要在线学每个 ad 的 CTR，同时最大化 revenue

**两种 advertiser 模型：**
1. **Myopic：** 每轮只最大化当轮 utility
   - 基于 UCB 的 online mechanism
   - Worst-case **O(√T)** regret tight bound
   - 当 CTR 静态且有 gap → **negative regret**（学得越多赚得越多）
2. **Non-myopic：** advertiser 考虑长期 utility，可能在早期策略性出价
   - 给出在 static valuation 设定下的 negative regret 算法
   - 关键挑战：advertiser 的策略性投标会扭曲样本

**意义：** 拍卖学习领域少见的兼顾 truthfulness 与 regret bound 的工作

**面试考点：** UCB / Thompson Sampling 在拍卖中的适配、Myopic vs Strategic bidder 的区别、Truthful mechanism 的设计原则

---

## 当日小结

- **主线 1：Auto-bidding 范式** — BiCB（#1）/ AuctionNet（#2）/ ABA（#4）三篇共同构成 auto-bidding 全景：从轻量解析解（直播窗口）、到大规模仿真 benchmark、到非平稳跨渠道 bandit
- **主线 2：CTR 模型 + 校准** — SUIN（#5）解决稀疏用户、UniMVT（#3）解决干预 confounding、CTR Survey（#8）/ Online Auction Learning（#9）给历史回顾与理论 regret 视角
- **主线 3：生成式 + LLM 在广告** — CTR-DPO（#6）做文案生成、TAAC-2025（#7）发布全模态 generative rec benchmark，两者共同指向"广告系统的 LLM 化"
