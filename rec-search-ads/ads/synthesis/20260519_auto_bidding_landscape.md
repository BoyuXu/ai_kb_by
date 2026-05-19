# Auto-Bidding 全景综合 — 2026-05-19

> 综合 3 篇当日学习论文：BiCB (直播)、AuctionNet (benchmark)、ABA (跨渠道 bandit)

## 一、技术演进

Auto-bidding 自 2018 年阿里 USCB 起经历四代：

| 代际 | 方案 | 关键特征 |
|------|------|---------|
| 第一代 | PID / USCB（2018） | 简单反馈控制，依赖线下流量预估 |
| 第二代 | DRL-Bid / IQL（2020） | 强化学习，离线训练 + 在线探索 |
| 第三代 | LP-Bid / 双对偶（2022） | 线性规划闭式解，理论可证 |
| 第四代 | **BiCB / Auctioneer / Diffusion-Bid（2024+）** | 场景化（直播/多渠道）+ 仿真 benchmark + 生成式 |

三篇论文覆盖第四代的三条主线：

- **BiCB：** 把 LP 思想迁到直播这种短窗口、流量高波动场景
- **AuctionNet：** 提供 benchmark 标准化对比 LP / RL / Generative 各类 bidder
- **ABA：** 跨渠道分预算的 combinatorial bandit，应对非平稳市场

## 二、核心公式

**1. 经典 auto-bidding 形式（PPC + Budget Constraint）：**

maximize Σ_i v_i · x_i
s.t.  Σ_i c_i · x_i ≤ B（预算）
      Σ_i x_i ≤ N（曝光约束）

其中 v_i 是 advertiser 估值，c_i 是 click cost，x_i 是赢得机会的指示量；ML 视角是估 v_i 与 c_i。

**2. BiCB 的二元约束闭式解：**

bid* = (v · CTR) / (1 + λ_upper − λ_lower)

λ_upper / λ_lower 来自 upper / lower bound 的对偶变量，通过统计学方法估计未来流量分布闭式求解。

**3. ABA 的饱和均值回报：**

r_k(b_k) = α_k · (1 − exp(−β_k · b_k))

每个渠道 k 的回报对预算 b_k 呈 concave 单调饱和（Hill function）；联合 budget B = Σ b_k。

**4. 非平稳 bandit 的 change-point detection：**

CUSUM_t = max(0, CUSUM_{t−1} + (r_t − μ̂) − δ)
若 CUSUM_t > h → 触发 reset + 增强探索

## 三、工业实践

**Bidder 选型决策树：**

| 场景 | 推荐方案 |
|------|---------|
| 算力受限、要求秒级响应 | LP / BiCB 闭式解 |
| 模型可在线训练 + 流量稳定 | DRL（PPO / SAC） |
| 高度非平稳市场 | ABA 风格 bandit + change-point |
| 长尾 advertiser、流量稀疏 | Generative bidder（Diffusion-Bid） |
| 多渠道联合 | Combinatorial bandit + 饱和函数 |
| 直播 / 短窗口 | BiCB |

**Benchmark 实践：**

- 用 **AuctionNet** 做线下 ablation（1000 万机会 × 48 agents），对比新方法 vs LP / RL baseline
- 配合 **NeurIPS Bidding Competition** 数据做 cross-validation
- 上线前在 shadow flight（影子拍卖）跑 1–2 周再小流量灰度

**KV trick 与陷阱：**

1. **流量预测漂移：** 节假日、大促、平台冷启需要单独建模
2. **Budget 早枯：** 多用 PID 平滑出价节奏，避免前期 over-bid
3. **Truthfulness：** 非 truthful mechanism 下 bidder 会策略性出价，扭曲数据收集
4. **Pacing：** 时间段维度上的分段优化（hour-level → 5min-level）

## 四、面试考点

1. PID、LP、DRL、Diffusion-Bid 四代 bidder 的演进逻辑？
2. GSP 与 VCG 的真实性差异？为什么 Google / Facebook 仍多用 GSP？
3. 非平稳 bandit 的解法（Discounted UCB / SW-UCB / EXP3）？
4. AuctionNet benchmark 解决了什么 reproducibility 问题？
5. 直播广告与传统 search ads 的 bidding 差异（窗口短、流量波动、anchor 维度约束）？
6. Combinatorial bandit 与 multi-armed bandit 的复杂度差异？
7. Auto-bidding 中的 budget pacing 工程实践？

## 参考

- [BiCB: Lightweight Auto-bidding in Live Advertising](https://arxiv.org/abs/2508.06069)
- [AuctionNet: Benchmark for Decision-Making in Ad Auctions](https://arxiv.org/abs/2412.10798)
- [ABA: Adaptive Budget Optimization via Combinatorial Bandits](https://arxiv.org/abs/2502.02920)
