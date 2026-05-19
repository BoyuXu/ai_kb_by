# 多任务 / 多行为推荐综合 — 2026-05-19

> 综合 5 篇当日学习论文：MBGen、Multi-Behavior Survey、MTL-DML、RMTL、MTDRS Survey

## 一、技术演进

多任务（MTL）与多行为（MBR）推荐是工业推荐绕不开的两个坐标：

- **MTL** 关心"同时优化 CTR / CVR / 时长 / 收藏 / 转发"等多目标
- **MBR** 关心"利用 click / cart / fav / buy 等多种行为联合建模"

二者本质都在做"知识共享 + 冲突缓解"，技术演进上交叉融合：

| 阶段 | MTL 主线 | MBR 主线 |
|------|---------|---------|
| 共享底层 | Shared-bottom（2017） | Joint factorization |
| 灵活共享 | Cross-stitch、MMoE（2018） | NMTR、EHCF |
| 任务隔离 | PLE / CGC（2020） | MB-GMN、CML |
| 校正偏差 | ESMM（2018）/ AITM | KMCLR、CIKM 多视图 |
| 梯度调和 | GradNorm / PCGrad / CAGrad | 行为图对比学习 |
| 跨塔互学 | **MTL-DML（2023.9）** | 多行为对比 / 蒸馏 |
| RL 调权 | **RMTL（WWW 2023）** | RL 多目标 |
| 生成式统一 | One-Model-Fits-All | **MBGen（CIKM 2024）** |

## 二、核心公式

**1. MMoE 输出：**

y_t = h_t( Σ_i g_t(x)_i · E_i(x) )

每个任务 t 有专属 gate g_t 选择 expert 组合；问题是 expert 与 tower 仍可能 negative transfer。

**2. PLE 任务隔离：**

将 experts 分为 shared experts S 和 task-specific experts T_t；gate 只在 (S ∪ T_t) 内做选择 → 避免任务专属知识被污染。

**3. MTL-DML 互学损失：**

L = Σ_t L_task_t + λ · Σ_{i ≠ j} KL( σ(y_i / τ) ‖ σ(y_j / τ) )

让 task i 与 task j 的预测分布相互蒸馏，温度 τ 控制软度。

**4. RMTL 动态权重：**

w_t^(s) = π_θ(state_s)_t

由 actor 网络根据 session state s 输出每个任务的权重；critic 估计 long-term reward 反传更新 π_θ。

**5. MBGen 统一序列：**

P(b_{n+1}, i_{n+1} | history) = P(b_{n+1} | history) · P(i_{n+1} | history, b_{n+1})

先生成 behavior token、再生成 item token，behavior 与 item 在序列中交错放置，autoregressive 训练。

## 三、工业实践

**MTL 工业范式选择：**

| 业务场景 | 推荐方案 |
|---------|---------|
| 信息流 CTR + 时长 | MMoE / PLE |
| 电商 CTR → CVR | ESMM / ESM2 / AITM（消除 SSB + DS） |
| 多入口推荐 | MMoE + scene-aware gating |
| 强冲突任务（点击 vs 收藏） | PLE + PCGrad |
| 长会话动态调权 | RMTL（RL 风格） |

**MBR 工业落地：**

1. **行为图 GNN（KMCLR / S-MBRec）：** 每种行为构图 → 跨行为消息传递
2. **多视图对比学习：** 用 click / buy 行为构造正例对，提升表征
3. **生成式范式（MBGen）：** 把行为 + item 统一 tokenize，inherit LLM 的 scaling
4. **行为加权 loss：** 高价值行为（buy）权重高于低价值行为（click）

**Negative Transfer 的 5 个工程缓解手段：**

1. PLE 任务隔离
2. PCGrad 梯度投影
3. GradNorm / Uncertainty Weighting 自适应权重
4. RMTL 风格 RL 调权
5. MTL-DML 跨塔蒸馏

实践中通常组合 1+3 或 1+5。

## 四、面试考点

1. MMoE → PLE → CGC 的演进动机？PLE 解决了 MMoE 什么具体问题？
2. ESMM 的 entire-space 训练解决了哪两类偏差（SSB / DS）？
3. PCGrad、CAGrad、Nash-MTL 等梯度调和方法的区别？
4. Negative Transfer 在多任务推荐里的根因？
5. 多行为推荐为什么不能简单 concat 不同行为序列？MBGen 用什么 trick 区分？
6. RMTL 中 actor-critic 选 PPO / SAC 还是 DDPG？session-level reward 怎么定义？
7. PLE + PCGrad 与 RMTL 在工业上的取舍？

## 参考

- [MBGen: Multi-Behavior Generative Recommendation](https://arxiv.org/abs/2405.16871)
- [Multi-Behavior Recommender Systems: A Survey](https://arxiv.org/abs/2503.06963)
- [Deep Mutual Learning across Task Towers](https://arxiv.org/abs/2309.10357)
- [Multi-Task Recommendations with Reinforcement Learning (RMTL)](https://arxiv.org/abs/2302.03328)
- [Multi-Task Deep Recommender Systems: A Survey](https://arxiv.org/abs/2302.03525)
