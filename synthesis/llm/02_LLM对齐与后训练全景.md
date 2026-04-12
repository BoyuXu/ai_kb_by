# LLM 对齐与后训练全景：从 RLHF 到 GRPO 到 RLVR

> 综合自 6 篇 synthesis | 更新：2026-04-13 | 领域：LLM 对齐/后训练
> 关联：[[concepts/multi_objective_optimization]] | [[01_LLM推理优化全景]]

---

## 谱系总览

```
                    RLHF (PPO, 2022)
                        │
        ┌───────────────┼───────────────┐
        │               │               │
  去 RM 路线      去 Critic 路线     混合/极简路线
        │               │               │
  DPO  SimPO       GRPO  DAPO     REINFORCE++ RLOO
  IPO  ORPO       Dr.GRPO PRIME   ReMax  RAFT
  KTO  CPO
```

### 三大路线

| 路线 | 砍掉了什么 | 代价 | 最佳场景 |
|------|-----------|------|----------|
| 去 RM（DPO 系） | RM + Critic | 只有二元偏好信号 | 有偏好标注数据的通用对齐 |
| 去 Critic（GRPO 系） | Value Network | 需在线采样 + RM 推理 | 有可验证 reward 的 reasoning |
| 混合极简 | 尽可能多砍 | 方差控制较难 | 快速实验、资源受限 |

---

## 一、经典基线

### PPO（Proximal Policy Optimization）

$$
\mathcal{L}_{PPO} = \mathbb{E}_t\left[\min\left(\rho_t \hat{A}_t,\ \text{clip}(\rho_t, 1\!-\!\epsilon, 1\!+\!\epsilon)\hat{A}_t\right)\right]
$$

- 4 模型：policy $\pi_\theta$, reference $\pi_{ref}$, reward model $R$, value model $V$
- 显存 ~4x $M_\theta$，通用性最强，超参敏感
- 代表：InstructGPT, ChatGPT, Claude

---

## 二、去 RM 路线（Direct Preference 系）

### DPO 推导

**核心洞察**：RLHF 最优策略有闭式解，可直接从偏好数据优化。

$$
\mathcal{L}_{DPO} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]
$$

**推导**：从 Bradley-Terry 偏好模型反演 reward → 代入 PPO → 消去 RM → 得到 DPO loss。本质是"无中间商赚差价"。

### DPO 系变体

| 方法 | 核心改进 | 特色 |
|------|---------|------|
| IPO | 去掉 BT 假设，平方损失 | 更强正则化，防 overfit |
| KTO | **不需要偏好对**，只需 good/bad 标签 | 前景理论，数据获取成本极低 |
| SimPO | **去掉 ref model**，用序列 log-prob 当 reward | 极简，AlpacaEval 超 DPO |
| ORPO | SFT + 对齐一步到位 | 单阶段训练 |
| SPPO | 模型自博弈（generator + judge） | 零标注，完全自举 |
| Online DPO | 当前 policy 在线采样 → RM 判断 → DPO 更新 | 弥补分布偏移 |

**极简化趋势**：DPO (2 model) → SimPO (1 model) → ORPO (单阶段)

---

## 三、去 Critic 路线（Group RL 系）

### GRPO 核心公式

$$
\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G}, \quad \mathcal{L}_{GRPO} = -\frac{1}{G}\sum_{i=1}^{G}\left[\min\left(r_i(\theta)\hat{A}_i,\ \text{clip}(r_i(\theta), 1\!\pm\!\epsilon)\hat{A}_i\right) - \beta \mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]\right]
$$

**直觉**：组内排名打分——不需要知道"绝对应该拿多少分"（Critic），只需知道"比同学高了多少"。

**PPO → GRPO 显存对比**：PPO ~4x $M_\theta$（4 模型），GRPO ~2x $M_\theta$（2 模型），70B 模型从 ~560GB 降至 ~280GB。

### DAPO（字节, 2025）—— 解决 GRPO 三大问题

| 问题 | DAPO 解法 |
|------|----------|
| Entropy collapse | Clip-Higher（$\epsilon_{high} > \epsilon_{low}$），鼓励探索 |
| 过长回答 | Overlong Penalty |
| 无信息样本 | Dynamic Sampling（丢弃全对/全错 prompt） |

$$
\mathcal{L}_{DAPO} = -\mathbb{E}\left[\min\left(\rho \hat{A},\ \text{clip}(\rho, 1\!-\!\epsilon_{low}, 1\!+\!\epsilon_{high})\hat{A}\right)\right] + \beta\sum_t KL_t + \lambda\max(0, |y|\!-\!L_{max})
$$

### Dr.GRPO（双重稳健估计）

修正 GRPO 的系统性偏差：advantage 中的 mean/std 依赖 $\pi_{old}$ 采样分布，用重要性权重修正。数学任务稳定提升 2-3%。

### PRIME（过程奖励 + GRPO）

不只看最终答案，每步推理都有奖励信号（PRM）。解决"前 9 步对、第 10 步错"时 ORM 给整体负分的问题。

---

## 四、RLVR — 可验证奖励的 RL

| 维度 | RLHF | RLVR |
|------|------|------|
| 奖励来源 | 学出来的 RM（可被欺骗） | 程序验证（客观不可欺骗） |
| 标注成本 | 高 | 极低（自动验证） |
| 适用 | 通用对齐 | 数学/代码/逻辑 |
| 代表 | ChatGPT | **DeepSeek-R1**, Kimi k1.5 |

### DeepSeek-R1 三阶段训练

```
Phase 1: Cold Start SFT (~1000 样本) → 学会 <think> 格式
Phase 2: GRPO (纯 RL) → 奖励=答案正确性(0/1)+格式完整性 → AIME 9%→80%
Phase 3: Rejection Sampling + SFT → 蒸馏到 7B/14B
```

**惊人发现**：R1-Zero 在纯 RL 训练中自发涌现 CoT（自我反思、回溯、多步探索），无需任何推理过程标注。

### LIMO 反直觉发现

817 条高质量样本 > 10 万低质量样本。原因：强预训练模型已内化推理能力，SFT 只需"格式激活"。

---

## 五、测试时计算扩展（Test-Time Compute Scaling）

### 两种机制

1. **搜索 + PRM 验证**：采样 N 个候选，PRM 逐步打分，Beam Search
2. **自适应分布修正**：模型迭代修正自己的回答

$$
\pi^*(q) = \arg\max_{\pi \in \Pi} \mathbb{E}_{y \sim \pi(q)}[R(y,q)] \quad \text{s.t.} \quad C(\pi) \leq B
$$

**关键发现**：小模型 + 充足 test-time compute **超越 14x 更大模型**（FLOPs-matched）。

### PITA — 推理时对齐

训练小型 Preference Guidance Policy，推理时修改 frozen LLM 的 token 概率，无需修改权重。

$$
p_{aligned}(x_t | x_{<t}) = p_{LLM}(x_t | x_{<t}) \cdot g_\phi(x_t | x_{<t}, \text{pref})
$$

**应用**：多用户个性化、快速适应新偏好、A/B 测试。

---

## 六、关键对比总表

| 方法 | RM | Critic | Ref | 在线采样 | 信号 | 最佳场景 |
|------|-----|--------|-----|---------|------|---------|
| PPO | ✅ | ✅ | ✅ | ✅ | 连续分数 | 通用对齐 |
| DPO | ❌ | ❌ | ✅ | ❌ | 二元偏好 | 有偏好数据 |
| KTO | ❌ | ❌ | ✅ | ❌ | good/bad | 无偏好对 |
| SimPO | ❌ | ❌ | ❌ | ❌ | 二元偏好 | 极简 |
| GRPO | ✅ | ❌ | ✅ | ✅ | 连续分数 | 数学/代码 |
| DAPO | ✅ | ❌ | ✅ | ✅ | 连续分数 | 长 reasoning |
| PRIME | ✅(PRM) | ❌ | ✅ | ✅ | 过程奖励 | 多步推理 |

---

## 七、决策树

```
有偏好对？
├── 是 → 需要在线采样？
│   ├── 不需要 → DPO（通用）/ IPO（防overfit）/ SimPO（省ref）
│   └── 需要 → Online DPO / GSPO
├── 只有 good/bad 标签 → KTO
├── 自采样 + 可验证 reward
│   ├── 只看最终答案 → GRPO
│   ├── GRPO entropy collapse → DAPO
│   ├── 需要过程监督 → PRIME
│   └── 修正 advantage bias → Dr.GRPO
└── 预算极其有限
    ├── 单 GPU → SimPO
    └── SFT+对齐一步到位 → ORPO
```

---

## 八、工业实践路径

```
Phase 1: SFT（指令遵循基础能力）
Phase 2: DPO/SimPO 打底（通用对齐，成本低）
Phase 3: GRPO/DAPO 拉升（reasoning 能力，需可验证 reward）
Phase 4: PRIME 精修（过程奖励，解决多步推理）
```

---

## 面试高频 Q&A

### Q1: DPO 和 RLHF 的本质区别？
**30秒**：DPO 证明 RLHF 最优策略有闭式解，把 RL 转化为分类问题。PPO 需 4 模型，DPO 只需 2 模型。代价是 DPO 只有二元信号，信息密度低。

### Q2: GRPO 为什么在 reasoning 上赢 DPO？
**30秒**：(1) 同时看 G 个样本的相对排序，信息量大于 win/lose；(2) 在线探索带来新分布；(3) 可验证 reward 零噪声。

### Q3: DAPO 解决了 GRPO 什么问题？
**30秒**：Entropy collapse（Clip-Higher 鼓励探索）、过长回答（overlong penalty）、无信息样本（Dynamic Sampling 丢弃全对/全错）。

### Q4: RLVR 和 RLHF 最核心的差异？
**30秒**：奖励来源。RLHF 用学出来的 RM（可被欺骗），RLVR 用程序验证（客观不可欺骗）。DeepSeek-R1 证明纯 RLVR 可自发涌现 CoT。

### Q5: Test-time compute scaling 的两种机制？
**30秒**：(1) 搜索+PRM 验证（采样多个候选逐步打分选最优）；(2) 自适应修正（模型迭代修正答案）。关键是按难度分配计算量。

### Q6: 1 张 A100 做对齐选什么？
**30秒**：SimPO（去掉 ref model）或 ORPO（SFT+对齐一步到位）。没偏好数据用 RAFT（采样→排序→top-k SFT）。

---

## 记忆助手

- **RLHF = 师徒制**：先学基本功(SFT) → 请裁判打分(RM) → 根据反馈改进(PPO)
- **DPO = 直接考试**：不需要裁判，直接看好坏答案配对题
- **GRPO = 小组互评**：组内对比打分，不需要 Critic
- **RLVR = 机器判卷**：数学/代码有标准答案，不可欺骗
- **对齐四代口诀**：RLHF(全家桶) → DPO(去RM) → GRPO(去Critic) → RLVR(去人类标注)

---

## 相关概念

- [[concepts/attention_in_recsys|Attention 在搜广推中的演进]]
- [[concepts/multi_objective_optimization|多目标优化]]
- [[06_知识蒸馏技术全景|知识蒸馏与对齐的融合 (P9)]]
