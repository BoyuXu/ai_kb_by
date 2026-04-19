# LLM 对齐 RL 方法全景图（2023-2026）

> 知识卡片 | 创建：2026-04-13 | 领域：llm-infra/alignment
> 关联：[[GRPO大模型推理RL算法]] | [[RLVR_vs_RLHF后训练路线]] | [[20260411_inference_optimization_and_alignment]]

---

## 一、谱系总览

```
                        RLHF (PPO, 2022)
                            │
            ┌───────────────┼───────────────┐
            │               │               │
      去 RM 路线      去 Critic 路线     混合/极简路线
            │               │               │
     ┌──────┴──────┐   ┌───┴────────┐    ┌─────┴─────┐
     DPO  SimPO    │  GRPO  DAPO   │  REINFORCE++ RLOO
     IPO  ORPO     │  Dr.GRPO     │  ReMax  RAFT
     KTO  CPO      │  PRIME       │  RSO
     SPPO TDPO     │  Critique-GRPO│
                   │  Off-Policy GRPO
                   │  TF-GRPO     │
            │      │            │
            └──┬───┘       ┌───┘
               │           │
          生成式偏好路线    │
            GPO  GSPO     │
            Online DPO ───┘
```

### 三大路线的底层逻辑

| 路线 | 砍掉了什么 | 代价 | 最佳场景 |
|------|-----------|------|----------|
| 去 RM（DPO 系） | Reward Model + Critic | 只有二元偏好信号，信息密度低 | 有偏好标注数据的通用对齐 |
| 去 Critic（GRPO 系） | Critic/Value Network | 需要在线采样 + RM 推理 | 有可验证 reward 的 reasoning |
| 混合极简 | 尽可能多砍 | 方差控制较难 | 快速实验、资源受限 |

---

## 二、逐方法详解

### A. 经典基线

#### PPO（Proximal Policy Optimization）
- **架构**：4 个模型 — policy π_θ, reference π_ref, reward model R, value model V
- **核心公式**：
  ```
  L_PPO = E[min(ρ·A, clip(ρ, 1±ε)·A)]
  其中 ρ = π_θ(y|x) / π_old(y|x), A = GAE(R, V)
  ```
- **优点**：通用性最强，理论完备
- **缺点**：4 模型显存爆炸，超参敏感，训练不稳定
- **代表作**：InstructGPT, ChatGPT, Claude

#### REINFORCE++
- **改进**：砍掉 value model，用 `reward - baseline` 替代 GAE
- **baseline**：batch 内 reward 均值（或指数滑动平均）
- **保留**：PPO 的 clip ratio + KL penalty
- **定位**：PPO 的 drop-in 极简替代，OpenRLHF 默认支持

#### RLOO（REINFORCE Leave-One-Out）
- **核心**：每个样本的 baseline = 同 prompt 下其余 K-1 个样本的 reward 均值
- **公式**：`baseline_i = (1/(K-1)) · Σ_{j≠i} R_j`
- **优势**：无偏估计，方差显著低于单点 baseline
- **与 GRPO 区别**：GRPO 用全组均值（含自身），RLOO 排除自身

#### ReMax
- **核心**：baseline = 组内最大 reward（贪心解码结果的 reward）
- **直觉**：和最优解比差多少，而非和平均水平比
- **效果**：在 summarization 任务上方差比 RLOO 更低

---

### B. 去 RM 路线（Direct Preference 系）

#### DPO（Direct Preference Optimization）— Stanford 2023
- **核心洞察**：RLHF 的最优 policy 有闭式解，可直接从偏好数据优化
- **公式**：
  ```
  L_DPO = -E[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
  ```
- **本质**：Bradley-Terry 偏好模型的闭式反解
- **优点**：只需 2 模型（policy + ref），训练稳定，pipeline 简单
- **缺点**：离线数据分布漂移、reward 信号粗（只有 win/lose）、容易 reward hacking
- **代表作**：Llama-2-Chat, Zephyr

#### IPO（Identity Preference Optimization）— Azar et al. 2023
- **改进**：去掉 Bradley-Terry 模型假设
- **公式**：直接优化偏好 margin 的平方损失
  ```
  L_IPO = E[(log(π_θ(y_w|x)/π_ref(y_w|x)) - log(π_θ(y_l|x)/π_ref(y_l|x)) - 1/(2β))²]
  ```
- **解决**：DPO 在偏好数据上容易 overfit 的问题（正则化更强）

#### KTO（Kahneman-Tversky Optimization）— Ethayarajh 2024
- **颠覆点**：**不需要偏好对**！只需要 "这个回答好" / "这个回答差" 的标签
- **理论基础**：前景理论的非对称价值函数 — 人对损失的敏感度 > 收益
- **公式**：对 good/bad 样本分别应用不同权重的 KL 散度损失
- **优势**：数据获取成本极低（不需要 A/B 配对），适合大规模标注
- **劣势**：理论保证弱于 DPO

#### SimPO（Simple Preference Optimization）— 2024
- **颠覆点**：连 reference model 都不要了
- **隐式 reward**：用序列平均 log-prob 本身当 reward（不除以 ref）
  ```
  r(y|x) = (1/|y|) · log π_θ(y|x)
  ```
- **优势**：省掉 ref model 的显存，inference 时 reward 和生成完全一致
- **效果**：AlpacaEval 2 上超过 DPO

#### ORPO（Odds Ratio Preference Optimization）— 2024
- **核心**：SFT + 对齐一步到位
- **方法**：在 SFT 的 NLL loss 上加一个 odds ratio 偏好正则项
- **优势**：单阶段训练，不需要先 SFT 再 alignment

#### SPPO（Self-Play Preference Optimization）— 2024
- **核心**：模型自博弈 — 同时充当 generator 和 judge
- **流程**：生成 → 自评 → 构造偏好对 → DPO 更新 → 迭代
- **优势**：零标注数据，完全自举

#### CPO（Contrastive Preference Optimization）— 2024
- **核心**：对比学习视角做偏好优化
- **方法**：将 chosen/rejected 视为正负样本对，用 InfoNCE-style loss

#### TDPO（Token-level DPO）— 2024
- **核心**：序列级偏好拆解为 token 级
- **优势**：更精细的信号传播，每个 token 都有梯度
- **实现**：forward KL divergence 在 token 粒度上的分解

#### Online DPO — 2024
- **问题**：离线 DPO 的偏好数据和当前 policy 分布不匹配
- **方案**：用当前 policy 在线采样 → 用 RM 判断偏好 → DPO 更新
- **本质**：DPO 的在线变体，弥补分布偏移

---

### C. 去 Critic 路线（Group RL 系）

#### GRPO（Group Relative Policy Optimization）— DeepSeek 2024
- **核心**：同 prompt 采样 G 个回答，组内归一化 advantage
- **公式**：
  ```
  A_i = (r_i - mean(r)) / std(r)          # 组内相对优势
  L = -E[min(ρ·A, clip(ρ, 1±ε)·A)] - β·KL(π_θ||π_ref)
  ```
- **砍掉**：Value model（用组内统计替代 GAE）
- **G 的选择**：DeepSeek 用 G=64，越大方差越低但采样成本越高
- **代表作**：DeepSeek-Math, DeepSeek-R1
- **详见**：[[GRPO大模型推理RL算法]]

#### DAPO（Decoupled Alignment Preference Optimization）— ByteDance 2025
- **解决 GRPO 三大问题**：
  1. **Entropy collapse**：策略快速坍缩到少数高分模式
  2. **过长回答**：模型倾向生成冗长输出刷分
  3. **Clip 不对称**：奖励好回答和惩罚差回答需要不同力度
- **四大改进**：
  - Clip-Higher：上下界解耦（如 ε_low=0.2, ε_high=0.28），鼓励探索
  - Dynamic Sampling：丢弃全对/全错的 prompt（信息量为零）
  - Token-level KL：不是序列级 KL，而是每个 token 独立计算
  - Overlong Penalty：超长回答单独施加惩罚
- **效果**：AIME 2024 上 50% → DeepSeek-R1-Zero 水平

#### Dr.GRPO（Doubly Robust GRPO）— 2025
- **问题**：GRPO 的组内归一化引入了系统性偏差
  ```
  原始 GRPO：A_i = (r_i - mean(r)) / std(r)
  这里 mean(r) 和 std(r) 依赖于 π_old 的采样分布，不是 π_θ 的
  ```
- **修正**：双重稳健估计（Doubly Robust），用重要性权重修正 advantage
- **效果**：数学任务上稳定提升 2-3%

#### PRIME（Process Reward + GRPO）— 2025
- **核心**：不只看最终答案，每一步推理都有奖励信号
- **架构**：GRPO 框架 + PRM（Process Reward Model）
- **PRM 来源**：用 Monte Carlo 回放估计每步的 value（哪步推理是关键转折点）
- **优势**：解决 outcome-based reward 的稀疏性问题
- **效果**：在多步数学推理上显著优于纯 GRPO

#### Critique-GRPO — Zhang et al. 2025 (arXiv: 2506.03106)
- **核心**：将自然语言 critique 注入 GRPO 的 RL 循环
- **方法**：失败样本 → LLM 生成 critique → critique 引导 refinement → 初始+修正回答共同参与 advantage 计算
- **解决**：纯数值 reward 的信息瓶颈（性能平台期、无法有效自我反思、持续性失败）
- **效果**：AIME 2024 上 +16.7% Pass@1（vs GRPO），Qwen 系列 +15~21.6%
- **详见**：[[20260420_grpo_variants_and_kv_cache]]

#### Off-Policy GRPO — Mroueh et al. 2025 (arXiv: 2505.22257)
- **核心**：将 GRPO 从 on-policy 扩展到 off-policy，用历史策略采样估计 advantage
- **理论**：证明 on-policy 和 off-policy 都有 policy improvement 下界
- **优势**：采样效率高（历史样本复用）、训练更稳定、显存更低
- **效果**：GSM8K/AIME/Math500 上显著优于或持平 on-policy GRPO
- **详见**：[[20260420_grpo_variants_and_kv_cache]]

#### Training-Free GRPO — Chen et al. 2025 (arXiv: 2510.08191)
- **范式转换**：从参数空间优化 → 上下文空间优化，模型权重完全冻结
- **方法**：策略 = frozen LLM + 可变经验上下文 C，用语义 advantage 迭代更新 C
- **优势**：无需模型权重访问（适用闭源 API），成本降 100 倍（$800→$8）
- **效果**：100 样本让 DeepSeek-V3.1 超越微调 32B 模型
- **详见**：[[20260420_grpo_variants_and_kv_cache]]

---

### D. 生成式偏好 & 统一框架

#### GPO（Generalized Preference Optimization）— 2024
- **核心**：统一框架，DPO/IPO/KTO/SPPO 都是其特例
- **方法**：参数化偏好损失函数 f(·)，不同 f 对应不同方法
  ```
  L_GPO = E[f(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
  f = -log σ → DPO;  f = (x-1/2β)² → IPO;  ...
  ```
- **价值**：理论统一 + 可以插值搜索最优 f

#### GSPO（Group Sampling Preference Optimization）— 2025
- **核心**：GRPO 的采样 + DPO 的 loss = 最佳混合
- **流程**：
  1. 同 prompt 采样 G 个回答（同 GRPO）
  2. RM 打分后构造组内 chosen/rejected 偏好对
  3. 用 DPO-style loss 更新（而非 PPO-style clip）
- **优势**：在线数据（解决 DPO 分布偏移）+ 稳定训练（DPO loss 比 clip 更稳）

---

### E. 其他重要变体

| 方法 | 核心思想 | 适用场景 |
|------|----------|----------|
| **RAFT** (Reward rAnked Fine-Tuning) | 采样 → RM 排序 → 只拿 top-k 做 SFT | 最简单的 RL 替代，低算力 |
| **RSO** (Rejection Sampling Optimization) | RAFT + 统计修正采样偏差 | 比 RAFT 更严谨 |
| **DNO** (Direct Nash Optimization) | 纳什均衡：模型对抗自身历史版本 | 自博弈 + 理论保证 |
| **WARP** (Weight Averaged Rewarded Policies) | 多轮 RLHF 后权重指数平均，防遗忘 | 长期多轮训练 |
| **NCA** (Noise Contrastive Alignment) | 噪声对比估计视角的偏好优化 | 大规模偏好数据 |
| **APO** (Anchored Preference Optimization) | 锚定 ref model，防止灾难性偏移 | 安全关键场景 |

---

## 三、关键对比总表

| 方法 | 需要 RM | 需要 Critic | 需要 Ref | 在线采样 | 信号类型 | 最佳场景 | 代表作 |
|------|---------|------------|---------|---------|---------|---------|--------|
| PPO | ✅ | ✅ | ✅ | ✅ | 连续分数 | 通用对齐 | ChatGPT |
| REINFORCE++ | ✅ | ❌ | ✅ | ✅ | 连续分数 | 资源受限 | - |
| RLOO | ✅ | ❌ | ✅ | ✅ | 连续分数 | 低方差在线 | - |
| DPO | ❌ | ❌ | ✅ | ❌ | 二元偏好 | 有偏好数据 | Llama-2-Chat |
| IPO | ❌ | ❌ | ✅ | ❌ | 二元偏好 | 防 overfit | - |
| KTO | ❌ | ❌ | ✅ | ❌ | good/bad 标签 | 无偏好对 | - |
| SimPO | ❌ | ❌ | ❌ | ❌ | 二元偏好 | 极简 | - |
| ORPO | ❌ | ❌ | ❌ | ❌ | 二元偏好 | 单阶段 | - |
| GRPO | ✅ | ❌ | ✅ | ✅ | 连续分数 | 数学/代码 | DeepSeek-R1 |
| DAPO | ✅ | ❌ | ✅(解耦) | ✅ | 连续分数 | 长 reasoning | - |
| Dr.GRPO | ✅ | ❌ | ✅ | ✅ | 连续分数 | 修正 GRPO bias | - |
| PRIME | ✅(PRM) | ❌ | ✅ | ✅ | 过程奖励 | 多步推理 | - |
| GSPO | ✅ | ❌ | ✅ | ✅ | 二元偏好(构造) | 混合 | - |
| Online DPO | ✅ | ❌ | ✅ | ✅ | 二元偏好 | 弥补分布偏移 | - |
| Critique-GRPO | ✅+NL | ❌ | ✅ | ✅ | 数值+自然语言 | 突破平台期 | Qwen/Llama |
| Off-Policy GRPO | ✅ | ❌ | ✅ | Off-policy | 连续分数 | 大规模高效训练 | IBM |
| TF-GRPO | ❌ | ❌ | ❌ | 冻结rollout | 语义advantage | 闭源API增强 | DeepSeek-V3.1 |

---

## 四、决策树

```
你有什么数据？
│
├── 有偏好对 (chosen, rejected)
│   ├── 需要在线采样吗？
│   │   ├── 不需要（离线）
│   │   │   ├── 数据质量高，怕 overfit → IPO
│   │   │   ├── 想省 ref model → SimPO / ORPO
│   │   │   └── 通用 → DPO
│   │   └── 需要（弥补分布偏移）→ Online DPO / GSPO
│   │
├── 只有 good/bad 标签（无配对）→ KTO
│
├── 没有标注数据，靠自采样
│   ├── 有可验证 reward（数学/代码/逻辑）
│   │   ├── 只看最终答案 → GRPO
│   │   ├── GRPO entropy collapse → DAPO
│   │   ├── GRPO 性能平台期 → Critique-GRPO
│   │   ├── 需要过程监督 → PRIME
│   │   ├── 想修正 advantage bias → Dr.GRPO
│   │   ├── 采样效率太低 → Off-Policy GRPO
│   │   └── 闭源 API / 无法微调 → Training-Free GRPO
│   │
│   ├── 有 RM 但非可验证 → PPO / REINFORCE++
│   │
│   └── 什么都没有，纯自举 → SPPO / RAFT
│
└── 预算极其有限
    ├── 单 GPU → SimPO（2 模型都不用）
    ├── SFT + 对齐一步到位 → ORPO
    └── 采样 top-k 做 SFT → RAFT
```

---

## 五、核心公式速查

### 5.1 PPO
```
L_PPO = E_t[min(ρ_t · A_t, clip(ρ_t, 1-ε, 1+ε) · A_t)]
ρ_t = π_θ(a_t|s_t) / π_old(a_t|s_t)
A_t = GAE(δ_t, λ) = Σ_l (γλ)^l · δ_{t+l}
δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

### 5.2 DPO
```
L_DPO = -E_{(x,y_w,y_l)}[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
```

### 5.3 GRPO
```
A_i = (r_i - μ_G) / σ_G        where μ_G = mean(r_1...r_G), σ_G = std(r_1...r_G)
L_GRPO = -E[min(ρ·A, clip(ρ, 1±ε)·A)] + β·KL(π_θ||π_ref)
```

### 5.4 DAPO（改进 GRPO）
```
L_DAPO = -E[min(ρ·A, clip(ρ, 1-ε_low, 1+ε_high)·A)]     # 解耦 clip
         + β·Σ_t KL_t(π_θ||π_ref)                          # token-level KL
         + λ·max(0, |y|-L_max)                              # overlong penalty
Dynamic Sampling: 丢弃 min(r)==max(r) 的 prompt
```

### 5.5 KTO
```
L_KTO = E_{y~good}[σ(β·log(π_θ/π_ref) - z_ref)] + λ·E_{y~bad}[σ(z_ref - β·log(π_θ/π_ref))]
z_ref = E_{x'}[β·KL(π_θ||π_ref)]
```

### 5.6 SimPO
```
r_SimPO(y|x) = (β/|y|) · log π_θ(y|x)           # 无需 ref model
L_SimPO = -E[log σ(r_SimPO(y_w|x) - r_SimPO(y_l|x) - γ)]   # γ = target margin
```

---

## 六、2025-2026 趋势与研判

### 6.1 GRPO 家族统治 Reasoning
- DeepSeek-R1 → DAPO → Dr.GRPO → PRIME → **Critique-GRPO / Off-Policy GRPO / TF-GRPO**
- 核心模式：**砍 Critic + 组内归一化 + 可验证 reward**
- 2025下半年三大新方向：NL反馈注入、Off-Policy高效采样、无训练上下文优化
- 一切数学/代码/逻辑任务都在用这个范式

### 6.2 DPO 系走向极简
```
DPO (2 model) → SimPO (1 model, 无 ref) → ORPO (单阶段)
                 KTO (无偏好对)
```
- 极限是：**一个模型、一阶段训练、只需 good/bad 标签**

### 6.3 融合是终局
- GSPO = GRPO 采样 + DPO loss
- PRIME = 过程奖励 + GRPO
- Online DPO = 在线采样 + DPO
- 未来会更多 mix-and-match

### 6.4 可验证 Reward 是关键变量
| 有可验证 Reward | 没有 |
|----------------|------|
| GRPO/DAPO 碾压一切 | DPO/KTO 性价比更高 |
| 自动标注，无限 scale | 受限于人类标注/LLM-as-Judge |
| 数学/代码/逻辑 | 聊天/安全/创意 |

### 6.5 工业实践路径
```
Phase 1: SFT（指令遵循基础能力）
Phase 2: DPO/SimPO 打底（通用对齐，成本低）
Phase 3: GRPO/DAPO 拉升（reasoning 能力，需可验证 reward）
Phase 4: PRIME 精修（过程奖励，解决多步推理）
```

---

## 七、面试高频题

### Q1: DPO 和 RLHF(PPO) 的本质区别？
**答**：DPO 证明了 RLHF 的最优策略有闭式解，把 RL 问题转化为分类问题。PPO 需要 4 个模型（policy + ref + RM + critic），DPO 只需 2 个（policy + ref）。代价是 DPO 只有二元偏好信号，信息密度低于 PPO 的连续 reward。

### Q2: GRPO 为什么在 reasoning 上赢 DPO？
**答**：三个原因：(1) 信号颗粒度 — 同时看 G 个样本的相对排序，梯度信息量大于 win/lose；(2) 在线探索 — 每轮采样带来新分布，DPO 离线偏好对容易让 policy 坍缩；(3) 可验证 reward — 数学答案自动验证 → 零噪声 → advantage 估计非常准。

### Q3: DAPO 解决了 GRPO 的什么问题？
**答**：三大问题：(1) Entropy collapse — 通过 Clip-Higher（上界 > 下界）鼓励探索；(2) 过长回答 — overlong penalty 惩罚冗长输出；(3) 无信息样本 — Dynamic Sampling 丢弃全对/全错 prompt。

### Q4: KTO 的创新在哪？数据要求和 DPO 有什么区别？
**答**：KTO 基于 Kahneman 前景理论，对 "损失" 和 "收益" 非对称加权。最大创新是**不需要偏好对**，只需要 "这个好/差" 的单点标签。DPO 需要同一个 prompt 下的 (chosen, rejected) 配对，KTO 可以用不同 prompt 的独立标注。

### Q5: 如果你只有 1 张 A100 要做对齐，选什么方法？
**答**：SimPO 或 ORPO。SimPO 去掉了 ref model（只需 1 个模型在显存里），ORPO 更进一步把 SFT 和对齐合成一个阶段。如果连偏好数据都没有，用 RAFT（采样 → 排序 → top-k SFT）。

### Q6: GSPO 和 GRPO 的区别？
**答**：都是在线采样 G 个回答，但 loss 不同。GRPO 用 PPO-style 的 clip ratio loss（保留了 RL 味道），GSPO 用 DPO-style 的 log-sigmoid loss（把采样出的组内最好/最差构造为偏好对）。GSPO 训练更稳定，GRPO 探索性更强。

### Q7: 过程奖励（PRM）和结果奖励（ORM）有什么区别？PRIME 为什么重要？
**答**：ORM 只看最终答案对不对（稀疏信号），PRM 对每一步推理打分（密集信号）。PRIME 把 PRM 接入 GRPO 框架，让每步推理都有梯度。关键优势：解决多步推理中 "前 9 步对、第 10 步错" 时 ORM 给整体负分的问题。

---

## 八、参考文献

1. Schulman et al. (2017) — PPO
2. Rafailov et al. (2023) — DPO
3. Azar et al. (2023) — IPO
4. Ethayarajh et al. (2024) — KTO
5. Shao et al. (2024) — GRPO (DeepSeek-Math)
6. DeepSeek (2025) — DeepSeek-R1
7. ByteDance (2025) — DAPO
8. Meng et al. (2024) — SimPO
9. Hong et al. (2024) — ORPO
10. Wu et al. (2024) — Self-Play (SPPO)
11. Tang et al. (2024) — GPO
12. Zeng et al. (2024) — TDPO
13. Dong et al. (2023) — RAFT
14. Liu et al. (2024) — RSO
15. Munos et al. (2024) — DNO
16. Ramé et al. (2024) — WARP
17. Wang et al. (2025) — PRIME
18. Zhong et al. (2025) — Dr.GRPO
19. Yu et al. (2025) — GSPO
20. Zhang et al. (2025) — Critique-GRPO (arXiv:2506.03106)
21. Mroueh et al. (2025) — Off-Policy GRPO (arXiv:2505.22257)
22. Chen et al. (2025) — Training-Free GRPO (arXiv:2510.08191)
