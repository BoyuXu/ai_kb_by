# RLVR vs RLHF：LLM 后训练的两条路
> 📚 参考文献
> - [Rlvr Reinforcement Learning With Verifiable Rew...](../papers/daily/20260323_rlvr_reinforcement_learning_with_verifiable_rewards.md) — RLVR: Reinforcement Learning with Verifiable Rewards for ...
> - [Grpo-Group-Relative-Policy-Optimization-Llm-Rea...](../papers/daily/20260321_grpo-group-relative-policy-optimization-llm-reasoning.md) — GRPO: Group Relative Policy Optimization for Large Langua...
> - [Grpo-Group-Relative-Policy-Optimization-For-Lar...](../papers/daily/20260321_grpo-group-relative-policy-optimization-for-large-language-model-reasoning.md) — GRPO: Group Relative Policy Optimization for Large Langua...
> - [Kimi K1.5 Scaling Reinforcement Learning With Llms](../papers/daily/20260323_kimi_k1.5_scaling_reinforcement_learning_with_llms.md) — KIMI k1.5: Scaling Reinforcement Learning with LLMs
> - [Grpo Group Relative Policy Optimization](../papers/daily/20260322_grpo_group_relative_policy_optimization.md) — GRPO: Group Relative Policy Optimization for Large Langua...
> - [Llmorbit-A-Circular-Taxonomy-Of-Large-Language-...](../papers/daily/20260321_llmorbit-a-circular-taxonomy-of-large-language-models-from-scaling-walls-to-agentic-ai-systems.md) — LLMOrbit: A Circular Taxonomy of Large Language Models fr...
> - [Kvcache Compression For Long-Context Llm Infere...](../papers/daily/20260323_kvcache_compression_for_long-context_llm_inference_.md) — KVCache Compression for Long-Context LLM Inference: Metho...
> - [Efficiently-Aligning-Draft-Models-Via-Parameter...](../papers/daily/20260321_efficiently-aligning-draft-models-via-parameter-and-data-efficient-adaptation-for-speculative-decoding.md) — Efficiently Aligning Draft Models via Parameter- and Data...

> 知识卡片 | 创建：2026-03-23 | 更新：2026-03-26 | 领域：llm-infra

---

## 🆕 2026-03-26 深度整合更新

### DeepSeek-R1：RLVR 路线的里程碑验证
**三阶段训练流程**（今日新增细节）：
```
Phase 1 - Cold Start（SFT，~1000 样本）
  → 学会 <think>...</think> 格式，否则 RL 训练初期格式崩塌
  
Phase 2 - GRPO（纯 RL，无 Critic）
  → 奖励 = 答案正确性（0/1）+ 格式完整性奖励
  → 自发出现：自我反思、回溯、多步探索等"涌现"行为
  → AIME 2024：9% → 80%（无任何推理过程标注！）
  
Phase 3 - Rejection Sampling + SFT（蒸馏）
  → 用 R1 生成高质量 CoT 数据，蒸馏到 7B/14B 小模型
  → R1-Distill-7B 超越 GPT-4o（数学）
```

**R1 的关键技术选择**：
- 格式奖励（Format Reward）= `<think>` 必须正确闭合，防止格式退化
- Token-level KL 惩罚：防止 policy 漂离 reference model 太远（避免语言退化）
- 拒绝 Reward Hacking：监控异常长度的输出（极短或极长的推理链）

### LIMO 的反直觉发现
```
传统假设：教会模型数学推理需要 10万+ CoT 样本
LIMO 发现：817 个高质量样本 > 10万+ 低质量样本

原因：Qwen2.5-32B 预训练时已见过大量数学内容
     推理能力已内化，SFT 只需"格式激活"
     
工程含义：
  - 数据质量 >> 数据数量（对强预训练模型）
  - 高质量样本标准：难度适中 + 步骤完整 + 多样化主题
  - 可以用 R1/GPT-4 生成合成高质量 CoT 数据替代人工标注
```

### Qwen3 的混合推理创新
```
Qwen3 核心设计：统一 thinking / non-thinking 两种模式
  - thinking=True：激活 <think> 推理链（类 R1）
  - thinking=False：直接输出（类 GPT-4，适合日常对话）
  
为什么重要：
  - 不需要维护两个独立模型（推理模型 + 对话模型）
  - 用户可按 API 参数按需控制计算成本
  - Qwen3-235B-A22B（MoE）：235B 参数，每次只激活 22B
    → 推理成本 = 22B Dense 模型，容量 = 235B
```

## 📐 核心公式与原理

### 📐 RLHF 目标函数推导

**核心目标函数：**

$$
\mathcal{L}_{\text{RLHF}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}}\left[\min\left(\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\hat{A}_{\text{RM}},\ \text{clip}(\cdot, 1-\epsilon, 1+\epsilon)\hat{A}_{\text{RM}}\right) - \beta\, \mathbb{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]\right]
$$

**推导步骤：**

1. **奖励模型训练（RM）**：用人类标注的偏好对比数据 $(y_w, y_l, x)$ 训练奖励模型 $r_{\phi}(y|x)$：
   $$\mathcal{L}_{\text{RM}} = -\mathbb{E}\left[\log\sigma(r_\phi(y_w|x) - r_\phi(y_l|x))\right]$$
   其中 $\sigma$ 是 logistic 函数，$y_w$ 是被人类评为更好的回答。

2. **Advantage 估计（关键差异）**：PPO 用 Critic 网络 $V(x)$ 估计价值：
   $$\hat{A}_{\text{RM}} = r_\phi(y|x) - V(x)$$
   这引入了额外的价值网络，内存占用翻倍。

3. **PPO 更新**：对每个样本应用 PPO 的 surrogate loss：
   $$\mathcal{L}_{\text{RLHF}} = \min\left(r_t \hat{A},\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}\right) - \beta\, \mathbb{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]$$
   其中 $r_t = \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 是重要性权重（importance ratio）。

4. **奖励黑箱的风险**：奖励模型可能被策略欺骗（reward hacking），生成表面高分但无用的回答。

**符号说明：**

| 符号 | 含义 |
|------|------|
| $\pi_\theta$ | 策略网络（要优化的 LLM） |
| $\pi_{\text{ref}}$ | 参考策略（固定不变的 SFT 初始模型） |
| $r_\phi(y\|x)$ | 奖励模型，需单独训练 |
| $\hat{A}_{\text{RM}}$ | 奖励模型输出的价值估计（可被欺骗） |
| $\epsilon$ | clip 范围，通常 0.2，防止单步更新过大 |
| $\beta$ | KL 惩罚系数，约束 $\pi_\theta$ 不能偏离 $\pi_{\text{ref}}$ 太远 |

**直观理解：** RLHF 像「请老师评作文」——老师有可能被花言巧语迷惑，给低质量但文采好的作文打高分。奖励模型也一样，会被模型的「欺骗能力」所迷惑。

---

### 📐 RLVR 目标函数推导

**核心目标函数（以 GRPO 为例）：**

$$
\mathcal{L}_{\text{RLVR}}(\theta) = \frac{1}{G}\sum_{i=1}^{G}\left[\min\left(r_i(\theta)\hat{A}_i,\ \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i\right) - \beta\, \mathbb{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]\right]
$$

其中 $\hat{A}_i$ 由**程序验证的客观奖励**而非学出来的模型计算。

**推导步骤：**

1. **客观奖励来源**：对同一个问题 $x$ 采样 $G$ 个回答 $\{y_1, \ldots, y_G\}$，用规则打分（e.g. 数学题答对 = 1，答错 = 0）：
   $$r_i = \begin{cases} 1 & \text{if } y_i \text{ 正确} \\ 0.5 & \text{if } y_i \text{ 部分正确} \\ 0 & \text{if } y_i \text{ 错误} \end{cases}$$
   关键：这些奖励**不可被欺骗**，由外部程序验证（数学答案、代码编译、单元测试等）。

2. **组内相对优势**（GRPO 的核心创新）：计算同 prompt 内的 advantage 归一化：
   $$\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r}, \quad \mu_r = \frac{1}{G}\sum_{j=1}^{G} r_j, \quad \sigma_r = \sqrt{\frac{1}{G}\sum_{j=1}^{G}(r_j - \mu_r)^2}$$
   **关键洞察**：不需要额外的 Critic 网络或价值函数，组内平均和方差就是 baseline。

3. **重要性权重（全序列粒度）**：计算整个回答的概率比：
   $$r_i(\theta) = \frac{\pi_\theta(y_i|x)}{\pi_{\text{ref}}(y_i|x)} = \prod_{t=1}^{|y_i|}\frac{\pi_\theta(y_{i,t}|x, y_{i,<t})}{\pi_{\text{ref}}(y_{i,t}|x, y_{i,<t})}$$

4. **KL 约束防止漂移**：
   $$\mathbb{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] = \mathbb{E}_{y \sim \pi_\theta}\left[\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right]$$
   确保策略不能为了追求高奖励而彻底改变原有的语言分布。

**符号说明：**

| 符号 | 含义 |
|------|------|
| $G$ | 同一问题的采样回答数（8-16） |
| $r_i$ | 规则验证的奖励（如答对/答错），**不可欺骗** |
| $\mu_r, \sigma_r$ | 组内奖励的均值和标准差，用作 baseline 代替 Critic |
| $\hat{A}_i$ | 组内归一化优势，范围约 [-3, 3]（3σ 截断） |
| $r_i(\theta)$ | 新策略与参考策略在整个回答上的概率比 |
| $\beta$ | KL 惩罚系数（0.01-0.1），防止策略漂移 |

**直观理解：** RLVR 像「对数学答案」——1+1 = 2 是不可否认的客观事实，模型无法欺骗。同时用组内排名作为学习信号（比同学好多少），而非依赖额外网络（Critic），既稳定又节省显存。

---

### 📐 RLHF vs RLVR 效率对比

**显存占用对比：**

| 方案 | 模型数量 | 显存占用 | 计算效率 | 适用范围 |
|------|---------|--------|--------|---------|
| RLHF + PPO | $\pi_\theta$, $\pi_{\text{ref}}$, $V_\phi$, $r_\phi$ | ~4x | 低（4个模型） | 开放生成 |
| RLVR + GRPO | $\pi_\theta$, $\pi_{\text{ref}}$ | ~2x | 高（无额外网络） | 可验证任务 |

**效果对比（DeepSeek-R1 数据）：**

$$
\text{AIME Accuracy: } \begin{cases} \text{GPT-4o (RLHF)} &= 9\% \\ \text{R1-Zero (RLVR)} &= 80\% \end{cases}
$$

在可验证的数学任务上，RLVR 的效果远超 RLHF（无需人工 CoT 标注）。

---

**一句话**：RLHF 用「人工打分员」（奖励模型）来判断好坏，RLVR 直接用「考试答案对错」来判断——后者更稳定，但只适用于有客观标准的任务（数学/代码/逻辑）。

**类比**：RLHF 像「请老师评作文」（主观，老师可能被文采骗），RLVR 像「对数学答案」（客观，1+1 就是 2）。DeepSeek-R1 和 Kimi K1.5 都用了 RLVR 路线，结果 LLM 自发「学会思考」。

---

## 核心机制对比

```
RLHF（Reinforcement Learning from Human Feedback）
├── 流程：SFT → 奖励模型训练 → PPO 优化 → 模型
├── 奖励来源：奖励模型（RM）打分，RM 由人类偏好数据训练
├── 风险：Reward Hacking（模型欺骗奖励模型，获高分但不完成任务）
├── 适用：开放生成（写作、对话、创意内容）
└── 代表：InstructGPT, ChatGPT

RLVR（Reinforcement Learning with Verifiable Rewards）
├── 流程：SFT（可选）→ RLVR 直接用客观奖励 → 模型
├── 奖励来源：程序验证（答案对错、测试通过与否）
├── 风险：覆盖任务有限（不适合开放问答）
├── 适用：数学、代码、形式逻辑等有客观标准的任务
└── 代表：DeepSeek-R1-Zero, Kimi K1.5
```

---

## RL 算法进化线

```
PPO（2017，Schulman）
├── 需要 Actor + Critic 两个网络（内存 2x）
├── 复杂超参（clip ratio, GAE λ, entropy coef）
└── 工业标准，但工程实现复杂

GRPO（Group Relative Policy Optimization，DeepSeek）
├── 去掉 Critic 网络 → 显存节省 40-50%
├── 核心：同一 prompt 采样 G 个回答，用组内相对奖励归一化 advantage
├── KL 惩罚防止策略漂移（token 级别）
└── 效果：DeepSeek-Math 7B MATH +5.2pp；计算效率 ~1.4x PPO

REINFORCE++（简化版 PPO）
├── 无 Critic，无 GAE，更简单
└── 适合 verifiable reward 场景（奖励稀疏但精确）

今日关键结论（RLVR 论文）：
- 过程奖励（PRM）> 结果奖励（ORM）：每步推理打分 > 只看最终答案
- 格式奖励（format reward）有效：强制<think>结构，质量稳定提升
- GRPO 在 RLVR 场景实际效果最佳
```

---

## DeepSeek-R1 的惊人现象：自发 CoT 涌现

```
实验（R1-Zero）：
  输入：纯数学问题，零 SFT，只有 RLVR
  输出：模型自发学会 <think>...</think> 格式，产生 CoT 推理

为什么会这样？
  强化学习压力 → 模型发现「先想清楚再回答」得到更高验证奖励
  → 推理是一种被奖励强化出来的「工具行为」

类比：让 AI 做数学题得分高 → AI 自己发明了演算纸
```

---

## 工业落地实践

| 能力需求 | 推荐方案 | 原因 |
|---------|---------|------|
| 数学/代码推理 | RLVR + GRPO | 客观奖励，稳定，无 reward hacking |
| 写作/对话助手 | RLHF + PPO/DPO | 无客观标准，需人类偏好 |
| 多任务混合 | RLHF（软奖励）+ RLVR（硬奖励）组合 | 互补覆盖 |
| 低成本微调 | DPO（无需在线 RL）| 效果 ~PPO 的 80%，实现简单 |

**广告推荐中的 RLVR 应用**：
- CTR/CVR 预测 → 客观奖励（用户行为）→ 可直接用 RLVR 框架
- 创意文案生成 → 主观好坏 → 仍需 RLHF 或 LLM-as-judge
- 出价策略优化（AutoBid）→ 可验证收益 → RLVR 天然适配

---

## 面试考点

1. **Q: RLVR 和 RLHF 最核心的差异？**
   A: 奖励来源不同：RLHF 用学出来的奖励模型（可被欺骗），RLVR 用程序验证（客观不可欺骗）

2. **Q: Reward Hacking 是什么？如何防止？**
   A: 模型找到奖励模型的「漏洞」，表面得分高但实际质量差；防止：奖励模型多样化、KL 约束防漂移、用 RLVR 替代

3. **Q: GRPO 为什么可以不需要 Critic？**
   A: PPO 的 Critic 估计 baseline（减少方差），GRPO 用同 prompt 的 G 个回答的平均奖励作 baseline，组内相对比较代替绝对值估计

4. **Q: 过程奖励（PRM）vs 结果奖励（ORM）？**
   A: ORM 只看最终答案（稀疏，第一步错也得负 reward）；PRM 对每步推理打分（密集，更精确的学习信号）

5. **Q: DeepSeek-R1-Zero 的 CoT 涌现说明了什么？**
   A: 推理能力可以通过纯 RL 自发获得，不一定需要 CoT 数据 SFT；意味着 scaling RL 可能比 scaling data 更有效

### Q1: KV Cache 为什么是推理瓶颈？
**30秒答案**：KV Cache 大小 = 2×layers×heads×dim×seq_len×dtype_size。长序列时内存爆炸。优化：①Multi-Query Attention；②量化（FP8/INT4）；③页注意力（vLLM PagedAttention）；④压缩（H2O/SnapKV）。

### Q2: RLHF 和 DPO 的区别？
**30秒答案**：RLHF：训练 reward model + PPO 优化，需要在线采样。DPO：直接用偏好数据优化策略，跳过 reward model，更简单稳定。效果接近但 DPO 训练成本更低。

### Q3: 模型量化的原理和影响？
**30秒答案**：FP32→FP16→INT8→INT4：每次减半存储和计算。①Post-training Quantization：训练后量化，简单但可能损失精度；②Quantization-Aware Training：训练中模拟量化，精度损失更小。

### Q4: Speculative Decoding 是什么？
**30秒答案**：用小模型（draft model）快速生成多个候选 token，大模型一次性验证。如果小模型猜对 n 个，等于大模型「跳过」了 n 步推理。加速比取决于小模型的准确率。

### Q5: MoE 的优势和挑战？
**30秒答案**：优势：同参数量下推理更快（只激活部分 Expert），或同计算量下容量更大。挑战：①负载均衡（部分 Expert 过热/闲置）；②通信开销（分布式 Expert 选择）；③训练不稳定。

### Q6: PagedAttention（vLLM）的核心思想？
**30秒答案**：借鉴操作系统虚拟内存分页，将 KV Cache 切分为固定大小的「页」，按需分配。解决传统方式预分配最大序列长度导致的内存浪费（平均浪费 60-80%）。

### Q7: Continuous Batching 是什么？
**30秒答案**：传统 Static Batching 等最长序列完成才处理下一批。Continuous Batching 每个 token step 都可以加入新请求，序列完成立即释放。将 GPU 利用率从 ~30% 提升到 ~80%。

### Q8: GRPO 和 PPO 的核心区别？
**30秒答案**：PPO 需要 value network 估计 advantage；GRPO 用 group 内的相对奖励替代 value network：采样 G 个输出，用组内排名作为 baseline。更简单、更稳定、不需要额外模型。

### Q9: RAG vs Fine-tuning 怎么选？
**30秒答案**：RAG：知识频繁更新、需要引用来源、不想改模型。Fine-tuning：任务固定、需要特定风格/格式、追求最低延迟。两者可结合：fine-tune 后的模型 + RAG 检索。

### Q10: LLM 推理的三大瓶颈？
**30秒答案**：①Prefill 阶段：计算密集（大量矩阵乘）；②Decode 阶段：内存密集（KV Cache 读写）；③通信：多卡推理时的 AllReduce。优化方向：FlashAttention（①）、PagedAttention（②）、TP/PP 并行（③）。
