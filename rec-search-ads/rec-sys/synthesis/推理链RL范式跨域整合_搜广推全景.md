# 推理链 × RL 范式跨域整合：搜广推的共同演进方向（2026-03-30）

> 🌐 跨域综述 | 覆盖：rec-sys × ads × search × llm-infra | 深度整合 | 老师视角

---

## 🔭 一、全景：今日31篇论文的共同信号

读完今天 31 篇论文，有一个压倒性的信号：

**推理链（Chain-of-Thought）+ 强化学习（RL/GRPO）正在成为搜广推全链路的统一训练范式。**

```
2023-2024 (第一波)              2025-2026 (今天的论文，第二波)
──────────────────────          ─────────────────────────────────────
LLM 作为特征提取器               LLM 作为推理引擎（会"想"才能行动）
RAG = 被动检索                   Search-R1 = 主动推理驱动搜索
CTR 模型 = 判别式打分            CADET = 生成式序列建模
RL 出价 = 策略梯度优化           GAVE = Diffusion + 价值引导探索
推荐 = 召回→粗排→精排 pipeline  生成式推荐 = 端到端 autoregressive
```

这不是"LLM 应用于推荐/搜索/广告"的表层尝试，而是**训练范式层面的替换**。

---

## 🧩 二、三个领域的同构模式

### 模式一：CoT + RL 训练推理质量

| 领域 | 论文 | 任务 | Reward |
|------|------|------|--------|
| Rec-Sys | OneRec-Think | 推荐生成 | Hit@k + NDCG |
| Rec-Sys | GR2 | 重排 | NDCG@10 |
| Search | Rank-R1 | 文档重排 | NDCG/MRR |
| Search | Search-R1 | 主动搜索 | 最终答案正确性 |
| Ads | — | 出价策略 | ROI / GMV |
| LLM-Infra | LIMO | 数学推理 | 答案正确性 |

**共同结构**：
```
大 LLM 生成推理链（教师）
  → SFT 蒸馏到小模型（学生）
    → GRPO 以任务指标为 reward 强化学习
      → Constrained Decoding 保证输出合法性
```

**为什么 GRPO 特别适合这些任务**：

$$
\hat{r}_i = \frac{r_i - \bar{r}}{\text{std}(r)}, \quad \text{无需 critic 网络，组内相对 reward 归一化}
$$

搜广推的 reward（Hit Rate、NDCG、ROI）都是"离散或稀疏"的——PPO 的 critic 网络在稀疏 reward 下难以收敛，GRPO 天然规避这一问题。

---

### 模式二：统一模型替代多阶段 Pipeline

| 领域 | 被替代的多阶段系统 | 统一模型 |
|------|-------------------|---------|
| Rec-Sys | 双塔召回 + DIN 精排 + 规则重排 | OneRec / UniGRF (end-to-end) |
| Rec-Sys | MMOE + task-specific head | MTFM (task token 路由) |
| Ads | CTR + CVR + LTV 三模型 | UniROM (统一 eCPM) |
| Search | 独立 retrieval + reranking | Search-R1 (主动推理统一) |

**工程动机**（比 novelty 更重要）：
1. 误差传播：每个阶段的误差在下一阶段放大
2. 目标不一致：召回优化 Recall，精排优化 AUC，各自为政
3. 维护成本：3个模型 = 3套训练 + 3套部署 + 3套监控
4. 优化空间：联合训练能找到单模型达不到的全局最优

---

### 模式三：大模型训练+小模型部署（蒸馏路径标准化）

```
训练阶段 (offline, 无延迟约束)      部署阶段 (online, <10ms)
──────────────────────────────        ──────────────────────
70B LLM 生成推理链（高质量）    →   7B 蒸馏模型（延迟友好）
URM: LLM 全量训练              →   TinyBERT 级蒸馏推理
EAGLE-3: 大目标模型            →   小草稿模型 (1/10 size)
GR2/Rank-R1: 70B 教师         →   7B 学生
LIMO: 大模型 SFT 激活          →   已激活的小模型
```

**关键数字**：蒸馏后模型 vs 完整模型，精度差距普遍 <2%，延迟降低 5-15x。这个比例已经工业可行。

---

## 📐 三、跨域核心公式串联

**1. 统一目标函数（搜广推都在优化的本质）**：

$$
\max_\pi \mathbb{E}_\pi[\text{Value}] \quad \text{s.t. Constraints}
$$

- Rec-Sys：Value = 用户满意度 (NDCG/CTR)，Constraint = 多样性/新鲜度
- Ads：Value = GMV/ROI，Constraint = 预算/CPA
- Search：Value = 答案质量，Constraint = 延迟/API 调用次数

**2. GRPO 训练范式（跨域统一训练信号）**：

$$
\mathcal{L}_\text{GRPO} = -\mathbb{E}\left[\frac{\pi_\theta(a)}{\pi_\text{old}(a)} \cdot \frac{r - \bar{r}}{\text{std}(r)}\right]
$$

**3. 蒸馏+KL 约束（防止 Goodhart's Law 的标配）**：

$$
\mathcal{L} = r_\text{task} - \beta \cdot \text{KL}(\pi_\theta || \pi_\text{ref}) + \gamma \cdot r_\text{quality}
$$

- 广告创意生成用于防止 reward hacking
- RLHF 用于防止 alignment tax
- 本质一样：约束优化空间，防止过拟合单一指标

**4. Scaling Laws（推荐系统的新 insight）**：

$$
\mathcal{L}(N_e, N_d) = A \cdot N_e^{-\alpha} + B \cdot N_d^{-\beta} + C, \quad \alpha > \beta
$$

推荐系统的"规模法则"与 LLM 不同——embedding table（稀疏参数）的 scaling 收益高于 dense DNN。

---

## 🎯 四、核心洞察（老师视角）

### 洞察1：推理是连接"训练期语义"和"推理期决策"的桥梁
- **传统 CTR 模型**：特征 → 嵌入 → MLP → sigmoid。推理过程完全隐式，不可解释。
- **CoT 推荐/排序**：用户行为 → 显式推理链（"用户最近对XX感兴趣，所以..."）→ 推荐决策。
- **本质**：推理链把模型的"黑盒激活"转化为"白盒推理路径"，可检查、可调试、可监控。这不只是可解释性，是工程可维护性的提升。

### 洞察2：Reward 设计比模型架构更重要
今天所有的 RL/GRPO 论文，拼的不是网络结构，而是：
1. **主 Reward 选择**：Hit@k vs NDCG@k vs F1 vs GMV（不同指标引导不同行为）
2. **Process Reward**：中间步骤奖励（推理链质量、搜索 query 质量）vs 只有最终奖励
3. **惩罚设计**：格式违规惩罚、浪费搜索惩罚、预算超支惩罚

**面试陷阱**：很多候选人能背 GRPO 公式，但说不清"在推荐任务里 reward 怎么设计"——这才是真正的 engineering challenge。

### 洞察3：统一模型的代价是系统复杂性从"横向"转移到"纵向"
多阶段系统：横向复杂（多个模型，接口对接复杂）
统一模型：纵向复杂（一个大模型，内部结构复杂，debug 更难，A/B 测试更难精准定位）

工业界迁移的节奏：**不是"全部替换"，而是"逐步合并"**——先合并精排+重排，再合并召回+精排，阶段性降低风险。

### 洞察4：LLM 基础设施的进步（EAGLE-3/MiniKV）直接使能了应用层创新
没有 Speculative Decoding 将延迟从 150ms 降到 40ms，OneRec-Think 的推理链就无法上线。
没有 INT2 KV Cache 压缩，长用户行为序列的 LLM 就放不进 GPU。

**技术栈的每一层都在互相解锁**：不要把 infra 和 application 割裂来学，要理解它们的依赖关系。

### 洞察5：数据质量 > 数据数量，在搜广推也适用
LIMO（817条）说明了推理领域的这个规律，但同样的逻辑适用于：
- ReasonIR：Reasoning-aware Hard Negative 的质量决定检索模型上限
- GR2：推理链 SFT 数据的质量决定重排效果
- IDProxy：广告 multimodal description 的质量决定冷启动效果

**工程师视角**：数据工程（如何采集、清洗、标注高质量数据）的 ROI 往往高于模型架构改进。

### 洞察6：开源vs闭源的边界在搜广推是模糊的
今天的论文来自：快手（OneRec）、Meta（HSTU）、阿里巴巴（URM/GNOLR）、小红书（IDProxy）——都是工业界主导。学术界 bench 已经在追赶工业界的实践，而不是反过来。

工业界的优势：真实用户数据、真实延迟约束、真实 A/B 测试反馈。这些是论文里的"实验结论"背后真正重要的东西。

---

## 🎓 五、常见考点（跨域综合，20题）

**Q1**: 为什么 GRPO 比 PPO 更适合搜广推的 RL 训练？
> PPO 需要 critic 网络估计 value function，在稀疏 reward（命中/未命中）下难以收敛；GRPO 用组内相对 reward 归一化，无需 critic，直接优化相对排名，更稳定，且与排序指标天然对齐

**Q2**: 推荐系统的 Scaling Laws 和 LLM 的 Scaling Laws 最大区别是什么？
> LLM：dense 参数（transformer 权重）是性能主体，计算和参数线性对应；推荐系统：embedding table（稀疏查表）是性能主体，scaling 体现在 embedding 维度/数量而非 DNN 深度，两者的 compute-optimal 分配比例完全不同

**Q3**: 为什么"统一模型"（UniGRF/UniROM/MTFM）在工业界比学术界更受关注？
> 学术界关注精度提升；工业界关注总体拥有成本（TCO）。统一模型减少了模型数量、接口复杂度、上线流程，运维人力成本下降 50%+，这些收益在论文里不会出现

**Q4**: Semantic ID 的设计如何影响生成式推荐的效果？
> RQ-VAE 使相似物品共享 ID 前缀（树状结构），让生成模型能泛化到新物品（新物品前缀匹配已知类别）；相比随机 ID，Semantic ID 的冷启动效果提升 +12%，是生成式推荐工业落地的关键

**Q5**: 为什么 Speculative Decoding 的加速比 = 1/(1-α)？
> 草稿模型生成 k 个 token，目标模型并行验证（一次 forward）。期望接受的 token 数 = α/(1-α)，总效率 ≈ (1 + α/(1-α)) / (1 + 草稿成本) ≈ 1/(1-α)（草稿成本忽略不计时）

**Q6**: 广告出价的 Lagrangian 对偶变量 λ 如何在线更新？
> λ 是预算的影子价格。实践中用 PID 控制器（比例-积分-微分）根据实际预算消耗速率 vs 目标速率调整 λ：消耗过快 → 提高 λ（降低出价）；消耗过慢 → 降低 λ（提高出价）

**Q7**: LLM 做推荐和传统 CTR 模型的本质区别在哪里？
> 传统 CTR：判别式模型，学习 P(click|features)，特征工程驱动；LLM 推荐：生成式模型，学习 P(next_item|history)，语义理解驱动。前者对已见 ID 精准，后者对新 ID/新场景泛化好，蒸馏方案融合二者优势

**Q8**: Hard Negative Mining 在不同检索任务中有什么区别？
> 普通检索 HN：BM25 高分但语义不相关；推理密集型 HN（ReasonIR）：表面语义相似但逻辑推导无关（更难！）；广告 CTR HN：同类目高质量广告但转化率低（选择偏差导致）

**Q9**: Diffusion 模型用于出价的工程挑战是什么？
> ① 采样延迟：标准 Diffusion 1000步 vs DDIM 10步，后者才能满足毫秒级出价要求；② 约束出价空间：出价 b≥0 需要 clamping；③ 非平稳市场：Diffusion policy 的训练数据分布会 drift，需要频繁重训

**Q10**: 为什么 MLA（Multi-head Latent Attention）是 KV Cache 压缩的"架构级"而非"运行时"解法？
> MLA 在模型训练时就将 KV 设计为低维 latent（$d_c \ll N_h d_h$），推理时只存 latent；量化/稀疏是推理时 patch，需要额外计算。MLA 无额外运行时开销，但只适用于从头训练的新模型

**Q11**: 多任务学习的负迁移如何判断是否发生？
> ① 指标变化：引入新任务后旧任务 AUC 下降；② 梯度分析：两个任务梯度 cos<0（方向冲突）；③ 梯度量级：某任务梯度范数 >> 其他任务（主导更新方向）。诊断工具：梯度 cosine similarity 监控

**Q12**: 广告系统中 position bias 和 exposure bias 有什么区别？
> Position bias：相同广告在不同位置 CTR 不同（位置本身影响点击）；Exposure bias：模型只能学习被曝光的广告，未曝光广告的表现未知（训练数据分布偏差）。两者都需要 IPS/反倾向加权校正

**Q13**: 生成式重排（GR2）比传统重排（SetRank/PRM）的核心优势是什么？
> GR2 生成排列（permutation）而非逐物品打分：① 天然 listwise（考虑物品间关系）；② 推理链提供显式意图对齐；③ Constrained decoding 保证排列合法。代价：延迟更高（需要 LLM decode）

**Q14**: LIMO 的"数据质量标准"对工业界标注有什么指导意义？
> ① 步骤完整性 > 答案正确性（跳步的正确答案对 SFT 有害）；② 多样性（同一知识点不同解法）> 数量；③ 可验证性（有程序/公式可验证）的样本值高于主观评估样本。工业界 annotation guideline 应优先保证这三点

**Q15**: 推荐系统中 next-item prediction 预训练和 CTR fine-tune 为什么组合有效？
> Next-item prediction 学习用户行为的"语言"（行为序列的分布）；CTR fine-tune 学习"方言"（特定广告的点击倾向）。预训练提供泛化基础，fine-tune 提供任务特异性，两者互补

**Q16**: 为什么 FlashAttention 需要为每代 GPU 重写？
> 每代 GPU 的访存层次（HBM/L2/SRAM 大小比例）、计算单元（WMMA→WGMMA）、IO 机制（同步→TMA 异步）都不同，最优 tile 大小和 IO/计算重叠策略都要重设计。这是 AI 基础设施永续工程的本质

**Q17**: 在线广告系统如何评估 Exploration 的价值？
> 用 Counterfactual Estimation（反事实）：在历史出价分布下估计"如果以不同 bid 竞价会发生什么"。IPW（逆倾向加权）校正 selection bias。上线时用 holdout experiment（保留部分流量做探索），用长期 LTV 而非短期 CTR 评估

**Q18**: 搜广推系统中"实时性"和"模型质量"的权衡如何平衡？
> 特征实时性分层：用户实时行为（毫秒级） > 物品统计特征（分钟级） > LLM 语义特征（小时级/离线）。不同特征用不同更新策略，LLM 特征离线计算后缓存，核心 CTR 特征实时更新

**Q19**: Tree-based Speculative Decoding（EAGLE-3）比单序列草稿快多少，为什么？
> 树状草稿提供多条候选路径，目标模型并行验证所有分支（一次 forward），选最长合法前缀；期望有效 token 数 = 单序列的 1.3-1.8x，在接受率相同的条件下加速比进一步提升

**Q20**: 如果让你设计一个"2027年的推荐系统"，架构上最重要的改变是什么？
> 核心变化：① 召回+排序+重排统一为 End-to-End 生成式模型（已在路上：OneRec/HSTU）；② 推理链 + RL 替代 feature engineering（模型自己学什么有用）；③ LLM 基础设施（Speculative Decoding + KV Cache 压缩）使实时推理可行；④ 数据飞轮从"点击反馈"升级为"对话式反馈"（用户明确表达偏好，类似 LLM-AUCTION 的对话逻辑）

---

## 🔗 六、跨域依赖关系图

```
LLM-Infra 进步 (EAGLE-3, MiniKV)
  ↓ 使能
Rec-Sys: 推理链推荐可部署 (OneRec-Think, GR2)
Ads: 生成式出价可部署 (GAVE, GRAD)  
Search: LLM 重排可部署 (Rank-R1)

GRPO 训练范式成熟 (来自 DeepSeek-R1/LIMO)
  ↓ 迁移
Rec-Sys: GRPO 训练推荐推理链
Search: GRPO 训练重排/主动搜索
Ads: GRPO 训练出价探索策略

Scaling Laws 确立 (LLM 的 Chinchilla + 推荐的 Kunlun)
  ↓ 指导
Rec-Sys: 优先 embedding scaling，不盲目堆 DNN
LLM-Infra: compute-optimal 资源分配
```

---

## 📚 七、参考文献（按域）

### rec-sys
> - [生成式推荐系统技术全景](./生成式推荐系统技术全景.md) — rec-sys 今日全景综述

### search
> - [推理增强检索技术综述](../../search/synthesis/推理增强检索技术综述.md) — search 今日全景综述

### ads
> - [LLM时代广告系统技术演进](../../ads/synthesis/LLM时代广告系统技术演进.md) — ads 今日全景综述

### llm-infra
> - [LLM推理加速与高效训练技术全景](../../../llm-agent/llm-infra/synthesis/LLM推理加速与高效训练技术全景.md) — infra 今日全景综述

## 📐 核心公式直观理解

### RLHF 在推荐中的应用

$$
\mathcal{L}_{\text{RLHF}} = -\mathbb{E}_{a \sim \pi_\theta}[R(s, a)] + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

**直观理解**：让推荐策略 $\pi_\theta$ 最大化用户满意度（reward $R$），同时不能偏离参考策略 $\pi_{\text{ref}}$ 太远（KL 正则）。KL 项防止模型为了追求 reward 而走极端（比如只推高 CTR 但低质量的标题党内容）。

### 推理链（CoT）增强排序

$$
\text{score}(d|q) = \text{LLM}(\text{"思考步骤："} + \text{reasoning} + \text{"相关性评分："})
$$

**直观理解**：让 LLM 先"想一想"为什么这个文档和 query 相关（或不相关），再给出评分。推理过程中 LLM 会发现细微的匹配关系（如"文档讨论的是 GPU 架构，query 问的是 CUDA 编程，两者密切相关因为..."），比直接打分更准确。

### 跨域迁移的统一 embedding

$$
e_{\text{unified}} = \text{Proj}(e_{\text{search}}) \approx \text{Proj}(e_{\text{rec}}) \approx \text{Proj}(e_{\text{ads}})
$$

**直观理解**：搜索、推荐、广告三个系统各有各的 embedding 空间。统一 embedding 让用户在搜索中的行为（"搜过跑步鞋"）能直接帮助推荐（推跑步装备）和广告（投运动品牌广告）——跨域信号复用，冷启动效果大幅提升。

