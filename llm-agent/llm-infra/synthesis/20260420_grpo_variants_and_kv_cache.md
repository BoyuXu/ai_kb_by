# GRPO 变体演进 + KV Cache 企业级优化 (2026-04-20)

> 知识卡片 | 创建：2026-04-20 | 领域：llm-infra/alignment + inference
> 关联：[[LLM对齐RL方法全景_2026]] | [[GRPO大模型推理RL算法]] | [[KVCache与LLM推理优化全景]] | [[20260419_KV_cache_quantization_adaptive_methods]]

---

## Theme A: GRPO 家族演进 — 三大变体如何扩展原始 GRPO

### 背景回顾

原始 GRPO (DeepSeek, 2024) 的核心：砍掉 Critic，用组内相对 reward 归一化作为 advantage：

$$
A_i = \frac{r_i - \mu_G}{\sigma_G}, \quad L_{\text{GRPO}} = -\mathbb{E}\left[\min\left(\rho \cdot A, \text{clip}(\rho, 1 \pm \epsilon) \cdot A\right)\right] + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

GRPO 的三大局限：
1. **纯数值反馈瓶颈**：标量 reward 无法传递"错在哪、怎么改"的信息 → 性能天花板
2. **On-policy 采样效率低**：每轮更新必须用当前策略重新采样 → 计算昂贵
3. **参数更新门槛高**：需要访问模型权重做梯度更新 → 无法用于闭源 API

三篇新论文分别从这三个角度突破。

---

### 1. Critique-GRPO：自然语言反馈注入 RL 循环 (arXiv: 2506.03106)

**问题**：标量 reward 有三个根本缺陷 — (1) 性能平台期，(2) 无法有效自发自我反思，(3) 持续性失败（同一类错误反复犯）。

**核心洞察**：已经 RL 训练到平台期的模型，给它自然语言 critique（"你第3步符号搞反了"），仍然能成功修正 → 说明信息瓶颈在 reward signal，不在模型能力。

**方法**：

```
标准 GRPO:  prompt → G 个回答 → 数值 reward → 组内归一化 advantage → 梯度更新
Critique-GRPO: prompt → G 个回答 → 数值 reward
                                  ↓ (失败样本)
                              critique 生成（LLM-as-Critic 或自批评）
                                  ↓
                              critique-guided refinement → 新回答 → reward
                                  ↓
                          初始回答 + 修正回答 共同参与 advantage 计算 → 更新
```

**关键创新**：
- **Verbal Credit Assignment**：自然语言 critique 充当"密集信号"，指出具体错误位置和修正方向
- **Refinement Trajectory**：通过 critique → refine 构造的高质量轨迹是标准探索（数值 reward）无法到达的
- **双流学习**：模型同时从 (1) 初始回答和 (2) critique 引导的修正回答中学习
- **自批评能力**：训练后期，模型可用自身生成 critique（Self-Critiquing），实现闭环自改进

**公式扩展**：

$$
L_{\text{Critique-GRPO}} = L_{\text{GRPO}}(\{y_i^{\text{init}}\}) + \alpha \cdot L_{\text{GRPO}}(\{y_j^{\text{refine}}\})
$$

其中 $y_j^{\text{refine}}$ 是 critique 引导修正后的回答，$\alpha$ 控制 refinement 学习的权重。

**结果**：
- Qwen 系列模型 Pass@1 提升 +15.0%~21.6%
- Llama-3.2-3B-Instruct Pass@1 提升 +7.3%
- AIME 2024 上 +16.7% Pass@1（vs 标准 GRPO）
- 自批评模式仍然有效 → 不依赖外部 Critic

---

### 2. Off-Policy GRPO：打破 On-Policy 采样瓶颈 (arXiv: 2505.22257)

**问题**：标准 GRPO 是 on-policy 的 — 每次策略更新后必须用新策略重新采样 G 个回答。这导致：
- 采样成本高（尤其 G=64 时）
- 显存占用大（需同时加载 policy 做推理和训练）
- 数据利用率低（采样一次只用一次）

**核心贡献**（IBM Research）：

1. **理论分析**：证明了 on-policy 和 off-policy GRPO 目标函数都能保证 policy improvement，给出了下界
2. **Off-Policy Advantage**：用历史策略 $\pi_{k-v}$（v 步前的策略）的采样来估计 advantage：

$$
\hat{A}_i^{\text{off}} = \frac{r_i - \mu_{\alpha}}{\sigma_{\alpha}}, \quad \text{where } \mu_{\alpha}, \sigma_{\alpha} \text{ computed from } \pi_{k-v} \text{ samples}
$$

3. **Clipped Surrogate 保持**：保留 PPO-style clip，但重要性比率 $\rho$ 的参考策略变为 $\pi_{k-v}$

**On-Policy vs Off-Policy 对比**：

| 维度 | On-Policy GRPO | Off-Policy GRPO |
|------|---------------|-----------------|
| 采样策略 | 当前 $\pi_k$ | 历史 $\pi_{k-v}$ |
| 每轮采样 | 必须重新采样 | 可复用历史样本 |
| 训练稳定性 | 标准 | 更稳定（理论保证） |
| 数据效率 | 低（用一次丢弃） | 高（多次复用） |
| 显存峰值 | 高（推理+训练同时） | 低（推理和训练可分离） |
| 理论保证 | $J(\pi_{k+1}) \geq J(\pi_k) - \epsilon$ | 同样有下界保证 |

**结果**：
- GSM8K、AIME 2024、Math500 上 off-policy 显著优于或持平 on-policy
- 训练更稳定，loss 曲线抖动更小
- 潜力：与 experience replay buffer 结合可进一步提升

---

### 3. Training-Free GRPO：从参数空间到上下文空间 (arXiv: 2510.08191)

**问题**：所有前述 GRPO 变体都需要访问模型权重做梯度更新。对于闭源 API（GPT-4、Claude）或超大模型（成本 $800+ 微调），这不现实。

**范式转换**：

```
传统 GRPO:  π_θ (可训练参数) + reward → 梯度更新 θ
TF-GRPO:   π = frozen_LLM(·| context_C) → 优化 C（上下文），θ 不变
```

**核心方法**：

1. **策略重定义**：$\pi(y|x) = \text{LLM}(y | x, C)$，其中 $C$ 是可变的经验上下文（experiential context）
2. **语义 Advantage**：不再计算数值 advantage，而是让 LLM 对一组 rollout 做语义反思（introspection），提取"什么做得好/差"
3. **经验蒸馏**：多轮迭代中，高质量经验知识被蒸馏为 token prior，注入 prompt 引导行为

$$
C^{(t+1)} = \text{Distill}\left(C^{(t)}, \{(y_i, r_i)\}_{i=1}^G, \text{LLM\_Introspect}\right)
$$

4. **多 epoch 学习**：在极少量标注数据上反复迭代，每轮用语义 advantage 更新 $C$

**与原始 GRPO 的类比**：

| 组件 | 原始 GRPO | Training-Free GRPO |
|------|----------|-------------------|
| 策略表示 | 模型参数 $\theta$ | 上下文 $C$ |
| Advantage | 数值归一化 $(r_i - \mu)/\sigma$ | 语义反思（自然语言） |
| 优化方式 | 梯度下降 | 上下文迭代更新 |
| 需要权重访问 | 是 | 否（仅 API 调用） |
| 成本 | $800+（微调 32B） | $8（100 样本 API 调用） |

**结果**：
- DeepSeek-V3.1-Terminus + 100 样本 → 超越微调 32B 模型的性能
- 训练成本降低 100 倍（$800 → $8）
- 数学推理 + Web 搜索任务上 OOD 泛化优秀
- 适用于任何 LLM API（仅需 prompt 注入能力）

---

### GRPO 变体总览对比表

| 维度 | 原始 GRPO | Critique-GRPO | Off-Policy GRPO | Training-Free GRPO |
|------|----------|--------------|----------------|-------------------|
| **解决的问题** | 砍 Critic 降成本 | 数值 reward 信息瓶颈 | On-policy 采样效率 | 闭源/大模型不可训练 |
| **Reward 信号** | 纯数值 | 数值 + 自然语言 critique | 纯数值 | 语义 advantage |
| **采样策略** | On-policy | On-policy | Off-policy（历史策略） | 冻结模型 rollout |
| **需要梯度更新** | 是 | 是 | 是 | 否 |
| **需要模型权重** | 是 | 是 | 是 | 否（API 即可） |
| **核心创新** | 组内归一化 advantage | Critique→Refine 双流 | 理论保证 off-policy | 上下文空间优化 |
| **最佳场景** | 可验证 reasoning | 突破平台期 | 大规模训练降本 | 闭源 API 增强 |
| **代表模型** | DeepSeek-R1 | Qwen + Llama | IBM 实验 | DeepSeek-V3.1 API |

---

## Theme B: KV Cache 企业级优化

### 4. LMCache：企业级 KV Cache 存储层 (arXiv: 2510.09665)

**问题**：KV Cache 传统上只存在 GPU 显存中。企业场景需要：
- 跨查询复用（同一 system prompt 的 KV cache）
- 跨引擎共享（多个 vLLM 实例间）
- 多级存储（GPU → CPU → 磁盘 → 远程）

**LMCache 是什么**：首个开源 KV Cache 独立存储层，解耦 cache 存储与推理引擎。

**架构**：

```
┌─────────────────────────────────────────┐
│  LLM Inference Engine (vLLM / SGLang)   │
│       ↕ KV Cache Connector (模块化)      │
├─────────────────────────────────────────┤
│            LMCache Layer                 │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌─────┐ │
│  │ GPU  │→ │ CPU  │→ │ Disk │→ │Redis│ │
│  │Memory│  │Memory│  │(NVMe)│  │(远程)│ │
│  └──────┘  └──────┘  └──────┘  └─────┘ │
│       Control API (pin/lookup/cleanup)   │
└─────────────────────────────────────────┘
```

**三大技术支柱**：

| 支柱 | 内容 | 为什么重要 |
|------|------|-----------|
| 高性能数据搬运 | Batched ops + Compute/IO Pipeline | KV cache 体积大（70B 模型单请求 ~数 GB），逐块搬太慢 |
| 模块化 Connector | 与引擎解耦 | vLLM/SGLang 迭代极快，LMCache 不被锁死 |
| First-class Control API | Pin/Lookup/Cleanup/Move/Compress | 企业需要精细控制缓存策略 |

**两大核心能力**：

1. **Cache Offloading（前缀复用）**：
   - 相同 system prompt 的 KV cache 只算一次，存到 CPU/磁盘
   - 新请求来时直接加载，跳过 prefill
   - 适用：多轮对话、RAG（共享长文档前缀）

2. **Prefill-Decode (PD) Disaggregation（跨引擎传输）**：
   - Prefill 引擎计算 KV cache → 通过网络传给 Decode 引擎
   - GPU 间用 NVLink/RDMA，远程用 Ethernet
   - 适用：大规模分布式部署

**结果**：
- 与 vLLM 结合，多轮 QA 和文档分析场景 **吞吐量提升 15x**
- 开源：github.com/LMCache/LMCache

---

### 5. KV Cache 优化策略全景综述 (arXiv: 2603.20397)

**定位**：系统性综述，将所有 KV Cache 优化技术分为五大方向，并映射到七大部署场景。

**五大优化方向**：

| 方向 | 核心思想 | 代表方法 | 压缩比 | 精度影响 |
|------|---------|---------|--------|---------|
| **Cache Eviction** | 淘汰低重要性 token | H2O, StreamingLLM, SnapKV | 2-10x | 轻微（<1%） |
| **Cache Compression** | 量化/低秩压缩 KV | KIVI, KVQuant, MiniKV | 4-8x | 0.5-3% |
| **Hybrid Memory** | 多级存储（GPU+CPU+磁盘） | LMCache, OffloadKV | N/A | 0%（等价） |
| **Novel Attention** | 架构层面减少 KV | GQA, MLA, MQA | 4-32x | <0.5% |
| **Combination** | 多技术组合 | RocketKV, KVzip, ShadowKV | 10-50x | <1% |

**七大部署场景与最优策略**：

| 场景 | 上下文长度 | 优先级 | 推荐方向 |
|------|-----------|--------|---------|
| 超长上下文单请求 | >100K | 精度 > 内存 | Eviction + Compression 组合 |
| 高吞吐数据中心 | 中等 | 吞吐 > 精度 | Hybrid Memory + Batching |
| 边缘设备 | 短-中 | 内存 > 一切 | 激进 Compression + Novel Attention |
| 多轮对话 | 累积增长 | 延迟 > 吞吐 | Cache Offloading（前缀复用） |
| 准确性关键推理 | 中-长 | 精度 > 一切 | 保守 Eviction（保留高注意力 token） |
| RAG/文档分析 | 长 | 吞吐 + 精度 | Hybrid Memory + Selective Eviction |
| Batch 多请求 | 短 | 吞吐 > 延迟 | PagedAttention + GQA |

**核心洞察**：
- **没有银弹**：没有单一技术在所有场景下最优
- **最优策略取决于**：上下文长度 × 硬件约束 × 工作负载特征
- **趋势**：自适应多阶段优化 pipeline（运行时动态选择策略）

---

### KV Cache 技术演进时间线

```
2023: PagedAttention (vLLM) — 解决内存碎片
  ↓
2024: GQA/MQA (LLaMA-3) — 架构层面减少 KV head
  ↓
2024: H2O/SnapKV — 智能 eviction（注意力引导）
  ↓
2024: KIVI/KVQuant — KV cache 量化
  ↓
2025: LMCache — 独立存储层 + 跨引擎共享
  ↓
2025-26: 自适应量化 (Don't Waste Bits, ARKV) — per-token 精度分配
  ↓
2026: 组合策略 (RocketKV, ShadowKV) — 多技术融合 + 场景化部署
```

---

## 综合分析：两大主题的交汇

GRPO 和 KV Cache 看似无关，但在 LLM 工程中有深层连接：

1. **GRPO 训练需要大量推理**：每步采样 G 个回答 → 推理效率直接影响 GRPO 训练成本 → KV Cache 优化降低采样阶段开销
2. **Off-Policy GRPO + KV Cache 复用**：历史采样的 KV cache 可以缓存复用，进一步降低 off-policy 训练成本
3. **Training-Free GRPO + LMCache**：冻结模型的 rollout 天然适合 KV cache 前缀复用（同一模型、同一 system prompt）
4. **推理时对齐**：KV Cache 优化使得推理时 compute budget 更充裕 → 可以做更多 rollout → GRPO-style 的推理时 self-improvement

---

## 面试 Q&A

### Q1: Critique-GRPO 和标准 GRPO 的核心区别是什么？为什么自然语言反馈比数值 reward 更有效？

**答**：标准 GRPO 只用数值 reward（对/错、分数），Critique-GRPO 额外引入自然语言 critique。有效性的原因是**信息密度差异**：数值 reward 只传递"做得好不好"（1 bit 信息），critique 传递"哪里错了、为什么错、怎么改"（可能几百 token 的信息）。实验证明已经 plateau 的 GRPO 模型，给自然语言 critique 后仍能修正失败样本 → 说明模型能力没到上限，是 reward signal 的信息瓶颈。Critique-GRPO 在 AIME 2024 上比标准 GRPO 高 16.7% Pass@1。

### Q2: Off-Policy GRPO 相比 On-Policy GRPO 的优势是什么？有什么理论保证？

**答**：三大优势：(1) **采样效率**：历史样本可复用，不需每轮重新采样；(2) **训练稳定性**：advantage 统计量来自更稳定的旧分布；(3) **显存优化**：推理和训练可时序分离。理论保证方面，论文证明了 on-policy 和 off-policy GRPO 都有 policy improvement 下界，即 $J(\pi_{k+1}) \geq J(\pi_k) - \epsilon$。实验表明 off-policy 在 GSM8K/AIME/Math500 上显著优于或持平 on-policy。

### Q3: Training-Free GRPO 是怎么做到不更新参数就优化策略的？和 prompt engineering 有什么区别？

**答**：关键区别在于**系统性优化 vs 手动设计**。Training-Free GRPO 将策略表示为 $\pi(y|x) = \text{LLM}(y|x, C)$，其中 $C$ 是通过 RL 循环迭代优化的经验上下文。优化过程：(1) 对同一问题做 G 次 rollout；(2) LLM 对 rollout 做语义反思（哪些好/差）；(3) 将反思蒸馏进 $C$。和手动 prompt engineering 的区别：TF-GRPO 是**自动化、数据驱动、迭代优化**的，每轮都基于 rollout 的语义 advantage 更新 $C$。100 样本就能让 DeepSeek-V3.1 超越微调 32B 模型，成本仅 $8。

### Q4: LMCache 解决了 KV Cache 的什么痛点？在企业部署中如何选择存储层级？

**答**：LMCache 解决两大痛点：(1) KV cache 锁死在单 GPU 显存中无法跨查询/引擎共享；(2) GPU 显存是最贵的存储资源。存储层级选择原则：
- **GPU Memory**：正在解码的活跃请求（延迟最低）
- **CPU Memory**：高频复用的前缀 cache（如 system prompt），延迟 ~100μs
- **NVMe Disk**：中等频率复用，延迟 ~1ms
- **Redis/远程**：跨机器共享，延迟 ~10ms
核心决策因素是**访问频率 × 延迟容忍度**。LMCache 与 vLLM 结合实现 15x 吞吐提升。

### Q5: KV Cache 五大优化方向各自的适用场景是什么？如何组合？

**答**：
- **Eviction（淘汰）**：适用于超长上下文、精度可接受微降；H2O 保留"heavy hitter" token
- **Compression（压缩）**：带宽受限环境；KIVI 对 Key 用 4bit、Value 用 2bit
- **Hybrid Memory**：高吞吐服务；LMCache 层级存储
- **Novel Attention**：新模型训练时选择；GQA/MLA 从架构层面减少 KV
- **Combination**：极端场景（百万 token）；RocketKV = 粗粒度选择 + 细粒度重建

组合原则：**架构（GQA）× 压缩（量化）× 运行时（eviction）** 三层叠加效果最佳。选择时看场景：延迟敏感选 eviction，内存紧张选 compression，多实例选 hybrid。

### Q6: GRPO 训练和 KV Cache 优化之间有什么工程联系？

**答**：GRPO 训练中的采样阶段（每步生成 G 个回答）本质是大规模推理任务。KV Cache 优化直接降低采样成本：(1) GQA 减少每个采样的 KV 显存；(2) 同 prompt 的 G 个采样可共享前缀 KV cache（prefix caching）；(3) Off-Policy GRPO 的历史样本 KV cache 可以缓存到 CPU/磁盘后续复用。在 DeepSeek-R1 的训练中，vLLM 的 PagedAttention 是 GRPO 大规模采样的基础设施。

---

## 参考文献

1. Zhang et al. (2025) — Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback. arXiv:2506.03106
2. Mroueh et al. (2025) — Revisiting Group Relative Policy Optimization: Insights into On-Policy and Off-Policy Training. arXiv:2505.22257
3. Chen et al. (2025) — Training-Free Group Relative Policy Optimization. arXiv:2510.08191
4. Cheng & Liu et al. (2025) — LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference. arXiv:2510.09665
5. Xu et al. (2026) — KV Cache Optimization Strategies for Scalable and Efficient LLM Inference. arXiv:2603.20397
6. Shao et al. (2024) — GRPO (DeepSeek-Math)
7. DeepSeek (2025) — DeepSeek-R1
