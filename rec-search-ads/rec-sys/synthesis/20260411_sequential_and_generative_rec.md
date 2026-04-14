# 2026.04.11 序列推荐建模演进 + 生成式推荐与工业排序

> **覆盖论文**：SORT、OneRec-V2、GLIDE (Spotify)、NEO (Spotify)、生成式推荐综述 (中文2025)、FuXi-Linear、BlossomRec、SIGMA、M2Rec
>
> **关联概念页**：[[sequence_modeling_evolution]] | [[generative_recsys]] | [[attention_in_recsys]] | [[embedding_everywhere]]
>
> **关联 synthesis**：[[20260407_GenRec_advances_synthesis]] | [[生成式推荐范式统一_20260403]] | [[推荐广告生成式范式统一全景]]

---

## Theme A：序列推荐建模演进 — Linear Attention / Sparse Attention / Mamba

### 1. 技术演进脉络

```
SASRec (Self-Attention, O(n²))
  ↓ 长序列瓶颈
  ├─→ Linear Attention 路线: LinRec → FuXi-γ → FuXi-Linear (2026.02)
  ├─→ Sparse Attention 路线: FlashAttention → BlossomRec (2025.12)
  └─→ SSM/Mamba 路线: Mamba4Rec → SIGMA (AAAI'25) → M2Rec (2025.05)
```

**核心矛盾**：标准 Self-Attention 的 O(n²) 复杂度限制了长序列（>1000）建模能力，而工业场景中用户行为序列越来越长（电商可达数千甚至数万）。三条路线各有取舍。

### 2. 核心方法对比表

| 模型 | 核心机制 | 复杂度 | 长序列建模 | 短序列稳定性 | 关键创新 |
|------|---------|--------|-----------|-------------|---------|
| **FuXi-Linear** | 双通道 Linear Attention | O(n) | ✅ 优秀（千级token验证scaling law） | ✅ | Temporal Retention Channel + Linear Positional Channel |
| **BlossomRec** | Block-level Sparse Attention | O(n·√n) | ✅ | ✅（显式建模长短期兴趣） | 双稀疏模式 + 门控融合 |
| **SIGMA** | 双向 Mamba + GRU | O(n) | ✅ | ✅ | PF-Mamba + DS Gate + FE-GRU |
| **M2Rec** | 多尺度 Mamba + FFT + LLM | O(n) | ✅ | ✅ | FFT频域建模 + LLM语义 + 自适应门控 |

### 3. FuXi-Linear 深度解析

**问题**：现有 Linear Attention 在推荐场景效果不佳，主要因为：(1) 时间信息和语义信息混合导致信号串扰 (crosstalk)；(2) 位置编码在线性复杂度下难以有效融入。

**解法：双通道架构**

```
Input Sequence
    ├─→ Temporal Retention Channel (TRC)
    │     - 用时间戳差异计算独立的衰减矩阵 D_t
    │     - 周期性注意力权重，避免时间-语义串扰
    │     - 关键公式: o_t = Σ_{s≤t} γ^{Δ(t,s)} · (K_s^T V_s) · Q_t
    │
    └─→ Linear Positional Channel (LPC)
          - 可学习核函数替代 softmax
          - 位置信息通过核函数编码，保持 O(n)
          - 关键公式: o_t = Σ_{s≤t} φ(Q_t)^T φ(K_s) · V_s
```

**Scaling Law**：在千级token序列上首次验证推荐领域的 power-law scaling property（之前 linear 模型未探索过）。

**工业价值**：prefill 加速 10×，decode 加速 21×，适合在线推理的长序列建模。

### 4. BlossomRec 深度解析

**核心洞察**：用户兴趣自然分为长期稳定偏好和短期情境兴趣，应该用不同的注意力模式分别建模。

**双稀疏模式**：
- **长期兴趣**：全局稀疏采样（Block-level，从全序列中均匀抽取关键交互）
- **短期兴趣**：局部窗口注意力（近期的密集交互）
- **门控融合**：可学习 gate 动态平衡长短期权重

$$\text{Output} = \sigma(g) \cdot \text{LongAttn}(x) + (1-\sigma(g)) \cdot \text{ShortAttn}(x)$$

**优势**：即插即用，可嵌入任何 Transformer-based 序列推荐模型（SASRec/BERT4Rec等），显著降低内存。

### 5. SIGMA 深度解析 (AAAI 2025)

**核心思路**：用 Mamba（SSM）替代 Transformer 做序列推荐，但解决 Mamba 的三个短板：

| Mamba 短板 | SIGMA 解法 |
|-----------|-----------|
| 单向建模 | **PF-Mamba**：Partially Flipped Mamba，构建双向上下文 |
| 方向权重固定 | **DS Gate**：Dense Selective Gate，输入自适应地调整前向/后向权重 |
| 长期遗忘 | **FE-GRU**：Feature Extract GRU，显式捕获短期依赖作为补充 |

**PF-Mamba 关键**：不是简单的双向扫描，而是"部分翻转"——只翻转 SSM 的扫描方向，不翻转输入序列，避免信息泄漏。

### 6. M2Rec 深度解析

**核心创新**：三源融合——时域 (Mamba) + 频域 (FFT) + 语义 (LLM)

```
                    ┌─ Mamba Branch ───→ 时序依赖特征
Input Sequence ──→  ├─ FFT Branch ────→ 周期模式特征
                    └─ LLM Branch ────→ 语义上下文特征
                              ↓
                    Adaptive Gating → 最终表示
```

- **FFT-Enhanced Mamba**：在频域显式建模用户行为周期模式（如每周末购物），分离趋势与噪声
- **LLM 嵌入**：用预训练 LLM 编码 item 文本描述，补充稀疏交互数据
- **自适应门控**：$\alpha_t, \beta_t, \gamma_t = \text{softmax}(W[h^{mamba}_t; h^{fft}_t; h^{llm}_t])$

**结果**：HR@10 比 Mamba4Rec 提升 3.2%，推理速度比 Transformer baseline 快 20%。

### 7. 面试高频考点 — 序列建模

**Q1: Linear Attention 和标准 Attention 的本质区别？**
> 标准 Attention: $\text{softmax}(QK^T/\sqrt{d})V$，softmax 导致 O(n²)。
> Linear Attention: 用核函数 $\phi$ 替换 softmax，$\phi(Q)(\phi(K)^TV)$，先算 $\phi(K)^TV$ 得 O(n)。
> 代价：失去 softmax 的非线性选择性，可能降低建模能力。FuXi-Linear 通过双通道缓解。

**Q2: Mamba 相比 Transformer 在推荐场景的优劣？**
> 优：O(n) 复杂度，适合长序列；隐状态压缩历史，推理高效。
> 劣：单向建模（推荐需要双向上下文）；选择性机制对短序列可能不如全局注意力。
> SIGMA 用 PF-Mamba 双向化 + FE-GRU 短期补充来弥补。

**Q3: 为什么 BlossomRec 的"双稀疏"比均匀稀疏好？**
> 用户兴趣有结构：远期兴趣稀疏但稳定（用全局采样），近期兴趣密集但多变（用局部窗口）。一刀切的均匀稀疏会丢失这种结构。

**Q4: M2Rec 为什么要引入 FFT？**
> 用户行为有周期性（日/周/月），时域模型（Mamba/RNN）隐式学习周期性效率低，FFT 在频域直接捕获周期模式。类比：语音处理中 STFT 的思路。

---

## Theme B：生成式推荐与工业排序

### 1. 技术演进脉络

```
传统级联: 召回→粗排→精排→重排 (判别式)
    ↓
Semantic ID: TIGER (RQ-VAE) → GLIDE (Spotify) → NEO (Spotify统一)
    ↓
端到端生成: OneRec-V1 (Enc-Dec) → OneRec-V2 (Lazy Decoder, 8B)
    ↓
工业 Transformer 排序: DIN/DIEN → HSTU → SORT (Alibaba)
```

### 2. 核心方法对比表

| 模型 | 来源 | 核心范式 | Item表示 | 架构 | 规模 | 关键指标 |
|------|------|---------|---------|------|------|---------|
| **SORT** | Alibaba | Transformer精排 | 特征交叉 | Transformer+MoE | — | 工业AB提升 |
| **OneRec-V2** | 快手 | 端到端生成式 | 生成式ID | Lazy Decoder-Only | 8B | App时长+0.47% |
| **GLIDE** | Spotify | 生成式召回 | Semantic ID | Decoder-Only | — | 新发现+14.3% |
| **NEO** | Spotify | 搜索+推荐+推理统一 | SID (RQ) | Decoder-Only LLM | 10M+目录 | HR@10提升36-58% |
| **生成式综述** | 中文 | 方法论综述 | 四类标记法 | — | — | — |

### 3. SORT 深度解析（Alibaba 工业精排）

**核心问题**：Transformer 在工业精排中面临三大挑战：
1. 高特征稀疏性（百亿级稀疏特征）
2. 低标签密度（曝光/点击比极低）
3. 计算效率（百毫秒延迟约束）

**系统优化四件套**：

| 优化 | 目标 | 方法 |
|------|------|------|
| **Request-Centric 组织** | 减少无关计算 | 按请求组织样本，同一请求内的候选共享上下文 |
| **Local Attention** | 降低复杂度 | 限制注意力范围到同一请求内的候选 |
| **Query Pruning** | 减少冗余 | 裁剪低信息量的 query token |
| **Generative Pre-training** | 缓解过拟合 | Next-item prediction 预训练，增强标签密度 |

**模型细节**：
- **Special Tokens**：引入 [CLS] 等特殊 token 聚合请求级信息
- **QKNorm**：对 Q、K 向量做归一化，稳定训练
- **Attention Gate**：门控注意力输出，控制信息流
- **MoE (Mixture of Experts)**：FFN 层用 MoE 扩展容量

**关键公式** — Generative Pre-training:
$$\mathcal{L}_{gen} = -\sum_t \log P(x_{t+1} | x_1, ..., x_t; \theta)$$
先 generative pre-training，再 discriminative fine-tuning，解决 label sparsity。

### 4. OneRec-V2 深度解析（快手，8B生成式推荐）

**V1 的瓶颈**：
- Encoder-Decoder 架构，97.66% 算力消耗在 Encoder 端（序列编码），仅 2.34% 用于生成
- RL 对齐依赖 reward model → reward hacking 风险

**V2 核心创新**：

**Lazy Decoder-Only Architecture**：
- 去掉 Encoder，全部用 Decoder-Only
- "Lazy" = 延迟计算：不对整个序列做全量编码，只在需要生成时按需编码上下文
- **效果**：总计算量降低 94%，训练资源降低 90%，成功 scale 到 8B 参数

**Preference Alignment with Real-World Interactions**：
- **Duration-Aware Reward Shaping**：用实际观看时长（而非二元点击）作为奖励信号
- **Adaptive Ratio Clipping**：改进 PPO 的 clip 策略，避免 reward hacking
- 不再依赖代理 reward model，直接用真实用户反馈

**AB 实验** (快手/快手极速版): App Stay Time +0.467% / +0.741%，多目标无跷跷板效应。

### 5. GLIDE 深度解析（Spotify 生成式播客召回）

**场景特点**：播客推荐 = 稳定偏好（固定收听的节目）+ 变化意图（探索新节目）

**技术方案**：
1. **Semantic ID 构建**：对播客目录做语义编码→离散化→生成层次化 token
2. **指令跟随**：将推荐建模为 instruction-following 任务
3. **Long-term Embedding as Soft Prompt**：长期用户偏好嵌入作为软提示注入，捕获稳定兴趣
4. **Constrained Decoding**：保证生成的 ID 在目录中有效

**工业约束与解法**：
- 延迟约束 → 轻量 decoder + 短期上下文编码
- 目录规模 → Semantic ID 压缩（从数百万播客到有限 token 空间）

**结果**：非习惯性播客流量 +5.4%，新节目发现 +14.3%。

### 6. NEO 深度解析（Spotify 统一搜索+推荐+推理）

**核心目标**：一个模型同时做搜索、推荐、推理，不需要外部工具。

**三阶段训练**：

| 阶段 | 名称 | 目标 |
|------|------|------|
| Stage 1 | Semantic Foundation | RQ 量化目录 item 嵌入 → SID token |
| Stage 2 | Domain Grounding | 对齐 SID token 和 LLM embedding 空间 |
| Stage 3 | Capability Induction | 指令微调：推荐/搜索/用户理解任务 |

**关键设计**：
- **Heterogeneous Catalog**：支持多实体类型（音乐/播客/有声书）
- **Constrained Decoding**：保证生成的 item ID 合法（不会"幻觉"出不存在的 item）
- **Task Control via Prompt**：自然语言 prompt 控制任务类型和输出格式

**结果** (10M+ item catalog): HR@10 提升 36-58%，NDCG@10 提升 46-97%，且有正向跨任务迁移。

### 7. 生成式推荐综述（中文 2025）— 四类标记方法

这篇综述系统梳理了生成式推荐的 Item 标记方法，是理解 Semantic ID 体系的最佳参考。

**四类标记方法对比**：

| 标记方法 | 原理 | 代表工作 | 优势 | 劣势 |
|---------|------|---------|------|------|
| **特征/语义描述** | 用自然语言描述 item | P5, GPT4Rec | 直观；利用 LLM 语言能力 | 描述冗长；歧义 |
| **向量量化 (VQ)** | VQ-VAE 离散化 | TIGER (RQ-VAE) | 层次化结构；压缩高效 | 码本崩塌；语义损失 |
| **残差量化 (RQ)** | 多阶段残差逼近 | TIGER, NEO | 更高重建精度；层次清晰 | 训练复杂度高 |
| **LLM 标记** | LLM 生成潜在标识符 | MQ-Tokenizer | 融合协同信号和语义 | 依赖 LLM 质量；计算贵 |

**架构规模趋势**：2023 百万级 → 2025 十亿级（OneRec-V2 8B），与 LLM scaling 趋势同步。

### 8. 面试高频考点 — 生成式推荐

**Q1: 生成式推荐比传统级联架构好在哪？**
> 传统级联（召回→粗排→精排→重排）每级都有信息损失，且各级独立优化目标不一致。生成式推荐端到端优化，直接从用户序列生成推荐结果，避免级联误差传播。OneRec-V2 在快手验证了端到端生成式可替代传统级联。

**Q2: Semantic ID 的构建方法？RQ-VAE 是什么？**
> Residual Quantization VAE：多层码本，每层量化上一层的残差。
> Item → Encoder → z → 第1层量化得 c1 → 残差 r1 = z - lookup(c1) → 第2层量化得 c2 → ...
> 最终 Semantic ID = [c1, c2, c3, ...]，层次化结构：粗→细。

**Q3: OneRec-V2 的 "Lazy Decoder" 为什么能省 94% 算力？**
> V1 的 Encoder 对整个用户序列做全量编码（97.66% 算力），但生成时只需要局部上下文。Lazy Decoder 去掉独立 Encoder，在 Decoder 中按需编码，避免冗余计算。本质上是从"先编码再生成"变为"生成时编码"。

**Q4: SORT 的 Generative Pre-training 为什么能缓解过拟合？**
> 工业精排的 label density 极低（曝光多、点击少），直接训练 Transformer 容易过拟合。GPT 预训练让模型在无标签的 next-item prediction 上学习行为模式，相当于数据增强。先 generative → 再 discriminative，双阶段训练。

**Q5: 如何保证生成式推荐不产生"幻觉"（生成不存在的 item）？**
> Constrained Decoding：在解码时限制 token 空间为合法 Semantic ID。GLIDE 和 NEO 都用了这个技术。本质上是在 beam search/sampling 时加上 trie 或 FSA 约束。

**Q6: NEO 如何在一个模型中统一搜索和推荐？**
> 关键是 Semantic ID 作为共享语言：搜索 = "给定 query，生成 SID"；推荐 = "给定用户历史，生成 SID"；推理 = "给定 SID，生成文本描述"。Prompt 控制任务类型，constrained decoding 保证 item 合法性。

---

## 综合洞察

### 两大趋势的交汇点

序列建模演进（Theme A）和生成式推荐（Theme B）正在融合：
1. **生成式推荐的序列建模**：OneRec-V2、GLIDE、NEO 都用 Decoder-Only（本质是自回归序列模型），序列建模的效率直接决定生成式推荐的可行性
2. **Linear/Mamba 为生成式推荐提速**：当用户序列达到数千 token，标准 Transformer 的 O(n²) 不可接受。FuXi-Linear、SIGMA 等 O(n) 方案是生成式推荐 scale 的基础
3. **Semantic ID 是桥梁**：将 item 离散化为 token 后，推荐问题完全转化为序列生成问题，可以复用 NLP 的所有序列建模技术

### 工业实践要点

| 要点 | 说明 |
|------|------|
| **序列长度分层** | 短序列用标准 Attention，长序列用 Linear/Mamba（如 FuXi-Linear 的 10x 加速） |
| **预训练+微调** | Generative pre-training → Discriminative fine-tuning (SORT) 是工业精排标配 |
| **多目标平衡** | OneRec-V2 的 Duration-Aware Reward + Adaptive Clipping 避免跷跷板 |
| **延迟约束** | Spotify GLIDE 在延迟约束下做生成式召回，需 lightweight decoder + constrained decoding |
| **统一架构** | NEO 证明搜索+推荐+推理可统一，减少系统复杂度，但需要 Semantic ID 基础设施 |

---

*Last updated: 2026-04-11*

---

## 相关概念

- [[vector_quantization_methods|向量量化方法]]
- [[multi_objective_optimization|多目标优化]]
