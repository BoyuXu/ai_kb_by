# 字节系搜广推 Transformer 优化演进

> 字节跳动在搜索/推荐/广告领域的 Transformer 架构创新系列。从特征交互到序列建模到统一架构，逐步解决工业级 CTR/CVR 预估的效率与效果难题。
> 关联：[[concepts/attention_in_recsys]] | [[concepts/sequence_modeling_evolution]] | [[synthesis/rec/03_精排系统_CTR到多目标生成]] | [[synthesis/ads/02_广告排序系统演进]]

---

## 目录

1. [演进脉络总览](#1-演进脉络总览)
2. [DHEN — 深层层级集成网络](#2-dhen--深层层级集成网络)
3. [RankMixer — 无参数 Token Mixing](#3-rankmixer--无参数-token-mixing)
4. [HyFormer — 混合 Transformer](#4-hyformer--混合-transformer)
5. [OneTrans — 统一 Transformer](#5-onetrans--统一-transformer)
6. [InterFormer — 异构信息双向交互](#6-interformer--异构信息双向交互)
7. [NS Tokenizer — 非序列特征 Token 化](#7-ns-tokenizer--非序列特征-token-化)
8. [HSTU — 推荐系统 Scaling Law 奠基](#8-hstu--推荐系统-scaling-law-奠基)
9. [FinalMLP — 双流 MLP 特征交互](#9-finalmlp--双流-mlp-特征交互)
10. [对比总结与面试速查](#10-对比总结与面试速查)

---

## 1. 演进脉络总览

字节搜广推 Transformer 的演进遵循一条清晰主线：**从模块分离到架构统一**。

```
传统堆叠架构（FM/DCN + DIN/BST 独立堆叠）
    |
    v
DHEN (2022): 多层异构专家 + 层级融合 — 深层特征交互
    |
    v
RankMixer (2024): 无参数 token mixing — 极简高效特征交互
    |
    v
HyFormer (2026): Query Decoding + Query Boosting 交替 — 序列与特征交互混合
    |
    v
OneTrans (2026): 单 Transformer 统一特征交互 + 序列建模 — 架构统一终局
    |
    v
InterFormer (2025): 双向交互 Cross Arch — 信息流双向增强

── 相关重要工作（非字节但深度关联）──

HSTU (2024, Meta): 1.5T 参数推荐 Transformer — 验证推荐 Scaling Law
FinalMLP (2023, 华为): 双流 MLP + 特征选择 — MLP 也能打败复杂模型
```

**核心问题演进**：
- **Phase 1**（DHEN）：特征交互不够深，浅层 FM/DCN 难以建模高阶交叉
- **Phase 2**（RankMixer）：Transformer 做特征交互太重，能否更轻量？
- **Phase 3**（HyFormer）：特征交互和序列建模各自独立，如何混合增强？
- **Phase 4**（OneTrans）：为什么要两套模块？能否一个 Transformer 搞定一切？
- **对照线**（HSTU）：Meta 证明推荐也有 Scaling Law，1.5T 参数持续提升
- **对照线**（FinalMLP）：简单 MLP 双流架构也能 SOTA，质疑复杂架构必要性

---

## 2. DHEN — 深层层级集成网络

> Deep Hierarchical Ensemble Network for CTR Prediction (KDD 2022, ByteDance)

### 核心创新

DHEN 发现：现有模型（DCN、AutoInt、xDeepFM）的特征交互模块通常只有 2-3 层就饱和，无法通过简单加深获得收益。原因是**单一交互范式**在深层会退化。

解法：**多种异构交互专家 + 层级融合**。每一层包含多个不同类型的特征交互模块（FM、Cross Network、Self-Attention、MLP 等），通过门控机制动态融合。

### 架构

```
Input Embedding
    |
    v
[Layer 1] Expert_FM  Expert_CrossNet  Expert_SelfAttn  Expert_MLP
    |           |            |              |
    +--- Gating Network (softmax 门控) ---+
    |
    v
[Layer 2] Expert_FM  Expert_CrossNet  Expert_SelfAttn  Expert_MLP
    |           |            |              |
    +--- Gating Network ---+
    |
    v  (可堆叠 N 层)
Output Head
```

**关键设计**：
- 每层 $K$ 个异构专家，各专家用不同交互范式
- 层间通过门控网络动态选择/融合专家输出：$\mathbf{h}_l = \sum_{k=1}^{K} g_k^{(l)} \cdot E_k^{(l)}(\mathbf{h}_{l-1})$
- 门控权重 $g_k^{(l)} = \text{softmax}(\mathbf{W}_g \mathbf{h}_{l-1})_k$
- 异构性防止深层退化（不同专家有不同的梯度特征）

### 与前序工作对比

| 方面 | DCN-V2 | AutoInt | DHEN |
|------|--------|---------|------|
| 交互范式 | 单一（Cross Layer） | 单一（Self-Attention） | 多种异构 |
| 可堆叠深度 | 2-3 层饱和 | 2-3 层饱和 | 7+ 层持续提升 |
| 融合方式 | 无 | 无 | 层级门控 |
| 参数量 | 中 | 中 | 较大（多专家） |

### 面试高频考点

1. **Q**: DHEN 为什么能堆叠更深而不退化？
   **A**: 异构专家提供不同梯度信号，避免单一交互范式在深层的表达力饱和。类比 MoE 的思想——多样性本身就是正则化。

2. **Q**: DHEN 的门控和 MoE 的门控有什么区别？
   **A**: DHEN 是层级门控（每层融合所有专家），MoE 是稀疏门控（每层只激活 Top-K 专家）。DHEN 更关注多样性融合，MoE 更关注计算效率。

3. **Q**: DHEN 的工程瓶颈？
   **A**: 多专家并行计算增加显存和延迟。工业部署需要对专家数量做 AUC-延迟 Pareto 取舍。

---

## 3. RankMixer — 无参数 Token Mixing

> 字节内部工作，未独立发表论文，但在 HyFormer 中作为核心组件被引用和使用。

### 核心创新

RankMixer 提出了一种**无需可学习参数**的 token mixing 方法，用于替代 Self-Attention 做特征交互。核心思想：将 token 维度和特征维度之间做 reshape + transpose，实现跨 token 的信息交换。

### 架构（Token Mixing 操作）

给定 $T$ 个 token，每个 token 维度为 $D$：

$$Q \in \mathbb{R}^{B \times T \times D} \xrightarrow{\text{reshape}} \mathbb{R}^{B \times T \times T \times d_{\text{sub}}} \xrightarrow{\text{transpose}(1,2)} \mathbb{R}^{B \times T \times T \times d_{\text{sub}}} \xrightarrow{\text{reshape}} \mathbb{R}^{B \times T \times D}$$

其中 $d_{\text{sub}} = D / T$。

**直觉理解**：把每个 token 的特征向量切成 $T$ 段，每段长 $d_{\text{sub}}$；然后在 $T$ 个 token 之间交换对应段的信息。相当于一次"洗牌"操作。

**约束**：$D$ 必须能被 $T$ 整除。

### RankMixerBlock 完整结构

```
Input tokens [B, T, D]
    |
    +-- Token Mixing (reshape + transpose, 无参数)
    |
    +-- Shared FFN (SwiGLU 激活)
    |
    +-- LayerNorm + Residual
    |
Output tokens [B, T, D]
```

### 与前序工作对比

| 方面 | Self-Attention | MLP-Mixer | RankMixer |
|------|---------------|-----------|-----------|
| 计算复杂度 | $O(T^2 D)$ | $O(T D^2)$ | $O(T D)$（reshape 近乎免费） |
| 可学习参数 | $4D^2$（QKV+O） | $T^2 + D^2$ | **0**（mixing 无参数） |
| 信息混合方式 | 加权聚合 | 全连接 | 固定排列交换 |
| 适用场景 | 通用 | CV | CTR 特征交互（token 数少） |

### 面试高频考点

1. **Q**: RankMixer 的 token mixing 为什么有效？为什么无参数也能工作？
   **A**: CTR 场景的 token 数量少（通常 <20），每个 token 有明确语义（用户特征组/物品特征组）。信息混合的关键在于让不同 token 的子空间特征交叉——reshape+transpose 恰好实现了这点。后续的 FFN 提供了非线性变换能力。

2. **Q**: RankMixer vs Self-Attention 在 CTR 场景的取舍？
   **A**: RankMixer 速度快 3-5 倍，适合延迟敏感场景。Self-Attention 在 token 语义差异大时更好（能学习 token 间的动态权重）。HyFormer 的设计是：序列用 Attention，特征交互用 RankMixer。

3. **Q**: $D$ 必须被 $T$ 整除这个约束在工程上如何处理？
   **A**: 设计时调整 $d_{\text{model}}$ 使其为 token 总数的倍数。或使用 padding 凑整。

---

## 4. HyFormer — 混合 Transformer

> HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction (arXiv 2026, ByteDance)
> arXiv:2601.12681

### 核心创新

HyFormer 的核心洞察：**序列建模和特征交互不应该是独立的两阶段，而应该交替进行、相互增强**。

传统堆叠架构：先做序列建模（DIN/BST），输出一个用户兴趣向量，再和其他特征做交叉。问题：序列建模时看不到非序列特征的信号，特征交互时序列信息已被压缩。

HyFormer 的解法：设计**可堆叠的 HyFormerBlock**，每个 Block 交替执行 Query Decoding（从序列提取兴趣）和 Query Boosting（跨 token 特征交互）。

### 架构

```
NS Tokenizer: 非序列特征 -> NS tokens [B, T_ns, D]
Seq Tokenizer: 行为序列 -> Seq tokens [B, L, D] (per domain)
    |
    v
Query Generator: NS + MeanPool(Seq) -> Q tokens [B, N_q * S, D]
    |
    v
HyFormerBlock x N (可堆叠):
    |
    +-- Step 1: Sequence Evolution (per-domain 编码器)
    |       SwiGLU / Transformer / LongerEncoder
    |
    +-- Step 2: Query Decoding (per-domain cross-attention)
    |       Q tokens attend to Seq tokens -> 解码兴趣
    |
    +-- Step 3: Token Fusion
    |       concat(decoded Q tokens, NS tokens) -> all tokens
    |
    +-- Step 4: Query Boosting (RankMixerBlock)
    |       token mixing + FFN -> 特征交互
    |
    v
Output Head -> CTR logit
```

### 关键设计细节

**Query Generator**：
- 全局信息 = Concat(NS_tokens_flat, MeanPool(Seq_i))
- 每个 query 由独立 FFN（SiLU 激活）生成
- 每个序列域独立生成 $N_q$ 个 query token

**Sequence Evolution 三种编码器选型**：

| 编码器 | 结构 | 复杂度 | 适用场景 |
|--------|------|--------|---------|
| SwiGLU | LN + SwiGLU + Residual | $O(LD)$ | 短序列，追求速度 |
| Transformer | Self-Attention + FFN | $O(L^2D)$ | 标准方案 |
| Longer | Top-K 压缩 + Self-Attention | $O(K^2D + LKD)$ | 长序列（L > 200） |

**LongerEncoder** 的长序列压缩：
- 首层：Cross-Attention（Q = latest top_k tokens, KV = all tokens）→ 压缩到 $K$ 个 token
- 后续层：退化为 Self-Attention（因为 $L \le K$）
- 支持 RoPE 位置编码

**Query Boosting（RankMixerBlock）**：
- 将所有 decoded Q tokens + NS tokens 拼接
- 执行 RankMixer 的无参数 token mixing + Shared FFN
- 实现跨域、跨类型的特征交互

### 与前序工作对比

| 方面 | DIN/SIM | BST | DCN+BST 堆叠 | HyFormer |
|------|---------|-----|-------------|----------|
| 序列建模 | Target Attention | Self-Attention | Self-Attention | Seq Evolution + Query Decoding |
| 特征交互 | MLP | 无 | DCN/FM | RankMixer (Query Boosting) |
| 序列-特征交互 | 无 | 无 | 串行（先序列后交互） | **交替进行**（可堆叠） |
| 长序列支持 | SIM 检索 | 截断 | 截断 | LongerEncoder 压缩 |
| 多序列域 | 分别建模 | 分别建模 | 分别建模 | per-domain 编码 + 统一 fusion |

### 面试高频考点

1. **Q**: HyFormer 相比 DIN+DCN 堆叠架构的核心优势？
   **A**: 交替执行让信息双向流动——序列建模利用了特征交互的结果（上一层 Query Boosting 更新的 Q tokens），特征交互利用了序列建模的结果（本层 Query Decoding 的输出）。堆叠架构是单向的信息流。

2. **Q**: HyFormer 的 Query Decoding 和 DIN 的 Target Attention 有什么区别？
   **A**: DIN 用 candidate item 做 query 对整个序列加权求和，得到一个向量。HyFormer 用多个可学习 query token 做 cross-attention，保留了多个不同维度的兴趣表示（多 query = 多兴趣）。更重要的是 HyFormer 的 query 是可堆叠更新的。

3. **Q**: RankMixer 在 HyFormer 中的角色？能否替换成 Self-Attention？
   **A**: RankMixer 在 Query Boosting 阶段做特征交互，替代 Self-Attention 可节省 3-5 倍延迟。可以替换成 Self-Attention，效果略好但延迟增加，需要在 Pareto 前沿选点。

4. **Q**: LongerEncoder 的 Top-K 压缩和 SIM 的 Top-K 检索有什么区别？
   **A**: SIM 是在 Embedding 空间做硬检索（最相关的 K 个 item），信息有损。LongerEncoder 是在 Attention 空间做软压缩（Cross-Attention 的 K 个 query attend 全序列），保留了全局信息。

5. **Q**: 为什么 HyFormer 是 TAAC 2026 比赛的 baseline（PCVRHyFormer）？
   **A**: 赛题要求"统一架构同时建模序列和特征交互"，HyFormer 天然满足。且其模块化设计（编码器可选、Block 可堆叠）方便参赛者在此基础上改进。

---

## 5. OneTrans — 统一 Transformer

> OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer (WWW 2026, ByteDance)
> arXiv:2510.26104

### 核心创新

OneTrans 把统一做到了极致：**一个 Transformer backbone 同时完成特征交互和序列建模**，不再需要独立的序列编码器和特征交互模块。

核心思想：将所有输入（用户特征、物品特征、上下文特征、行为序列）统一编码为 token 序列，输入同一个 Transformer，通过精心设计的 Attention Mask 控制信息流。

### 架构

```
统一 Tokenizer
    |
    +-- user_feat_tokens (非序列)
    +-- behavior_seq_tokens (序列, 按时间排列)
    +-- ctx_tokens (上下文)
    +-- target_ad_token (候选广告)
    |
    v
[user_feat] [seq_1, seq_2, ..., seq_T] [ctx] [target]
    |
    v
Shared Transformer (N layers)
    |  Attention Mask:
    |  - 行为序列: Causal Mask (防止未来泄露)
    |  - 非序列特征: Full Mask (无时序约束)
    |  - 跨域: 特征 token 可 attend 序列, 序列不看未来上下文
    |
    v
[CLS] / target_token -> CTR Head -> logit
    |
    +-- 辅助任务: 序列预测头 (预测下一个行为 item)
```

### 关键设计

**统一 Tokenizer**：
- 非序列特征：每个字段或字段组映射为一个 token
- 序列特征：每个行为 item 映射为一个 token（concat 多个 side-info embedding）
- 所有 token 投影到相同维度 $d_{\text{model}}$

**Attention Mask 策略**（核心创新）：

| Source Token | Target Token | 能否 Attend | 原因 |
|-------------|-------------|-------------|------|
| 序列 token $i$ | 序列 token $j \le i$ | 是 | Causal: 看历史 |
| 序列 token $i$ | 序列 token $j > i$ | 否 | 防未来泄露 |
| 非序列 token | 所有 token | 是 | 无时序约束 |
| 序列 token | 非序列 token | 是 | 可利用静态特征 |

**位置编码适配**：
- 行为序列 token：RoPE（编码时序位置）
- 非序列特征 token：无位置编码（顺序无关）

**双任务训练**：

$$\mathcal{L} = \mathcal{L}_{\text{CTR}} + \lambda \cdot \mathcal{L}_{\text{seqPred}}$$

- 序列预测辅助任务：利用 Causal Mask 预测下一个行为 item
- $\lambda$ 建议 0.1-0.3

### 与前序工作对比

| 方面 | DIN+DCN 堆叠 | HyFormer | OneTrans |
|------|-------------|----------|----------|
| 架构 | 独立模块堆叠 | 交替执行 | 单一 Transformer |
| 参数共享 | 无 | 部分（RankMixer） | **完全共享** |
| 参数量 | 大（各模块独立参数） | 中 | **小（-35%）** |
| 信息流 | 单向（先序列后交互） | 交替双向 | **全局双向** |
| 训练速度 | 1x | ~1.2x | **1.4x** |
| 推理优化 | 困难 | 中等 | **KV Cache 友好** |

### 面试高频考点

1. **Q**: OneTrans 如何用一个 Transformer 同时做特征交互和序列建模？
   **A**: Self-Attention 本质上就是所有 token pair 之间的交互——当 token 包含不同类型特征时，Self-Attention 自然实现了特征交互。通过 Causal Mask 控制序列 token 的信息流方向，保留了时序建模能力。统一后，特征交互和序列建模在每一层都在同时发生。

2. **Q**: OneTrans 的 Attention Mask 如何设计？为什么这样设计？
   **A**: 序列 token 用 Causal Mask（看历史不看未来），防止信息泄露，同时支持辅助序列预测任务。非序列 token 用 Full Mask（看所有），因为静态特征没有时序约束。这种混合 mask 是 OneTrans 统一的关键。

3. **Q**: OneTrans 的辅助序列预测任务为什么能提升 CTR？
   **A**: (1) 迫使模型更好地建模行为序列的时序规律，学到更好的序列表示作为 CTR 的上游特征。(2) 多任务学习的正则化效果减少过拟合。(3) Causal Mask 让同一个模型可以同时服务两个任务，零额外推理开销（推理时去掉序列预测头）。

4. **Q**: OneTrans 参数量减少 35% 的原因？
   **A**: 传统架构中，序列编码器（Transformer layers）和特征交互网络（DCN/FM layers）各有独立参数。OneTrans 用共享的 Transformer layers 同时完成两者，消除了参数冗余。

5. **Q**: OneTrans vs HyFormer 各自适合什么场景？
   **A**: OneTrans 适合延迟极敏感、追求极简架构的场景（如广告精排）。HyFormer 适合多序列域、需要灵活配置不同序列编码器的场景（如多业务线推荐）。HyFormer 的模块化设计更方便实验和迭代。

---

## 6. InterFormer — 异构信息双向交互

> InterFormer: Effective Heterogeneous Interaction Learning for CTR Prediction (CIKM 2025)
> arXiv:2411.09852

### 核心创新

InterFormer 关注的问题：序列建模分支和特征交互分支之间的**信息隔离**。传统堆叠架构中，序列分支（Sequence Arch）和交互分支（Interaction Arch）独立运行，信息只在最后做一次拼接。

解法：引入 **Cross Arch**，在每一层的序列分支和交互分支之间建立双向信息桥梁。

### 架构

```
Sequence Arch          Cross Arch          Interaction Arch
(序列建模)          (双向桥梁)          (特征交互)
     |                  |                      |
  [Layer 1]    <-- Cross Attn -->         [Layer 1]
     |                  |                      |
  [Layer 2]    <-- Cross Attn -->         [Layer 2]
     |                  |                      |
  [Layer N]    <-- Cross Attn -->         [Layer N]
     |                  |                      |
     +------------ Concat + Head ----------+
```

**Cross Arch 的双向交互**：
- 序列 → 交互：序列分支的输出作为 KV，交互分支的 token 作为 Q → 交互分支获得序列信号
- 交互 → 序列：交互分支的输出作为 KV，序列分支的 token 作为 Q → 序列分支获得全局特征信号

### 与前序工作对比

| 方面 | 堆叠架构 | HyFormer | OneTrans | InterFormer |
|------|---------|----------|----------|-------------|
| 信息流方向 | 单向 | 交替（隐式双向） | 全局双向 | **显式双向** |
| 两个分支独立性 | 完全独立 | 共享 query | 完全融合 | 独立 + 桥梁 |
| 灵活性 | 高（可选任意模块） | 中 | 低（必须统一） | **最高** |
| 模块可替换性 | 是 | 部分 | 否 | 是 |

### 面试高频考点

1. **Q**: InterFormer 和 HyFormer 都解决序列-特征交互的问题，核心区别是什么？
   **A**: HyFormer 通过**共享 query token** 实现隐式双向（query 在 decoding 和 boosting 之间传递信息）。InterFormer 通过**显式 Cross Attention** 在两个独立分支之间建桥。InterFormer 保持了两个分支的独立性，更灵活；HyFormer 融合更紧密，效率更高。

2. **Q**: InterFormer 的 Cross Arch 增加了多少延迟？
   **A**: 每层增加两次 Cross Attention（序列→交互 + 交互→序列），额外计算量约为主分支的 30-50%。可以通过减少 Cross Arch 的层数（如每 2 层做一次交互）来控制。

---

## 7. NS Tokenizer — 非序列特征 Token 化

> 非独立论文，是 RankMixer / HyFormer 中的关键组件

### 核心创新

传统 CTR 模型将特征 Embedding 拼接为一个长向量输入 MLP。NS Tokenizer 将非序列特征**分组编码为语义 token**，让后续模块可以用 Attention / Mixing 机制做 token-level 交互。

### 两种变体

**GroupNSTokenizer**（语义分组）：
```
按业务语义分组（如 ns_groups.json 定义）:
  Group 1 (用户基础): age, gender, city -> mean_pool -> Linear -> token_1
  Group 2 (用户行为统计): click_cnt, buy_cnt -> mean_pool -> Linear -> token_2
  Group 3 (物品属性): category, brand, price -> mean_pool -> Linear -> token_3
  ...
```
- 每个语义组产生 1 个 token
- 组内先各自 Embedding，再 mean pooling，最后投影到 $d_{\text{model}}$
- 需要人工定义分组策略

**RankMixerNSTokenizer**（自动均分）：
```
所有特征 Embedding 拼接: [e_1, e_2, ..., e_F] -> concat -> 长向量
均分为 T 段: [seg_1 | seg_2 | ... | seg_T]
每段独立投影: seg_i -> Linear -> token_i
```
- 无需人工分组，$T$ 可自由指定
- 每段对应的特征组合由 Embedding 拼接顺序决定

### 面试高频考点

1. **Q**: GroupNSTokenizer vs RankMixerNSTokenizer 如何选择？
   **A**: GroupNSTokenizer 语义更清晰，适合特征含义明确的场景，但依赖人工分组经验。RankMixerNSTokenizer 全自动，适合特征数量多、语义边界模糊的场景。工业实践中建议先用 RankMixerNSTokenizer 快速实验，再用 GroupNSTokenizer 做精细优化。

2. **Q**: 为什么要做特征 Token 化？直接拼接 Embedding 做 MLP 不行吗？
   **A**: Token 化后可以用 Attention/Mixing 机制做结构化的 token-level 交互，每对 token 之间的交互是显式的。MLP 的交互是隐式的，且对 token 顺序不敏感。Token 化是 Transformer 处理特征的前提。

---

## 8. 对比总结与面试速查

### 全景对比表

| 模型 | 年份 | 核心贡献 | 序列建模 | 特征交互 | 序列-特征耦合 | 参数效率 | 推理友好 |
|------|------|---------|---------|---------|-------------|---------|---------|
| DHEN | 2022 | 异构专家深层交互 | 无（需外接） | 多专家门控 | 无 | 低 | 中 |
| RankMixer | 2024 | 无参数 token mixing | 无 | reshape+transpose | 无 | **极高** | **极高** |
| HyFormer | 2026 | 交替 decode+boost | Transformer/Longer | RankMixer | 交替（隐式双向） | 高 | 高 |
| OneTrans | 2026 | 单 Transformer 统一 | 共享 Transformer | 共享 Transformer | **完全统一** | **最高** | **最高** |
| InterFormer | 2025 | 显式双向 Cross Arch | 独立 Seq Arch | 独立 Int Arch | 显式双向桥梁 | 中 | 中 |

### 面试万能框架：字节 Transformer 演进的底层逻辑

**三句话讲清楚**：
1. **问题**：工业 CTR 系统中，特征交互（FM/DCN）和序列建模（DIN/BST）是两套独立模块，信息单向流动（先序列→后交互），导致表达力受限。
2. **演进**：从 DHEN（加深交互）→ RankMixer（轻量交互）→ HyFormer（交替混合）→ OneTrans（完全统一），逐步打破两套模块的边界。
3. **终局**：OneTrans 证明了一个 Transformer 可以同时完成两者，通过 Attention Mask 控制信息流，参数减少 35%，速度提升 1.4x。

### TAAC 2026 比赛关联

- **Baseline**：PCVRHyFormer = HyFormer 在 PCVR（Post-Click CVR）场景的实现
- **赛题方向**：Towards Unifying Sequence Modeling and Feature Interaction（与 OneTrans 完全契合）
- **改进思路**：
  - HyFormer → OneTrans 统一架构
  - InterFormer 的双向交互思路改进 Query Decoding
  - DCN-V2 替换 RankMixerBlock 做显式高阶交叉
  - LongerEncoder 参数调优（top_k、层数）

---

> 最后更新：2026-05-01
> 关联文档：[[concepts/attention_in_recsys]] | [[concepts/sequence_modeling_evolution]] | [[synthesis/rec/03_精排系统_CTR到多目标生成]] | [[synthesis/ads/taac2026_kdd_competition]]
