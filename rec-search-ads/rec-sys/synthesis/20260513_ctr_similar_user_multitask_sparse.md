# CTR 建模前沿：相似用户增强 + 稀疏长序列注意力 + 多任务知识迁移

> **日期**：2026-05-13
> **覆盖论文**：SUIN (2604.23810) / SparseCTR (2601.17836, WWW 2026) / EKTM (2605.05730)
> **三大主题**：(1) 相似用户增强 CTR (2) 稀疏注意力长序列 CTR (3) 多任务知识迁移 CVR

**相关概念页**：[[推荐中的注意力机制]] | [[序列建模演进]] | [[Embedding无处不在]] | [[多目标优化]]
**相关 synthesis**：[[20260503_scaling_sequence_multitask_frontier]] | [[20260504_scaling_coldstart_ctr_frontier]] | [[20260420_feature_interaction_ctr_advances]]

---

## 总览表

| # | 论文 | 核心贡献 | 场景 | 关键指标 |
|---|------|----------|------|----------|
| 1 | **SUIN** (2604.23810) | 相似用户行为序列增强目标用户表示 | 短/长序列 CTR | 多个 benchmark SOTA |
| 2 | **SparseCTR** (2601.17836, WWW 2026) | 时间感知稀疏注意力处理超长行为序列 | 长序列 CTR | CTR +1.72%, CPM +1.41% |
| 3 | **EKTM** (2605.05730) | Router+Transmitter 跨任务知识迁移 | 多任务 CVR 预估 | 转化率预估提升 |

---

## 1. SUIN: Similar Users-Augmented Interest Network

### Problem
用户行为序列稀疏是 CTR 预估的核心挑战。传统序列模型（DIN/DIEN/SIM/HSTU）仅依赖目标用户自身行为序列，当用户行为稀疏（新用户、低活跃用户）时，序列信息量不足以准确刻画兴趣。

### Method

SUIN 的核心思路：**从相似用户"借"行为来增强目标用户的序列表示**。

**Step 1: 相似用户检索**
- 使用序列编码器将用户行为序列编码为 embedding
- 在用户检索池中，基于 embedding 相似度检索 Top-K 相似用户
- 将相似用户的行为序列按相似度降序拼接到目标用户序列后

$$\mathbf{S}_{\text{aug}} = [\mathbf{S}_{\text{target}}; \mathbf{S}_{u_1}; \mathbf{S}_{u_2}; \cdots; \mathbf{S}_{u_K}]$$

**Step 2: User-Specific Target-Aware Position Encoding**
- 位置编码不仅编码行为在序列中的位置，还标识行为来源用户
- 捕捉每个行为与目标物品的相对位置关系

$$\text{PE}(i, j) = f(\text{user\_id}_i, \text{pos}_j, \text{target\_item})$$

**Step 3: User-Aware Target Attention**
- 联合考虑 item-item 和 user-user 两个维度的相关性
- Item-item：行为物品与目标物品的语义相似性
- User-user：行为来源用户与目标用户的相似度
- 两个维度的注意力权重共同决定每个行为的贡献

$$\alpha_{ij} = \text{softmax}\left(\frac{q_{\text{target}} \cdot k_{ij}}{\sqrt{d}} + \beta \cdot \text{sim}(u_{\text{target}}, u_i)\right)$$

### Innovation
1. **首次将相似用户行为序列作为 CTR 模型的一等公民输入**，而非简单的 collaborative filtering 特征
2. **用户感知注意力**同时建模 item-item 和 user-user 相关性，有效过滤相似用户中的噪声行为
3. **位置编码区分行为来源**，模型能学习"自己的行为 vs 相似用户的行为"的不同权重

### Results
- 在短序列和长序列 benchmark 上均显著超越 SOTA 序列 CTR 模型
- 对行为稀疏用户（<10 次交互）提升尤为显著

### Keywords
Similar User Retrieval, Behavior Augmentation, Target Attention, Position Encoding, CTR, Cold-Start Mitigation

### 面试要点
- Q: SUIN 和 SIM (Search-based Interest Model) 有什么区别？
  A: SIM 是从目标用户自身长序列中搜索相关行为（减噪），SUIN 是从其他用户的序列中借行为（增量）。两者可组合使用。
- Q: 相似用户检索的时效性问题？
  A: 用户检索池需要周期性更新，embedding 可用 ANN 索引加速。在线推理时检索开销需控制在可接受范围。

---

## 2. SparseCTR: 时间感知稀疏注意力 (WWW 2026)

> **注**：SparseCTR 已在 [[20260503_scaling_sequence_multitask_frontier]] 中有基础覆盖，此处补充技术细节和与 SUIN 的对比视角。

### Problem
标准 self-attention $O(n^2)$ 无法处理 10K+ 的长行为序列。NLP/CV 领域的稀疏注意力方案（如 Longformer 的滑动窗口）不适配推荐场景的**非均匀时间间隔**特性。

### Method: EvoAttention (Evolutionary Sparse Self-Attention)

**TimeChunking（时间感知分块）**
- 不是按固定大小分块，而是按用户行为的时间间隔自适应分块
- 时间间隔大的地方（如用户一周没活跃）自然形成 chunk 边界
- 每个 chunk 内的行为在时间上紧密相关

**三分支注意力**
1. **Global Attention**：少量全局 token 与所有行为交互，捕捉长期兴趣
2. **Transition Attention**：相邻 chunk 之间的交互，捕捉兴趣转换
3. **Local Attention**：chunk 内部的自注意力，捕捉短期兴趣

**RelTemporal（相对时间编码）**
- 将行为之间的时间差编码进注意力计算
- 时间差越小，注意力权重天然越大

$$\text{Attn}(q_i, k_j) = \frac{q_i \cdot k_j}{\sqrt{d}} + \text{RelTemporal}(t_i - t_j)$$

### Results
- 三个数量级的 FLOPs 范围内持续性能提升（scaling law 现象）
- 在线 A/B 测试：CTR +1.72%，CPM +1.41%
- 代码开源：github.com/laiweijiang/SparseCTR

### SUIN vs SparseCTR 技术对比

| 维度 | SUIN | SparseCTR |
|------|------|-----------|
| 解决的稀疏性 | 用户行为稀疏（行为少） | 计算稀疏（序列太长） |
| 核心思路 | 从外部借行为 | 从内部精选关注 |
| 注意力类型 | Target Attention | Self-Attention (Sparse) |
| 适用场景 | 新/低活用户 | 高活用户长序列 |
| 互补性 | 两者可组合：先用 SUIN 增强稀疏用户序列，再用 SparseCTR 高效处理长序列 |

---

## 3. EKTM: Effective Knowledge Transfer for Multi-Task Recommendation

### Problem
CVR（转化率）预估面临严重的数据稀疏问题：用户点击→转化的比例极低（通常 <5%），导致单任务 CVR 模型训练数据不足。多任务学习（MTL）是自然的解法，但传统 MTL 架构（Shared-Bottom/MMoE/PLE）存在两个问题：
1. **负迁移**：不相关任务的梯度干扰
2. **知识路由不明确**：Shared Expert 的知识被动扩散，缺乏主动的跨任务知识传递机制

### Method

**Router 模块（知识聚合）**
- 全局知识中枢，从所有任务的表示中聚合有用信息
- 类似于 MoE 中的全局 Expert，但专注于跨任务知识整合

**Transmitter 模块（知识转化）**
- 每个 CVR 任务配备独立的 Transmitter
- 将 Router 聚合的通用知识转化为该任务特定的增强信号
- 相当于给每个任务配了一个"翻译器"

$$\mathbf{h}_{\text{task}_i} = \text{Transmitter}_i(\text{Router}(\mathbf{h}_{\text{task}_1}, \mathbf{h}_{\text{task}_2}, \ldots, \mathbf{h}_{\text{task}_N}))$$

**关键设计**：
- Router 是共享的（全局视角），Transmitter 是任务私有的（任务特异性）
- 知识流向明确：各任务 → Router 聚合 → Transmitter 分发 → 各任务增强
- 这种显式的"汇聚-分发"比 MMoE 的隐式 gating 更可控

### Innovation
1. **显式知识路由**：不依赖 gating network 的隐式选择，而是显式定义知识流向
2. **Router-Transmitter 解耦**：全局聚合（Router）与局部转化（Transmitter）分离，避免负迁移
3. **CVR 任务间的直接互益**：每个 CVR 任务不仅从点击等辅助任务获益，还从其他 CVR 任务获益

### Results
- 在电商平台的 CVR 预估任务上显著提升
- 相比 MMoE/PLE，在多 CVR 任务设置下效果更优

### Keywords
Multi-Task Learning, Knowledge Transfer, CVR Prediction, Router-Transmitter, Negative Transfer

### 面试要点
- Q: EKTM 和 PLE 的区别？
  A: PLE 用 Extraction Network 做隐式的任务间知识共享，依赖 gating 选择 expert。EKTM 用 Router 显式聚合所有任务知识，再用 Transmitter 做任务特定转化，知识流向更明确。
- Q: 为什么不直接用 Shared Expert？
  A: Shared Expert 是被动共享（所有任务都用同一个 expert），EKTM 的 Router 是主动聚合（先整合再分发），且每个任务有独立的 Transmitter 做适配，减少负迁移。

---

## 综合洞察

### 技术趋势

**1. CTR 建模正在从"单用户单序列"走向"多源信息融合"**
- SUIN：融合相似用户行为（social/collaborative 信号）
- SparseCTR：融合超长时间跨度行为（temporal 信号）
- EKTM：融合多任务监督信号（task 信号）

**2. 注意力机制的专用化**
- 推荐场景的 attention 已经和 NLP 分化：SUIN 的 user-aware target attention、SparseCTR 的 time-aware sparse attention 都是推荐特有的设计

**3. 稀疏性问题的三个层面和解法**

| 稀疏性类型 | 具体表现 | 解法 |
|-----------|----------|------|
| 行为稀疏 | 用户交互少 | SUIN（借行为）|
| 计算稀疏 | 序列太长无法全注意力 | SparseCTR（稀疏注意力）|
| 标签稀疏 | CVR 转化样本少 | EKTM（跨任务知识迁移）|

### 面试串讲建议
这三篇论文可以用一个统一框架讲："CTR/CVR 预估的核心挑战是各种形式的稀疏性，SUIN 解决行为稀疏、SparseCTR 解决计算稀疏、EKTM 解决标签稀疏，分别从数据增强、模型效率、任务协同三个角度突破。"
