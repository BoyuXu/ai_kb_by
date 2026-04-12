# 多任务学习与 MoE

> **创建日期**: 2026-04-13 | **合并来源**: 推荐广告系统多任务学习与MoE专家混合, 精排模型进阶深度解析（多任务部分）, 推荐广告生成式范式统一全景（MoE部分）
>
> **核心命题**: 多任务学习从 SharedBottom 到 MoE 到稀疏专家，核心挑战是任务间负迁移与规模化扩展

---

## 一、多任务架构演进

```
SharedBottom (共享底层)
  → MMOE (多门控混合专家, 2018)
    → PLE (渐进分层萃取, 2020)
      → SNR (子网络路由, 2021)
        → AITM (自适应信息传递, 2022)
          → DHEN (深度层次专家, 2023)
            → HoME (同构混合专家, 2024)
              → SMES (稀疏门控专家, 2025)
                → MGOE (宏观图专家, 2025)
                  → Meta Lattice / RankMixer (2025-2026)
```

---

## 二、经典架构详解

### 2.1 MMOE (Multi-gate Mixture-of-Experts)

```
所有 Expert 对所有任务可见，每任务独立 Gating：

Expert1  Expert2  Expert3
   ↓        ↓        ↓
   └────┬───┴────┬───┘
        │        │
   Gate A [0.6,0.3,0.1]   Gate B [0.1,0.5,0.4]
        ↓                       ↓
     Task A                  Task B
```

$$
y_k = h_k\left(\sum_{i=1}^{n} g_k^{(i)}(x) \cdot f_i(x)\right)
$$

$g_k^{(i)}$ 是任务 $k$ 对 expert $i$ 的门控权重，$f_i$ 是第 $i$ 个 expert。

### 2.2 PLE (Progressive Layered Extraction)

**核心改进**：显式分离私有 Expert 和共享 Expert。

```
Task A 私有 Expert | 共享 Expert | Task B 私有 Expert
        ↓                ↓                ↓
    Selector A (只从自己+共享中选)    Selector B
        ↓                                ↓
     Task A                           Task B
```

- 每个任务只从"自己的私有 Expert + 共享 Expert"中选择
- 多层堆叠：每层的输出作为下一层的输入（渐进萃取）
- 解决 MMOE 中任务差异大时的负迁移问题

### 2.3 SNR (Sub-Network Routing)

**核心创新**：显式 Controller 决定子网络连接，而非简单加权融合。

```python
# Controller 生成路由权重
routing_logits = self.controller(x)  # [batch, task_num * subnet_num]
routing_weights = softmax(routing_logits)

# 子网络输出 + 路由融合
for task_id in range(task_num):
    fused = sum(routing_weights[task_id, i] * subnet_output[i])
    pred = tower[task_id](fused)
```

**vs MMOE**：MMOE 是单层 Expert 加权，SNR 是子网络级路由，可学习更复杂的连接模式。

### 2.4 AITM (Adaptive Information Transfer)

**核心创新**：建模任务间显式依赖关系（曝光→点击→转化）。

$$
h_{\text{task}_k} = \text{AIT}(h_{\text{current}}, \{h_{\text{task}_1}, ..., h_{\text{task}_{k-1}}\})
$$

AIT 用 Attention 机制从前序任务选择性接收信息：
- Query: 当前任务表示
- Key/Value: 前序任务表示
- 自适应传递，后序任务可以借用前序任务的信号

### 2.5 四者对比

| 维度 | MMOE | PLE | SNR | AITM |
|------|------|-----|-----|------|
| Expert 可见性 | 全部共享 | 私有+共享 | 子网络内共享 | 任务独立+传递 |
| 任务关系 | 隐式学习 | 部分显式 | Controller 控制 | 显式依赖链 |
| 适用场景 | 任务相关性中等 | 任务差异大 | 复杂动态路由 | 序列依赖任务 |
| 代表任务 | CTR+CVR | 点击+收藏+评论 | 复杂多任务 | 曝光→点击→转化 |

**选型建议**：
- 任务间相关性高（CTR+CVR）：MMOE 或 AITM
- 任务间差异大（点击+评论+分享）：PLE
- 任务关系复杂且动态：SNR
- 有明显序列依赖（曝光→点击→转化）：AITM

---

## 三、前沿 MoE 架构

### 3.1 SMES — 稀疏门控多任务 (推荐侧)

**问题**：任务数从 3-5 增长到 20-50 时，传统 MMOE/PLE 面临参数冗余和负迁移。

**方案**：每任务只激活 Top-K Expert：
$$
g_{t,m} = \begin{cases} s_{t,m} & \text{if } s_{t,m} \in \text{TopK}(\{s_{t,j}\}_{j=1}^{M}) \\ 0 & \text{otherwise} \end{cases}
$$

- 层次化 Expert：Shared / Task-Group / Task-Specific
- Load Balancing Loss 防止 expert collapse
- 20+ 任务下 AUC +0.54%，FLOPs -60%

### 3.2 MGOE — 宏观图专家 (广告侧)

**任务级全局路由**（vs SMES 的样本级动态路由）：

$$
w_{t,m}^{(l)} = \frac{\exp((\alpha_{t,m}^{(l)} + g_m) / \tau)}{\sum_{m'} \exp((\alpha_{t,m'}^{(l)} + g_{m'}) / \tau)}
$$

- Gumbel-Softmax 可微分学习
- 同一任务所有样本走相同路径（更稳定）
- 适合任务数固定、规模极大（亿级参数）的场景

### 3.3 SMES vs MGOE

| 维度 | SMES (推荐侧) | MGOE (广告侧) |
|------|-------------|-------------|
| 路由粒度 | 样本级动态稀疏 | 任务级静态稀疏 |
| 适用场景 | 样本差异大 | 任务数固定，规模大 |
| 稳定性 | 较低（路由随样本变化） | 高（路由固定） |
| 本质 | 减少样本搜索空间 | 减少任务搜索空间 |

### 3.4 HoME (Homogeneous MoE)

所有 Expert 结构相同但参数不同，通过 Gating 组合。简化了 Expert 设计，便于并行计算。

### 3.5 Meta Lattice / RankMixer

最新方向：用 lattice 结构（格结构）建模任务间关系，自动发现任务分组和共享模式。

---

## 四、梯度冲突处理

多任务学习中不同任务梯度可能冲突：$g_i \cdot g_j < 0$

### 4.1 PCGrad (Project Conflicting Gradients)

当两任务梯度内积为负时，将冲突梯度投影到正交方向：

$$
g_i' = g_i - \frac{g_i \cdot g_j}{\|g_j\|^2} g_j
$$

修改后 $g_i' \perp g_j$，两任务更新不再冲突。

### 4.2 GradNorm

自适应调整任务权重，使各任务以相似速度学习：

$$
\tilde{G}_i(t) = \bar{G}(t) \times \left[\frac{L_i(t)}{L_i(0)}\right]^\alpha
$$

学习慢的任务获得更大权重（更大的目标梯度范数）。

### 4.3 Uncertainty Weighting

基于任务不确定性自动加权：

$$
\mathcal{L} = \sum_i \left(\frac{1}{2\sigma_i^2} L_i + \log \sigma_i\right)
$$

不确定性高的任务权重自动降低（precision = $1/\sigma^2$）。

### 4.4 梯度方法对比

| 方法 | 核心思想 | 适用场景 |
|------|---------|---------|
| PCGrad | 投影冲突梯度 | 梯度方向明显冲突 |
| GradNorm | 平衡训练速度 | 任务学习速度差异大 |
| Uncertainty | 不确定性加权 | 任务噪声水平不同 |

### 4.5 负迁移诊断

如何判断负迁移是否发生：
1. **指标变化**：引入新任务后旧任务 AUC 下降
2. **梯度分析**：两任务梯度 cos < 0（方向冲突）
3. **梯度量级**：某任务梯度范数 >> 其他（主导更新方向）

---

## 五、工业经验

### 5.1 Loss 权重设计

工业中多目标融合分常用乘法公式：
$$
\text{score} = \text{pCTR}^{w_1} \times \text{pCVR}^{w_2} \times \text{duration}^{w_3}
$$

权重调整方式：
- 人工调参（快速但不稳定）
- GradNorm / Uncertainty Weighting（自动但需要额外计算）
- Pareto 优化（理论最优但工程复杂）

### 5.2 统一模型 vs 独立模型

| 方案 | 优势 | 劣势 |
|------|------|------|
| 独立模型 | 无负迁移，各自优化 | 维护成本高，接口复杂 |
| MMOE/PLE | 共享表示，减少冗余 | 可能负迁移 |
| SMES 稀疏 | 规模化扩展 | 路由学习不稳定 |
| 统一大模型 | 维护成本最低 | Debug 更难，A/B 定位难 |

### 5.3 SMES vs LLM MoE

**相同点**：都用 Top-K 稀疏激活 + load balancing loss。
**不同点**：
- SMES gating 是 task-aware（每任务独立 gating），LLM MoE 是 token-level
- SMES 有层次化 Expert（Shared/Task-Group/Task-Specific），LLM MoE 的 expert 对等
- SMES 目标是避免负迁移，LLM MoE 目标是增加模型容量

---

## 六、面试高频考点

**Q1: MMOE 和 PLE 的核心区别？**
A: MMOE 所有 Expert 对所有任务可见，通过 gating 隐式学习共享/独有。PLE 显式分离私有和共享 Expert，每任务只从自己+共享中选，避免大任务差异下的负迁移。

**Q2: 多任务学习的负迁移如何判断？**
A: (1) 指标：引入新任务后旧任务 AUC 下降；(2) 梯度 cos < 0（方向冲突）；(3) 某任务梯度范数 >> 其他。PCGrad 通过投影解决方向冲突，GradNorm 通过自适应权重解决速度差异。

**Q3: SMES 和 LLM 的 MoE 有何异同？**
A: 相同：Top-K 稀疏激活 + load balancing。不同：SMES 是 task-aware gating，LLM 是 token-level；SMES 有层次化 Expert，LLM Expert 对等；目标不同（避免负迁移 vs 增加容量）。

**Q4: AITM 的自适应信息传递如何工作？**
A: 用 Attention 机制：当前任务表示作 Query，前序任务表示作 Key/Value。后序任务（如转化）可以选择性接收前序任务（如点击）的信息。适合曝光→点击→转化的序列依赖结构。

**Q5: 梯度冲突的三种处理方法各适合什么场景？**
A: PCGrad（投影）适合梯度方向明显冲突；GradNorm（平衡速度）适合任务学习速度差异大；Uncertainty Weighting（不确定性加权）适合任务噪声水平不同。

**Q6: SMES 如何解决 20+ 任务的规模化？**
A: Top-K 稀疏激活让计算复杂度与任务数解耦，层次化 Expert 组织减少冗余，Load Balancing Loss 防止 expert collapse。相比 PLE，20+ 任务下 AUC +0.54% 且 FLOPs -60%。

---

## 相关概念

- [[concepts/multi_objective_optimization|多目标优化]]
- [[concepts/attention_in_recsys|Attention 在搜广推中的演进]]
- [[concepts/generative_recsys|生成式推荐统一视角]]
