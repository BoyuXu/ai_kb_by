# 精排系统：CTR 到多目标生成

> **创建日期**: 2026-04-13 | **合并来源**: CTR模型深度解析, 推荐系统排序范式演进, 推荐系统全链路架构概览, 20260411_industrial_recsys_fullstack, 精排模型进阶深度解析, recsys-infrastructure-landscape-2025-2026
>
> **核心命题**: 精排从特征交叉到深度学习到 Transformer 到生成式模型，核心约束始终是延迟 vs 效果的平衡

---

## 一、全链路架构：级联漏斗

```
全量物料池 (~百万/千万)
    ↓ 召回 (Recall)
候选集 (~数千)     ← 多路召回: 双塔/图/序列/热门
    ↓ 粗排 (Pre-Ranking)
候选集 (~数百)     ← 轻量模型蒸馏: 双塔/轻量交叉
    ↓ 精排 (Ranking)
候选集 (~数十)     ← 重模型: DIN/DIEN/DCN/Transformer + 多目标
    ↓ 重排 (Re-Ranking)
展示结果 (~十几)   ← 多样性/去重/业务规则/上下文
```

**核心原则**：每一级用更复杂的模型处理更少的候选。上层负责"不漏"，下层负责"精准"。

---

## 二、精排模型演进

```
LR → FM → DeepFM → Wide&Deep → DCN → DIN → DIEN → BST → DCN-V2 → HSTU → SORT
```

### 2.1 特征交叉演进

| 模型 | 年份 | 核心机制 | 交叉方式 |
|------|------|---------|---------|
| LR | - | 手动特征交叉 | 人工构造 |
| FM | 2010 | 隐向量内积 | 二阶自动交叉 |
| DeepFM | 2017 | FM + DNN 并行 | 低阶(FM)+高阶(DNN) |
| Wide&Deep | 2016 | Wide(记忆) + Deep(泛化) | 双路并行 |
| DCN | 2017 | Cross Network | 显式高阶交叉 |
| DCN-V2 | 2021 | 混合专家 Cross Network | 更高效的高阶交叉 |

### 2.2 用户行为建模演进

| 模型 | 核心机制 | 关键创新 |
|------|---------|---------|
| DIN | Target Attention | 候选 item 引导的注意力加权 |
| DIEN | GRU + AUGRU | 兴趣提取 + 兴趣演化 |
| BST | Transformer | 自注意力序列建模 |
| SIM | 两阶段检索 | 万级超长序列 |
| HSTU | 去 softmax Transformer | 统一召回+排序 |
| SORT | Request-centric Transformer | 全栈工业优化 |

### 2.3 DSSM 双塔（召回→粗排桥梁）

```
User Tower:              Item Tower:
[user_id, age, ...]      [item_id, category, ...]
    ↓ MLP                    ↓ MLP
user_embedding           item_embedding
    ↓                        ↓
    └──── cosine sim ────────┘
              ↓
          similarity score
```

粗排使用双塔打分，特征比召回更多但比精排少。精排蒸馏是粗排的工业主流方案。

---

## 三、精排核心公式

### 3.1 DIN Target Attention

$$
\alpha_i = \frac{\exp(f(e_i, e_{target}))}{\sum_j \exp(f(e_j, e_{target}))}
$$

$$
v_U = \sum_i \alpha_i \cdot e_i
$$

### 3.2 多目标融合公式

工业常用加权融合：
$$
\text{score} = \text{pCTR}^{w_1} \times \text{pCVR}^{w_2} \times \text{durationPred}^{w_3}
$$

### 3.3 多任务学习损失

$$
L = \sum_k \lambda_k L_k(\theta_{\text{shared}}, \theta_k)
$$

---

## 四、粗排设计

### 4.1 典型方案

| 方案 | 思路 | 优劣 |
|------|------|------|
| 双塔打分 | 和召回类似但特征更多 | 快但交叉不够 |
| **精排蒸馏** | 精排 teacher 训练轻量 student | 效果好，工业主流 |
| 轻量交叉 | 浅层 DCN | 平衡速度和效果 |

### 4.2 蒸馏要点
- Teacher: 精排模型（重特征交叉）
- Student: 轻量双塔或浅层网络
- Loss: KD loss (soft label) + 硬标签 loss 联合
- 核心 gain: 把精排的"排序知识"压缩进粗排

---

## 五、工业精排约束与实践

### 5.1 精排的工业约束

| 约束 | 影响 |
|------|------|
| 延迟 <100ms | 限制模型复杂度 |
| 特征稀疏 | 需要 embedding + 交叉 |
| 样本选择偏差 | 需要纠偏（IPW/DR） |
| 多目标冲突 | 需要 Pareto 优化 |

### 5.2 在线学习

推荐精排的数据分布变化快，需要模型快速更新：
- **全量训练**：每天/每周重训（离线）
- **增量训练**：每小时/每分钟增量更新（近线）
- **实时特征**：毫秒级特征更新（在线）

DeepRec 框架支持分钟级增量更新，10TB+ 模型 serving。

### 5.3 模型校准

精排输出的 pCTR/pCVR 需要校准后才能用于多目标融合：
- **Platt Scaling**：$\hat{p} = \sigma(a \cdot f(x) + b)$
- **温度缩放**：$\hat{p} = \sigma(f(x) / T)$
- **Isotonic Regression**：非参数单调校准

### 5.4 基础设施选型

| 框架 | 核心定位 | 适用场景 |
|------|---------|---------|
| NVIDIA Merlin | 端到端 GPU 加速 | 大规模实时推荐 |
| TorchRec | 分布式 embedding 训练 | 超大规模 embedding 表 |
| DeepRec | TF 生态 + 在线学习 | 阿里系/TF 用户 |
| FuxiCTR | CTR 预测标准化基准 | 模型对标/特征工程优化 |

### 5.5 Kamae: Training-Serving 一致性
- 问题：特征预处理在 Spark（训练）和 Keras（推理）之间不一致
- 方案：在 Keras 内统一所有预处理逻辑，pipeline 导出为模型 bundle

---

## 六、Scaling Law 与模型扩展

### 6.1 推荐系统 Scaling Law (Wukong)

$$
\mathcal{L}(N_e, N_d) = A \cdot N_e^{-\alpha} + B \cdot N_d^{-\beta} + C, \quad \alpha > \beta
$$

**关键发现**：
- Embedding table（稀疏参数）的 scaling 收益**高于** dense DNN
- $\alpha > \beta$：增大 embedding 比增深 MLP 更有效
- 与 LLM 不同：推荐的 scaling 体现在 embedding 维度/数量，而非 DNN 深度

### 6.2 OneTrans — 统一特征交叉与序列建模 (字节)

异构注意力掩码（Heterogeneous Attention Mask）：
- Feature-Feature：全连接（特征交叉）
- Feature-Action：全连接（target-aware 序列聚合）
- Action-Action：因果掩码（序列建模）
- Action-Feature：全连接（上下文感知序列）

参数量 -35%，推理延迟 -20%，AUC +0.38%。

### 6.3 MTGR — 双流融合 (美团)

DLRM 特征交叉流 + HSTU 序列建模流，自适应 gating 融合：
- GAUC 提升 2.88pp，GMV 提升 2.1%
- 渐进式训练：先预训练序列流，再联合训练

---

## 七、面试高频考点

**Q1: 推荐系统为什么要分召回/粗排/精排/重排四级？**
A: 计算量约束。全量百万级物料不可能对每个都跑重模型。级联漏斗逐级缩减候选，每级用更复杂模型处理更少候选，效果与效率平衡。

**Q2: DIN 的 target attention 和标准 attention 区别？**
A: DIN 用候选 item 作 query 对历史行为加权（target-aware），标准 attention 是自注意力。

**Q3: 精排多目标怎么平衡？CTR 和时长冲突怎么办？**
A: MMOE 多 expert 共享+门控选择；PLE 任务特定+共享 expert+渐进萃取。融合分常用：$\text{score} = \text{CTR}^{w_1} \times \text{CVR}^{w_2} \times \text{duration}^{w_3}$，权重人工或自动调。

**Q4: 级联架构 vs 端到端生成式（OneRec-V2），工业中选哪个？**
A: 目前绝大多数公司仍用级联架构（成熟、可解释、各级可独立优化）。端到端生成式是前沿方向（快手 OneRec-V2 已落地），优势是避免级间误差传播，但算力要求高、调试复杂。短期面试答级联为主+了解生成式趋势。

**Q5: 推荐系统的 Scaling Law 和 LLM 有什么区别？**
A: LLM dense 参数是性能主体，计算和参数线性对应。推荐系统 embedding table（稀疏查表）是性能主体，scaling 体现在 embedding 维度/数量而非 DNN 深度，compute-optimal 分配比例完全不同。

**Q6: SORT 如何解决排序 Transformer 的稀疏问题？**
A: 四步：(1) request-centric 组织；(2) local attention 替代 full attention；(3) query pruning 剪低价值查询；(4) generative pre-training。结果：orders +6.35%，latency -44.67%。

**Q7: 粗排蒸馏的核心是什么？**
A: 用精排模型（重特征交叉）作 teacher 训练轻量 student（双塔/浅层），KD loss + 硬标签联合训练。gain 来自把精排的排序知识压缩进粗排。

**Q8: Training-Serving Skew 是什么？如何解决？**
A: 特征预处理在训练（Spark）和推理（Keras）之间不一致导致效果降级。Kamae 方案：在 Keras 内统一所有预处理逻辑，pipeline 导出为模型 bundle，训练推理共用。

---

## 相关概念

- [[concepts/attention_in_recsys|Attention 在搜广推中的演进]]
- [[concepts/multi_objective_optimization|多目标优化]]
- [[concepts/embedding_everywhere|Embedding 技术全景]]
- [[concepts/sequence_modeling_evolution|序列建模演进]]
- [[concepts/generative_recsys|生成式推荐统一视角]]
