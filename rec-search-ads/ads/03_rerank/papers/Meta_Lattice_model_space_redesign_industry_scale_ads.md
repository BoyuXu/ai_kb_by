# Meta Lattice: Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations
> 来源：https://arxiv.org/abs/2512.09200 | 领域：ads | 学习日期：20260329

## 问题定义

Meta 广告推荐系统面临两大核心挑战：

1. **数据碎片化（Data Fragmentation）**：多个产品线（Facebook、Instagram、Reels、Marketplace等）的数据相互隔离，每个产品单独训练模型导致数据利用率低
2. **基础设施成本激增**：在多产品、多目标（Multi-Domain, Multi-Objective, MDMO）场景下，持续扩大模型规模的成本不可持续

## 核心方法与创新点

### Lattice 框架：模型空间重设计

**超越传统 MDMO 的五大核心技术：**

**① 跨域知识共享（Cross-Domain Knowledge Sharing）**
- 统一跨产品线的用户行为建模
- 共享的 User Tower 捕获通用用户偏好，领域特定 Ad Tower 保留业务差异

**② 数据整合（Data Consolidation）**
- 打破产品线数据孤岛，统一数据格式和特征空间
- 联合训练时通过 domain-adaptive 技术避免负迁移

**③ 模型统一（Model Unification）**
- 从 N×M 个独立模型（N产品×M目标）收缩为统一的单一模型架构
- 使用动态路由/门控机制在推理时区分不同产品/目标的需求

**④ 知识蒸馏（Distillation）**
- 统一大模型 → 轻量级服务模型
- 保留效果的同时大幅降低推理成本

$$
\mathcal{L}_{Lattice} = \mathcal{L}_{task} + \lambda_{distill} \mathcal{L}_{KD} + \lambda_{domain} \mathcal{L}_{domain\_adapt}
$$

**⑤ 系统优化（System Optimizations）**
- 统一模型减少存储冗余（不再维护 N×M 套独立参数）
- 推理路径简化，降低服务延迟

### 模型空间重设计示意

```
传统 MDMO:              Lattice:
FB-CTR-Model            ┌─────────────────────┐
FB-CVR-Model            │   Unified Lattice   │
IG-CTR-Model    →       │      Model          │
IG-CVR-Model            │  (一个模型覆盖全部)   │
Reels-CTR-Model         └─────────────────────┘
...                      ↓ domain/task routing
```

## 实验结论

Meta 生产环境部署结果：

| 业务指标 | 提升幅度 |
|---------|---------|
| 收入驱动顶线指标 | **+10%** |
| 用户满意度 | **+11.5%** |
| 转化率 (CVR) | **+6%** |
| 算力节省 | **-20%** |

发表于 KDD 2026。这是 Meta 生产环境的真实 A/B 测试结果，规模极大。

## 工程落地要点

1. **负迁移防范**：跨域数据整合时，需监控弱域被强域主导的问题，使用 domain-adaptive loss weighting
2. **模型统一的路由设计**：推理时需要轻量级 domain/task 路由机制（如 soft routing 或 hard gating）
3. **蒸馏目标设计**：统一大模型的 soft label 应该保留 domain-specific 的信息，而非简单平均
4. **AB测试复杂度**：统一模型的 AB 测试需要同时验证各产品线效果，设计分层实验方案
5. **冷启动域处理**：新产品线加入时，利用统一模型做 warm-start，避免数据稀疏

## 面试考点

**Q1: Lattice 的"模型空间重设计"相比传统 Multi-Task Learning 有什么本质区别？**
A: 传统 MTL 在单个产品内联合优化多个任务；Lattice 是跨产品（Multi-Domain）+跨目标（Multi-Objective）的全局优化，同时通过数据整合和模型统一解决了从 N×M 个模型到 1 个模型的范式转变，并配套系统优化实现可部署性。

**Q2: 跨产品线数据整合时，如何避免强域（如Facebook主feed）压制弱域（如Marketplace）？**
A: ①Domain-adaptive 样本权重：弱域样本给予更高权重 ②Domain-specific head：共享底层表示，但每个域有独立的上层网络 ③Gradient surgery：当梯度方向冲突时做梯度投影，只保留对弱域有利的方向 ④Domain temperature：控制各域在联合训练中的"学习速率"

**Q3: Lattice 如何实现"算力节省20%"？技术机制是什么？**
A: ①从 N×M 个独立模型到 1 个统一模型，减少重复的 Embedding 存储（特征 embedding 共享）②统一的推理路径，减少不同模型之间的数据搬运 ③通过蒸馏压缩模型尺寸，小模型服务 ④共享 KV Cache（统一模型在服务时可复用历史计算）

**Q4: 在 Meta 这样的超大规模系统中，如何保证统一模型的推理延迟满足 SLA？**
A: ①轻量级路由（domain/task 路由使用浅层网络，延迟<1ms）②分层服务（重型统一模型负责粗排，轻型蒸馏模型负责精排/重排）③硬件亲和性优化（统一模型便于做整体的 TensorRT/量化优化）④动态计算图（只激活当前请求相关的子图）

**Q5: Lattice 框架的11.5%用户满意度提升来自哪里？**
A: 核心机制：跨域数据整合使模型看到更完整的用户行为图谱（在 Instagram 上的行为可以辅助 Facebook feed 推荐），从而更准确理解用户真实兴趣，减少推荐噪声，提升整体满意度。同时统一的 MDMO 优化避免了不同产品线在争夺同一用户注意力时的内部竞争。
