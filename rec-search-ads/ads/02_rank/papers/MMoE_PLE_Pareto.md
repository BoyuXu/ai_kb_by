# 多目标广告排序：MMoE、PLE 与 Pareto 优化

> 来源：技术综述 | 日期：20260316 | 领域：ads

## 问题定义

广告排序需要同时优化多个互相冲突的目标：
- 广告主目标：CTR（点击）、CVR（转化）、ROAS（广告回报率）
- 平台目标：收入（RPM）、用户体验（不干扰）、生态健康（多样性）

**挑战**：
1. **跷跷板效应**：提升 CTR 往往损害 CVR（标题党问题）。
2. **负迁移**：多任务共享参数时，不相关任务互相干扰，反而比单任务差。
3. **目标权重设定**：不同目标的权重如何确定？

## 核心方法与创新点

### 1. Hard Parameter Sharing（硬共享）

底层共享 embedding + MLP，上层各任务独立 head：
```
shared_representation = MLP(input_features)
ctr_pred = CTR_head(shared_representation)
cvr_pred = CVR_head(shared_representation)
```
简单高效，但负迁移风险高。

### 2. MMoE（Multi-gate Mixture of Experts）

Google 2018 提出，用多个专家网络 + 门控机制：
```
# K 个 Expert Network，每个任务有独立 Gate
expert_outputs = [Expert_k(input) for k in range(K)]
gate_k = softmax(Gate_k(input))  # 任务 k 的门控权重
task_input_k = Σ gate_k[i] × expert_outputs[i]
output_k = Tower_k(task_input_k)
```
不同任务通过 gate 选择不同专家，缓解负迁移。

### 3. PLE（Progressive Layered Extraction）

腾讯 2020 提出，将专家分为 **任务特定专家** 和 **共享专家**：
```
# 第 l 层：
task_specific_experts[k] = [Expert_k_i for i in range(m)]
shared_experts = [Expert_shared_j for j in range(n)]

# 任务 k 的门控
gate_k = concat(task_specific_experts[k], shared_experts)
extraction_k = attention(gate_k)  # 选择性融合
```
解决 MMoE 中共享专家被所有任务"争抢"导致专家退化的问题。

### 4. Pareto 优化（多目标权衡）

- **线性加权**：score = Σ w_i × objective_i，权重需人工调优。
- **Constrained Optimization**：maximize CTR s.t. CVR > threshold，用 Lagrangian 松弛。
- **帕累托前沿探索**：MGDA（Multiple Gradient Descent）自动找帕累托最优方向。
- **MOO-MTL（Multi-Objective Multi-Task Learning）**：学习每个目标的 Pareto 权重，无需手动设定。

## 实验结论

- PLE 在腾讯广告上相比 MMoE：主任务 AUC +0.1-0.3%，辅助任务也有提升（负迁移降低）。
- 帕累托优化在工业实验中：CTR/CVR 同时提升，无需人工权衡，上线效率提升 30%（减少调参时间）。

## 工程落地要点

- Expert 数量：通常 K=4-8，太多则参数膨胀，训练慢；太少则表达能力不足。
- Gate 初始化：用均匀初始化（1/K），防止训练初期某个专家主导。
- 线上推理：所有任务共用一次前向传播，额外计算很少（仅多几个 head）。
- 任务权重调优：建议用 **Uncertainty Weighting**（Kendall et al. 2018），自动学习各任务损失权重，减少手工调参。

## 常见考点

- Q: 什么是负迁移（Negative Transfer）？在多任务学习中如何检测？
  A: 负迁移指多任务联合训练后，某任务性能反而比单任务训练更差。检测方法：将多任务模型与单任务基线对比，如果主任务 AUC 下降则发生负迁移。MMoE/PLE 通过专家隔离缓解负迁移。

- Q: 广告排序中 CTR × CVR 的乘法怎么理解？
  A: 广告的 eCPM（有效千次展示收益）= bid × pCTR × pCVR（对于 CPA 广告）。这是因为广告主按转化付费：每次展示的期望收入 = 转化bid × P(click) × P(convert|click)。排序时最大化 eCPM 即同时考虑了 CTR 和 CVR。

- Q: 如何处理多目标排序中的"探索-利用"权衡？
  A: 广告排序中也有探索需求（避免系统锁定在当前最优而错过更好选项）。常用方法：(1) ε-greedy（小概率随机探索）；(2) Thompson Sampling（基于 uncertainty 探索）；(3) 在排序分中加入 exploration bonus（如 UCB）。
