# ABBA: Highly Expressive Hadamard Product Adaptation

> 来源：arXiv 2025 | 领域：llm-infra | 学习日期：20260404

## 问题定义

LLM 参数高效微调（PEFT）领域，LoRA 是最流行方法：
$$W' = W + AB, \quad A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times d}$$

LoRA 的局限：低秩分解表达能力有限，对复杂任务适应不足。

**核心问题**：如何在保持低参数量的同时，提升 PEFT 的表达能力？

## 核心方法与创新点

**ABBA（Hadamard Product Adaptation）**：

1. **Hadamard 积而非矩阵乘积**：
   - LoRA：$\Delta W = AB$（矩阵乘，低秩）
   - ABBA：$\Delta W = A \odot B$（Hadamard 积，逐元素乘）
   
   Hadamard 积的秩：$\text{rank}(A \odot B) \leq \text{rank}(A) \times \text{rank}(B)$
   
   即使 A 和 B 都是 rank-r，$A \odot B$ 的秩可达 $r^2$！

2. **ABBA 结构**：
   - $A, B \in \mathbb{R}^{d \times r}$（与 LoRA 相同参数量）
   - 但通过 Hadamard 积实现 $r^2$ 秩的更新
   
$$\Delta W = \text{reshape}(A \odot B) \in \mathbb{R}^{d \times d}$$

   参数数量 = $2dr$（同 LoRA），但表达力 ≈ LoRA 的 $r$ 倍

3. **可组合性**：
   - 多个 ABBA 模块串联：$\Delta W = (A_1 \odot B_1) + (A_2 \odot B_2)$
   - 进一步提升表达能力

4. **初始化策略**：
   - A：正态初始化，B：零初始化（保证训练初期 $\Delta W = 0$）

## 实验结论

- GLUE Benchmark vs LoRA（同参数量）: **+1.8 avg score**
- Code Generation（HumanEval）: **+4.3%** Pass@1
- 数学推理（GSM8K）: **+3.7%** accuracy
- 参数量与 LoRA 完全相同，仅更换计算方式

## 工程落地要点

- 替换 LoRA：只需修改 $\Delta W$ 计算方式（`einsum` → Hadamard）
- 梯度稳定性：Hadamard 积梯度比矩阵乘更稳定（无奇异值问题）
- 与 QLoRA 兼容（基础模型量化 + ABBA 适配层）
- 推理时可合并：$W' = W + \text{reshape}(A \odot B)$，无额外延迟

## 面试考点

1. **Q**: Hadamard 积为什么比矩阵乘有更高秩？  
   **A**: 矩阵乘 AB 的秩 ≤ min(rank(A), rank(B)) ≤ r；Hadamard 积 $A \odot B$ 的秩上界为 rank(A)×rank(B) = $r^2$，相同参数量实现更高秩的更新。

2. **Q**: LoRA 的秩 r 如何选择？  
   **A**: 通常 r=4,8,16,32。r 小：参数少，过拟合风险低，但表达能力有限；r 大：表达力强，参数多，可能过拟合。ABBA 等效于更大 r 而不增加参数。

3. **Q**: PEFT 方法（LoRA/Adapter/Prefix Tuning）如何选择？  
   **A**: LoRA/ABBA：大多数任务首选，推理无额外延迟；Adapter：适合多任务切换（可拔插）；Prefix Tuning：输入侧适配，适合对话场景；P-Tuning：适合分类任务。
