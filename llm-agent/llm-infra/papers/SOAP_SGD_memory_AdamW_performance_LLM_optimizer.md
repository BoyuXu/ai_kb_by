# SOAP: SGD-like Memory, AdamW-level Performance (LLM Optimizer)

> 来源：arXiv 2024 | 领域：llm-infra | 学习日期：20260404

## 问题定义

大模型训练的优化器困境：
- **AdamW**：性能优秀，但需要两份额外内存（一阶矩 m + 二阶矩 v），内存占用 = 参数量 × 3
- **SGD/SGD+Momentum**：内存低（参数量 × 1），但训练质量差，收敛慢
- **Adafactor**：压缩二阶矩（行列分解），节省内存但性能下降

能否实现 **SGD 级别内存 + AdamW 级别性能**？

## 核心方法与创新点

**SOAP（Shampoo Optimizer with Adam Preconditioner）**：

1. **Shampoo 基础**：
   - 为矩阵参数 $W \in \mathbb{R}^{m \times n}$ 维护两个小矩阵前置条件器：
   
$$L_t = \frac{1}{t}\sum_{\tau=1}^t G_\tau G_\tau^T \in \mathbb{R}^{m \times m}$$
$$R_t = \frac{1}{t}\sum_{\tau=1}^t G_\tau^T G_\tau \in \mathbb{R}^{n \times n}$$

   - 更新：$W \leftarrow W - \eta L_t^{-1/4} G_t R_t^{-1/4}$

2. **SOAP 关键创新**：
   - 在 Shampoo 的特征基（Eigenbasis）中运行 Adam
   - 将 L 和 R 特征分解：$L = Q_L \Lambda_L Q_L^T$
   - 在旋转坐标系中进行 Adam 更新（等效于自适应二阶优化）

3. **内存效率**：
   - $L \in \mathbb{R}^{m \times m}$，$R \in \mathbb{R}^{n \times n}$ 比全 Hessian $\mathbb{R}^{mn \times mn}$ 小得多
   - 对 Linear Layer（$m, n \ll mn$），内存节省显著

4. **计算优化**：
   - 特征分解周期性执行（每 T 步一次，不是每步）
   - 避免矩阵逆计算瓶颈

## 实验结论

- GPT 训练收敛速度 vs AdamW: 等价甚至略快（**+2-5%** tokens efficiency）
- 内存 vs AdamW: **减少 40-60%**（取决于矩阵形状）
- vs Adafactor: 收敛质量显著更好（PPL 降低 **5-8%**）
- 3B 模型训练：可在 8×A100 上完成 AdamW 需要 16×A100 的任务

## 工程落地要点

- 前置条件器更新频率 T：推荐每 20-50 步（平衡效果和开销）
- 矩阵分解开销：特征分解 $O(n^3)$，大矩阵（MLP）需特别关注
- 混合使用：Embedding 层用 AdamW，Linear 层用 SOAP
- 分布式训练：前置条件器需同步（allreduce）

## 面试考点

1. **Q**: 为什么 AdamW 内存占用是参数量 3 倍？  
   **A**: 参数本身 1 份 + 一阶矩 m（动量）1 份 + 二阶矩 v（自适应学习率）1 份 = 3 份（fp32 训练时实际 4 份，含 fp16 主参数）。

2. **Q**: Shampoo 优化器的核心思想？  
   **A**: 利用矩阵参数的行列结构，用低秩矩阵（L, R）近似全 Hessian，实现比 SGD 更好的曲率信息利用，代价远低于真正的二阶方法（牛顿法）。

3. **Q**: 大模型训练为什么不用二阶优化（牛顿法）？  
   **A**: 参数量 70B，完整 Hessian 矩阵内存需求 $70B^2 \times 4$ bytes = 不可行。SOAP/Shampoo 用矩阵结构近似，是实用二阶方法的最佳折中。
