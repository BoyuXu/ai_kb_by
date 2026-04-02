# EAGLE-3: Scaling up Inference Acceleration via Training-Time Test
> 来源：arXiv:2503.01840 | 领域：llm-infra | 学习日期：20260330

## 问题定义
Speculative Decoding（推测解码）通过草稿模型生成候选 token、目标模型验证的方式加速 LLM 推理，但草稿模型的接受率（acceptance rate）是瓶颈。EAGLE-3 提出在训练时融入"测试时行为"，大幅提升草稿模型与目标模型的一致性，使接受率和加速比达到新 SOTA。

## 核心方法与创新点
1. **EAGLE 系列演进**：EAGLE → EAGLE-2（动态长度草稿）→ EAGLE-3（训练时测试融合）。核心：草稿模型用目标模型的 feature（不是 logit），学习"预测目标模型的下一个状态"。
2. **Training-Time Test（TTT）**：训练时模拟推测解码的"接受-拒绝"过程，将被拒绝的草稿 token 也纳入训练信号，使草稿模型学习"如何通过验证"。
3. **Feature-level Speculation**：草稿模型不直接预测 token，而是预测目标模型隐层 feature，再通过 LM head 映射到 token。减少 vocab 空间的误差，接受率更高。
4. **Tree-based Drafting**：生成树状候选（多个分支），目标模型并行验证所有分支，选最长合法前缀，进一步提升平均接受 token 数。
5. **动态预算分配**：根据当前上下文难度动态调整草稿长度（简单 context → 长草稿；困难 context → 短草稿），避免无效草稿。

## 实验结论
- Llama-3.1-70B 推理：加速比 3.8x（对比自回归解码），EAGLE-2 为 2.9x
- 接受率：EAGLE-3 平均 token 接受率 0.82（EAGLE-2 为 0.71）
- 质量损失：生成文本与原始模型完全一致（Speculative Decoding 保证 exact match）

## 工程落地要点
- 草稿模型规模选择：目标模型 1/10 大小（70B → 7B draft）接受率和效率最优
- Tree-based verification 需要 batch attention（多分支并行），需支持 attention mask 变体
- 动态草稿长度需要运行时决策逻辑，建议用简单启发式（历史接受率移动平均）
- 与 KV Cache 优化兼容：草稿模型和目标模型共享 prefix KV Cache

## 常见考点
- Q: Speculative Decoding 的原理和保证？
  - A: 草稿模型快速生成 k 个 token，目标模型并行验证（一次 forward 处理 k 个 token），用 rejection sampling 保证输出分布与目标模型完全一致（无近似误差）
- Q: 什么因素影响 Speculative Decoding 的加速比？
  - A: ① 接受率 α（越高加速越大）；② 草稿模型速度（越快越好）；③ 验证批次大小（目标模型并行验证 k token vs 串行）；④ 硬件利用率（是否 compute-bound）
- Q: EAGLE 和 Medusa 的区别？
  - A: Medusa：在目标模型上加多头解码（每头预测不同位置 token）；EAGLE：独立草稿模型基于 target feature 预测，接受率更高、灵活性更好

## 数学公式

$$
\text{Speedup} = \frac{1 + \alpha + \alpha^2 + ... + \alpha^k}{1 + c} \approx \frac{1}{1-\alpha} \cdot \frac{1}{1+c}
$$

$\alpha$：token 接受率，$c$：草稿模型相对计算成本，$k$：草稿长度
