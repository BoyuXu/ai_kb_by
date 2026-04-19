# Predicting LLM Output Length via Entropy-Guided Representations (PLP)

> 来源：https://arxiv.org/abs/2602.11812 | 领域：llm-infra | 学习日期：20260420 | ICLR 2026

## 问题定义

LLM 推理中的长尾输出长度分布导致 batched inference 的严重计算浪费（padding overhead）。如果能预测输出长度，就能更高效地做 batch 调度。

**核心挑战**：
1. 输出长度是 prompt 和采样策略共同决定的 —— 同一 prompt 不同采样路径长度差异大
2. 静态预测（只看 prompt）无法处理"一对多"的随机场景

## 核心方法与创新点

### 1. EGTP (Entropy-Guided Token Pooling) — 静态预测

复用模型自身的激活表示（而非训练额外的预测模型），通过 token 级别的熵引导池化：

$$
\text{weight}_i = \frac{H(p_i)}{\sum_j H(p_j)}, \quad \hat{L} = f\left(\sum_i \text{weight}_i \cdot h_i\right)
$$

- 高熵 token（模型不确定的位置）携带更多关于输出长度的信息
- 零额外模型开销，直接复用前向传播的中间结果

### 2. PLP (Progressive Length Prediction) — 动态预测

自回归过程中逐步更新剩余长度预测：

$$
\hat{L}_t = g(h_t, \hat{L}_{t-1}), \quad t = 1, 2, ..., T
$$

每生成一个 token，根据当前激活更新对剩余长度的估计。适用于采样（非确定性）场景。

### 3. ForeLen Benchmark

构建专用基准测试，覆盖不同模型、不同采样策略、不同 prompt 类型。

## 广告系统的潜在应用

1. **LLM 广告生成的延迟控制**：预测广告文案/创意的生成长度，提前分配计算资源
2. **对话式广告的 SLA 管理**：在 LLM 回答中插入广告时，预测回答长度以确定最佳插入点
3. **LLM serving 效率优化**：batch scheduling 基于预测长度分组，减少 padding 浪费

## 核心 Insight

1. **熵是长度的信号** —— 高熵 token 意味着模型面临分叉点（多种后续路径），这些分叉点的密度和模式决定了最终输出长度
2. **Progressive > Static** —— 静态预测只能给均值估计，动态预测能跟踪采样路径的实际走向
3. **LLM Serving 的下一个优化维度** —— 从"如何更快推理"到"如何更精准调度"

## 面试考点

- Q: 为什么不用一个独立小模型预测长度？
  > 独立模型需要额外训练数据和推理开销，且无法泛化到新 prompt 分布。EGTP/PLP 复用 LLM 自身的激活，零额外成本且天然泛化。
- Q: Progressive Length Prediction 如何影响 batch scheduling？
  > 当预测剩余长度短的请求可以提前释放 GPU 资源给等待中的请求，实现更细粒度的动态 batching（类似 continuous batching 的增强版）。

---

## 相关链接

- [[ad_insertion_llm_generated_responses]] — LLM 回答中插入广告（长度预测可用于确定插入点）
