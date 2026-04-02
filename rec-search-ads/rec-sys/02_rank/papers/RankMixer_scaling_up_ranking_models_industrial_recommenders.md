# RankMixer: Scaling Up Ranking Models in Industrial Recommenders

> 来源：https://arxiv.org/abs/2507.15551 | 领域：rec-sys | 学习日期：20260329

## 问题定义

工业推荐系统精排模型扩大规模面临两大实际障碍：
1. **延迟与成本约束**：工业推荐必须满足严苛的延迟上限（ms 级）和高 QPS 需求，无法简单堆叠参数量
2. **GPU 利用率低（MFU 低）**：现有精排模型的特征交叉模块（FM、CIN 等）继承自 CPU 时代，无法高效利用现代 GPU，MFU 仅约 4.5%，导致扩展性差

**核心矛盾**：想要像 LLM 一样 scaling，但推荐模型的架构不适配 GPU 并行计算。

## 核心方法与创新点

### RankMixer：硬件感知的统一特征交互架构

**三大技术创新**：

**1. Multi-Head Token Mixing 替代 Self-Attention**
- 保留 Transformer 的高并行性
- 用 token mixing（MLP-Mixer 风格）替代 O(n²) 自注意力
- 兼顾效率与特征交互表达能力
- MFU 从 4.5% 大幅提升至 **45%**（10× 提升）

**2. Per-token FFN（逐特征前馈网络）**
- 同时维护特征内部子空间建模（intra-feature）和跨特征空间交互（inter-feature）
- 不同 token（特征）使用独立 FFN 参数，提升特征差异化建模能力

**3. Sparse-MoE 变体扩展到十亿参数**
- 通过混合专家机制（MoE）进行高效参数扩展
- **动态路由策略**：解决专家训练不均衡问题
- 在维持推理延迟基本不变的前提下，参数量扩大 100×

## 实验结论

**大规模生产数据集**（万亿级样本量）：
- MFU：从 4.5% → **45%**（提升 10×）
- 参数规模：扩大 **100×**，推理延迟基本持平

**在线 A/B 测试**（推荐 + 广告两大业务场景）：
- **1B Dense-Parameters RankMixer** 全流量上线：
  - 用户活跃天数（User Active Days）提升 **+0.3%**
  - 应用内总使用时长（Total In-App Duration）提升 **+1.08%**
- 在服务成本不增加的前提下实现上述增益

## 工程落地要点

1. **替换低 MFU 模块**：直接用 RankMixer 替换 FM/DCN/CIN 等传统特征交叉组件，无需改变整体架构
2. **MFU 监控**：将 MFU 作为核心工程指标，指导模型架构选型
3. **MoE 路由平衡**：需关注专家负载均衡，引入辅助损失（auxiliary loss）防止专家退化
4. **延迟 profiling**：1B 参数模型上线前需做全链路延迟压测，确保 P99 延迟达标
5. **渐进式扩规模**：先验证小参数版本线上效果，再逐步扩展到 1B 规模
6. **多场景验证**：同一架构在推荐和广告两个场景均有正收益，说明通用性强

## 常见考点

**Q1：为什么传统推荐排序模型（FM/DIN 等）在 GPU 上 MFU 很低？**
> A：FM、DIN、CIN 等特征交叉模块在设计时针对 CPU 计算模式，大量使用循环、稀疏索引、不规则内存访问，这些操作在 GPU 上并行度低。计算量与显存带宽不匹配，GPU 算力大量空置，导致 MFU 仅约 4.5%。

**Q2：RankMixer 用 Token Mixing 替代 Self-Attention 的权衡是什么？**
> A：Self-Attention 是 O(n²) 复杂度，对推荐特征（通常几百个）实际运行快，但 GPU 利用率不优。Token Mixing 用矩阵乘法形式的通道混合，保持高 GPU 并行度且复杂度可控，实现 MFU 10× 提升。Trade-off 是可能损失部分 attention 的动态权重表达，但实验证明综合效果更好。

**Q3：Sparse-MoE 如何在不增加延迟的情况下扩展 100× 参数？**
> A：MoE 的每个 token 只激活少数专家（Top-K routing），所以 FLOP 与 Dense 模型相比没有线性增长，推理时只计算被激活的专家子网络。参数量增大主要影响显存，不影响计算量。动态路由负载均衡确保专家均匀激活，避免某些专家始终空置。

**Q4：MFU（Model Flops Utilization）如何计算？有什么实际意义？**
> A：MFU = 实际有效 FLOP / 理论峰值 FLOP。推荐排序场景实际 MFU 只有 4.5% 意味着 95.5% 的 GPU 算力被浪费在内存等待、调度开销等。提升 MFU 可以在相同硬件成本下运行更大模型，是推荐系统工业化的关键工程指标。

**Q5：Per-token FFN 与共享 FFN 的区别？为什么 Per-token FFN 更适合推荐？**
> A：共享 FFN 对所有特征使用同一权重，特征间处理方式无差异；Per-token FFN 为每个特征（token）单独维护 FFN 参数，允许模型对不同类型特征（用户行为序列、物品属性、上下文等）学习差异化表示。推荐场景中特征异构性强，Per-token FFN 能更好保留各特征的独立语义。
