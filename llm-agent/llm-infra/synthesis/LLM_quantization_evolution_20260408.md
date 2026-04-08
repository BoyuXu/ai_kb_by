# LLM 量化技术演进综合分析

> 综合日期：2026-04-08 | 涵盖论文：KVQuant, PolarQuant, MF-QAT, BitNet

## 技术演进路线

### 第一阶段：权重量化（Weight Quantization）
- **目标**：减少模型存储和推理内存
- **代表**：PolarQuant, BitNet
- PolarQuant: Block Normalization → Hadamard Rotation → Gaussian-Matched Quantization
- BitNet: 极端路线，权重直接约束到 {-1, 0, +1}

### 第二阶段：KV Cache 量化（Activation/Cache Quantization）
- **目标**：解决长上下文推理的内存瓶颈
- **代表**：KVQuant
- 关键洞察：RoPE 位置编码会扭曲 Key 分布，Pre-RoPE 量化效果更好
- KV Cache 内存公式：$\text{Mem}_{KV} = 2 \times L \times H \times d \times N \times \text{bits}/8$

### 第三阶段：弹性量化（Elastic Quantization）
- **目标**：一次训练，多平台部署
- **代表**：MF-QAT
- Slice-and-Scale 实现格式间无缝转换
- 支持 FP16 → INT4 的弹性伸缩

## 核心公式对比

| 方法 | 量化对象 | 精度 | 核心公式/操作 |
|------|---------|------|--------------|
| PolarQuant | 权重 | 4-bit | $W' = Q(\text{Hadamard}(\text{Norm}(W)))$ |
| KVQuant | KV Cache | 3-bit | Per-channel, Pre-RoPE quantization |
| MF-QAT | 权重+激活 | 多格式 | Slice-and-Scale 格式转换 |
| BitNet | 权重 | 1.58-bit | $W \in \{-1, 0, +1\}$ 三值约束 |

## 工业实践指南

1. **推理服务**：KVQuant 用于长上下文场景，可与 FlashAttention 组合
2. **模型压缩**：PolarQuant 用于 4-bit 部署，Hadamard 旋转是关键
3. **跨平台部署**：MF-QAT 用于异构硬件环境
4. **极致压缩**：BitNet 用于 CPU-only 或边缘设备

## 面试考点

1. **量化误差分析**：为什么 Hadamard 旋转能减少量化误差？（均匀化分布，减少 outlier）
2. **KV Cache 量化**：为什么 Key 比 Value 更难量化？（RoPE 引入的分布偏斜）
3. **弹性量化**：如何在不重新训练的情况下切换量化格式？（Slice-and-Scale）
4. **极端量化**：1-bit 模型为什么还能工作？（冗余参数、二值化理论）
5. **系统协同**：量化与 FlashAttention、PagedAttention 等的兼容性
