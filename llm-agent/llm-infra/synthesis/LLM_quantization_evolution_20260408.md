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

### 第四阶段：自动化混合精度压缩框架（Automated Mixed-Precision Compression）
- **目标**：一行代码完成模型压缩，自动适配硬件约束
- **代表**：OneComp (Fujitsu, 2026)
- 论文：OneComp: One-Line Revolution for Generative AI Model Compression (arxiv 2603.28845)

**核心方法**：给定模型标识符和硬件 VRAM 约束，OneComp 自动完成：模型检测 → 混合精度分配 → 渐进量化执行（layer-wise → block-wise → global refinement）

**关键组件 — AutoBit**：
- 基于 ILP（整数线性规划）的 per-layer bitwidth 分配
- 自动从用户指定的 VRAM 约束推导目标平均比特宽度
- 利用层间量化敏感度差异：早期层和 attention output projection 更脆弱（高精度），中间 MLP 层更耐压缩（低精度）
- 目标函数：min 全局量化误差 s.t. 总内存 <= VRAM budget

**渐进量化流水线**：
1. Layer-wise PTQ：逐层压缩单个 Linear 层
2. Block-wise PTQ：以 Transformer Block 为单位联合优化
3. Global PTQ：全模型联合校准微调
- 每个阶段产出都是可部署的 checkpoint（pivot 设计），后续阶段只会更好

**面试考点 — OneComp/AutoBit**：
- 为什么 uniform bitwidth 不是最优？答：Transformer 各层量化敏感度差异大（early layers 和 attn_o_proj 脆弱，mid MLP 层耐压），uniform 分配会在敏感层过度压缩
- 混合精度分配为什么用 ILP？答：离散优化问题（每层比特宽度是整数），ILP 能在内存约束下找到全局最优分配
- 渐进量化 vs 一步到位的优势？答：每个阶段都有可部署 checkpoint，可根据质量-压缩 tradeoff 随时停止；后续阶段在前阶段基础上 refine，收敛更稳定

## 核心公式对比（更新）

| 方法 | 量化对象 | 精度 | 核心公式/操作 |
|------|---------|------|--------------|
| PolarQuant | 权重 | 4-bit | $W' = Q(\text{Hadamard}(\text{Norm}(W)))$ |
| KVQuant | KV Cache | 3-bit | Per-channel, Pre-RoPE quantization |
| MF-QAT | 权重+激活 | 多格式 | Slice-and-Scale 格式转换 |
| BitNet | 权重 | 1.58-bit | $W \in \{-1, 0, +1\}$ 三值约束 |
| **OneComp/AutoBit** | **权重（混合精度）** | **自适应 2-8bit** | **ILP: min $\sum_l \epsilon_l(b_l)$ s.t. $\sum_l m_l(b_l) \leq \text{VRAM}$** |

## 工业实践指南（更新）

1. **推理服务**：KVQuant 用于长上下文场景，可与 FlashAttention 组合
2. **模型压缩**：PolarQuant 用于 4-bit 部署，Hadamard 旋转是关键
3. **跨平台部署**：MF-QAT 用于异构硬件环境
4. **极致压缩**：BitNet 用于 CPU-only 或边缘设备
5. **一键压缩部署**：OneComp 用于快速适配目标硬件 VRAM，自动混合精度，开发效率最高

## 面试考点

1. **量化误差分析**：为什么 Hadamard 旋转能减少量化误差？（均匀化分布，减少 outlier）
2. **KV Cache 量化**：为什么 Key 比 Value 更难量化？（RoPE 引入的分布偏斜）
3. **弹性量化**：如何在不重新训练的情况下切换量化格式？（Slice-and-Scale）
4. **极端量化**：1-bit 模型为什么还能工作？（冗余参数、二值化理论）
5. **系统协同**：量化与 FlashAttention、PagedAttention 等的兼容性
6. **混合精度分配**：为什么不同层应该用不同 bitwidth？如何自动确定？（AutoBit/ILP，层敏感度差异）
7. **渐进量化**：OneComp 的三阶段流水线设计有什么优势？（每阶段可部署、单调改善、可提前停止）
