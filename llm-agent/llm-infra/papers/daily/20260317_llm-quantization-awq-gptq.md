# LLM 量化：AWQ 与 GPTQ

> 来源：工程实践 / MLSys 2024 | 日期：20260317

## 问题定义

LLM 参数量巨大（7B~70B+），FP16 存储需要 14GB~140GB 显存，部署成本高。量化（Quantization）将权重/激活值从 FP16 压缩到 INT8/INT4，可将内存需求降低 2~4x，同时加速推理（计算 INT8 比 FP16 更快）。

核心挑战：
- **权重 outlier**：LLM 权重中存在极大/极小值，naive 量化会引入大误差
- **激活 outlier**：激活值中的异常大值使量化范围很宽，精度损失大

## 核心方法

### GPTQ（GPT Quantization，2022 ETH）

1. **基于 OBQ（Optimal Brain Quantization）框架**
   - 逐层量化，每层优化量化误差
   - 对权重矩阵逐列量化，量化每列后用 Hessian 矩阵补偿其他列的误差

2. **量化流程**
   - 计算各层的 Hessian 矩阵 $H = 2X^TX$（X 是层的输入激活）
   - 按列顺序量化：$\hat{W}_q = \text{round}(W / \delta) \times \delta$
   - 量化误差传播补偿：$W_{i+1:} -= \frac{e_q}{[H^{-1}]_{ii}} \cdot H^{-1}_{i, i+1:}$

3. **特点**
   - 支持 INT4/INT3 超低比特量化
   - 量化后可直接推理，无需微调
   - 适合大模型（30B+），因为逐层优化可并行

### AWQ（Activation-Aware Weight Quantization，2023 MIT）

1. **核心发现**
   - 只有约 1% 的权重通道对应"激活 outlier"（这些通道的激活值异常大）
   - 这 1% 的权重对量化精度影响不成比例地大（Salient Weights）

2. **保护显著权重**
   - 对显著权重通道（Salient Channels）使用更高精度（FP16）或更小量化步长
   - 显著性判断：通过校准集激活值幅度识别 $s = \arg\max |X_i|$

3. **Per-channel 缩放技巧**
   - 量化前对显著权重通道放大 $\alpha$（缩放），量化后除以 $\alpha$
   - 等价于降低该通道的量化误差，无需混合精度，全程 INT4

4. **特点**
   - 量化速度快（无需 Hessian 计算），适合快速部署
   - 无数据集依赖（只需少量校准数据），泛化性好
   - 支持高效 CUDA kernel（AutoAWQ）

## 对比

| 特性 | GPTQ | AWQ |
|------|------|-----|
| 量化精度 | 略高 | 略低 |
| 量化速度 | 慢（逐列优化） | 快（channel scaling） |
| 内存需求 | 需要 Hessian（内存大） | 仅需校准集 |
| 适用规模 | 大模型（30B+） | 各种规模 |
| Kernel 支持 | ExLlama/CUDA | AutoAWQ/CUDA |

## 工程落地要点

1. **校准集准备**：128~512 条有代表性的输入样本，覆盖实际使用场景
2. **量化评估**：在 perplexity（语言建模）+ 下游任务精度双维度评估
3. **Kernel 选择**：AWQ → AutoAWQ；GPTQ → ExLlamaV2 kernel；均支持 vLLM 集成
4. **部署推荐**：边缘设备（内存<16GB）优先 AWQ INT4；服务器优先 GPTQ INT4 或 FP8

## 常见考点

- **Q: INT4 量化 vs INT8 量化的精度权衡？**
  A: INT4 将值映射到 16 个级别（vs INT8 的 256 个），理论精度损失更大。实践中小模型（7B）INT4 量化有明显精度损失，大模型（70B）INT4 量化精度接近原 FP16（更多参数提供更多冗余）。

- **Q: 为什么 LLM 权重中会有 outlier？**
  A: Transformer 的 Layer Norm 和 Attention 机制会在特定维度累积大幅度激活，导致对应的权重通道输入动态范围很宽。这在 GPT-style 模型中普遍存在，越大的模型 outlier 越显著（LLM.int8() 论文的发现）。

- **Q: AWQ 的 per-channel 缩放如何等价于混合精度？**
  A: 对显著通道权重乘以 $s > 1$（放大），量化时用相同步长量化，放大的权重相对量化步长的误差更小（信噪比更高）；推理时激活值除以 $s$ 补偿，等价于对该通道用更小的量化步长，即更高精度，而无需实际使用 FP16。
