# 投机解码自适应化、KV Cache 量化与端侧高效 LLM (2025-2026)

> 覆盖论文：SpecKV (2605.02888), EfficientLLM (2505.13840), Adaptive KV Quantization (2604.04722), SmallThinker (2507.20984), QuantSpec (2502.10424)
> 交叉引用：[[20260504_speculative_decoding_longcontext_quant.md]]、[[20260504_kv_cache_and_speculative_serving.md]]、[[20260503_kv_cache_frontier_2026.md]]、[[20260419_KV_cache_quantization_adaptive_methods.md]]、[[MoE架构设计与推理优化.md]]、[[LLM推理效率三角.md]]

---

## 1. 技术演进总览

5 篇论文覆盖 LLM 推理效率的三大前沿方向：

```
LLM 推理效率前沿 (2025-2026)
|
|-- 投机解码自适应化
|   |-- SpecKV: 自适应 gamma 选择，MLP 控制器 +56% 吞吐
|   |-- QuantSpec: 自投机解码 + 分层量化 KV Cache，2.5x 加速
|
|-- KV Cache 量化精细化
|   |-- Adaptive KV Quantization: 类 Huffman 变长位宽分配，按 token 重要性
|   |-- QuantSpec (双重角色): 分层量化 + bit-sharing 消除 draft 额外内存
|
|-- 端侧高效 LLM 原生设计
|   |-- SmallThinker: 双层稀疏 MoE + 预注意力路由 + NoPE-RoPE 混合注意力
|   |-- EfficientLLM: 100+ 模型-技术对效率全景评估基准
```

核心趋势：
1. **投机解码从固定到自适应**：固定 $\gamma=4$ 是次优的，SpecKV 用轻量 MLP 实现逐步自适应选择，几乎零开销
2. **KV Cache 量化从静态到逐 token 动态**：不再一刀切 INT4/INT8，而是按 token 重要性分配 2/4/8/FP16 位宽
3. **端侧 LLM 需要原生架构设计**：SmallThinker 证明"先训大模型再压缩"不如"从头为端侧约束设计"
4. **效率无银弹**：EfficientLLM 揭示没有单一方法在所有维度最优，选型必须结合任务和规模

---

## 2. 逐篇精读

### 2.1 SpecKV: 自适应投机解码 (2605.02888)

**Problem**: 投机解码的推测长度 $\gamma$ 通常固定为 4，但最优值随任务类型和模型压缩级别变化。固定 $\gamma$ 导致：小了浪费并行验证能力，大了接受率低白做功。

**Method**:
- 系统性 profiling：4 类任务 $\times$ 4 种 $\gamma$ $\times$ 3 种压缩级别（FP16/INT8/NF4），收集 5,112 条步级记录
- 提取 draft 模型信号：per-step acceptance rate、draft entropy、draft confidence
- 轻量 MLP 控制器，每步根据 draft 信号选择最优 $\gamma$，最大化 expected tokens per step

**Innovation**:
- 首次将 $\gamma$ 选择建模为可学习的决策问题（而非启发式规则）
- 压缩感知（compression-aware）：不同量化级别下最优 $\gamma$ 不同，控制器自动适配
- 极低开销：0.34 ms/decision，< 0.5% 步时间

**Results**:
- 比固定 $\gamma=4$ 基线提升 **56.0%** expected tokens/step
- $p < 0.001$（paired bootstrap test）
- 全部数据、模型、notebook 开源

**Keywords**: adaptive speculative decoding, gamma selection, compression-aware, draft signal

---

### 2.2 EfficientLLM: 效率评估全景 (2505.13840)

**Problem**: LLM 效率技术（架构/微调/量化）缺乏统一基准，各论文实验设置不可比。

**Method**:
- 48x GH200 + 8x H200 生产级集群评估
- 三个轴：
  1. **Architecture**: MQA / GQA / MLA / NSA / MoE
  2. **Fine-tuning**: LoRA / RSLoRA / DoRA
  3. **Quantization**: INT4 / FP16
- 六个细粒度指标：Memory Utilization / Compute Utilization / Latency / Throughput / Energy / Compression Rate
- 100+ 模型-技术对，0.5B-72B 参数

**Innovation**:
- 首个同时覆盖架构 + 微调 + 量化的统一效率基准
- 揭示三条核心洞察而非给出单一最优解
- 扩展至视觉模型和 VLM 验证泛化性

**Results / Key Findings**:

| 技术 | 优势 | 代价 | 适用场景 |
|------|------|------|----------|
| MoE | FLOPs 降低，精度提升 | VRAM 增加 40% | 算力充足、内存富裕 |
| INT4 量化 | 内存/能耗降 3.9x | 精度降 3-5% | 资源受限部署 |
| MQA | 最优内存-延迟权衡 | Head 数受限 | 端侧设备 |
| MLA | 最低困惑度 | 延迟略高 | 质量优先 |
| RSLoRA | 效率超 LoRA | 仅 > 14B 有效 | 大模型微调 |

**Keywords**: LLM efficiency benchmark, MQA/GQA/MLA comparison, quantization trade-offs

---

### 2.3 Adaptive KV-Cache Quantization: 端侧自适应 (2604.04722)

**Problem**: KV Cache 内存随上下文线性增长，是端侧推理的主要瓶颈。静态量化（全 INT4 或全 INT8）对所有 token 一视同仁，浪费重要 token 的精度。

**Method**:
- 类 Huffman 编码思路：高频/重要 token 分配更多比特，低重要性 token 极致压缩
- Token 级特征提取：token frequency、quality score、attention variance、entropy-based uncertainty
- 轻量数据驱动控制器，从 {2-bit, 4-bit, 8-bit, FP16} 中动态选择每个 token 的 KV 精度

**Innovation**:
- 首次将 Huffman 编码的变长分配原则应用到 KV Cache 量化
- 逐 token 粒度（而非逐 layer 或逐 channel），精度分配最细化
- 控制器足够轻量，适合端侧部署

**Results**: SmolLM-360M on HellaSwag:
- 解码延迟降低 **17.75%**（vs 静态量化）
- 精度提升 **7.60 points**
- 仅比 FP16 低 **0.30 points**

**Keywords**: adaptive KV quantization, Huffman-inspired bit allocation, on-device LLM, token importance

---

### 2.4 SmallThinker: 端侧原生高效 LLM (2507.20984)

**Problem**: 现有端侧 LLM 方案多为"训大模型 → 蒸馏/量化 → 压缩上端侧"，压缩后性能损失大。端侧三大约束：弱算力、有限内存、慢存储。

**Method**:
- **双层稀疏结构**：
  1. Fine-grained MoE：稀疏激活专家
  2. Sparse FFN：进一步稀疏化前馈网络
- **Pre-attention Router**：注意力计算前即决定专家路由，推理引擎可以在计算 attention 的同时预取专家参数，隐藏存储延迟
- **NoPE-RoPE 混合稀疏注意力**：减少 KV Cache 需求
- 两个变体：4B-A0.6B（激活 0.6B）和 21B-A3B（激活 3B）

**Innovation**:
- 首个"原生端侧设计"的 LLM 系列（非压缩适配）
- Pre-attention Router 是关键工程创新：将存储 I/O 与 attention 计算重叠
- 双层稀疏 = MoE 稀疏 + FFN 稀疏，极致压缩计算量

**Results**:
- Q4_0 量化下，两模型均 > **20 tokens/s on CPU**
- 内存占用：4B-A0.6B 仅 **1GB**，21B-A3B 仅 **8GB**
- 性能超越同规模甚至更大 LLM

**Keywords**: on-device LLM, dual-layer sparsity, pre-attention routing, MoE-FFN hybrid

---

### 2.5 QuantSpec: 量化自投机解码 (2502.10424)

**Problem**: 长上下文推理中 KV Cache 是 GPU 内存和延迟的主要瓶颈。自投机解码（self-speculative）需要 draft 模型，额外 KV Cache 进一步加剧内存压力。

**Method**:
- Draft 模型与 target 模型共享架构，但使用分层 4-bit 量化 KV Cache + 4-bit 量化权重加速
- **分层量化 KV Cache**：target 和 draft 共享 KV Cache 的高位比特（bit-sharing），draft 无需额外内存
- **Double full-precision buffer**：最近 KV Cache 保持全精度，提升接受率并避免无用的量化/反量化

**Innovation**:
- Bit-sharing 消除了 self-speculative 方法中 draft 模型的额外内存开销
- Double buffer 是关键工程优化：最近 token 全精度 → 高接受率，历史 token 4-bit → 低内存
- 将投机解码和 KV Cache 量化统一到一个框架

**Results**:
- 接受率 > **90%**
- 端到端加速 **~2.5x**
- 内存减少 **~1.3x**（vs 其他 self-speculative 方法）

**Keywords**: self-speculative decoding, hierarchical KV quantization, bit-sharing, long-context inference

---

## 3. 横向对比与统一视角

### 3.1 投机解码演进路线

| 阶段 | 代表方法 | 核心思路 | 局限 |
|------|----------|----------|------|
| 固定 Draft Model | SpecDec (Leviathan) | 小模型 draft + 大模型 verify | 需要额外模型 |
| Self-Speculative | Medusa, Eagle | 模型内部生成 draft | $\gamma$ 固定，KV Cache 额外开销 |
| **量化 Self-Spec** | **QuantSpec** | 4-bit 量化 draft + bit-sharing KV | 量化误差累积 |
| **自适应 $\gamma$** | **SpecKV** | MLP 控制器逐步选最优 $\gamma$ | 需要 profiling 数据训练 |
| 统一理论 | UniVer (前作) | OT 统一 multi-step + multi-draft | 理论贡献为主 |

### 3.2 KV Cache 量化路线

```
均匀量化 (INT4/INT8 全 token 一致)
    ↓
逐 layer 混合精度 (KIVI, QAQ)
    ↓
逐 channel 混合精度 (WindowQuant)
    ↓
逐 token 自适应 (Adaptive KV Quant) ← 本批论文
    ↓
bit-sharing 跨模型 (QuantSpec) ← 量化+投机解码统一
```

### 3.3 端侧 LLM 设计哲学

| 路线 | 代表 | 核心思路 | 性能/效率 |
|------|------|----------|-----------|
| 训大压小 | LLaMA-3.2-1B | 蒸馏 + 量化 | 受限于原始架构 |
| 紧凑预训练 | Phi-3-mini | 高质量数据 + 小架构 | 数据依赖 |
| **原生端侧** | **SmallThinker** | 双层稀疏 + 预注意力路由 | 20 tok/s on CPU, 1GB |

### 3.4 面试高频问题

**Q: 投机解码的 $\gamma$ 如何选择？固定值有什么问题？**
A: 固定 $\gamma=4$ 是次优的：简单 token 接受率高应多推测，困难 token 应少推测。SpecKV 用 draft entropy/confidence 训练 MLP 控制器，逐步自适应选择，提升 56%。

**Q: KV Cache 量化的粒度选择？**
A: 演进路线：均匀 → 逐 layer → 逐 channel → 逐 token。Adaptive KV Quantization 证明逐 token 按重要性分配位宽效果最好（类 Huffman 原理）。

**Q: 端侧 LLM 为什么不能直接把大模型量化到 4-bit 就完事？**
A: 因为端侧瓶颈不只是内存。SmallThinker 识别了三个瓶颈（算力/内存/存储 I/O），用 pre-attention routing 隐藏存储延迟、双层稀疏减少计算量、NoPE-RoPE 减少 KV Cache。EfficientLLM 实证：INT4 量化精度降 3-5%，MoE 增加 40% VRAM，没有银弹。

**Q: QuantSpec 如何避免 draft 模型的额外 KV Cache 开销？**
A: Bit-sharing：target 的 FP16 KV Cache 高 4 位与 draft 的 4-bit KV Cache 共享存储，draft 无额外内存。最近 token 用 double full-precision buffer 保持全精度保证接受率。

---

## 4. 与现有知识的关联

- **投机解码统一视角**：SpecKV 的自适应 $\gamma$ 和 QuantSpec 的量化自投机是 [[20260504_speculative_decoding_longcontext_quant.md]] 中 UniVer/HierSpec 的实用化延伸
- **KV Cache 量化演进**：Adaptive KV Quantization 将 [[20260419_KV_cache_quantization_adaptive_methods.md]] 的混合精度推到逐 token 粒度
- **MoE 架构**：SmallThinker 的双层稀疏 MoE 是 [[MoE架构设计与推理优化.md]] 中稀疏激活思路在端侧的极致应用
- **效率评估**：EfficientLLM 为 [[LLM推理效率三角.md]] 提供了首个大规模实证基础
