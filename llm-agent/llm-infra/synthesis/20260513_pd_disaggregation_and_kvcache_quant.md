# P/D Disaggregation 与 KV Cache 量化前沿 (2025-2026)

> **覆盖论文**: 10 篇 (5 篇深度 + 5 篇简要)
> **核心主题**: Prefill-Decode 分离架构 / KV Cache 自适应量化 / KV Cache 管理综述
> **关联概念**: [[embedding_everywhere]] | [[attention_in_recsys]]

---

## 一、技术演进脉络

```
Monolithic Serving (vLLM)
  ↓ Prefill 与 Decode 干扰严重
Inter-GPU P/D Disaggregation (DistServe, Splitwise)
  ↓ 跨 GPU 通信开销大
Intra-GPU P/D Disaggregation (Nexus)     ← 单 GPU 内分离
  ↓ 结合两者
Unified Aggregation-Disaggregation (TaiChi) ← 动态切换
  ↓ 进一步优化模型层面
Targeted Pruning for P/D (PDTrim)        ← 针对性剪枝
  ↓
KV Cache 量化与管理 (端侧优化)
  ├── 自适应量化 (AdaptiveKV)
  └── 系统级 KV Cache 管理 Survey
```

---

## 二、深度学习论文

### Paper 6: TaiChi: Unifying P/D Aggregation and Disaggregation for LLM Serving
**[2508.01989] Wang et al. (2025.08)**

**Problem**: P/D 分离 (disaggregation) 降低延迟但浪费 GPU; 聚合 (aggregation) 提高利用率但引入干扰。如何两全?

**Key Innovation - Latency Shifting**:
- 核心洞察: 不是所有请求都需要相同的 SLO 余量
- **Latency Shifting**: 将"SLO 余裕充足"的请求的 GPU 资源转移给"SLO 快超标"的请求
- 两类 GPU 实例:
  - **Prefill-heavy**: 快速 prefill, 但 decode 受干扰
  - **Decode-heavy**: 低干扰 decode, 但 prefill 慢

**架构**:
```
Request Router
  ├── SLO 分析器 → 预测 TTFT/TPOT 是否达标
  ├── Prefill-heavy GPU pool (适合短输入长输出)
  └── Decode-heavy GPU pool (适合长输入短输出)
      ↕ 动态迁移 (latency shifting)
```

**核心公式**:
$$\text{Goodput} = \frac{\text{满足 TTFT 和 TPOT SLO 的请求数}}{\text{总请求数}}$$

TaiChi 优化目标: 最大化 Goodput, 而非简单的吞吐量或延迟。

**Results**:
- Goodput 提升 **77%** (vs SOTA, balanced SLO 下)
- 关键场景: TTFT 和 TPOT 同时有严格 SLO 时优势最大

**面试考点**:
- Q: P/D 分离 vs 聚合的 trade-off? A: 分离降低干扰但增加 GPU 数量和 KV 传输开销; 聚合共享 GPU 但 decode 受 prefill 干扰。TaiChi 动态选择。

---

### Paper 7: Nexus: Proactive Intra-GPU P/D Disaggregation
**[2507.06608] Shi et al. (2025.07)**

**Problem**: 跨 GPU 分离通信开销大; 单 GPU 内的 chunked prefill 虽共享但干扰难控。

**Key Innovation - Proactive Resource Splitting**:
- 发现 GPU 资源存在 **收益递减点 (saturation point)**: 超过阈值后, 增加资源对延迟改善极小
- 在单 GPU 内将计算资源动态分割给 prefill 和 decode
- **主动式** (proactive) vs 被动式 (reactive): 不是等 SLO 违反再调整, 而是预测并提前分配

**核心机制**:
```
单 GPU
  ├── Prefill partition (SM cores 子集)
  ├── Decode partition (SM cores 子集)
  └── Dynamic Rebalancer (每个 batch 重新分配)
```

**Results**:
- TBT (Time-Between-Tokens) 降低 **2.2x**
- 吞吐量提升 **1.4x** (vs vLLM-disaggregation), 且只需 **一半 GPU 数量**

**面试考点**:
- Q: Intra-GPU vs Inter-GPU 分离的核心区别? A: Intra-GPU 避免 KV cache 网络传输, 但需要 GPU 内部资源隔离 (SM partitioning)
- Q: 收益递减效应的工程意义? A: 说明 GPU 资源可以安全地在 P/D 间分享, 不需要 100% 独占

---

### Paper 8: PDTrim: Targeted Pruning for P/D Disaggregation
**[2509.04467] Zhang et al. (2025.09)**

**Problem**: 现有模型剪枝方法忽略 P/D 分离场景的特殊性。Prefill 和 decode 对不同层的敏感度不同。

**Key Innovation**:
- **异构剪枝敏感度分析**: Prefill 和 decode 阶段对 transformer block 的移除敏感度差异显著
- Prefill 更敏感于浅层 (信息提取), decode 更敏感于深层 (生成质量)
- 为 P 和 D 分别识别可移除的 block, 生成两个不同的剪枝模型

**Method**:
1. 构建剪枝集和蒸馏集
2. 迭代 block 移除: 每次移除对 P 或 D 影响最小的 block
3. 知识蒸馏恢复质量

**Results**:
- 推理加速 **20.56%**
- KV cache 传输带宽降低 **4.95x** (关键: P/D 分离中最大的开销就是 KV 传输)
- 质量损失可控 (蒸馏恢复)

**面试考点**:
- Q: 为什么剪枝要区分 P 和 D? A: 两阶段的计算模式不同 (P 是矩阵乘, D 是向量乘), 对不同层的依赖不同
- Q: 带宽降 4.95x 怎么实现的? A: 剪掉的层不需要传输 KV cache, P 和 D 各自使用更小的模型

---

### Paper 9: Don't Waste Bits! Adaptive KV-Cache Quantization for On-Device LLMs
**[2604.04722] Boroujeni et al. (2026.04)**

**Problem**: 端侧 LLM 推理中 KV cache 内存/带宽是主要瓶颈, 固定精度量化浪费 bit 或过度压缩。

**Key Innovation**:
- **Token-level 自适应量化控制器**: 基于轻量特征动态选择精度
- 特征: token 频率、质量分数、attention variance、entropy 不确定性
- 精度选择: {2-bit, 4-bit, 8-bit, FP16}

**架构**:
```
每个 token 进入 KV cache 时:
  ├── 提取 4 个轻量特征
  ├── 送入 compact controller (小 MLP)
  └── 输出精度决策: 2/4/8/FP16 bit
```

**核心公式**:
$$b_t = f_\theta(\text{freq}_t, \text{quality}_t, \text{var}_t, \text{entropy}_t) \in \{2, 4, 8, 16\}$$

**Results** (SmolLM-360M on HellaSwag):
- Decoding 延迟降低 **17.75%** (vs static quantization)
- 准确率提升 **7.60 points**
- 与 FP16 推理仅差 **0.30 points**

**面试考点**:
- Q: 为什么自适应比固定精度好? A: 重要 token (高 attention) 用高精度保留信息, 不重要 token (低 entropy) 用低精度节省空间
- Q: 端侧部署的特殊约束? A: 内存极有限 (几 GB), KV cache 随序列长度线性增长, 必须激进压缩

---

### Paper 10: Survey on LLM Acceleration based on KV Cache Management
**[2412.19442] Li et al. (2024.12, revised 2025.07)**

**Problem**: KV Cache 管理技术的全景综述。

**Key Innovation - 三层分类法**:

| 层级 | 策略 | 代表方法 |
|------|------|---------|
| **Token-level** | 选择 / 预算分配 / 合并 / 量化 / 低秩分解 | StreamingLLM, H2O, KIVI, GQA |
| **Model-level** | 架构创新 / 注意力机制优化 | MQA, GQA, MLA (DeepSeek) |
| **System-level** | 内存管理 / 调度 / 硬件感知 | vLLM PagedAttention, SGLang RadixAttention |

**核心技术详解**:

**Token-level 选择**:
- **StreamingLLM**: 保留 attention sink (开头几个 token) + 最近 window
- **H2O (Heavy Hitter Oracle)**: 保留累积 attention 最高的 token
- **SnapKV**: 基于 attention pattern 的 token 重要性自动识别

**Token-level 量化**:
- **KIVI**: Key 用更高精度 (4-bit), Value 用更低精度 (2-bit)
- **KVQuant**: 分通道量化 + outlier 处理

**Model-level 架构**:
- **MQA → GQA → MLA**: KV head 数量 从 $n_h$ → $n_h/g$ → 低秩压缩
- MLA (Multi-head Latent Attention): DeepSeek-V2 使用, 将 KV 投影到低秩空间

**System-level**:
- **PagedAttention (vLLM)**: 类操作系统分页, 消除内存碎片
- **RadixAttention (SGLang)**: 利用 prefix 共享, 多请求复用 KV

**面试考点**:
- Q: KV Cache 为什么是推理瓶颈? A: 大小 = $2 \times n_\text{layers} \times n_\text{heads} \times d_\text{head} \times \text{seq\_len}$, 随序列长度线性增长, 128K context 下可达数十 GB
- Q: MQA vs GQA vs MLA 的演进逻辑? A: MQA 太激进 (1个KV head) 损失质量; GQA 折中 (分组共享); MLA 用低秩投影, 保持质量同时极致压缩

---

## 三、简要记录论文 (LLM-Infra 方向其余论文)

> 以下论文与本批 KV Cache / Serving 主题相关, 已在之前批次 synthesis 中深度覆盖:
> - KV Cache 前沿 2026: [[20260503_kv_cache_frontier_2026]]
> - KV Cache + 投机解码: [[20260504_kv_cache_and_speculative_serving]]
> - 投机解码 + 长上下文 + 量化: [[20260504_speculative_decoding_longcontext_quant]]

本批 5 篇深度论文覆盖了 P/D Disaggregation 和 KV Cache 量化的最新进展, 与之前批次形成完整的 LLM Serving 优化知识图谱。

---

## 四、P/D Disaggregation 技术对比

| 方法 | 粒度 | 核心思路 | 优势 | 劣势 |
|------|------|---------|------|------|
| DistServe | Inter-GPU | P 和 D 在不同 GPU | 零干扰 | KV 传输开销大, GPU 利用率低 |
| Nexus | Intra-GPU | 单 GPU 内 SM 分区 | 无网络传输 | 需要 GPU 支持资源隔离 |
| TaiChi | Hybrid | 动态选择聚合/分离 | 最优 goodput | 系统复杂度高 |
| PDTrim | Model-level | 为 P/D 分别剪枝模型 | 降传输带宽 4.95x | 需要离线剪枝+蒸馏 |

---

## 五、KV Cache 优化技术全景

```
KV Cache 优化
├── 减少 KV 数量 (Token-level)
│   ├── 驱逐: StreamingLLM, H2O
│   ├── 合并: CaM, D2O
│   └── 选择: SnapKV, PyramidKV
├── 压缩 KV 精度 (Quantization)
│   ├── 固定: KIVI (K4V2)
│   ├── 自适应: AdaptiveKV [本批], CacheQuant
│   └── 混合: MiniKV (2-bit + importance)
├── 架构减少 KV (Model-level)
│   ├── MQA → GQA → MLA
│   └── Linear Attention (无需 KV cache)
└── 系统级管理 (System-level)
    ├── PagedAttention (vLLM)
    ├── RadixAttention (SGLang)
    └── Offloading (FlexGen, InfiniGen)
```

---

## 六、面试考点总结 (Q&A)

**Q1: P/D Disaggregation 解决什么问题?**
A: Prefill (计算密集, 矩阵乘) 和 Decode (内存密集, 向量乘) 混合执行时互相干扰: prefill 抢占 GPU 导致 decode 延迟飙升, decode 占用内存导致 prefill batch 变小。分离后各自优化。

**Q2: TaiChi 的 Latency Shifting 原理?**
A: 发现不是所有请求都紧贴 SLO 边界。SLO 余裕大的请求可以"贡献"资源给紧迫请求。通过在 prefill-heavy 和 decode-heavy GPU 间动态路由实现。Goodput 提升 77%。

**Q3: Nexus 如何在单 GPU 内做 P/D 分离?**
A: 利用 GPU SM (Streaming Multiprocessor) 的收益递减特性: 超过饱和点后增加 SM 对延迟改善极小。因此可以安全地将部分 SM 分给另一阶段, 通过 Dynamic Rebalancer 每个 batch 重新分配。

**Q4: PDTrim 的异构剪枝有什么意义?**
A: Prefill 对浅层敏感 (信息提取), Decode 对深层敏感 (生成质量)。分别剪枝后, P 模型去掉深层 block, D 模型去掉浅层 block, KV cache 只需传递两个模型共有的层, 带宽降 4.95x。

**Q5: 端侧 LLM 的 KV Cache 量化为什么需要自适应?**
A: 固定精度 (如全部 4-bit) 对重要 token (attention sink, 关键信息点) 过度压缩导致质量下降, 对不重要 token 又浪费精度。自适应方法根据 token 重要性动态分配精度, 同样的平均 bit 数下质量更好。

**Q6: KV Cache 大小怎么估算?**
A: $\text{KV size} = 2 \times L \times n_h \times d_h \times s \times \text{dtype\_bytes}$, 其中 $L$ 是层数, $n_h$ 是 KV head 数, $d_h$ 是 head 维度, $s$ 是序列长度。LLaMA-70B (GQA 8 heads) 在 128K context 下约 40GB。

**Q7: MLA (Multi-head Latent Attention) 是什么?**
A: DeepSeek-V2 提出。将 K, V 投影到低维潜空间 $c = W_\text{compress} \cdot [K; V]$, 推理时只缓存 $c$ (维度远小于原始 KV), 解码时再投影回来。兼顾 MHA 的质量和 MQA 的效率。
