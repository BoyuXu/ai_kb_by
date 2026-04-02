# KV Cache 优化技术：从显存瓶颈到高效推理

> 来源：技术综述 | 日期：20260316 | 领域：llm-infra

## 问题定义

Transformer 在推理时，KV Cache 存储历史 token 的 Key 和 Value，大小随序列长度线性增长：
```
KV cache size = 2 × batch_size × num_layers × seq_len × head_dim
            = 2 × 32 × 80 × 4096 × 128  ≈ 83 GB（LLaMA-70B, seq_len=4k）
```

**问题**：
1. **显存瓶颈**：KV cache 占显存 60-80%，严重限制 batch size 和 sequence length。
2. **推理延迟**：访问显存延迟 500-1000 ns，比计算慢 100x（计算密度低）。

## 核心方法与创新点

### 1. KV Cache 量化（KV Quantization）

将 KV embedding 从 float32 量化为 int8 或更低精度：

**INT8 量化**：每个 token 的 KV 分别用一个缩放因子 s，实际值 = 量化值 × s。
```python
# 量化
kv_quantized = (kv_float / scale).round().clip(-128, 127).astype(int8)

# 反量化（推理时）
kv_dequant = kv_quantized * scale  # 关键：每次读取时反量化
```

精度损失通常 < 0.5% AUC（大多数模型）。显存节省 **4-8x**。

**激进量化（4-bit / 2-bit）**：精度损失 1-3%，节省 8-16x。

### 2. KV Cache 分层（KV Cache Hierarchy）

将访问频率分层：
```
L1: 寄存器/L1 Cache（热 token）
L2: SRAM（温 token）
L3: HBM 显存（冷 token，很少访问的历史）
L4: 主机内存/磁盘（极冷，需要 swap）
```

对每一层使用不同存储策略：
- **Hot tokens**（最近N个）：保持高精度，全精度存储。
- **Warm tokens**：INT8 量化，存在 SRAM。
- **Cold tokens**：极度压缩（PagedAttention style），存在主机内存。

### 3. PagedAttention（页面式 KV Cache）

VLLM 提出的高效管理方案：

将 KV cache 分割成固定大小的"页"（如 16 tokens），支持：
- **动态显存分配**：不预先分配全部显存，用到时才分配。
- **显存碎片整理**：页式管理避免了序列长度不均导致的内存浪费。
- **共享 KV**：多个请求共享相同前缀的 KV（Prompt 重复时）。

### 4. Multi-Query Attention（MQA）/ Group-Query Attention（GQA）

减少 KV head 数量：
```
标准 MHA:     Q heads: 32,  K heads: 32,  V heads: 32  → KV size = 32×d
GQA:          Q heads: 32,  K heads: 4,   V heads: 4   → KV size = 4×d（8倍减少）
MQA:          Q heads: 32,  K heads: 1,   V heads: 1   → KV size = 1×d（32倍减少）
```

GQA（Llama 2 采用）速度快 5-10%，质量损失 < 1%。

## 实验结论

- INT8 KV 量化：显存节省 4x，吞吐量（throughput）+50%，困惑度（perplexity）影响 < 0.5%。
- PagedAttention（VLLM）：相比标准 attention，内存利用率 +60-80%（显存节省）。
- GQA vs MHA：LLaMA-2：在相同显存约束下，GQA 吞吐量更高（因为 KV 更小）。

## 工程落地要点

- **小模型量化收益更大**：小模型 KV cache 占比更高（计算少）；大模型可能优先用其他优化。
- **批量推理显存优化建议**：量化 + PagedAttention + GQA 组合，显存节省可达 10-16x，使原来无法装下的 70B 模型可以在 80G 显存上跑。
- **精度检验**：对关键应用（医学诊断、代码生成）量化后需要 benchmark，验证精度衰减是否可接受。
- **框架支持**：vLLM 内置了 KV quantization 和 paged attention；HuggingFace Transformers 的 bitsandbytes 支持量化；TensorRT-LLM 对 INT8/INT4 有特殊优化。

## 常见考点

- Q: 为什么 KV cache 这么占显存？
  A: 自回归 LLM 生成时，第 i 个 token 需要 attend 到前 i-1 个 token。如果每次都重新计算 K,V，会浪费计算（许多重复）。KV cache 保存历史 K,V，避免重复计算，但需要存储所有历史。Batch 推理时，多个请求的历史长度不同，无法完全共享。

- Q: PagedAttention 和标准 attention 在 KV cache 管理上的区别？
  A: 标准 attention 为每个序列预先分配一块连续显存（max_seq_len），浪费严重（序列通常短于 max_len）。PagedAttention 将 KV 分成固定大小的"页"，动态分配，只用所需的页数。多个序列可共享访问同一个页（当 prefix 相同时），进一步减少显存。

- Q: GQA 为什么有效？多头 attention 的作用是什么？
  A: 多头 attention 让模型同时从多个"表示子空间"获取信息（类似多个滤波器）。但每个 query head 不一定都需要独立的 K,V head——可以多个 query head 共享一个 K,V head，减少参数和显存，同时保留多头带来的表达力。GQA（分组共享）是折中方案。
