# LLM 推理优化全景：从 Attention 到 Serving 系统

> 综合自 9+ 篇 synthesis | 更新：2026-04-13 | 领域：LLM 推理优化
> 关联：[[concepts/attention_in_recsys]] | [[07_MoE架构与稀疏激活]]

---

## 推理优化技术全景

| 优化方向 | 朴素方案 | 创新方案 | 加速倍数 |
|---------|---------|---------|---------|
| Attention 计算 | 标准 O(n²) | FlashAttention-3（Tiling + Warp 专业化） | 1.5-2x vs FA2 |
| KV Cache 内存 | 全量缓存 | GQA + 量化 + PagedAttention + XQuant | 4-32x 内存省 |
| 解码速度 | 逐 token 自回归 | Speculative Decoding（EAGLE-3, Double SD） | 2-5x |
| 批处理 | Static Batching | Continuous Batching（vLLM） | 2-3x 吞吐 |
| 分布式 | Tensor Parallelism | Prefill-Decode 分离 + MoE 解耦 | 1.5-2x |
| 长上下文 | Full Attention | MoBA 块级稀疏 + PLENA 分层 KV | 3x+ 吞吐 |

---

## 一、FlashAttention 系列

### 核心原理：IO-Aware Tiling

标准 Attention 需要将完整 n×n 矩阵写入 HBM：

$$
\text{IO}_{naive} = O(n^2 d + n^2) = O(n^2) \text{ HBM 访问}
$$

FlashAttention 将 Q/K/V 分块，在 SRAM 中完成计算（Online Softmax）：

$$
m_{\text{new}} = \max(m_{\text{old}},\ \max_j S_{ij}), \quad \ell_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \ell_{\text{old}} + \sum_j e^{S_{ij} - m_{\text{new}}}
$$

$$
\text{IO}_{FA} = \Theta\!\left(\frac{n^2 d^2}{M}\right), \quad M = \text{SRAM 容量}
$$

### FlashAttention 演进

| 版本 | 核心创新 | GPU 利用率 | 加速 |
|------|---------|-----------|------|
| FA-1 (2022) | Tiling + Online Softmax | ~35% | 2-3x vs 标准 |
| FA-2 (2023) | 更优分块, A100 优化 | ~35% | 改进 FA-1 |
| FA-3 (2024) | Warp 专业化 + Softmax 延迟隐藏 + FP8 | **~75%** | 1.5-2x vs FA-2 |

**FA-3 三大创新**：
1. **Warp 专业化**：Producer warp（TMA 加载数据）+ Consumer warp（WGMMA 矩阵乘），计算与 IO 真正重叠
2. **Softmax 延迟隐藏**：max-reduction 穿插在 WGMMA 等待期
3. **FP8 支持**：Q/K 用 E4M3（精度高），V 用 E5M2（范围大），Softmax FP32

---

## 二、KV Cache 优化

### 内存精确计算

$$
\text{KV}_{Mem} = 2 \times L \times H_{KV} \times d_k \times N \times s
$$

| 符号 | 含义 | 示例 |
|------|------|------|
| $L$ | 层数 | 80 (LLaMA-3-70B) |
| $H_{KV}$ | KV 头数（GQA 后） | 8 |
| $d_k$ | Head 维度 | 128 |
| $N$ | 序列长度 | 32768 |
| $s$ | 字节/元素 | 2 (BF16) |

LLaMA-3-70B, 32K tokens: $\approx$ 107 GB（单请求 KV Cache）

### GQA/MQA：注意力头共享

$$
\text{KV}_{GQA} = \frac{G}{H} \times \text{KV}_{MHA}
$$

- $G=1$（MQA）：最大压缩，精度损 ~2%
- $G=H/4$（GQA）：工业最优解，精度损 <0.5%，LLaMA-3/Mistral 标配

### 三条压缩路线

| 路线 | 方法 | 内存节省 | 精度损失 |
|------|------|---------|---------|
| A: 量化 | INT8/INT4 KV Cache | 2-4x | <1-3% |
| B: Token 驱逐 | H2O/SnapKV/StreamingLLM | 2-5x | <3% |
| C: 结构优化 | MLA (DeepSeek) | 8-16x | ~0% |
| 新: 重物化 | XQuant (存 X 不存 KV) | 7.7-12.5x | <0.1% |

**XQuant 核心洞察**：缓存层输入 X 而非 KV，推理时重算 K=XW_K, V=XW_V。X 只有一个张量（vs KV 两个），分布更平滑（无 RoPE 扭曲），量化友好。

### MoBA：混合块注意力（Moonshot, 2025）

将 MoE 思想迁移到注意力——将上下文切分为 KV Block，每个 Query token 通过 Router 动态选择 Top-K 个最相关 Block。已部署 Kimi。与 FlashAttention 可组合使用。

---

## 三、Speculative Decoding（投机解码）

### 数学原理

$$
\alpha = \mathbb{E}\left[\min\left(1, \frac{p_{target}(x)}{q_{draft}(x)}\right)\right], \quad \text{Speedup} \approx \frac{\gamma+1}{1 + (1-\alpha)/\alpha \cdot c}
$$

拒绝后重采样保证输出分布与大模型完全一致：$p'(x) = \text{norm}(\max(0, p(x) - q(x)))$

### EAGLE 系列演进

| 版本 | 核心改进 | 接受率 |
|------|---------|-------|
| EAGLE | 草稿模型 + target logit 对齐 | 0.71 |
| EAGLE-2 | 动态草稿长度 | 0.75 |
| EAGLE-3 | TTT（训练时模拟验证） | **0.82** |

**Draft 对齐**：用 LoRA 对小模型做 KL 散度对齐，接受率 62% → 83%，端到端加速 1.8x → 2.7x。

### 其他变体

- **Double SD**：两级草稿模型（超小→小→大），更高加速比
- **FastMTP**：Multi-Token Prediction，一次预测多个 token

---

## 四、Serving 系统

### 系统对比

| 系统 | 核心技术 | 吞吐(相对) | 特色 |
|------|---------|-----------|------|
| HuggingFace | Static Batching | 1x | 简单易用 |
| vLLM | PagedAttention + Continuous Batching | 3-5x | 开源标杆 |
| TensorRT-LLM | Kernel Fusion + INT8/FP8 | 5-8x | NVIDIA 官方 |
| SGLang | RadixAttention 前缀复用 | 4-6x | Agent/多轮最优 |

### PagedAttention 核心

借鉴 OS 虚拟内存分页，KV Cache 按 page 动态分配/回收：
- 内存利用率 60% → 95%
- 多请求共享 Prefix KV
- Copy-on-Write 支持 Beam Search

### Continuous Batching

$$
\text{Throughput} = \frac{B_{eff} \times \bar{L}_{output}}{\bar{T}_{latency}}
$$

每个 decode step 可加入/退出请求，GPU 利用率 30% → 80%+。

### Prefill-Decode 解耦

Prefill（compute-bound）和 Decode（memory-bound）分到不同 GPU 集群。PLENA 进一步实现三层 KV Cache（HBM/DRAM/SSD），128K 上下文吞吐提升 3.2x。

---

## 五、推理效率三角

$$
\text{Quality} \times \text{Throughput} \times \text{Latency}^{-1} \leq C_{hardware}
$$

### Arithmetic Intensity 判断瓶颈

$$
I = \frac{\text{FLOPs}}{\text{Bytes}_{accessed}} \quad \Rightarrow \quad I > I_{roofline}: \text{compute-bound}; \quad I < I_{roofline}: \text{memory-bound}
$$

Prefill = compute-bound（大矩阵乘），Decode = memory-bound（KV Cache 读写）。

### 延迟 Benchmark（7B, A100 40GB）

| 优化组合 | TTFT | TPOT | 吞吐 |
|---------|------|------|------|
| 无优化 | 500ms | 20ms | 50 tok/s |
| +Continuous Batching | 300ms | 15ms | 200 |
| +PagedAttention | 250ms | 12ms | 300 |
| +INT8 量化 | 200ms | 8ms | 450 |
| +投机解码 | 200ms | 3ms | 1200 |

---

## 六、技术演进脉络

```
2020: 朴素 Attention + 静态 KV Cache
  → 2022: FlashAttention-1（IO-aware tiling）
    → 2023: vLLM PagedAttention + FA-2 + GQA + Speculative Decoding
      → 2024: FA-3 (H100 优化) + MoE 主流化 + EAGLE-2
        → 2025: XQuant 重物化 + MoBA 块级稀疏 + PLENA 分层 KV
          → 趋势: 学习型 KV 驱逐 + 重物化-稀疏组合
```

---

## 七、工业落地选型

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| 通用生产 API | PagedAttention + INT8 KV | 安全、高利用率 |
| 长对话助手(>32K) | H2O 驱逐 + Sliding Window | 显存可控 |
| Agent/ReAct | SGLang + RadixAttention | 重复 prefix，命中率高 |
| 批量离线推理 | Continuous Batching + FP8 | 最高吞吐 |
| 实时广告(<10ms) | 不压缩 + 小模型 | SLA 严格 |

---

## 面试高频 Q&A

### Q1: FlashAttention 的核心原理？
**30秒**：分块计算避免写出完整 n×n 矩阵到 HBM，Online Softmax 维护 running max/sum。FA-3 进一步让计算和 IO 流水线重叠，H100 利用率 35%→75%。

### Q2: KV Cache 显存占用公式？
**30秒**：$2 \times L \times H_{KV} \times d \times N \times s$。LLaMA-3-70B 32K tokens ≈ 107GB。优化路线：GQA（头共享）、量化（2-4bit）、PagedAttention（分页）、XQuant（重物化）。

### Q3: Speculative Decoding 的适用条件？
**30秒**：Draft 和 target 分布足够接近（接受率 >70%）。数学保证输出分布一致（rejection sampling）。加速比取决于接受率：$\alpha=0.83$ 时理论 ~6x，实际 2-3x。

### Q4: Prefill 和 Decode 阶段区别？
**30秒**：Prefill 处理整个 prompt（compute-bound），Decode 逐 token 生成（memory-bound）。解耦部署各自用最适合的硬件。

### Q5: PagedAttention vs OS 虚拟内存？
**30秒**：KV Block = 内存页，Block Manager = 页表，Copy-on-Write = Beam Search 分叉。消除 60-80% 内存浪费，并发 1-2 → 16-64。

### Q6: SGLang 相比 vLLM 的优势？
**30秒**：Radix Tree 支持跨 session 的精细前缀共享（vLLM 只支持静态 prefix）。对 ReAct Agent 场景吞吐量 ~3x，原生支持 constrained decoding。

### Q7: XQuant 为什么存 X 比存 KV 好？
**30秒**：(1) 一个张量 vs 两个，直接 2x 节省；(2) X 无 RoPE 扭曲，量化友好；(3) 跨层 X 高度相似可 delta 压缩至 10-12.5x。代价是重算 KV 的 FLOPs，但 decode 是 memory-bound，几乎免费。

### Q8: 推理优化策略优先级？
**30秒**：① INT8 量化（最易，2x 加速）→ ② PagedAttention（显存利用率）→ ③ Speculative Decoding（延迟 3-4x）→ ④ FlashAttention（稳定提速）。

---

## 记忆助手

- **KV Cache = 笔记本**：对话越长越厚（内存增长），PagedAttention = 活页本（按需加页），GQA = 共享笔记（多人共用一份）
- **FlashAttention = 流水线工厂**：搬运工搬材料时，计算员同时加工上一批
- **Speculative Decoding = 先写草稿再审核**：小模型快速生成 N 个 token，大模型一次性验证
- **推理三大瓶颈口诀**：Prefill 算力密集，Decode 内存密集，多卡通信密集

---

## 相关概念

- [[concepts/attention_in_recsys|Attention 在搜广推中的演进]]
- [[concepts/embedding_everywhere|Embedding 技术全景]]
- [[07_MoE架构与稀疏激活|MoE 架构与推理优化]]
