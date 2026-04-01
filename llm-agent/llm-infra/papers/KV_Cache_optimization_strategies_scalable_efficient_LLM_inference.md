# KV Cache Optimization Strategies for Scalable and Efficient LLM Inference
> 来源：arXiv:2603.20397 | 领域：llm-infra | 学习日期：20260330

## 问题定义
KV Cache 是 LLM 推理的核心性能组件，也是显存瓶颈。随着上下文长度从 4K 增长到 1M token，KV Cache 管理成为工程重点。本文系统综述 KV Cache 优化的全景策略，涵盖压缩、稀疏化、卸载、复用等多个维度。

## 核心方法与创新点

### 1. KV Cache 压缩
- **量化压缩**：INT4/INT2（见 MiniKV），8x 压缩比，精度损失 <2%
- **低秩分解**：将 K/V 矩阵分解为低秩近似 $K \approx AB$（rank=32），节省存储
- **Token 合并**：将相似 KV 向量合并（k-means pooling），减少 token 数

### 2. KV Cache 稀疏化（Sparse Attention）
- **局部窗口**：只保留最近 W 个 token 的 KV（Sliding Window Attention）
- **Sparse 注意力**：只保留 attention score 高的 KV（H2O, SnapKV）——token-level 稀疏
- **Sink Token**：发现 "attention sink"（初始 token 注意力分数恒高），始终保留 + 滑动窗口

### 3. KV Cache 卸载（Offloading）
- **CPU 卸载**：不活跃的 KV 页面换出到 CPU DRAM，活跃时预取回 GPU（PagedAttention 变体）
- **SSD 卸载**：超长上下文将 KV 分层存储（GPU HBM → CPU → SSD），延迟换取容量

### 4. KV Cache 共享与复用
- **Prefix Caching**：同一系统 Prompt 的 KV 跨请求复用（vLLM RadixAttention）
- **跨层共享**：相邻 Transformer 层共享 KV（MLA：Multi-head Latent Attention，DeepSeek）
- **Cross-request 复用**：RAG 场景中同一文档 KV 对多个 query 复用

## 实验结论
- 量化（INT4）+ 稀疏化（50% token drop）联合：H100 吞吐 +3.2x，精度损失 <3%
- Prefix Caching：同 system prompt 场景延迟降低 80%，显存节省 50%
- SSD 卸载：支持 1M token 上下文（原来 8K），代价：解码延迟增加 3x

## 工程落地要点
- 优先 Prefix Caching（零精度损失，高收益），其次量化（低精度损失），再考虑稀疏化
- Paged KV（PagedAttention）是工业标准，vLLM/TGI/TensorRT-LLM 均实现
- 稀疏化策略需要评估 task sensitivity（摘要任务 > 代码生成 > 问答 对 KV 稀疏更敏感）
- 卸载场景需要异步 prefetch，隐藏 IO 延迟

## 面试考点
- Q: PagedAttention 的原理和优势？
  - A: 类比 OS 虚拟内存分页：将 KV Cache 切成固定大小 block（默认 16 token），按需分配，避免内存碎片；支持请求间 block 共享（prefix caching）
- Q: StreamingLLM（Sink Token）的核心发现？
  - A: LLM 在 causal attention 中，最初几个 token（"sink tokens"）的注意力分数异常高，是模型用来存储"重置状态"的机制。保留 sink + 滑动窗口，即可无限长上下文推理
- Q: MLA（Multi-head Latent Attention，DeepSeek）如何减少 KV Cache？
  - A: 将 KV 压缩为低维 latent vector（$c_{KV} = W_{KV} h$），推理时从 latent 还原 KV，KV Cache 只存 latent（维度压缩 8-16x），显存大幅减少

## 数学公式

$$
\text{Standard KV: } M_\text{KV} = 2 \times L \times N_h \times d_h \times T \times \text{bytes}
$$

$$
\text{MLA KV: } M_\text{MLA} = L \times d_c \times T \times \text{bytes}, \quad d_c \ll N_h \times d_h
$$
