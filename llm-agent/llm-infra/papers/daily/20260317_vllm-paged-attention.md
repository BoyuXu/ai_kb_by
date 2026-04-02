# vLLM PagedAttention：LLM 推理内存管理革命

> 来源：工程实践 / SOSP 2023 UC Berkeley | 日期：20260317

## 问题定义

LLM 推理的内存瓶颈：KV Cache 存储随 batch size × seq_len 线性增长，传统方案需要**预先为每个请求分配最大可能长度的连续内存**，导致：
1. **内存浪费**：大多数请求实际生成长度远小于最大长度，预分配内存大量闲置
2. **碎片化**：不同长度请求的内存块在 GPU 上形成碎片，有效利用率仅 20~40%
3. **低 Throughput**：由于内存限制，batch size 受限，GPU 利用率低

## 核心方法与创新点

1. **PagedAttention**
   - 借鉴操作系统的虚拟内存分页管理思想
   - KV Cache 分割为固定大小的 **Block**（Page），每个 Block 存储固定数量 token 的 KV（如 16 tokens）
   - 每个请求的 KV Cache 由多个不连续的 Block 组成，通过**页表**（Block Table）映射逻辑地址到物理地址

2. **物理 Block 按需分配**
   - 请求到来时仅分配当前需要的 Block
   - 生成新 token 时，若当前 Block 已满，分配新 Block（可非连续）
   - 请求完成后，释放所有 Block 供其他请求使用

3. **Copy-on-Write 共享**
   - **Beam Search**：多个 beam 共享相同前缀的物理 Block，写入时才复制（Copy-on-Write）
   - **Parallel Sampling**：同一 prompt 生成多个独立序列，共享 prompt 部分的 KV Block
   - **Prefix Caching**：相同系统 prompt 的请求共享 KV Block

4. **Preemption + Swapping**
   - 当 GPU 内存不足时，可将优先级低的请求 KV Block 换出到 CPU（Swap）
   - 或者驱逐后重新计算（Recompute）

## 实验结论

- GPU 内存利用率从 ~40% 提升至 ~90%+
- 相比 HuggingFace Accelerate：吞吐量提升 2~4x
- Parallel Sampling（n=4）场景：内存节省约 55%（通过 CoW 共享前缀）

## 工程落地要点

1. **Block 大小选择**：16 tokens/block 是常用配置，更大 block 减少碎片但浪费更多（最后一个 block 往往不满）
2. **GPU 内存规划**：`vllm.LLM(gpu_memory_utilization=0.9)` 控制预留比例，预留 10% 给模型权重加载
3. **Prefix Caching 开启**：`enable_prefix_caching=True`，适合 System Prompt 固定的应用
4. **Chunked Prefill**：长 prompt prefill 分块处理，减少 TTFT 同时维持 decode 吞吐

## 常见考点

- **Q: PagedAttention 与操作系统虚拟内存的类比？**
  A: OS 虚拟内存：进程使用逻辑地址空间，通过页表映射到物理内存，不同进程可共享物理页（mmap），物理页按需分配，空闲页可换出。PagedAttention：请求使用逻辑 KV 序列，通过 Block Table 映射到物理 KV Block，不同请求可共享 Block（CoW），Block 按需分配，低优先级 Block 可换出。

- **Q: 为什么传统 LLM 推理内存利用率低？**
  A: 传统方案为每个请求预分配 max_seq_len 的连续内存（为了高效的批量矩阵乘法），但请求实际长度不到 max_seq_len，剩余内存空置；不同长度请求之间的碎片无法被其他请求使用。

- **Q: Copy-on-Write 在 Beam Search 中如何节省内存？**
  A: Beam Search 的多个 beam 有相同的前缀（已生成的部分），CoW 让它们共享前缀的物理 Block；只有当 beam 分叉（生成不同 token）时才复制出独立的 Block。若 beam 数 n=4，前缀占 80%，则内存节省约 60%。
