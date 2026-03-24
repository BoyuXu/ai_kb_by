# LLM Serving 系统：从 vLLM 到 TensorRT-LLM 的工程实践

> 创建：2026-03-24 | 领域：LLM | 类型：综合分析
> 来源：vLLM, TensorRT-LLM, TGI, SGLang, Orca 系列

---

## 🎯 核心洞察（4条）

1. **LLM Serving 的核心指标是 TTFT 和 TPS**：TTFT（Time To First Token，首 token 延迟）决定用户感知，TPS（Tokens Per Second，生成速度）决定体验流畅度
2. **vLLM PagedAttention 是 Serving 的基础设施**：KV Cache 分页管理 + Continuous Batching 使 GPU 利用率从 30% 提升到 80%+
3. **Prefill 和 Decode 解耦是趋势**：Prefill（prompt 处理）是 compute-bound，Decode（生成）是 memory-bound，分离部署可以各自优化
4. **多租户和 SLA 管理是生产关键**：不同请求有不同的优先级和延迟要求，需要优先级队列 + 抢占机制

---

## 🎓 面试考点（5条）

### Q1: vLLM 的核心创新？
**30秒答案**：PagedAttention——将 KV Cache 按 page 管理（类似 OS 虚拟内存），动态分配/回收 page，解决了 KV Cache 碎片化和预分配浪费问题。配合 Continuous Batching 实现请求动态加入/退出。

### Q2: Continuous Batching vs Static Batching？
**30秒答案**：Static Batching 等所有请求完成生成才处理下一批，短请求被长请求拖累。Continuous Batching 允许每个 iteration 独立加入新请求/移除完成的请求，GPU 利用率更高。

### Q3: TensorRT-LLM 的优化技术？
**30秒答案**：①Kernel Fusion（多个操作合并为一个 GPU kernel）；②INT8/FP8 量化（硬件原生支持）；③Inflight Batching（Continuous Batching 的 NVIDIA 实现）；④Multi-GPU Tensor Parallelism。

### Q4: Prefill-Decode 解耦怎么做？
**30秒答案**：Prefill 节点处理 prompt（需要大算力），生成完整 KV Cache 后通过高速网络传给 Decode 节点。Decode 节点只做逐 token 生成（需要大内存带宽）。好处：各节点用最适合的硬件。

### Q5: LLM Serving 的成本优化策略？
**30秒答案**：①量化（FP16→INT8 节省一半显存和成本）；②Speculative Decoding（小模型猜大模型验，减少大模型推理次数）；③KV Cache 复用（相同 prefix 的请求共享 KV Cache）；④动态 batch size 调整。

---

## 🌐 知识体系连接

- **上游依赖**：GPU 架构、模型量化、分布式系统
- **下游应用**：ChatBot 部署、API 服务、Agent 推理
- **相关 synthesis**：std_llm_inference_optimization.md, std_llm_moe_architecture.md
