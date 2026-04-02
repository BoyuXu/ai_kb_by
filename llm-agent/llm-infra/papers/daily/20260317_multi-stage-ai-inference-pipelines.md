# Understanding and Optimizing Multi-Stage AI Inference Pipelines

> 来源：arXiv | 日期：20260317

## 问题定义

现代 AI 推理系统不再是单一模型推理，而是由多个阶段组成的**流水线（Pipeline）**，如 RAG（检索 → 重排 → 生成）、Agent（规划 → 工具调用 → 综合）、推荐系统（召回 → 粗排 → 精排 → 重排）。多阶段流水线面临：

1. **瓶颈分析**：哪个阶段是端到端延迟的主要瓶颈？
2. **资源调度**：不同阶段的计算需求差异大，如何高效分配 GPU/CPU 资源？
3. **批量策略**：各阶段最优 batch size 不同，如何协调？
4. **容错设计**：一个阶段的失败如何处理？

## 核心方法与创新点

1. **流水线建模**
   - 将多阶段流水线建模为排队网络（Queueing Network）
   - 每个阶段 = 一个服务节点，有到达率 λ、服务率 μ
   - 端到端延迟 = $\sum_i \text{Latency}}_{\text{i + \text{Queue Time}}_i$

2. **瓶颈识别**
   - 利用率（Utilization）= λ / μ，利用率最高的阶段是瓶颈
   - 水平扩展瓶颈阶段（增加实例）

3. **异步流水线（Async Pipeline）**
   - 各阶段异步执行，结果通过消息队列传递（Kafka/Redis）
   - 避免同步等待，提升整体吞吐量
   - 代价：增加端到端延迟（队列等待时间）

4. **预取（Prefetching）**
   - 利用流水线的阶段顺序，提前执行后续阶段准备工作
   - 如 RAG：LLM 生成阶段开始前，提前 prefill 检索文档的 KV Cache

5. **批量策略优化**
   - Continuous Batching：在 LLM decode 阶段动态加入新请求（vLLM 的核心技术）
   - Chunked Prefill：将长 prefill 分块，与 decode 交错执行，降低 TTFT（Time to First Token）

6. **资源解耦**
   - Prefill 和 Decode 分离部署（Disaggregated Prefill-Decode）：GPU 利用率更高
   - 检索/重排在 CPU 集群，LLM 在 GPU 集群，各自独立扩展

## 实验结论

- 异步流水线相比同步：吞吐量提升约 2.5x，端到端 P50 延迟增加约 20%
- Continuous Batching：GPU 利用率从 40% 提升至 80%+
- Disaggregated Prefill-Decode：整体 token 吞吐量提升约 1.8x

## 工程落地要点

1. **SLO 拆解**：端到端 SLO（如 P99<2s）需分配给各阶段，通常按比例：检索<200ms, 重排<300ms, LLM<1500ms
2. **背压（Backpressure）**：下游阶段处理满时，向上游传递压力，避免内存撑爆
3. **降级策略**：重排超时时跳过重排直接使用召回结果；LLM 超时时返回 cached 答案
4. **监控指标**：各阶段 P50/P99 延迟、吞吐量、队列深度、错误率

## 常见考点

- **Q: Continuous Batching 的核心思想？**
  A: 传统 Static Batching 需要等待 batch 内所有请求完成才能接受新请求，导致 GPU 空闲（短请求完成后等长请求）。Continuous Batching 允许新请求在任意时刻加入正在执行的 batch，GPU 持续处于最高利用率。

- **Q: Prefill 和 Decode 分离部署的动机？**
  A: Prefill（处理输入 prompt）是计算密集型（矩阵乘法），适合大 batch；Decode（逐 token 生成）是内存带宽密集型（KV Cache 访问），受限于内存带宽而非算力。分离部署让两种操作各自在最优配置下运行，避免相互干扰。

- **Q: 多阶段流水线的端到端延迟如何优化？**
  A: 1) 识别并消除瓶颈阶段（水平扩展）；2) 并行化独立阶段；3) 使用预取减少后续阶段等待时间；4) 动态调整各阶段的资源配比（如流量低峰时缩减重排实例，高峰时扩容）。
