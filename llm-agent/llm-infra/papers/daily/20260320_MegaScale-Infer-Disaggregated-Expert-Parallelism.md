# MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism

> 来源：https://arxiv.org/abs/2504.02263 | 日期：20260320 | 领域：llm-infra

## 问题定义

Mixture-of-Experts (MoE) 模型通过稀疏激活架构实现了大规模LLM的扩展，但这也带来了新的推理挑战：

1. **内存密集化问题**：MoE的稀疏激活特性使得FFN层从计算密集型转变为内存密集型，导致GPU利用率大幅降低
2. **部署成本高昂**：传统MoE推理系统的操作成本显著增加
3. **通信开销大**：token dispatch等数据传输操作带来显著的通信延迟
4. **异构资源利用困难**：Attention和FFN模块的特性差异大，难以统一优化

## 核心方法与创新点

### 1. 解耦式架构设计 (Disaggregated Architecture)
- **独立扩展**：将Attention和FFN模块在每个模型层内解耦，允许独立扩缩容
- **异构部署**：针对两个模块的不同特性，采用不同的并行策略和硬件配置
- **定制化并行**：为Attention和FFN分别设计最优的模型并行策略

### 2. 乒乓流水线并行 (Ping-Pong Pipeline Parallelism)
- **微批处理**：将请求批次划分为多个micro-batches
- **双向传输**：在Attention和FFN模块之间穿梭传输micro-batches进行推理
- **通信隐藏**：通过流水线调度有效隐藏通信开销
- **GPU利用率最大化**：充分利用MoE稀疏性带来的计算间隙

### 3. 高性能M2N通信库
- **零拷贝传输**：消除不必要的GPU-to-CPU数据复制
- **轻量初始化**：减少group initialization开销
- **异步同步**：优化GPU同步机制，降低延迟
- **专为token dispatch优化**：适配解耦架构的数据传输需求

## 实验结论

- **吞吐量提升**：相比SOTA方案，单GPU吞吐量提升高达 **1.90x**
- **成本效益**：显著降低大规模MoE serving的运营成本
- **扩展性**：支持大规模MoE模型的高效推理部署
- **稳定性**：在生产环境中验证了解耦架构的可行性

## 工程落地要点

1. **架构适配**：需要对现有推理框架进行改造以支持解耦部署
2. **调度优化**：乒乓流水线需要精细的调度策略以避免bubble
3. **网络优化**：M2N通信库是性能关键，需要RDMA等高速网络支持
4. **监控体系**：需要建立针对解耦架构的监控和调优工具
5. **容量规划**：独立扩展需要更复杂的容量规划模型

## 面试考点

**Q1：MoE推理中为什么会出现GPU利用率低的问题？**
A：MoE的稀疏激活特性使得只有部分expert被激活，FFN层从计算密集型变为内存密集型。传统推理系统按层调度，导致大量时间花在内存访问和通信上，而非计算。

**Q2：解耦Attention和FFN模块的核心优势是什么？**
A：1) 独立扩缩容，根据各自负载动态调整资源；2) 采用不同的并行策略（如EP for FFN, TP for Attention）；3) 异构硬件部署，FFN可用高内存GPU，Attention可用高算力GPU。

**Q3：乒乓流水线并行如何隐藏通信开销？**
A：通过将batch划分为micro-batches，在FFN处理当前micro-batch时，attention可以同时处理下一个micro-batch，形成流水线overlap，从而隐藏传输延迟。

**Q4：M2N通信库解决了哪些传统通信的痛点？**
A：传统NCCL在MoE场景下存在多余拷贝、初始化开销大、同步粒度粗等问题。M2N库通过零拷贝、轻量初始化和异步同步优化了token dispatch性能。
