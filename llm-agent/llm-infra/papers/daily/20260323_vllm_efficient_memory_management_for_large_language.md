# vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention
> 来源：https://arxiv.org/abs/2309.06180 | 领域：llm-infra | 日期：20260323

## 问题定义
传统LLM serving存在严重的内存浪费：预分配的KV cache会有大量内部碎片（50-70%浪费）。vLLM提出PagedAttention，类比操作系统的虚拟内存，将KV cache分页管理，大幅提升内存利用率和吞吐量。

## 核心方法与创新点
- PagedAttention：KV cache按固定大小的block（page）管理，类似OS页表
- 按需分配：不预分配，生成几个token就分配几个KV block
- 物理-逻辑地址映射：block table维护逻辑序列到物理KV blocks的映射
- Continuous Batching集成：与Orca的连续batching结合，最大化GPU利用率

## 实验结论
vLLM相比FasterTransformer，吞吐量提升2-4x；相比原生HuggingFace提升约24x；内存碎片从70%降至<4%；能支持更大batch size（同等GPU内存下batch增加3-5x）。

## 工程落地要点
- vLLM是目前最广泛部署的LLM serving框架（GitHub 30k+ stars）
- block size默认16，可根据序列长度分布调优
- 多模型/多LoRA serving需要配置adapter管理策略

## 面试考点
1. **Q: PagedAttention解决的核心问题是什么？** A: KV cache的内存碎片（固定大小预分配导致浪费）和内存膨胀
2. **Q: Block Table的数据结构？** A: 类似页表：logical block number → physical block number的映射，每个序列独立
3. **Q: Copy-on-Write（写时复制）在vLLM中如何使用？** A: beam search时，多个候选共享KV block，写入时才复制，节省内存
4. **Q: vLLM的主要性能指标？** A: 吞吐量（tokens/sec）、TTFT（首token延迟）、TPOT（每token延迟）
5. **Q: vLLM如何处理不同长度请求的batch？** A: Continuous batching：固定batch中的新请求在有空位时即时插入，无需等待
