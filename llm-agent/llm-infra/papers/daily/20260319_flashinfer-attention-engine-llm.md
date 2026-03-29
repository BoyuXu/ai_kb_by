# FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving
> 来源：https://arxiv.org/abs/2501.01005 | 日期：20260319

## 问题定义
LLM推理服务需要处理多种复杂场景：不同长度的批次、不同注意力变体（GQA、MLA、滑动窗口注意力）、动态KV Cache管理（PagedAttention）。现有的注意力内核（如Flash Attention）难以覆盖所有场景，且不易定制。FlashInfer提供一个灵活、高性能、可定制的注意力计算引擎。

## 核心方法与创新点
1. **统一注意力抽象**：
   - 将注意力计算分解为：Q/K/V加载、注意力分数计算、Softmax、值聚合
   - 每个步骤可独立定制（如自定义mask、位置编码、注意力偏置）
   - 支持多种注意力变体：MHA、GQA、MQA、MLA（DeepSeek的多头潜在注意力）

2. **动态稀疏注意力**：
   - 支持基于内容的动态稀疏模式（只计算相关的Q-K对）
   - PagedKVCache原生支持：KV以页为单位存储，支持动态增删
   - 批次中不同长度的请求高效打包（batched prefill + decode）

3. **工程优化**：
   - 内核融合：将多个操作（RoPE + Attention + mask）融合为单一内核
   - 向量化：利用Tensor Core的矩阵运算加速
   - CPU-GPU Overlap：计算和数据传输并行

4. **Python API**：
   - 高级Python接口，支持快速原型
   - JIT编译：自定义注意力变体自动编译为高效CUDA内核

## 实验结论
- 在A100 GPU上，prefill阶段比标准PyTorch快约3-4倍
- decode阶段（单步生成）比vLLM的PagedAttention快约1.5-2倍
- 支持100K+上下文长度的注意力计算
- 批次吞吐量（tokens/s）提升约30-50%

## 工程落地要点
1. **vLLM集成**：FlashInfer已集成到vLLM，安装vLLM即可使用
2. **部署要求**：需要CUDA 11.8+，支持Hopper（H100）和Ampere（A100）架构
3. **模型适配**：新注意力变体（如MLA）需要验证FlashInfer是否支持或自定义内核
4. **性能调优**：根据实际批次大小和序列长度选择最优的分块大小

## 面试考点
Q1: PagedAttention（vLLM）是如何工作的？解决了什么问题？
> 传统KV Cache连续分配内存，不同请求长度不同导致内存碎片（最坏情况浪费约60%显存）。PagedAttention将KV Cache分成固定大小的页，按需分配，类似操作系统的虚拟内存分页。优势：内存利用率提升约55%，支持更大批次，吞吐量提升2-4倍。

Q2: GQA（Grouped Query Attention）和MQA（Multi-Query Attention）的原理？
> MHA（多头注意力）：每个head有独立的K、V。MQA：所有head共享一组K、V，减少KV Cache大小。GQA（分组查询注意力）：将heads分组，每组共享K、V，是MHA和MQA的折中。好处：减少KV Cache显存，加速解码，质量接近MHA。LLaMA-2-70B等大模型使用GQA。

Q3: LLM推理服务的主要性能瓶颈是什么？
> (1) Memory Bandwidth：decode阶段每步只生成1个token，GPU利用率低，主要受限于KV Cache读取带宽；(2) KV Cache显存：长上下文时KV Cache占用大量显存；(3) 批次延迟不均：不同请求长度不同，短请求要等长请求完成（head-of-line blocking）。解决方案：continuous batching（动态批次）、PagedAttention、推测解码。
