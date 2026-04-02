# KVCache Compression for Long-Context LLM Inference: Methods and Benchmarks
> 来源：https://arxiv.org/search/?query=KV+cache+compression+long+context+inference&searchtype=all | 领域：llm-infra | 日期：20260323

## 问题定义
LLM的KV cache是长上下文推理的内存瓶颈（128K tokens约需要数十GB显存）。本文综述KV cache压缩方法，包括量化、驱逐（eviction）和稀疏化等策略，并提供统一benchmark。

## 核心方法与创新点
- 分类框架：量化压缩（INT8/INT4 KV）、token驱逐（丢弃不重要token的KV）、稀疏注意力（只计算部分KV对）
- 重要性评估：基于注意力分数识别重要KV，保留关键信息
- 流式KV：H2O/StreamingLLM等方法，固定KV cache大小实现无限上下文
- 分层压缩：不同层使用不同压缩比（浅层比深层更重要）

## 实验结论
KV INT8量化：质量几乎无损（perplexity变化<0.5），显存减少50%。Token驱逐（H2O保留20% token）：短任务质量基本保持，长文档任务质量下降约3%。组合方案（量化+驱逐）可减少显存80%，质量损失约5%。

## 工程落地要点
- KV量化是最安全的选择，几乎无质量损失，建议生产环境默认开启
- H2O等驱逐方法对注意力sink（开头token）要保留，否则质量崩溃
- 不同任务的KV重要性分布不同，需要任务自适应的驱逐策略

## 常见考点
1. **Q: KV cache的内存计算公式？** A: 2（K+V）× layers × heads × head_dim × seq_len × precision_bytes
2. **Q: H2O（Heavy Hitter Oracle）的核心思想？** A: 保留累积注意力分数最高的token的KV（这些是"heavy hitter"）
3. **Q: Attention sink是什么？为什么必须保留？** A: 开头几个token（通常是BOS/system prompt）几乎被所有后续token高度关注
4. **Q: KV cache量化（INT8/INT4）的特殊挑战？** A: KV激活值分布irregular（outlier多），需要per-token或per-channel量化
5. **Q: PagedAttention（vLLM）如何管理KV cache？** A: 类似OS页表，将KV cache分成固定大小的页（block），虚拟连续实际分散
