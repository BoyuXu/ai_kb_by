# LLM 基础设施论文笔记 — 2026-04-16

## 1. FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference

**来源：** https://arxiv.org/abs/2505.13109 (May 2025, v3 Mar 2026)
**领域：** KV Cache 优化 × 推理加速
**核心问题：** 长上下文 LLM 推理中 KV Cache 检索成为瓶颈，现有方法在选择和召回过程中阻塞推理

**FreeKV 算法-系统协同优化：**

**算法层面：**
- **投机检索（Speculative Retrieval）：** 利用相邻解码步骤 query 向量的高相似性，将 KV 选择和召回移出关键路径
- **细粒度校正：** 对纯 KV 复用可能的精度损失进行修正，保持模型准确性

**系统层面：**
- **混合 KV 布局：** CPU/GPU 内存混合存储，消除碎片化数据传输
- **双缓冲流式召回：** 计算与数据传输重叠，实现完全延迟隐藏

**性能结果：**
- 在多种场景和模型上实现近无损精度
- 对比 SOTA KV 检索方法加速最高 13×
- Qwen-2.5-7B: 长输入/长生成对比 ShadowKV 加速 2.9×/4.9×
- Llama-3.1-8B: 加速 5×/8.4×

**面试考点：** KV Cache 的内存瓶颈分析、投机执行在推理中的应用、CPU-GPU 混合内存架构

---

## 2. RelayCaching: Accelerating LLM Collaboration via Decoding KV Cache Reuse

**来源：** https://arxiv.org/abs/2603.13289 (Feb 2026)
**领域：** 多 Agent LLM × KV Cache 复用
**核心问题：** 多 Agent LLM 系统中，前置 Agent 生成的内容被后续 Agent 重新 prefill，造成冗余计算和 TTFT 增加

**核心洞察：** 相同内容在 prefill 和 decode 阶段产生的 KV cache 高度一致，差异稀疏且局部化

**RelayCaching 两核心组件：**
1. **层范围分析器（Layer-Range Profiler）：** 利用 U 型分布识别关键层范围，基于层间相关性选择检测层
2. **Token 选择器（Token Selector）：** 结合偏差驱动和影响力驱动的选择策略，精准定位需要校正的 token

**性能：**
- 保持与完整 prefill 相当的生成质量
- KV cache 复用率超 80%
- 最高 4.7× TTFT 加速

**面试考点：** 多 Agent LLM 的 KV cache 冗余问题、prefill vs decode 阶段 KV cache 差异分析、训练无关方法的优势

---

## 3. PrefillShare: A Shared Prefill Module for KV Reuse in Multi-LLM Disaggregated Serving

**来源：** https://arxiv.org/abs/2602.12029 (Feb 2026)
**领域：** 分离式推理 × 多模型 KV 复用
**核心问题：** 多模型 Agent 工作负载中，相同 prompt 被多个模型分别 prefill，浪费 GPU 资源

**PrefillShare 方案：**
- 将共享 prefill 模块从任务特定 decoder 中解耦
- 同一 prompt 的 KV cache 跨模型复用
- 在相同总 GPU 预算下，共享 prefill 降低整体计算量

**适用场景：** Multi-Agent 系统中多个 LLM 处理相同上下文的场景（如 AutoGen/CrewAI 工作流）

**面试考点：** 分离式推理架构设计、跨模型 KV cache 兼容性、多 Agent 推理的资源调度

---

## 4. KV Cache Optimization Strategies for Scalable and Efficient LLM Inference

**来源：** https://arxiv.org/abs/2603.20397 (Mar 2026)
**领域：** KV Cache 优化综述
**核心定位：** 系统性分类 KV Cache 优化策略

**三层优化分类：**
1. **Token 级优化：**
   - KV Cache 选择（eviction/selection）
   - 预算分配（budget allocation）
   - 合并（merging）
   - 量化（quantization）
   - 低秩分解（low-rank decomposition）
2. **模型级优化：**
   - 注意力头共享/分组（GQA/MQA）
   - 架构改进（线性注意力、稀疏注意力）
3. **系统级优化：**
   - 内存管理（PagedAttention/vLLM）
   - 分离式推理（prefill-decode disaggregation）
   - 缓存调度和预取

**面试考点：** KV Cache 优化策略的分类框架、Token 级 vs 模型级 vs 系统级的适用场景、量化对模型质量的影响

---

## 5. Taming the Titans: A Survey of Efficient LLM Inference Serving

**来源：** https://arxiv.org/abs/2504.19720 (Apr 2025, v1)
**领域：** LLM 推理服务综述
**核心定位：** 全面综述 LLM 推理服务的效率优化技术

**核心主题：**
- **Prefill 优化：** chunked prefill、prefix caching、prompt compression
- **Decode 优化：** speculative decoding、continuous batching、early exit
- **KV Cache 管理：** PagedAttention、offloading、压缩
- **分离式推理：** prefill/decode 分离部署
- **MoE 推理：** 专家调度、负载均衡、通信优化
- **硬件优化：** 量化、kernel 融合、异构计算

**关键指标：** TTFT（首 token 延迟）、TPS（每秒 token 数）、吞吐量、SLO 达成率

**面试考点：** LLM 推理的 prefill/decode 两阶段特性、continuous batching 原理、speculative decoding 的正确性保证

---

**今日 llm-infra 总结：** 5 篇论文聚焦 KV Cache 优化和多 Agent 推理加速。关键趋势：(1) KV Cache 复用从单模型扩展到多 Agent 系统（RelayCaching/PrefillShare）；(2) 算法-系统协同优化成为主流（FreeKV）；(3) 分离式推理架构持续演进，从单模型到多模型场景。
