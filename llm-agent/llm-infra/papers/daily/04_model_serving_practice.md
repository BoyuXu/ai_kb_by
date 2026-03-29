# 模型 Serving / 推理优化工程实践

> 更新时间：2026-03-13 | 面向算法工程师面试

---

## 核心概念

### 1. 推理优化的核心目标

推理优化围绕三个核心指标展开：
- **吞吐量（Throughput）**：单位时间处理的请求数（tokens/s）
- **延迟（Latency）**：首 token 延迟（TTFT）和每 token 延迟（TPOT）
- **显存效率（Memory Efficiency）**：GPU HBM 的利用率，推理瓶颈往往在显存带宽而非算力

推理分为两个阶段：
- **Prefill 阶段**：处理 prompt，计算密集（compute-bound）
- **Decode 阶段**：自回归生成 token，带宽密集（memory-bound）

### 2. KV Cache 与 PagedAttention

**KV Cache** 是推理优化的核心机制：将 Attention 计算中的 Key/Value 矩阵缓存，避免重复计算。

**问题**：传统 KV Cache 存在严重的显存碎片化——预分配最大长度导致 60%-80% 显存浪费。

**PagedAttention（vLLM 核心创新）**：
- 借鉴操作系统虚拟内存分页机制
- 将 KV Cache 切分为固定大小的 block（默认 16 tokens/block）
- 通过 block table 实现非连续物理显存的逻辑连续映射
- 支持多个请求的 KV Cache 共享（Prefix Caching）
- 显存利用率从 <40% 提升到 >90%，吞吐量提升 2-4x

```
传统 KV Cache:  [  padding  |  actual tokens  ]  → 大量浪费
PagedAttention: [ block1 | block2 | block3 ] → 按需分配，支持共享
```

### 3. 模型量化（Quantization）

| 方法 | 精度 | 显存节省 | 速度提升 | 适用场景 |
|------|------|----------|----------|----------|
| FP16/BF16 | 高 | baseline | baseline | 在线服务默认 |
| INT8（W8A8）| 略降 | 50% | 1.5-2x | 对话/推荐 |
| INT4（GPTQ/AWQ）| 中等损失 | 75% | 2-3x | 资源受限部署 |
| FP8 | 接近 FP16 | 50% | 1.5x | H100/H800 首选 |

**GPTQ**：逐层量化，使用二阶信息（Hessian）最小化量化误差  
**AWQ**：激活感知量化，保护重要权重通道不量化  
**SmoothQuant**：将激活的量化难度迁移到权重上，通过逐通道缩放解决激活异常值问题

### 4. 模型并行策略

**Tensor Parallelism（TP）**：
- 将权重矩阵按列/行切分到多卡
- 适合单机多卡（NVLink 通信开销小）
- Megatron-LM 实现：每层通信 2 次 All-Reduce

**Pipeline Parallelism（PP）**：
- 将 Transformer 层分配到不同设备
- 存在 pipeline bubble（空泡），影响 GPU 利用率
- 适合多机多卡（网络带宽是瓶颈）

**Data Parallelism（DP）**：
- 多副本处理不同 batch
- 适合超大 batch 吞吐场景

**实际部署通常组合使用**：如 TP=8（单机）+ PP=4（多机）的混合并行

### 5. 推测解码（Speculative Decoding）

核心思想：用小模型（draft model）快速生成候选 token，再用大模型并行验证，接受或拒绝。

- **加速比**：理论 2-4x，实测 1.5-3x（取决于接受率 α）
- **无损**：数学上等价于原始大模型的分布
- **变体**：Medusa（多头解码）、EAGLE（树形推测）、LoRA Speculative Decoding

---

## 工程实践

### 主流推理框架对比

| 框架 | 核心优势 | 适用场景 |
|------|---------|---------|
| **vLLM** | PagedAttention、高吞吐、OpenAI 兼容 API | 在线服务，LLM 推理首选 |
| **TensorRT-LLM** | NVIDIA 深度优化，FP8 支持最好 | NVIDIA GPU，极致性能 |
| **SGLang** | RadixAttention，复杂提示共享，速度快 | RAG/Agent 场景 |
| **DeepSpeed-MII** | ZeRO 推理，极大模型支持 | 百亿参数以上 |
| **Ollama** | 开箱即用，CPU 支持 | 本地开发/小规模部署 |

### vLLM 典型部署配置

```bash
# 启动 vLLM OpenAI 兼容 Server
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 2 \      # TP=2，双卡
    --gpu-memory-utilization 0.90 \ # 90% 显存给 KV Cache
    --max-model-len 8192 \          # 最大上下文长度
    --enable-prefix-caching \       # 开启前缀缓存（System Prompt 共享）
    --quantization fp8              # FP8 量化
```

### 推荐系统模型推理特殊考虑

推荐系统与 LLM 推理有显著不同：

1. **Embedding 表极大**：工业级 Embedding 表可达 TB 级，需要 CPU/GPU 异构存储
2. **稀疏特征查找**：特征 ID 稀疏，显存带宽是瓶颈而非算力
3. **请求并发高**：推荐 QPS 通常远高于 LLM（万级 vs 百级）

**工业实践**：
- Meta DLRM：将 Embedding 表放 CPU，Dense 特征放 GPU，异步通信
- ByteDance：Parameter Server 架构 + GPU Dense 层
- Triton Inference Server：支持 Ensemble，将 Embedding 查找与 DNN 计算分离

### TorchServe / Triton Inference Server

```
Client → Load Balancer → [Triton Server × N]
                              ↓
                    [TensorRT Engine | ONNX Runtime | PyTorch]
                              ↓
                    [Dynamic Batching / Sequence Batching]
```

**Triton 关键特性**：
- Dynamic Batching：自动合并小请求为大 batch，提升 GPU 利用率
- Model Ensemble：多模型流水线（如 Embedding → Rank → Rerank）
- 支持 gRPC/HTTP，Prometheus 监控集成

### 关键性能指标与 SLA

```
在线推荐场景 SLA 参考：
- P99 延迟 < 50ms（召回）
- P99 延迟 < 100ms（排序）
- 吞吐量 > 5000 QPS/节点

LLM 对话场景 SLA 参考：
- TTFT < 500ms
- TPOT < 100ms/token
- 并发用户 > 100/节点
```

---

## 面试高频考点

### Q1：vLLM 的 PagedAttention 核心原理是什么？它解决了什么问题？

**A**：
PagedAttention 解决了传统 KV Cache 显存碎片化问题。传统方案需要为每个请求预分配最大长度的连续显存，即使实际序列很短也浪费大量空间，且不同请求的 KV Cache 无法共享。

PagedAttention 借鉴操作系统虚拟内存分页的思想：
1. 将 KV Cache 划分为固定大小（如 16 tokens）的 block
2. 通过 block table 实现逻辑地址到物理 block 的映射
3. 不同请求可以共享相同前缀的 KV block（Prefix Caching）
4. 显存利用率从 <40% 提升到 >90%，吞吐量提升 2-4x

### Q2：模型量化的原理是什么？INT4 和 INT8 各有什么优缺点？

**A**：
量化的本质是用低精度（如 INT8/INT4）近似表示高精度（FP32/FP16）的权重和激活值。

- **INT8（W8A8）**：权重和激活都量化到 8 bit，精度损失较小（<1% PPL 提升），推理速度提升约 1.5-2x，适合生产环境
- **INT4（GPTQ/AWQ）**：显存压缩 4x，但精度损失较大，适合资源受限场景或对精度不敏感的任务
- **AWQ 的优势**：观察到只有约 0.1% 的权重对精度至关重要，对这些"显著权重"保持高精度，其余量化到 INT4，精度损失很小

关键：量化 LLM 时激活值比权重更难量化（异常值问题），SmoothQuant 通过将激活的"量化难度"迁移到权重来解决此问题。

### Q3：推理时 Tensor Parallelism 和 Pipeline Parallelism 各适合什么场景？

**A**：
- **Tensor Parallelism（TP）**：把单层的权重矩阵切分到多卡，每层都需要通信（All-Reduce）。通信频繁，要求卡间带宽高（NVLink）。优点：延迟低，无 pipeline bubble。适合**单机多卡**场景。

- **Pipeline Parallelism（PP）**：把不同层分配到不同设备，通信只在层间边界发生（发送 activation），通信量少。缺点：存在流水线空泡，需要 micro-batch 技术减少气泡。适合**多机多卡**、跨机器带宽受限场景。

实际部署通常混合使用：如 70B 模型，单机 8 卡 TP=8，多机 PP=2，再加 DP=N 扩容。

### Q4：推测解码（Speculative Decoding）的原理及局限性？

**A**：
**原理**：用小 draft 模型自回归生成 γ 个候选 token，大 target 模型并行（一次前向）验证所有候选。通过拒绝采样保证输出分布等价于直接用大模型生成。

**加速原理**：大模型 1 次前向 = 验证 γ 个 token，若接受率 α 高，等效减少了 forward pass 次数。

**局限性**：
1. 接受率 α 依赖任务，代码生成/翻译等高度确定性任务效果好，创意写作效果差
2. 需要维护额外的 draft 模型，增加显存开销
3. 若 draft 和 target 分布差异大，接受率低，可能不如直接推理

### Q5：推荐系统在线推理的延迟优化有哪些关键手段？

**A**：
1. **模型结构优化**：MobileNet/轻量级骨干、知识蒸馏（大模型 → 小模型）
2. **算子融合**：将多个连续算子（如 BN+ReLU）融合为单一 CUDA kernel，减少显存读写
3. **Embedding 查找优化**：
   - Embedding 压缩（减少 embedding_dim 或量化）
   - 热点 Embedding 缓存到 GPU L2 Cache
   - Embedding 查找与 DNN 计算异步 overlap
4. **批处理（Batching）**：Dynamic Batching 将并发请求合并，提升 GPU 利用率
5. **级联排序**：粗排用轻量模型，精排用复杂模型，通过漏斗结构减少候选量
6. **预计算与缓存**：用户/物品侧特征提前计算并缓存（Feature Store）

### Q6：Continuous Batching 与 Static Batching 的区别？

**A**：
- **Static Batching**：等一个 batch 的所有请求都完成才处理下一个 batch。问题：长请求拖累整个 batch，GPU 空转等待。

- **Continuous Batching（iteration-level scheduling）**：每个 decode step 都可以动态添加新请求、移除完成请求。短请求完成后立即插入新请求填充 GPU，不等待长请求。

效果：GPU 利用率从 ~50% 提升到 >80%，吞吐量提升 2-5x。vLLM、TensorRT-LLM 均实现了 Continuous Batching。

### Q7：如何监控和评估线上推理服务的健康状态？

**A**：
关键监控指标：
1. **延迟指标**：P50/P95/P99 延迟，TTFT，TPOT（LLM）
2. **吞吐量**：RPS（requests/s），tokens/s
3. **资源利用率**：GPU 利用率，显存使用量，KV Cache 命中率
4. **错误率**：超时率，OOM 率，服务降级率
5. **队列深度**：请求积压量，反映负载水位

工具链：Prometheus + Grafana 监控，Jaeger 链路追踪，vLLM 内置 `/metrics` 端点暴露 OpenMetrics 格式指标。

---

## 参考资料

1. **Efficient Memory Management for Large Language Model Serving with PagedAttention** - vLLM Paper (SOSP 2023)
   - https://arxiv.org/abs/2309.06180

2. **vLLM 官方文档**
   - https://docs.vllm.ai/

3. **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration**
   - https://arxiv.org/abs/2306.00978

4. **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models**
   - https://arxiv.org/abs/2211.10438

5. **Grouped-Query Attention (GQA)**
   - https://arxiv.org/abs/2305.13245

6. **Chip Huyen - Real-time machine learning: challenges and solutions**
   - https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html

7. **TensorRT-LLM 文档**
   - https://nvidia.github.io/TensorRT-LLM/

8. **Monolith: Real Time Recommendation System With Collisionless Embedding Table** (ByteDance)
   - https://arxiv.org/abs/2209.07663
