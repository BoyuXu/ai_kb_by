# Prefill-Decode 分离的全谱演进 — 2026-05-19

> 综合 3 篇当日学习论文：TaiChi、Multi-Vendor P-D、Nexus

## 一、技术演进

LLM 推理天然分两阶段：

- **Prefill：** compute-bound，处理 prompt 中所有 token 并算出 KV cache
- **Decode：** memory-bound，每 step 只算一个 token，但要访问全部 KV cache

两阶段同 GPU 时互相干扰（Chunked Prefill 范式部分缓解）；解法逐步演进：

| 阶段 | 方案 | 时间 |
|------|------|------|
| 同 GPU 顺序 | vLLM continuous batching | 2023 |
| Chunked Prefill | Sarathi-Serve | 2024.03 |
| **跨 GPU 分离** | DistServe / Splitwise / Mooncake | 2024.06 起 |
| **统一架构** | TaiChi（Aug 2025） | 2025.08 |
| **异构 GPU 分离** | Multi-Vendor P-D（Sep 2025） | 2025.09 |
| **单 GPU 内分区** | Nexus（Jul 2025） | 2025.07 |

三种新范式覆盖三类场景：SLO 灵活调配（TaiChi）、异构成本优化（Multi-Vendor）、小集群微观调度（Nexus）。

## 二、核心公式

**1. Goodput 定义：**

Goodput = 每秒满足 (TTFT ≤ S_ttft) ∧ (TPOT ≤ S_tpot) 的请求数

是工业 serving 的真实优化目标，而非纯 throughput。

**2. TaiChi 的 Latency Shifting 决策：**

对每个未完成请求 i，定义 slack_i = (S_ttft − TTFT_predicted_i) + (S_tpot − TPOT_predicted_i)；把 slack > 0 的请求 GPU 配额向 slack < 0 的请求转移。

**3. Nexus 的 SM 分区代价模型：**

minimize α·max(TTFT_violation) + β·max(TPOT_violation)
s.t.  SM_prefill + SM_decode ≤ SM_total

通过 greedy search 在 ms 级求解。

**4. 异构 PD 部署的算力 / 带宽匹配：**

对 model size M、context length L：
- Prefill arithmetic intensity ≈ 2·M·L → 选 compute-strong GPU（H100 / GH200）
- Decode arithmetic intensity ≈ 2·M / batch → 选 memory-bandwidth-strong GPU（MI250 / 国产 HBM）

## 三、工业实践

**部署决策树：**

1. **同质大集群、SLO 紧 TTFT：** PD Aggregation（chunked prefill）
2. **同质大集群、SLO 紧 TPOT：** PD Disaggregation 跨 GPU（DistServe / TaiChi）
3. **同质大集群、SLO 均衡：** TaiChi 统一架构 + latency shifting
4. **异构 GPU 池：** Multi-Vendor P-D，prefill / decode 卡分工，配 HC 传输模块
5. **资源受限单 GPU：** Nexus 的 intra-GPU SM 分区

**KV Cache 传输路径优化：**

- 单机：NVLink / NVSwitch（H100 4.0 TB/s）
- 跨机：RDMA over InfiniBand 或 RoCEv2（200–800 Gbps）
- 异构：需精度对齐（FP8 ↔ BF16）+ layout 转换 + 分块流水

**实际收益参考：**

- TaiChi：balanced SLO 区 goodput +77%
- Nexus：vs vLLM throughput +2.2×, TTFT −20×, TBT −2.5×

## 四、面试考点

1. Prefill 与 Decode 为什么必须分开？两者的 arithmetic intensity 差多少？
2. Chunked Prefill 与 PD Disaggregation 的核心权衡是什么？
3. Goodput vs Throughput vs TPS 的区别？工业 serving SLO 一般定义为什么？
4. Multi-Vendor PD 中，KV cache 跨厂商 GPU 传输的工程挑战？
5. Nexus 的 SM 分区原理（CUDA Streams / MPS / MIG）？什么时候反而退化？
6. 在 latency 敏感 + 算力受限场景，PD 应该跨 GPU 分还是单 GPU 内分？

## 参考

- [TaiChi: Prefill-Decode Aggregation or Disaggregation?](https://arxiv.org/abs/2508.01989)
- [Disaggregated P-D Inference on Multi-Vendor GPUs](https://arxiv.org/abs/2509.17542)
- [Nexus: Proactive Intra-GPU Disaggregation](https://arxiv.org/abs/2507.06608)
