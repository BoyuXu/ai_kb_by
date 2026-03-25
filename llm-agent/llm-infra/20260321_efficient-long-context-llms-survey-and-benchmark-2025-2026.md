# Efficient Long-Context LLMs: Survey and Benchmark 2025-2026

> 来源：arxiv | 日期：20260321 | 领域：llm-infra

## 问题定义

LLM 的上下文窗口从 GPT-3 的 2k token 扩展到 2025 年的 1M+ token（Gemini 1.5 Pro），带来了巨大的内存和计算挑战：

1. **KV Cache 爆炸**：1M token 上下文，LLaMA-70B，每 token KV Cache ~0.5MB，总计需要 500GB+（远超单卡甚至多卡显存）
2. **Attention 计算**：O(n²) 复杂度，1M token 时单次 Attention 需要 ~10^12 次乘加运算
3. **理解质量**：长上下文中的"Lost in the Middle"问题——模型往往只能有效利用首尾信息，中间内容被忽视

本文是 2025-2026 年长上下文 LLM 技术的系统综述，覆盖近 200 篇工作，建立统一评估基准。

## 核心方法与创新点

**[推断]** 综述覆盖的主要技术方向：

1. **高效注意力架构**：
   - 线性注意力（Linear Attention）：State Space Model（Mamba）近似 Attention，O(n)
   - 稀疏注意力（Sparse Attention）：只计算局部+全局 token 的注意力（BigBird/LongFormer）
   - 滑动窗口注意力（Sliding Window）：Mistral 的 SWA，局部 W 个 token 完整 Attention

2. **KV Cache 压缩**：
   - KV Cache 量化（INT4/INT8 KV）
   - KV Cache 驱逐（Eviction）：保留重要的 KV（StreamingLLM 的"注意力汇"保留策略）
   - KV Cache 合并（Merging）：将近似重复的 KV 合并

3. **位置编码外推**：
   - RoPE 的上下文扩展：YaRN、LongRoPE、NTK-aware Scaling
   - 无需重新训练的位置插值（PI）

4. **RAG vs Long Context**：
   - RAG（检索增强）：只取相关片段放入短上下文
   - 长上下文：全量信息端到端处理
   - 2025 年趋势：两者融合（RAG + 长上下文精读）

5. **测试基准（Benchmark）**：
   - RULER：多任务长上下文理解基准
   - LongBench v2（2025 升级版）
   - Loong：真实工业长文档理解任务

## 实验结论

**[推断]** 关键对比发现：
- 1M token 场景：MambaFormer 混合架构（Attention + SSM）内存减少 **~60%**，速度 **~3x**，效果与纯 Attention 相当
- KV Cache 量化到 INT4：性能损失 <1%，内存减少 **~4x**
- "Lost in the Middle" 问题：100k token 时，中间 50k 的内容有效利用率仅 **~40%**（首尾各 25k 约 **~80%**）
- RAG vs 128k Long Context：RAG 在精确检索任务上 **+15%**；长上下文在全文理解任务上 **+22%**

## 工程落地要点

1. **选型原则**：<32k token 用现有方案（FlashAttention + GQA 足够）；32k-128k 需要 KV Cache 量化；>128k 考虑 Hybrid Mamba-Attention 或分段 RAG
2. **KV Cache 调度**：超出显存的长上下文 KV Cache 需要 CPU offload（PCIe 带宽 ~64GB/s，影响延迟）或 KV Cache 驱逐（丢弃低注意力分数的历史 KV）
3. **分布式推理**：超长上下文需要 Sequence Parallelism（序列并行），将序列分段到不同 GPU，用 Ring Attention 通信
4. **预填充 vs 解码优化**：长上下文的预填充（Prefill）阶段计算量大，需要分离 Prefill 和 Decode 服务（Chunked Prefill/Disaggregated Serving）

## 面试考点

- Q: KV Cache 是什么？为什么是长上下文的主要瓶颈？
  A: KV Cache 是推理时缓存的注意力 Key/Value 矩阵，避免每次生成新 token 时重新计算历史 token 的 K/V。大小 = 2 × num_layers × num_kv_heads × seq_len × head_dim × dtype_bytes。长上下文时 KV Cache 线性增长，128k token × 70B 模型可达 ~100GB，远超单卡显存。

- Q: Mamba 和 Transformer Attention 的比较？
  A: Mamba 基于 SSM（状态空间模型）：固定大小隐状态（O(1) 推理内存）、O(n) 训练和推理复杂度、不需要 KV Cache。代价：随机访问能力弱（无法精确回忆长距离特定 token），对精确匹配任务效果不如 Attention。混合架构（部分 Attention + 部分 SSM）是 2024-2025 年的主流方向。

- Q: "Lost in the Middle" 问题是什么？如何缓解？
  A: 长上下文中模型主要利用文档开头和结尾的信息，中间内容利用率低（注意力分数分散）。缓解方法：(1) 重排序（Reranking）将重要片段放到首尾；(2) 多次 Attention Pass（先全文一遍，重要片段再详读）；(3) 训练数据增强（随机置换文档顺序）；(4) 位置感知的注意力偏置。
