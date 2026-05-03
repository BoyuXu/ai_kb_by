# KV Cache 优化前沿（2026）

> 5 篇论文综合：DASH-KV + DepthKV + TTKV + KV Survey + LMCache

---

## 核心趋势

KV Cache 优化从简单驱逐策略进入**多维协同优化**时代：

1. **非均匀分配**：层间（DepthKV）+ 时间维度（TTKV）差异化对待
2. **算法范式转换**：从"压缩后计算"到"哈希后查找"（DASH-KV）
3. **系统级基础设施**：KV Cache 成为一等公民资源（LMCache 15x 吞吐）

---

## 论文精读

### 1. DASH-KV: 非对称哈希加速长上下文 (arxiv 2604.19351)

**ACL 2026 Findings**

**核心创新**：将 attention 重新定义为近似最近邻搜索（ANN），用哈希替代浮点运算

**方法**：
- **非对称编码**：Q 和 K 使用不同的哈希函数（Q 需要高精度，K 需要高效复用）
- **深度哈希**：可学习的哈希函数，不是 LSH
- **动态混合精度**：关键 token 保留全精度，其余用哈希近似

**结果**：
- LongBench (NarrativeQA, HotpotQA, Qasper) 上显著优于 SOTA 基线
- 质量接近 full attention
- 测试：Qwen2-7B, LLaMA-3.1-8B, Qwen2.5-14B

**面试考点**：
- 为什么 Q 和 K 需要不同编码？→ Q 只用一次需要精度，K 被多次复用需要效率
- 与 Flash Attention 的区别？→ FA 优化 IO，DASH-KV 优化计算本身
- 参考：[[KVCache与LLM推理优化全景.md]]

---

### 2. DepthKV: 层间差异化 KV 剪枝 (arxiv 2604.24647)

**核心洞察**：不同层对 KV 剪枝的敏感度差异巨大

**方法**：
- 固定全局 KV 预算，按层敏感度分配
- 支持三种策略：位置保护（保留中间层）、指标引导、混合策略
- **Plug-and-play**：可与任意 base 剪枝方法组合

**关键公式**：
给定总预算 $B$ 和 $L$ 层，每层分配 $b_l$ 满足 $\sum_l b_l = B$，按敏感度 $s_l$ 分配：
$$b_l = B \cdot \frac{s_l}{\sum_{l'} s_{l'}}$$

**结果**：激进压缩比下增益最大（说明均匀剪枝在高压缩时损失信息不均匀）

**面试考点**：为什么不同层敏感度不同？→ 浅层捕获局部模式可压缩，深层捕获全局语义需保留

---

### 3. TTKV: 时间分层 KV Cache (arxiv 2604.19769)

**核心创新**：将人类记忆模型映射到 KV Cache 管理

**生物启发设计**：
| 时间层 | 人类记忆 | KV Cache | 存储 | 精度 |
|--------|---------|---------|------|------|
| 近期 | 短期记忆 | 最近 token | HBM | FP16 |
| 中期 | 工作记忆 | 中间段 | HBM | INT8 |
| 远期 | 长期记忆 | 早期 token | DRAM | INT4 |

**方法**：
- 分层容量 + 分层精度 + 分层存储位置
- Block-wise streaming attention 重叠通信和计算
- HBM/DRAM 异构硬件利用

**面试考点**：为什么时间维度是合理的分层依据？→ LLM 注意力分布通常呈 recency bias

---

### 4. KV Cache Optimization Survey (arxiv 2603.20397)

**来源**：Dell Technologies（工业视角）

**五轴分类**：
| 方向 | 代表方法 | 核心思想 |
|------|---------|---------|
| Cache Eviction | H2O, StreamingLLM | 丢弃不重要 token |
| Cache Compression | KIVI, MiniKV | 量化/低秩 |
| Hybrid Memory | CPU/Disk offloading | 分层存储 |
| Novel Attention | Sparse/Linear | 改变计算模式 |
| Combination | 多策略组合 | 驱逐+压缩+卸载 |

**核心结论**：生产环境最佳效果来自**多策略组合**

---

### 5. LMCache: 企业级 KV Cache 管理层 (arxiv 2510.09665)

**核心创新**：首个生产级开源 KV Cache 管理层

**架构**：
```
LLM Engine (vLLM/SGLang)
        ↕ Connector API
    LMCache Layer
        ↕
GPU ↔ CPU ↔ Disk ↔ Redis ↔ RDMA
```

**两大能力**：
1. **前缀复用**（跨请求）：相同 system prompt 的 KV 共享
2. **Prefill-Decode 解耦**（跨引擎）：Prefill 引擎计算 KV，Decode 引擎消费

**结果**：多轮 QA 和文档分析场景 **15x 吞吐提升**

**面试考点**：
- 为什么 KV Cache 需要独立管理层？→ 跨请求/跨引擎复用是巨大优化空间
- 与 vLLM PagedAttention 的关系？→ PA 管理单引擎内分配，LMCache 管理跨引擎/跨存储
- 参考：[[KVCache与LLM推理优化全景.md]] | [[LLMServing系统实践.md]]

---

## 趋势总结

KV Cache 管理本质上是**内存层次结构问题**，类比 CPU Cache 设计：
- **DASH-KV**：改变计算模型（类比 SIMD 指令优化）
- **DepthKV + TTKV**：差异化分配（类比 Cache 替换策略）
- **LMCache**：系统级管理（类比 OS 虚拟内存）
- **五轴组合**：多策略协同（类比多级 Cache 协议）

生产部署最优解 = 量化 + 驱逐 + 分层存储 + 系统管理

→ 参考：[[KVCache与LLM推理优化全景.md]] | [[FlashAttention3与LLM推理基础设施.md]]
