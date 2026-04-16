# Synthesis: 多 Agent 场景下的 KV Cache 优化

**生成日期：** 2026-04-16
**涵盖论文：** FreeKV (2505.13109), RelayCaching (2603.13289), PrefillShare (2602.12029), KV Cache Optimization Survey (2603.20397), Taming the Titans Survey (2504.19720)

---

## 1. 技术演进路径

```
静态 KV Cache → PagedAttention → 压缩/量化 → 投机检索 → 跨 Agent 复用
  (朴素缓存)      (vLLM)        (RocketKV)    (FreeKV)   (RelayCaching/PrefillShare)
```

**从单模型到多 Agent：**
- 早期 KV Cache 优化关注单模型推理效率
- 2026 年趋势：多 Agent LLM 系统催生跨模型/跨阶段 KV 复用需求

## 2. 核心方法对比

| 方法 | 场景 | 核心技术 | 加速倍数 | 精度影响 |
|------|------|---------|---------|---------|
| FreeKV | 单模型长上下文 | 投机检索 + 混合内存布局 | 最高 13× | 近无损 |
| RelayCaching | 多 Agent 协作 | decode KV 跨阶段复用 | 最高 4.7× TTFT | 与完整 prefill 相当 |
| PrefillShare | 多模型 Agent | 共享 prefill 跨模型复用 | 降低整体计算量 | 需验证跨模型兼容 |

## 3. 核心技术详解

### 3.1 FreeKV 的投机检索

```
Step t: query_t → 选择 KV 子集 → 计算注意力 → output_t
Step t+1: query_{t+1} ≈ query_t（高相似性假设）
    → 复用 step t 的 KV 选择结果（移出关键路径）
    → 异步校正：检测复用误差 → 细粒度修正
```

**系统优化：**
```
CPU Memory ←→ GPU Memory
  [冷 KV]      [热 KV]
     ↑              ↑
  双缓冲流式召回（预取下一步需要的 KV）
```

### 3.2 RelayCaching 的 U 型层分析

**关键发现：** decode KV 与 prefill KV 的偏差在层维度上呈 U 型分布
- 底层和顶层偏差较大
- 中间层高度一致，可直接复用

```
层偏差分布：
高 |  *                                *
   |   *                             *
   |    *                          *
低 |      * * * * * * * * * * * *
   +------------------------------------
     L1  L2  ...  L_mid  ...  L_{N-1}  L_N
```

**校正策略：** 仅对高偏差层进行选择性 prefill，其余层直接复用

### 3.3 PrefillShare 的跨模型架构

```
Prompt "请分析这段文本..."
    ↓
[共享 Prefill 模块] → KV Cache
    ↓                    ↓              ↓
[Decoder A]         [Decoder B]     [Decoder C]
(摘要模型)          (分类模型)       (翻译模型)
```

## 4. KV Cache 优化策略全景（Survey 2603.20397）

### Token 级策略
- **选择（Eviction）：** 基于注意力分数淘汰低重要性 token（StreamingLLM, H2O）
- **预算分配：** 动态调整每层 KV 保留数量
- **合并（Merging）：** 相似 KV 向量聚合，减少总数
- **量化：** INT4/INT8 量化 KV 值（KIVI, QServe）
- **低秩分解：** KV 矩阵低秩近似

### 模型级策略
- **GQA/MQA：** 多 query 共享 key-value head
- **线性注意力：** O(n) 复杂度替代 O(n²)
- **稀疏注意力：** 仅计算局部/滑动窗口注意力

### 系统级策略
- **PagedAttention：** KV Cache 分页管理（vLLM）
- **Offloading：** GPU ↔ CPU ↔ SSD 分级存储
- **分离式推理：** prefill 和 decode 独立部署
- **前缀缓存：** 相同 system prompt 的 KV 复用

## 5. 工业实践要点

### 5.1 多 Agent LLM 部署架构
```
用户请求 → Agent Orchestrator
    → Agent 1 (prefill + decode) → output_1
    → Agent 2 (RelayCaching: 复用 Agent 1 的 KV + 增量 prefill) → output_2
    → Agent 3 (PrefillShare: 复用共享 prefill KV) → output_3
```

### 5.2 选型决策树
```
问：系统是单模型还是多 Agent？
├── 单模型 → 长上下文？
│   ├── 是 → FreeKV（投机检索 + 混合内存）
│   └── 否 → PagedAttention + KV 量化
└── 多 Agent → 同一模型还是不同模型？
    ├── 同一模型 → RelayCaching（decode KV 复用）
    └── 不同模型 → PrefillShare（共享 prefill）
```

### 5.3 关键工程指标
- **TTFT（Time to First Token）：** 受 prefill 阶段影响，RelayCaching 优化目标
- **TPS（Tokens Per Second）：** 受 decode 阶段影响，FreeKV 优化目标
- **内存效率：** KV Cache 占用 vs GPU 显存总量
- **SLO 达成率：** P99 延迟是否满足服务水平目标

## 6. 面试考点总结

1. **KV Cache 为什么是 LLM 推理瓶颈？**
   - 线性增长的内存占用（序列长度 × 层数 × 头数 × 维度 × 2）
   - decode 阶段 memory-bound，KV 读取是瓶颈

2. **FreeKV 的投机检索为什么有效？**
   - 相邻 step 的 query 向量高度相似（自回归生成的特性）
   - 所以上一步选出的 important KV 在当前步大概率仍然 important

3. **RelayCaching 的 U 型偏差如何解释？**
   - 底层关注局部语法特征，对前缀敏感
   - 顶层关注全局语义，对上下文变化敏感
   - 中间层提取通用表征，对前缀不敏感

4. **多 Agent 系统 KV 复用的前提条件？**
   - 共享内容占比足够高
   - 模型架构兼容（同一 tokenizer + 类似注意力结构）
   - 偏差在可校正范围内

5. **分离式推理（disaggregated inference）的核心优势？**
   - prefill 是 compute-bound，decode 是 memory-bound
   - 分离部署可以为每个阶段选择最优硬件配置

---

*本 synthesis 文档由 MelonEgg 每日学习自动生成，覆盖 2026-04-16 llm-infra 领域 5 篇核心论文*
