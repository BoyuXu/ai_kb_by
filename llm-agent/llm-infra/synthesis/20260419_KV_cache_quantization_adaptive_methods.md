# KV Cache自适应量化技术前沿 (2026-04-19)

## 核心趋势：从固定精度到自适应精度分配

### 今日相关论文

| 论文 | 核心方法 | 场景 | 关键结果 |
|------|---------|------|---------|
| Don't Waste Bits (2604.04722) | Token-wise自适应量化 | 端侧LLM | 延迟↓17.75%, 精度-0.30 vs FP16 |
| ARKV (2603.08727) | Per-layer自适应管理 | 长上下文 | 有限内存下质量保持 |
| VecInfer (2510.06175) | 离群值抑制+向量量化 | 通用推理 | 融合CUDA kernel优化 |

### 技术演进路径

```
统一量化 → 差异化量化 → 自适应量化 → 动态自适应量化
(INT8)     (Key高/Value低)  (Token重要性)   (实时解码决策)
```

### KV Cache优化方法全景

| 类别 | 代表方法 | 核心思想 |
|------|---------|---------|
| 缓存驱逐 | StreamingLLM, H2O | 淘汰低重要性token |
| 量化压缩 | KIVI, KVQuant, VecInfer | 降低精度存储 |
| 自适应量化 | Don't Waste Bits, ARKV | 按重要性分配精度 |
| 混合内存 | OffloadKV | CPU/GPU混合存储 |
| 新注意力机制 | GQA, MLA | 从架构层面减少KV |

### 核心技术细节

**Token重要性评估特征 (Don't Waste Bits)：**
1. Token频率 - 高频token通常更重要
2. Quality Score - 基于注意力分数的质量评分
3. Attention Variance - 注意力权重的方差（稳定 vs 波动）
4. Entropy-based Uncertainty - 基于熵的不确定性

**Key vs Value量化差异：**
- Key参与注意力分数计算 $\alpha = \text{softmax}(QK^T/\sqrt{d})$ → 对误差敏感 → 需更高精度
- Value参与加权求和 $\text{output} = \alpha V$ → 对误差更鲁棒 → 可更激进压缩

### 面试考点

**Q: KV Cache量化的核心挑战？**
1. 离群值处理（某些维度值极端大）
2. Key和Value需要不同策略
3. 量化粒度选择（per-token vs per-channel vs per-group）
4. 在线量化的计算开销

**Q: 如何评估KV Cache压缩的质量？**
- Perplexity保持度
- 下游任务准确率（如HellaSwag）
- 长文本Needle-in-a-Haystack测试
- 解码延迟和吞吐量

**Q: 端侧LLM推理的核心瓶颈？**
- 内存带宽（memory-bound）→ KV Cache量化直接减少数据传输量
- 而非计算量（compute-bound）→ 量化对延迟的改善来自带宽节省
