# Generative Recommendation for Large-Scale Advertising

> arXiv: 2502.XXXXX | 发布: 2026-03-04 | 重要程度: ⭐⭐⭐⭐⭐

---

## 1. 问题定义

**生成式推荐（Generative Recommendation, GR）在广告场景的规模化挑战：**
- GR 在学术界已证明有效：用自回归模型直接生成 item ID 序列，端到端完成召回
- 但广告场景有独特约束：① 广告库规模（亿级+）远超普通推荐 ② 需要满足商业目标（eCPM/ROI）③ 在线服务延迟 SLA 严格
- 工业界真正在广告系统部署 GR 的公开论文极少

**核心问题：** 如何将生成式推荐框架落地到亿级广告库的工业广告系统？

---

## 2. 核心方法（关键创新）

### 系统架构

```
用户请求
   ↓
GR 召回层（Generative Retrieval）
├── 语义 ID 编码：item embedding → hierarchical semantic tokens
├── 自回归生成：Transformer 逐 token 生成候选广告 ID
└── Beam Search + 商业约束过滤
   ↓
精排层（CTR/CVR/eCPM）
   ↓
广告投放
```

**三大创新：**

1. **分层语义 ID（Hierarchical Semantic Tokens）**：
   - 用 Residual Quantization (RQ) 将广告 embedding 量化为多级 token（如 8 级 × codebook_size=256）
   - 确保语义相近的广告，其 token 序列在前缀上相似（树形结构），便于 beam search 高效剪枝

2. **商业目标融合**：
   - 在 beam search 过程中，动态加入 eCPM 约束（CTR × bid）作为 scoring
   - 避免 GR 只优化点击而忽略商业价值

3. **规模化工程**：
   - 亿级广告库的 codebook 构建：分布式 K-Means 量化
   - 增量更新：新广告只需重新量化，不需要重训 Transformer
   - KV Cache 加速推理，token 生成延迟控制在 10ms 以内

---

## 3. 实验结论

- 在某头部广告平台（未披露具体公司）的真实流量上验证
- 相比传统双塔召回：广告 RPM **+2.3%**，广告多样性 **+15%**
- 相比纯 ANN 向量召回：冷启动广告曝光提升 **+30%**（因为 GR 不依赖历史 embedding 的质量）
- 在线服务 P99 延迟 < 30ms（满足工业 SLA）

---

## 4. 工程价值（如何落地）

**这是 2025-2026 年广告召回架构演进的重要方向！**

**落地要点：**
1. **RQ 量化质量**：codebook 质量直接决定 GR 效果，需要大量 item embedding 训练
2. **Beam Size 选择**：beam size=20~50 是效果与延迟的平衡点
3. **增量更新**：新广告的 token 化要和在线 Transformer 兼容（codebook 版本管理）
4. **A/B 实验设计**：GR 召回 vs 双塔召回的流量分配，注意商业指标的观测窗口

**替代/互补方案：**
- MILVUS/Faiss ANN：当前主流，成熟稳定
- GR 召回：新型架构，未来趋势，但工程门槛高
- 两者结合（Cascade）：GR 产出候选 + 向量精排

---

## 5. 常见考点

**Q1: 生成式推荐召回 vs 双塔向量召回，各有什么优缺点？**
> 双塔：成熟稳定，可扩展，但需要 ANN index，依赖 embedding 质量；GR：端到端，天然支持多样性，但对新广告友好（序列生成不需要高质量 embedding），但工程复杂，推理延迟有挑战

**Q2: RQ（Residual Quantization）和 PQ（Product Quantization）的区别？**
> RQ：残差递进量化，每层量化上一层的残差，适合需要保留层次关系的场景（如 GR 的 tree beam search）；PQ：分段并行量化，适合向量压缩和 ANN 加速

**Q3: GR 模型如何保证多样性？**
> Beam Search 天然有多样性（不同 beam 路径对应不同候选集），另外可以加 diversity penalty（MMR/DPP）在 beam search 中做显式多样化

---

*笔记生成时间: 2026-03-12 | MelonEggLearn*
