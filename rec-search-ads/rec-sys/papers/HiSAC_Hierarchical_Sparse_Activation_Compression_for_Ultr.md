# HiSAC: Hierarchical Sparse Activation Compression for Ultra-long Sequence Modeling in Recommenders

> arXiv: 2502.XXXXX | 发布: 2026-02-24 | 重要程度: ⭐⭐⭐⭐⭐

---

## 1. 问题定义

**超长序列建模的工程瓶颈：**
- 用户历史行为序列是推荐系统最重要的信号，序列越长效果越好
- 但 Transformer Attention 复杂度 O(n²)，序列长度超过 1000 后，训练和推理成本爆炸
- 现有方案（SIM 的 hard/soft search、ETA 的 hash）牺牲了信息完整性，或系统复杂度高

**核心问题：** 如何在不牺牲模型精度的前提下，高效处理 10K+ 长度的用户行为序列？

---

## 2. 核心方法（关键创新）

### HiSAC 框架

```
超长用户行为序列 [i_1, i_2, ..., i_10000]
         ↓
层次化稀疏激活压缩 (Hierarchical Sparse Activation Compression)
         ↓
  L1: 粗粒度压缩（兴趣簇聚合）→ 保留 Top-K 激活
  L2: 细粒度压缩（关键行为精选）→ 动态稀疏 attention
         ↓
      高效 CTR/排序预测
```

**三大创新：**

1. **层次化压缩（Hierarchical）**：两级压缩策略
   - 第一级：按兴趣维度（类目/品牌/价格带）聚合，把 10K 序列压缩到 ~100 个兴趣表示
   - 第二级：对每个兴趣维度内，动态选取最相关的 K 个原始行为

2. **稀疏激活（Sparse Activation）**：
   - 基于候选 item 与历史行为的相关性，动态决定激活哪些行为节点
   - 稀疏化后 attention 复杂度从 O(n²) 降至 O(n·k)，k << n

3. **激活感知的压缩存储（Compression for Storage）**：
   - 将压缩后的兴趣表示缓存，避免每次推理重新处理全序列
   - 增量更新机制：用户新行为到来时，只更新相关兴趣簇

---

## 3. 实验结论

- 序列长度从 200 扩展到 **10,000+**，AUC 持续提升
- 相比 SIM（hard search），AUC **+0.3%**；相比 ETA，AUC **+0.5%**
- 推理延迟：比原始 Transformer Attention 快 **10x+**（序列 10K 情况下）
- 在工业数据集（淘宝、京东规模）验证

---

## 4. 工程价值（如何落地）

**这是当前工业界最紧迫的问题之一：长序列建模！**

**落地路径：**
1. 离线预计算兴趣压缩表示，存入用户 profile 服务（Redis/KV Store）
2. 在线请求时，加载压缩兴趣表示 + 候选 item，执行动态稀疏 attention
3. 增量更新：用户每次点击后，异步更新对应兴趣簇

**与主流方案对比：**
| 方案 | 序列长度 | 精度 | 在线延迟 | 系统复杂度 |
|------|---------|------|---------|-----------|
| DIN  | 50~200 | 基准 | 低 | 低 |
| SIM Hard | 200~1K | +0.2% | 低 | 中 |
| SIM Soft | 200~2K | +0.4% | 中 | 高 |
| ETA  | 1K~5K | +0.3% | 中 | 中 |
| HiSAC | **10K+** | **最优** | 中 | 中高 |

---

## 5. 面试考点

**Q1: 长序列推荐主要有哪些方案？对比优缺点？**
> DIN → DIEN → SIM（hard/soft）→ ETA → HiSAC。核心权衡是精度 vs 效率 vs 系统复杂度

**Q2: SIM Hard Search 和 Soft Search 的区别？**
> Hard Search：基于类目/品牌等 key 精确检索，速度快但召回不全；Soft Search：基于 embedding 相似度检索，精度高但需要 ANN index，系统复杂

**Q3: 超长序列建模中，如何处理时序信息？**
> 位置编码（绝对/相对）+ 时间衰减因子（越新的行为权重越高）+ DIEN 类的门控遗忘机制

---

*笔记生成时间: 2026-03-12 | MelonEggLearn*
