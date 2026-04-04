# PerSRec: Efficient Sequential Recommendation for Long Term User Interest via Personalization

> 来源：arXiv 2025 | 领域：rec-sys | 学习日期：20260404

## 问题定义

长序列用户兴趣建模是序列推荐的核心挑战。用户历史行为序列长达数千，现有 Transformer 的 Self-Attention 复杂度为 $O(n^2)$，在长序列下计算成本极高。

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V \quad \text{复杂度} O(n^2d)$$

## 核心方法与创新点

**PerSRec** 的核心思想：将长序列压缩为 **个性化用户记忆**，而不是每次重新 attend 所有历史。

1. **个性化记忆压缩（PMC）**：
   - 训练一个轻量 Condenser，将长序列 $[i_1, \ldots, i_n]$ 压缩为固定长度记忆 $M_u \in \mathbb{R}^{k \times d}$（k << n）
   - 记忆在用户每次交互后增量更新，无需全量重算

2. **记忆感知注意力**：
   - 当前会话序列 attend 到个性化记忆而非全历史
   
$$\text{Output} = \text{Attention}(Q_{\text{session}}, K_M, V_M)$$

其中 $K_M, V_M$ 来自压缩记忆，复杂度降至 $O(n \cdot k)$

3. **分层时序感知**：
   - 短期兴趣：当前会话内的精细交互
   - 长期兴趣：压缩记忆中的宏观偏好
   - 两者通过门控融合：$h = \sigma(W_g[h_{\text{short}}; h_{\text{long}}])$

## 实验结论

- 序列长度 1000 时推理速度比 SASRec 快 **6.8x**
- Recall@10 在 Amazon Books 数据集上提升 **+4.2%**（vs SASRec）
- 记忆压缩比 k/n = 1/20 时性能-效率最优
- 增量更新延迟 < 1ms（适合在线服务）

## 工程落地要点

- 用户记忆 $M_u$ 存储在 KV 存储中（Redis/RocksDB），大小 = $k \times d \times 4$ bytes
- k=32, d=128 时每用户约 16KB，亿级用户约 1.6TB（可接受）
- 记忆更新可异步化：用户退出后批量更新，不影响在线延迟
- 冷启动：新用户记忆初始化为全局平均记忆

## 面试考点

1. **Q**: 如何处理超长用户历史序列（>1000 items）？  
   **A**: PerSRec 式记忆压缩——将历史压缩为固定长度 k 的个性化记忆，Attention 复杂度从 $O(n^2)$ 降至 $O(nk)$。

2. **Q**: 短期兴趣与长期兴趣如何融合？  
   **A**: 门控融合，短期来自当前会话 Transformer，长期来自压缩记忆，门控权重由用户状态决定。

3. **Q**: 这和 Mamba/SSM 方法相比有何不同？  
   **A**: PerSRec 保持 Transformer 架构，通过 **外部记忆** 解决长序列；Mamba 通过 **选择性状态空间** 内嵌长程依赖，都有各自适用场景。
