# R2LED: Equipping Retrieval and Refinement in Lifelong User Modeling

> 来源：arXiv 2025 | 领域：ads | 学习日期：20260404

## 问题定义

终身用户建模（Lifelong User Modeling）：建模用户数年累积的行为历史（序列长度 10k-100k）。

核心挑战：
- 序列过长无法直接 Feed 进模型
- 用户兴趣多元且随时间演变（旧兴趣不能直接丢弃）
- 长序列中的噪声行为影响模型质量

$$h_u = \text{Model}(i_{1}, i_{2}, \ldots, i_{N}) \quad N \sim 10^4$$

## 核心方法与创新点

**R2LED（Retrieval + Refinement for Lifelong Encoding & Distillation）**：

1. **检索阶段（Retrieval）**：
   - 给定目标广告 $a$，从全量历史中检索最相关 Top-K 行为
   
$$\mathcal{S} = \text{TopK}_{i \in \text{hist}} [\text{sim}(e_a, e_i)]$$

   - 检索器：ANN（近似最近邻）+ GPU Index（Faiss）

2. **精炼阶段（Refinement）**：
   - 检索结果 → Transformer 精炼（考虑时序 + 互补性）
   - 去除噪声检索结果（与当前意图不一致的行为）
   
$$h_u^{\text{refined}} = \text{Transformer}(\mathcal{S}, a)$$

3. **蒸馏（Distillation）**：
   - 长序列 Teacher → 短精炼序列 Student
   - 确保精炼后的表示保留长序列的关键信息
   
$$\mathcal{L}_{\text{distill}} = ||h_u^{\text{teacher}} - h_u^{\text{student}}||_2^2$$

4. **动态记忆更新**：
   - 关键行为（高置信度兴趣）写入长期记忆
   - 噪声行为（孤立、非重复）自动过期

## 实验结论

- CTR AUC vs SIM（长序列基线）: **+1.4‰**
- 序列长度 10k vs 1k：R2LED 额外收益 **+0.9‰**（长序列价值被充分利用）
- 在线延迟：检索 8ms + 精炼 12ms = 20ms（满足工业要求）

## 工程落地要点

- Faiss GPU Index：每用户最多存 10k 条 Embedding，内存 ~400GB（全量用户需分布式）
- 检索 K=50，精炼后保留 K=20
- 蒸馏 loss 权重在 [0.1, 0.5]，防止过拟合蒸馏目标
- 长期记忆 vs 短期序列分开存储，查询接口统一

## 面试考点

1. **Q**: 终身用户建模（10k+ 行为）的核心方案有哪些？  
   **A**: 检索式（SIM/R2LED）：给定目标 item 检索最相关历史；记忆压缩式（MIMN/PerSRec）：压缩历史为记忆向量；层次式：近期精细 + 远期粗粒度。

2. **Q**: 检索式方法的主要缺陷？  
   **A**: 检索相关性依赖目标 item Embedding 质量；检索结果缺乏时序上下文（可能召回孤立的旧行为）；冷启动 item 检索不准。R2LED 的 Refinement 阶段缓解了前两个问题。

3. **Q**: 知识蒸馏在这里的作用？  
   **A**: 保证精炼序列（短）的表示质量接近全量历史（长）的表示质量，防止检索+精炼引入信息损失。
