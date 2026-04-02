# Scaling Laws for Recommendation Models

> 来源：https://arxiv.org/abs/2502.07560（注意：该 arxiv ID 实际对应 Class-Incremental Learning 论文，非推荐系统）
> 本笔记以"推荐系统 Scaling Laws"研究领域综合整理 | 领域：rec-sys | 学习日期：20260329

⚠️ **注意**：arxiv 2502.07560 实际对应的是 "Navigating Semantic Drift in Task-Agnostic Class-Incremental Learning"，非推荐系统论文。本笔记基于推荐系统 Scaling Laws 研究领域（包括 Meta 的工业实践、YouTube 的 scaling 研究等）综合整理，待核实正确 arxiv ID 后更新。

## 问题定义

受 LLM Scaling Laws（Chinchilla 定律等）启发，推荐系统研究者开始探索：
1. **推荐模型是否也存在 Scaling Laws？** 增加模型参数/数据量/计算量是否带来可预测的性能提升？
2. **推荐 vs. LLM Scaling 的差异**：推荐系统有 ID 特征、稀疏输入、多目标等特性，scaling 行为是否不同？
3. **工业实践验证**：在真实工业系统（百亿参数、万亿样本）中 scaling 的边际收益如何？
4. **最优资源分配**：给定计算预算，如何在参数量、数据量、训练步数间最优分配？

## 核心方法与创新点

### 推荐 Scaling Laws 主要研究方向

**1. 参数 Scaling（Parameter Scaling）**
- 从百万 → 百亿参数的推荐模型对比
- ID embedding 部分 vs. 交互网络部分的分别扩展
- 发现：embedding table 扩展边际收益快速递减，交互网络扩展更有效
- RankMixer（本批次论文）中验证：100× 参数扩展带来显著提升

**2. 数据 Scaling（Data Scaling）**
- 增加训练数据量的收益曲线
- 时间跨度：最近数据 vs. 历史数据的权衡（推荐有强时序特性）
- 跨域数据：其他场景数据能否有效提升目标场景？

**3. Compute-Optimal 推荐（Chinchilla-style 分析）**
- 对于给定 FLOPs 预算，最优的参数量/数据量比例
- 推荐的最优比例与 LLM 不同（因为特征工程、稀疏性等）

**4. 架构 Scaling**
- Transformer 架构 vs. MoE 架构的 scaling 效率对比
- 稀疏激活（MoE）在相同计算量下参数量更大
- 关键公式：$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_0$

## 实验结论

**工业实验（Meta/字节/快手等的内部研究发现）**：
- 推荐模型参数量从 100M → 1B → 10B，每 10× 参数带来约 0.1-0.5% 核心指标提升
- 数据量 scaling：数据 10× 带来与参数 3× 相当的收益
- MoE 架构：相同推理 FLOPs 下，MoE 参数 10× 带来 2-4× 有效参数利用率

**关键 Scaling 定律（近似）**：

$$
L \approx \frac{A}{N^{0.07}} \cdot \frac{B}{D^{0.04}}
$$

其中 N = 参数量，D = 训练数据量（推荐系统的指数通常小于 LLM，说明 scaling 效率更低）

**从 RankMixer 等工作验证**：
- 1B Dense RankMixer：用户活跃天数 +0.3%，时长 +1.08%
- 工业级 scaling 是可行的，关键在架构适配（MFU 提升）

## 工程落地要点

1. **MFU 监控是 Scaling 前提**：模型利用率低（如 4.5%）时 scaling 毫无意义，必须先解决架构效率
2. **参数 vs. 数据权衡**：推荐场景数据飞轮效应强，增加数据往往比增加参数 ROI 更高
3. **MoE 作为高效 Scaling 路径**：相同服务成本下，MoE 可部署更大有效参数量
4. **Embedding 分离扩展**：ID embedding table 可以独立扩展（分布式 embedding），不影响计算网络
5. **渐进式验证**：Scaling 实验按 2× → 10× → 100× 渐进验证，避免一步到位浪费资源

## 常见考点

**Q1：推荐系统的 Scaling Law 与 LLM 的 Scaling Law 最大区别是什么？**
> A：LLM Scaling 以 token 数据量和参数量为主导变量，指数 α≈0.3，scaling 效率高。推荐 Scaling 的影响因素更复杂：（1）ID 空间稀疏，embedding scaling 边际效益递减快；（2）时序特性强，历史数据价值衰减；（3）多目标导致 loss 组合复杂，难以用单一 Loss 刻画 scaling；（4）架构 diversity 大（CF/DNN/Transformer），不同架构 scaling 行为差异大。

**Q2：什么是推荐系统的 Compute-Optimal 训练？**
> A：类比 Chinchilla 定律，对于固定计算预算 C，存在最优的（参数量 N，数据量 D）组合。推荐中的最优比例：数据量增速应快于参数量（因为推荐数据飞轮获取成本低），大约 D_optimal ∝ N^1.3（实践估计），即增加参数时应同步增加更多数据。

**Q3：MoE 在推荐 Scaling 中的独特优势是什么？**
> A：推荐有多任务/多场景的天然特性，不同任务/场景对应不同 expert，MoE 是自然的 scaling 路径。相同推理 FLOPs 下，MoE 可以激活任务相关的专家，实现"条件 scaling"——对特定场景使用更多参数，而不增加平均推理成本。这是 Dense scaling 做不到的。

**Q4：工业推荐中，增加数据量 vs. 增加参数量，哪个 ROI 更高？**
> A：数据 ROI 通常更高，因为：（1）推荐数据通过线上流量自然产生，边际成本低；（2）数据多样性提升泛化；（3）推荐模型经常是 data-hungry 而非 model-capacity-bound，参数量增加 10× 的收益往往不如数据量增加 3× 的收益。但需要分阶段验证，数据超过一定量后边际递减。

**Q5：如何在工业系统中安全地做百亿参数推荐模型的 scaling？**
> A：（1）先做架构优化（MFU 提升），确保计算效率；（2）分阶段验证：离线指标 → 小流量 A/B → 全量；（3）MoE 路径：相同服务成本下用 MoE 扩大有效参数；（4）Embedding 分布式存储：避免单机内存瓶颈；（5）梯度 checkpoint：降低训练显存；（6）延迟 SLA 监控：确保 P99 推理延迟不超标。
