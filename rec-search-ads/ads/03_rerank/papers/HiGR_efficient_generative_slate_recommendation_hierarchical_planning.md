# HiGR: Efficient Generative Slate Recommendation via Hierarchical Planning
> 来源：https://arxiv.org/abs/2512.24787 | 领域：ads | 学习日期：20260329

## 问题定义

Slate Recommendation（列表推荐）需要一次性向用户展示一个排序好的物品列表（如抖音推荐页、YouTube首页）。现有生成式方法存在三大痛点：

1. **语义纠缠的 Item Tokenization**：传统 RQ-VAE 生成的 Semantic ID 空间存在前缀语义不清晰问题（多义性/同义性）
2. **低效的顺序解码**：M个物品、每个D个token，需 D×M 步自回归（如10物品×3层=30步），推理延迟过高
3. **缺乏全局 Slate 规划**：从左到右生成无法优化整体列表结构

## 核心方法与创新点

### ① CRQ-VAE (Contrastive RQ-VAE) - 对比残差量化自编码器

- **Prefix-level 对比学习**：在前 D-1 层引入 InfoNCE loss，强制相似样本共享相同前缀，相异样本前缀分离
- **全局量化损失**：直接优化 latent 级别的全局量化误差，避免残差塌陷（residual vanishing）
- **分层语义结构**：高层前缀捕捉粗粒度语义，最后一层保留细粒度区分能力

$$
\mathcal{L}_{CRQ-VAE} = \mathcal{L}_{recon} + \lambda_1 \mathcal{L}_{global\_quan} + \lambda_2 \mathcal{L}_{cont}
$$

### ② HSD (Hierarchical Slate Decoder) - 分层 Slate 解码器

**Coarse-to-Fine 两阶段架构：**
- **粗粒度 Slate Planner**：自回归生成每个物品的 preference embedding，捕捉全局列表结构
- **细粒度 Item Generator**：M个参数共享的 generator，基于 preference embedding 并行生成 SID 序列

**GSBI 推理策略 (Greedy-Slate Beam-Item)**：
- Slate Planner 贪心解码（top-1）
- Item Generator Beam Search（宽度B），M个物品独立并行解码

### ③ ORPO-based 多目标偏好对齐

无需参考模型，三目标负样本构造：
1. 乱序负样本 → 排名目标
2. 负反馈替换 → 兴趣过滤
3. 语义不相似替换 → 多样性促进

$$
\mathcal{L}_{post} = -\log \pi_\theta(y^+|x) - \alpha \log \sigma(z_\theta(x,y^+) - z_\theta(x,y^-))
$$

## 实验结论

| 指标 | 数值 |
|------|------|
| 相比 SOTA 推荐质量提升 | **>10%** |
| 推理速度提升 | **5×** |
| 腾讯平台平均观看时长提升 | **+1.22%** |
| 腾讯平台平均视频播放量提升 | **+1.73%** |

部署规模：腾讯平台数亿用户在线 A/B 测试验证。

## 工程落地要点

1. **离线阶段**：CRQ-VAE 训练生成 SID 映射表 → HiGR 预训练 → ORPO 偏好对齐
2. **在线服务**：Context Encoder → Slate Planner → 并行 Item Generator → SID 映射返回
3. **冷启动**：CRQ-VAE 基于 bge-m3 内容 embedding，新物品直接编码无需重训练
4. **延迟保障**：GSBI 策略将生成复杂度从 O(D×M) 降至 O(M + D)，Item-level 解码并行化
5. **多目标平衡**：通过 ORPO 负样本构造灵活调节多样性/相关性权衡

## 常见考点

**Q1: HiGR 解决了现有生成式推荐的哪三个核心问题？**
A: ①语义纠缠的Tokenization ②低效顺序解码(D×M步) ③缺乏全局Slate规划

**Q2: CRQ-VAE 中的 Prefix-level 对比学习为什么排除最后一层？**
A: 前D-1层捕捉粗粒度语义，最后一层保留细粒度区分能力。若最后层也加对比约束会过度平滑，无法区分相似物品。

**Q3: HSD 的 GSBI 推理策略如何提升效率？**
A: Slate Planner 贪心解码，M个 Item Generator 独立并行 Beam Search，消除跨物品依赖，实现5×加速。

**Q4: ORPO 相比 DPO/RLHF 有什么优势？**
A: ①无需参考模型（显存减半）②集成监督学习缓解目标漂移 ③直接用 odds ratio 优化更稳定

**Q5: HiGR 的三目标偏好对齐具体如何构造负样本？**
A: ①随机乱序正样本（排名目标）②替换为负反馈物品（兴趣目标）③保留第一个item替换其余为语义不相似物品（多样性目标）

## 模型架构详解

### 重排输入
- **候选集**：精排 Top-K 结果（通常 K=50~200）
- **上下文**：用户实时兴趣、已展示列表、多样性约束

### 列表级建模
- Set-to-Sequence：将候选集编码为集合，解码为有序序列
- Attention 交互：候选间 Self-Attention 捕捉互补/竞争关系
- 位置感知：考虑展示位置对点击率的影响（position bias debiasing）

### 优化目标
- Listwise Loss：NDCG/MAP 的可微近似（ApproxNDCG、LambdaLoss）
- 多目标权衡：点击率 × 时长 × 多样性 × 新鲜度的加权组合
- 约束满足：品类多样性、广告密度、内容安全等硬约束

### 在线推理
- Beam Search / Greedy 解码生成最终列表
- 延迟预算：重排需在 10~50ms 内完成


## 与相关工作对比

| 维度 | 本文方法 | Pointwise排序 | 优势 |
|------|---------|-------------|------|
| 建模粒度 | Listwise | Pointwise | 捕获候选间交互 |
| 多样性 | 内生约束 | 后处理 | 端到端优化 |
| 位置偏差 | 显式建模 | 忽略 | 更准确的效果评估 |
| 推理延迟 | 可控 | 低 | 精度-延迟平衡 |


## 面试深度追问

- **Q: 重排阶段如何保证推理延迟？**
  A: 1) 限制重排候选集大小（通常 50~200）；2) 高效 Attention（线性 Attention 或稀疏 Attention）；3) 模型蒸馏；4) 贪心解码替代 Beam Search。

- **Q: 如何在重排中融入多样性约束？**
  A: 1) MMR（Maximal Marginal Relevance）：相关性-冗余度权衡；2) DPP（Determinantal Point Process）：行列式点过程建模集合多样性；3) 约束解码：每步生成时检查多样性约束。

- **Q: Listwise vs Pointwise vs Pairwise Loss 的选择？**
  A: Listwise（LambdaLoss/ApproxNDCG）直接优化排序指标，但训练不稳定；Pairwise（BPR）稳定但忽略位置信息；工程中常用 Pointwise 训练 + Listwise 微调。

- **Q: 位置偏差如何消除？**
  A: 1) IPW（逆倾向加权）：用位置 CTR 作为倾向得分加权；2) PAL（Position-Aware Learning）：显式建模位置 bias 项；3) 无偏数据收集：随机打散部分流量。
