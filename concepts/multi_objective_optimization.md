# 多目标优化：搜广推的共同难题

> **一句话总结**：你想让用户既点击又下单还不退货，同时广告主ROI达标、平台收入最大化——这些目标经常打架，多目标优化就是找到最佳平衡点。
>
> **为什么要学**：几乎所有搜广推的在线系统都面临多目标冲突。面试必问"MMoE和PLE区别"、"多目标怎么权衡"。

**相关概念页**：[Attention in RecSys](attention_in_recsys.md) | [Embedding全景](embedding_everywhere.md) | [生成式推荐](generative_recsys.md)

---

## 1. 为什么会有多目标？

### 推荐系统
- **用户侧**：点击率(CTR)、完播率、互动率、留存率
- **内容侧**：多样性、新鲜度、生态健康
- **平台侧**：DAU、时长、商业化收入

### 广告系统
- **广告主**：转化率(CVR)、ROI、LTV
- **平台**：eCPM（收入）、广告质量
- **用户**：广告体验、不打扰

### 搜索系统
- **相关性**：query-doc 匹配度
- **时效性**：新内容优先
- **多样性**：结果不重复
- **权威性**：来源可信度

**核心矛盾**：优化 A 目标时，B 目标往往会下降（跷跷板效应 / seesaw effect）。

---

## 2. 方法一：加权求和（最简单）

$$\mathcal{L} = \sum_k w_k \cdot \mathcal{L}_k$$

- 优点：实现简单，一行代码
- 缺点：权重 $w_k$ 怎么定？不同量级的 loss 混在一起不公平
- 改进：**不确定性加权**（Kendall 2018）：$w_k = \frac{1}{2\sigma_k^2}$，自动学习

**实际线上融合公式（推荐排序）**：

$$\text{score} = \text{CTR}^{w_1} \times \text{CVR}^{w_2} \times \text{时长}^{w_3} \times \text{bid}$$

广告中更常见的是 **eCPM = pCTR × pCVR × bid**，直接乘法融合。

---

## 3. 方法二：共享底座多任务模型

### Shared-Bottom（基线）
所有任务共享底层网络，顶部各自一个 tower。

问题：任务差异大时，底层被"拉扯"，谁也学不好。

### MMoE（Google 2018）⭐ 必掌握

$$h^{(t)}(x) = \sum_{k=1}^{K} g_k^{(t)}(x) \cdot f_k(x)$$

$$g^{(t)}(x) = \text{softmax}(W_g^{(t)} \cdot x)$$

- $K$ 个 Expert 网络（每个是独立的全连接网络）
- 每个任务 $t$ 有自己的 Gate，决定用哪些 Expert
- **软分离**：Gate 是 softmax，可以选择性地使用不同 Expert

**直觉**：CTR 任务可能主要用 Expert 1 和 3，CVR 任务用 Expert 2 和 3，共享 Expert 3 的通用知识。

### PLE（腾讯 2020）⭐ 必掌握

**MMoE 的问题**：所有 Expert 对所有任务开放 → 任务差异大时仍然有干扰。

**PLE 改进**：
- **私有 Expert**：每个任务独占（不给别的任务用）
- **共享 Expert**：所有任务共用
- Gate 同时接收私有和共享 Expert 的输出

**MMoE vs PLE 一句话区别**：MMoE 是软分离（通过 gate 隐式分配），PLE 是显式分离（硬性划分私有/共享）。

### SMES（美团 2024）
20+ 任务场景下 PLE 参数爆炸 → 稀疏门控，每个任务只激活部分 Expert。

📄 详见 [rec-sys/04_multi-task/synthesis/推荐广告系统多任务学习与MoE专家混合.md](../rec-search-ads/rec-sys/04_multi-task/synthesis/推荐广告系统多任务学习与MoE专家混合.md)

---

## 4. 方法三：梯度操纵

当加权求和导致跷跷板时，从梯度层面解决冲突：

| 方法 | 核心思想 | 适用场景 |
|------|---------|---------|
| GradNorm | 动态调权让各任务训练速度一致 | 任务收敛速度差异大 |
| PCGrad | 投影冲突梯度到正交方向 | 梯度方向经常冲突 |
| CAGrad | 找公共下降方向 | 理论最优但计算贵 |
| Nash-MTL | 博弈论均衡 | 研究导向 |

**工业界更常用的方案**：不做梯度操纵，而是用 PLE 结构 + 仔细调权 + AB测试 → 简单有效。

---

## 5. 方法四：Pareto 优化

当你无法确定权重时，找 **Pareto 前沿**——所有"不可能同时提升所有目标"的点的集合。

$$\text{Pareto 最优}：\nexists \mathbf{w}' \text{ s.t. } \forall k: \mathcal{L}_k(\mathbf{w}') \leq \mathcal{L}_k(\mathbf{w}) \text{ 且 } \exists j: \mathcal{L}_j(\mathbf{w}') < \mathcal{L}_j(\mathbf{w})$$

**直觉**：Pareto 前沿上的每个点代表一种 trade-off，选哪个由业务需求决定。

**工业应用**：线下训 Pareto 前沿 → 线上通过调权重在前沿上滑动。

---

## 6. 方法五：约束优化（广告系统核心）

广告系统的多目标本质是 **约束优化**——在满足约告主 ROI 约束下最大化平台收入：

$$\max_\theta \sum_i \text{eCPM}_i \quad \text{s.t.} \quad \text{ROI}_j \geq \text{target}_j, \forall j$$

**求解方法**：
1. **Lagrange 对偶**：$\mathcal{L} = \sum_i \text{eCPM}_i + \sum_j \lambda_j (\text{ROI}_j - \text{target}_j)$
2. **CMDP（约束马尔可夫决策过程）**：RL 框架下的约束优化
3. **PID 控制**：实时调 Lagrange 乘子 $\lambda$

📄 详见 [ads/02_rank/synthesis/广告系统多目标优化.md](../rec-search-ads/ads/02_rank/synthesis/广告系统多目标优化.md)

---

## 7. 各领域多目标对比

| 维度 | 推荐 | 广告 | 搜索 |
|------|------|------|------|
| 典型目标数 | 3-8 | 2-4 | 3-5 |
| 主要冲突 | CTR vs 多样性 | ROI vs 收入 | 相关性 vs 时效性 |
| 主流方法 | PLE + 加权融合 | 约束优化 + PID | 加权 RRF + 规则 |
| 调权方式 | AB测试 | 广告主设定约束 | 人工规则 + 学习 |
| 实时性要求 | 中 | 高（竞价实时） | 中 |

---

## 演进总结

```
固定权重加权求和
    │
    ▼
自动权重学习 (不确定性加权)
    │
    ├─ Shared-Bottom → MMoE → PLE → SMES (模型结构)
    │
    ├─ GradNorm → PCGrad (梯度操纵)
    │
    ├─ Pareto 前沿 (无需指定权重)
    │
    └─ 约束优化 / CMDP (广告系统主流)
```

## 面试高频问题

1. **MMoE 和 PLE 的区别？** → MMoE 所有 Expert 共享，通过 Gate 软分离；PLE 显式区分私有/共享 Expert，硬分离减少任务干扰。
2. **多任务学习中跷跷板效应怎么解决？** → 三个层面：①模型结构（PLE 显式分离）②损失权重（不确定性加权/GradNorm）③训练策略（分阶段训练/交替训练）。
3. **广告系统的多目标和推荐有什么不同？** → 广告本质是约束优化（ROI约束），推荐更多是软权衡（加权融合）。
4. **线上怎么调多目标权重？** → AB测试 + 在线调参。大公司用 Bayesian Optimization 自动搜索权重。
