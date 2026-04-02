# MMoE：多门控混合专家（Multi-gate Mixture-of-Experts）

> 来源：KDD 2018, Google | 年份：2018 | 领域：rec-sys/04_multi-task（多任务学习）

## 问题定义

推荐系统需要同时优化多个目标（CTR、CVR、时长、收藏、分享等），面临三大挑战：
1. **资源浪费**：分别训练多个单任务模型，Embedding 不共享，存储和训练成本翻倍
2. **负迁移**：Hard Parameter Sharing（共享底层+独立Tower）在任务相关性低时互相干扰
3. **任务冲突**：不同任务的最优梯度方向可能相反（如CTR要推热门，CVR要推精准）

**MMoE 核心思想**：让不同任务通过独立的门控网络，自适应地选择和组合专家网络，避免强制共享导致的负迁移。

## 模型结构图

```
┌───────────────────────────────────────────────────────┐
│                    MMoE Architecture                   │
│                                                       │
│  Task 1 Output        Task 2 Output                  │
│       ↑                    ↑                          │
│  ┌────┴────┐          ┌────┴────┐                    │
│  │ Tower 1 │          │ Tower 2 │                    │
│  └────┬────┘          └────┬────┘                    │
│       ↑                    ↑                          │
│  Σ g₁ᵢ·Eᵢ            Σ g₂ᵢ·Eᵢ                      │
│       ↑                    ↑                          │
│  ┌────┴────┐          ┌────┴────┐                    │
│  │ Gate 1  │          │ Gate 2  │                    │
│  │ g₁=σ(W₁x)│        │ g₂=σ(W₂x)│                  │
│  └────┬────┘          └────┬────┘                    │
│       └─────────┬──────────┘                          │
│                 ↓                                      │
│  ┌───────┬──────┴──────┬───────┐                     │
│  │ E₁    │  E₂        │  E₃   │  ... Eₖ (Experts)  │
│  │(MLP)  │  (MLP)     │ (MLP) │                     │
│  └───┬───┴──────┬──────┴───┬───┘                     │
│      └──────────┼──────────┘                          │
│                 ↑                                      │
│         Input Features x                               │
│         [User, Item, Context]                          │
└───────────────────────────────────────────────────────┘
```

## 核心方法与完整公式

### 公式1：MMoE 专家融合

$$f^k(x) = \sum_{i=1}^{K} g_i^k(x) \cdot E_i(x)$$

**解释：**
- $E_i(x)$：第 $i$ 个专家网络的输出（MLP）
- $g^k(x) = \text{Softmax}(W_{g^k} \cdot x) \in \mathbb{R}^K$：任务 $k$ 的门控权重
- 不同任务通过不同门控权重选择不同专家组合

### 公式2：任务 Tower 输出

$$\hat{y}^k = \sigma(W^k_T \cdot f^k(x) + b^k_T)$$

**解释：**
- 每个任务有独立的 Tower 网络（MLP）
- Tower 输入是门控融合后的专家输出 $f^k(x)$
- $\sigma$：sigmoid 输出概率

### 公式3：多任务联合损失

$$\mathcal{L} = \sum_{k=1}^{T} w_k \cdot \mathcal{L}_k(\hat{y}^k, y^k)$$

**解释：**
- $w_k$：任务 $k$ 的权重（可固定或可学习）
- $\mathcal{L}_k$：任务 $k$ 的损失函数（通常交叉熵）
- 权重调优关键：可用 Uncertainty Weighting 自动学习

### 公式4：PLE 改进（Progressive Layered Extraction）

$$f_l^k = \text{Gate}_l^k([E_l^{shared}(x); E_l^{k}(x)])$$

$$f_l^{shared} = \text{Gate}_l^{shared}([E_l^{shared}(x); E_l^{1}(x); \ldots; E_l^{T}(x)])$$

**解释：**
- $E_l^{shared}$：第 $l$ 层的共享专家
- $E_l^k$：第 $l$ 层任务 $k$ 的特定专家
- 任务特定门控只看共享+自己的专家
- 共享门控看所有专家 → 信息充分流通

### 公式5：Load Balancing Loss（防止专家坍缩）

$$\mathcal{L}_{balance} = K \cdot \sum_{i=1}^{K} f_i \cdot P_i$$

$$f_i = \frac{1}{N}\sum_{x} \mathbb{1}[\arg\max g(x) = i], \quad P_i = \frac{1}{N}\sum_{x} g_i(x)$$

**解释：**
- $f_i$：专家 $i$ 被选为 top-1 的频率
- $P_i$：专家 $i$ 的平均门控权重
- 均匀分布时 $\mathcal{L}_{balance}$ 最小，鼓励所有专家被均匀使用

## 与基线方法对比

| 方法 | 共享机制 | 负迁移控制 | 参数效率 | 任务自适应 |
|------|---------|-----------|---------|-----------|
| **单任务模型** | 无共享 | 无 | 低（多份Embedding） | N/A |
| **Shared-Bottom** | 共享底层MLP | 无 | 高 | 无 |
| **One-gate MoE** | 共享门控 | 部分 | 中等 | 部分 |
| **MMoE** | 多门控多专家 | 有 | 中等 | **有** |
| **PLE** | 共享+特定专家 | 更好 | 稍低 | **有** |

## 实验结论

- **Census-income 数据集**（低相关任务对）：MMoE 优于 Shared-Bottom 约 1-2% AUC
- **YouTube 视频推荐**：同时优化 CTR + 观看时长，MMoE 相比单任务更优
- **专家数量**：K=8 通常是工业实践 sweet spot
- **PLE vs MMoE**（腾讯广告）：主任务 AUC +0.1-0.3%，消除了辅助任务负迁移

## 工程落地要点

1. **任务权重**：各任务 loss weight 需调优，推荐 Uncertainty Weighting 或 GradNorm 自动学习
2. **专家数量**：K=4-8 为常见选择，过多专家会导致坍缩（某些专家利用率为 0）
3. **门控温度**：门控 softmax 前加温度参数，防止 winner-take-all
4. **梯度归一化**：主任务和辅助任务梯度量级差异大时需做 gradient normalization
5. **在线推理**：所有任务共用一次前向传播（共享 Embedding + Experts），仅多几个 Tower Head
6. **监控专家利用率**：定期检查 Gate 权重分布，防止 Expert 坍缩

## 面试考点

**Q1：为什么 Shared-Bottom 在任务相关性低时效果差？**
> 任务目标梯度方向不一致（甚至相反）时，共享层参数受矛盾梯度信号，导致每个任务都无法充分利用底层表征 → 负迁移。

**Q2：MMoE 中所有门控都选同一专家会怎样？**
> 退化成 Shared-Bottom，其他专家得不到训练。解决：Load Balancing Loss 或 Expert Dropout 鼓励专家多样性。

**Q3：PLE 相比 MMoE 的改进？**
> PLE 将 shared experts 和 task-specific experts 分开：task-specific experts 只服务特定任务，不受其他任务梯度污染；shared experts 学通用表征；多层 extraction 渐进式融合。

**Q4：如何检测多任务学习中的负迁移？**
> 对比多任务 vs 单任务基线 AUC：如果某任务多任务 AUC < 单任务 AUC，则发生了负迁移。也可监控训练过程中各任务 loss 曲线：一个任务 loss 下降同时另一个上升 = 负迁移。

**Q5：MMoE 的 Gate 网络设计有什么注意点？**
> ① 输入：通常用原始输入特征 x，不用 Expert 输出（避免循环依赖）② 网络结构：简单线性层 + Softmax 即可，过复杂的 Gate 反而过拟合 ③ 初始化：均匀初始化（1/K），防止某专家主导 ④ 正则化：Gate 输出加 entropy 正则化鼓励多样性。

**Q6：工业界多任务模型的迭代路线？**
> 阶段1：Shared-Bottom 快速验证 → 阶段2：MMoE 缓解负迁移 → 阶段3：PLE 精细化专家分离 → 阶段4：STAR（共享+特定参数网络）→ 阶段5：加入 Pareto/自动权重调优。每阶段 A/B 验证后再进入下一阶段。
