# 多目标广告排序：MMoE、PLE 与 Pareto 优化

> 来源：KDD 2018 (MMoE, Google) / RecSys 2020 (PLE, Tencent) | 年份：2018-2020 | 领域：ads/02_rank（多任务学习）

## 问题定义

广告排序需要同时优化多个互相冲突的目标：
- **广告主目标**：CTR（点击）、CVR（转化）、ROAS（广告回报率）
- **平台目标**：收入（RPM）、用户体验（不干扰）、生态健康（多样性）
- **用户目标**：相关性、新鲜度、不重复

**三大挑战**：
1. **跷跷板效应（Seesaw Effect）**：提升 CTR 往往损害 CVR（标题党问题），优化一个任务损害另一个
2. **负迁移（Negative Transfer）**：多任务共享参数时，不相关任务互相干扰，反而比单任务差
3. **目标权重设定**：不同目标的权重如何确定？人工调参成本高且不稳定

## 模型结构图

```
Hard Sharing        MMoE                    PLE (Progressive Layered Extraction)
                                           
  [Input]           [Input]                  [Input]
     ↓                ↓                        ↓
┌─────────┐    ┌─────────────┐         ┌─────────────────┐
│ Shared  │    │ E₁  E₂  E₃ │         │ S₁ S₂ │ E₁ │ E₂│  ← Shared + Task-specific Experts
│  MLP    │    │ (Experts)   │         │(Shared)│(T1)│(T2)│
└────┬────┘    └──┬───┬───┬──┘         └─┬──┬───┴──┬─┴──┬┘
     │            │   │   │              │  │      │    │
  ┌──┴──┐     ┌──┴┐ ┌┴──┐             ┌─┴──┴─┐ ┌──┴────┴─┐
  │T1 T2│     │G₁ │ │G₂ │  ←Gates     │Gate₁ │ │  Gate₂  │ ← Each gate selects from
  └─────┘     └─┬─┘ └─┬─┘             └──┬───┘ └────┬────┘   shared + own experts
               ↓      ↓                  ↓          ↓
            Tower₁  Tower₂           Tower₁      Tower₂
               ↓      ↓                  ↓          ↓
            pCTR    pCVR               pCTR       pCVR
```

## 核心方法与完整公式

### 公式1：MMoE 专家混合

$$f^k(x) = \sum_{i=1}^{n} g^k_i(x) \cdot f_i(x)$$

$$g^k(x) = \text{softmax}(W_{g^k} \cdot x)$$

**解释：**
- $f_i(x)$：第 $i$ 个 Expert Network 的输出
- $g^k(x)$：任务 $k$ 的 Gate Network 输出（softmax 归一化的权重）
- $W_{g^k}$：Gate $k$ 的可学习参数矩阵
- $n$：Expert 总数（通常 4-8）
- 不同任务通过不同的 Gate 权重选择不同的 Expert 组合

### 公式2：PLE 层级提取

$$f^k_l(x) = \sum_{i=1}^{m_k} g^k_{l,i}(x) \cdot E^k_{l,i}(x) + \sum_{j=1}^{m_s} g^k_{l,j+m_k}(x) \cdot E^s_{l,j}(x)$$

**解释：**
- $E^k_{l,i}$：第 $l$ 层任务 $k$ 的第 $i$ 个**任务特定专家**
- $E^s_{l,j}$：第 $l$ 层的第 $j$ 个**共享专家**
- $m_k$：任务 $k$ 的专家数量
- $m_s$：共享专家数量
- $g^k_l$：任务 $k$ 在第 $l$ 层的 Gate，对任务特定专家和共享专家都做加权

### 公式3：PLE 逐层提取（多层堆叠）

$$h^k_l = f^k_l(h^k_{l-1}, h^s_{l-1})$$
$$h^s_l = f^s_l(h^1_{l-1}, h^2_{l-1}, \ldots, h^K_{l-1}, h^s_{l-1})$$

**解释：**
- 每一层的输入来自上一层的**所有任务输出和共享输出**
- 共享专家可以看到所有任务的信息 → 信息流更充分
- 逐层提取（Progressive）：低层学通用特征，高层学任务特定特征

### 公式4：Pareto 多目标优化 - MGDA

$$\min_{\alpha \in \Delta^K} \left\| \sum_{k=1}^{K} \alpha_k \nabla_\theta \mathcal{L}_k(\theta) \right\|^2$$

**解释：**
- $\alpha_k$：第 $k$ 个任务的梯度权重，$\alpha \in \Delta^K$（单纯形约束）
- $\nabla_\theta \mathcal{L}_k$：第 $k$ 个任务的梯度
- MGDA 找到使所有任务梯度加权和范数最小的 $\alpha$ → Pareto 最优更新方向
- 自动平衡任务间冲突，无需手动设定权重

### 公式5：Uncertainty Weighting（Kendall 2018）

$$\mathcal{L} = \sum_{k=1}^{K} \frac{1}{2\sigma_k^2} \mathcal{L}_k + \log \sigma_k$$

**解释：**
- $\sigma_k$：任务 $k$ 的可学习不确定性参数
- 不确定性高的任务自动降权（$1/\sigma_k^2$ 小）
- $\log \sigma_k$ 项防止所有 $\sigma_k$ 趋向无穷大

## 与基线方法对比

| 方法 | 共享机制 | 负迁移控制 | 优势 | 劣势 |
|------|---------|-----------|------|------|
| **Hard Sharing** | 共享底层MLP | 无 | 简单高效 | 负迁移风险高 |
| **MMoE** | 多Expert+Gate | Gate软选择 | 灵活的任务关系 | Expert退化/坍缩 |
| **PLE** | 共享+特定Expert | Gate+Expert分离 | 显式隔离 | 参数量更大 |
| **ESMM** | 共享Embedding | 乘法结构 | 解决SSB | 任务关系固定 |
| **Cross-Stitch** | 学习共享比例 | 线性组合 | 自适应共享 | 只支持两任务 |

## 实验结论

- **PLE vs MMoE**（腾讯广告）：主任务 AUC +0.1-0.3%，辅助任务也有提升
- **PLE 消融**：去掉任务特定 Expert → 退化为 MMoE，AUC 下降 0.15%
- **Expert 坍缩现象**：MMoE 中部分 Expert 在所有任务的 Gate 权重趋于一致 → 实际退化为 Hard Sharing
- **Pareto 优化**：CTR/CVR 同时提升，调参时间减少 30%

## 工程落地要点

1. **Expert 数量**：通常 K=4-8，太多则参数膨胀训练慢，太少表达能力不足。PLE 中共享专家 2-4 个，每任务特定专家 1-2 个
2. **Gate 初始化**：用均匀初始化（1/K），防止训练初期某 Expert 主导导致坍缩
3. **线上推理**：所有任务共用一次前向传播（共享 Embedding + Experts），额外计算仅多几个 Tower Head
4. **任务权重调优**：推荐 Uncertainty Weighting 自动学习，避免手工网格搜索
5. **监控 Expert 利用率**：定期检查每个 Expert 被各任务 Gate 选中的频率分布，防止 Expert 坍缩

## 面试考点

**Q1：什么是负迁移？如何检测和缓解？**
> 负迁移指多任务联合训练后某任务性能反而比单任务训练更差。检测：对比多任务 vs 单任务基线 AUC。缓解：MMoE（Gate 软选择 Expert）→ PLE（任务特定 Expert 隔离）→ STAR（共享+特定参数网络）。

**Q2：MMoE 的 Expert 坍缩问题是什么？**
> 训练过程中所有 Gate 对 Expert 的权重分布趋于一致，多个 Expert 被等权使用 → 退化为 Hard Sharing。原因：Expert 初始化相似 + Gate 梯度趋同。PLE 通过显式分离共享/特定 Expert 缓解此问题。

**Q3：广告 eCPM 排序公式怎么用多任务预估？**
> eCPM = bid × pCTR × pCVR（CPA 出价）或 bid × pCTR（CPC 出价）。多任务模型同时预估 pCTR 和 pCVR，代入公式计算 eCPM 后排序。优势：共享 Embedding 提升小样本任务（CVR）效果。

**Q4：PLE 比 MMoE 好在哪里？**
> ① 显式分离共享 Expert 和任务特定 Expert → 任务特定 Expert 不被其他任务"争抢" ② 逐层提取，低层学通用特征，高层学任务特定特征 ③ 共享 Expert 的 Gate 也受所有任务梯度更新，信息流更充分。

**Q5：Pareto 优化和固定权重的区别？**
> 固定权重（$\mathcal{L} = w_1 L_1 + w_2 L_2$）需要人工设定 $w_1, w_2$，不同阶段最优权重不同。Pareto 优化（如 MGDA）在每步动态计算使所有任务都不恶化的更新方向，自动平衡冲突。但 MGDA 计算开销更大（需求解 QP）。

**Q6：多任务学习中如何处理"探索-利用"权衡？**
> ① ε-greedy：小概率随机探索 ② Thompson Sampling：基于 uncertainty 探索 ③ UCB：排序分中加 exploration bonus ④ 多任务中可为"辅助探索任务"设更高权重。

**Q7：工业界多任务模型的迭代策略？**
> 阶段1：Hard Sharing 快速验证多任务是否有增益 → 阶段2：MMoE 解决负迁移 → 阶段3：PLE 精细化 Expert 分离 → 阶段4：加入 Pareto 优化自动调权。每阶段通过 A/B 测试验证增益后再进入下一阶段。
