# An Efficient LLM-based Evolutional Recommendation with Locate-Forget-Update Paradigm

> 来源：arXiv 2025 | 领域：rec-sys | 学习日期：20260404

## 问题定义

LLM 推荐系统的核心矛盾：用户偏好动态变化，但 LLM 参数静态（训练后固定）。持续微调代价高昂（每次全量 SFT 需要大量算力），遗忘旧知识（灾难性遗忘）。

$$\text{Challenge: } \theta_{t+1} \leftarrow \text{update}(\theta_t, D_{t+1}) \text{ without forgetting } D_{1:t}$$

## 核心方法与创新点

**LFU（Locate-Forget-Update）** 三步范式：

1. **Locate（定位）**：
   - 识别 LLM 中负责特定用户/物品偏好的参数子集
   - 用 **Fisher Information Matrix** 定位关键参数层
   
$$\mathcal{F}_\theta = \mathbb{E}\left[\left(\nabla_\theta \log p(y|x)\right)^2\right]$$

2. **Forget（遗忘）**：
   - 对过时偏好对应的参数进行 **梯度反转**（Gradient Reversal）
   - 选择性遗忘：仅遗忘过时信息，保留通用知识
   
$$\Delta\theta_{\text{forget}} = -\eta \nabla_\theta \mathcal{L}_{\text{old}}$$

3. **Update（更新）**：
   - 在 Forget 后的参数基础上，用新数据进行轻量微调（LoRA）
   - 更新量极小：只修改 0.5-2% 参数
   
$$\theta_{t+1} = \theta_t + \Delta\theta_{\text{forget}} + \Delta\theta_{\text{update}}^{\text{LoRA}}$$

**增量更新机制**：每日增量数据触发 LFU，全量更新变为周批次。

## 实验结论

- 用户偏好漂移场景 NDCG@10: **+14.7%** vs 静态 LLM
- 比全量 SFT 计算量减少 **94%**
- 灾难性遗忘减少：旧任务性能保持 **97.3%**（vs 全量 SFT 的 89.1%）

## 工程落地要点

- Fisher 信息矩阵计算成本高，用对角近似（Diagonal Fisher）
- LoRA rank=4 足够增量更新（低秩捕获偏好漂移）
- 更新频率：日增量数据 > 1000 条时触发，避免噪声更新
- 监控：更新后跑留存指标，检测是否引入负迁移

## 面试考点

1. **Q**: 如何让 LLM 推荐系统适应用户偏好变化？  
   **A**: LFU 范式：定位相关参数（Fisher）→选择性遗忘过时偏好（梯度反转）→轻量更新（LoRA），避免全量 SFT。

2. **Q**: 什么是灾难性遗忘？如何在推荐场景避免？  
   **A**: 新任务训练覆盖旧知识。避免方法：选择性遗忘（只改变目标参数）+ LoRA（限制更新范围）+ EWC（弹性权重固化）。

3. **Q**: Fisher Information 在这里的作用？  
   **A**: 衡量参数对当前预测的重要性，高 Fisher 的参数是关键参数（谨慎修改），低 Fisher 的参数是可更新参数。
