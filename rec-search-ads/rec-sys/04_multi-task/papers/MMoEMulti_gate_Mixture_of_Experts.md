# MMoE：多门控混合专家（Multi-gate Mixture-of-Experts）

> 来源：工程实践 / KDD 2018 Google | 日期：20260317

## 问题定义

推荐系统通常需要同时优化多个目标（CTR、CVR、时长、收藏、分享等），朴素方法是分别训练多个单任务模型（资源浪费）或共享底层 + 各任务独立 tower（Hard Parameter Sharing，任务冲突时效果差）。当任务相关性低时，强制共享底层表征会导致**负迁移（Negative Transfer）**。

## 核心方法与创新点

1. **专家网络（Expert Networks）**
   - 设 K 个专家网络 $\{E_1, ..., E_K\}$，每个专家是独立的 MLP
   - 专家学习输入的不同表征子空间，天然捕获不同任务需要的特征

2. **多门控机制（Multi-gate）**
   - 每个任务有独立的门控网络：$g^k(x) = \text{Softmax}(W_g^k x)$，$\in \mathbb{R}^K$
   - 各任务的融合表征：$f^k(x) = \sum_i g^k_i(x) \cdot E_i(x)$
   - 各任务 tower 在 $f^k(x)$ 之上独立建模

3. **与 Shared-Bottom 和 MoE 的对比**
   - **Shared-Bottom**：所有任务共享底层，单一门控（无法任务自适应）
   - **One-gate MoE**：所有任务共享同一门控，仍然无法差异化
   - **MMoE**：每个任务有独立门控，实现任务自适应专家选择

4. **进化：PLE（Progressive Layered Extraction）**
   - 腾讯提出，区分 shared experts 和 task-specific experts，避免 task-specific 信息污染 shared experts
   - Multi-level extraction network，逐层融合

## 实验结论

- 在 Census-income 数据集（相关性低任务对）MMoE 优于 Shared-Bottom
- YouTube 视频推荐实验：同时优化 CTR + 观看时长，MMoE 相比单任务更优
- 专家数量 K=8 通常是工业实践中的 sweet spot

## 工程落地要点

1. **任务权重调整**：多任务联合训练时，各任务 loss weight 需要调优，通常用 grid search 或 AutoML
2. **专家数量**：K=4~8 为常见选择，过多专家会导致某些专家塌陷（collapse）
3. **门控温度**：门控 softmax 前可加温度参数，防止 winner-take-all 导致某专家利用率极低
4. **任务排序**：主任务（如 CTR）和辅助任务（如 CVR）梯度量级差异大，需做 gradient normalization
5. **工业实践**：阿里 ESMM、腾讯 PLE、快手 AITM 均是 MMoE 思想的延伸

## 常见考点

- **Q: 为什么 Shared-Bottom 在任务相关性低时效果差？**
  A: 任务目标梯度方向不一致（甚至相反）时，共享层参数会受到矛盾梯度信号，导致每个任务都无法充分利用底层表征，即负迁移。

- **Q: MMoE 中如果所有门控都选同一个专家会怎样？**
  A: 等价于退化成 Shared-Bottom，其他专家参数得不到充分训练。实际中通过 load balancing loss（类似 Switch Transformer）或 expert dropout 来鼓励专家多样性。

- **Q: PLE 相比 MMoE 的改进是什么？**
  A: PLE 将 shared experts 和 task-specific experts 分开，task-specific experts 只服务特定任务，不受其他任务梯度污染；shared experts 专注学习任务无关的通用表征；多层 extraction 实现渐进式特征融合。
