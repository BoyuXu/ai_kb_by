# MaRI: Accelerating Ranking Model Inference via Structural Re-parameterization in Large Scale RS

> 来源：arXiv | 日期：20260317

## 问题定义

工业级精排模型（Ranking Model）通常包含大量 Embedding + MLP + Attention 结构，推理延迟成为在线服务的瓶颈。传统压缩方法（剪枝/量化）会损失精度；知识蒸馏需要额外训练成本。**结构重参数化（Structural Re-parameterization）** 在CV领域已被证明可将训练时的多分支结构合并为单路径，推理时无额外代价，但在推荐系统中的应用存在特殊挑战（稀疏 Embedding、异构特征、点击率任务）。

## 核心方法与创新点

1. **训练时多分支增强**
   - 在 MLP 各层引入并行残差分支（Identity + BN、1×1 conv 等效的 Linear shortcut）
   - 在 Embedding 层增加 auxiliary embedding table（低秩分支），捕获不同粒度的特征交互

2. **推理时结构合并（Re-param）**
   - 训练完成后，将多分支线性变换代数合并为单一矩阵乘法
   - Embedding 分支合并：auxiliary embedding 权重合并回主 embedding，不增加 lookup 表大小
   - BN 层融入 Linear 层：$W' = W \cdot \gamma / \sqrt{\sigma^2+\epsilon}$，$b' = b - \mu\cdot\gamma/\sqrt{...} + \beta$

3. **推荐系统适配**
   - 针对 ID 特征稀疏性，设计 sparse-aware 重参数化，避免 dense re-param 引入的 embedding 膨胀
   - 对 DNN 各层独立 re-param，保持层间结构不变

4. **与 DLRM/DeepFM 结合**
   - 实验在 DLRM、DeepFM、DCN-v2 上均有效，通用性好

## 实验结论

- 在公开数据集（Criteo、Avazu）上 AUC 提升 0.1~0.3%，无推理额外开销
- 工业数据集推理 QPS 提升约 20%，延迟降低 15%（TensorRT 部署）
- 与量化（INT8）组合可叠加提速，QPS 综合提升 35%+

## 工程落地要点

1. **训练脚本改造**：在 MLP forward 中加入 re-param 分支，训练结束后调用 `reparameterize()` 接口合并
2. **Embedding 合并注意索引对齐**：auxiliary table 若使用不同 vocab，合并前需映射
3. **BN + Linear 合并**：生产环境建议单测验证合并前后输出一致性（误差 <1e-5）
4. **AB 实验**：Re-param 模型与原模型推理结构完全不同，需灰度放量观察稳定性

## 常见考点

- **Q: 结构重参数化的核心思想是什么？**
  A: 训练时引入多分支结构提升表达能力，推理时利用线性代数将多分支合并为等价的单分支，使推理无额外开销。

- **Q: BN 和 Linear 如何合并？**
  A: BN: $y = \gamma(x-\mu)/\sigma + \beta$；Linear: $z = Wy + b$。合并后 $W' = W\gamma/\sigma$，$b' = W(\beta - \gamma\mu/\sigma) + b$。

- **Q: 推荐系统中 re-param 的特殊挑战？**
  A: Embedding 是离散稀疏的，不能用连续 conv re-param；需要设计 sparse-aware 合并方案，且 auxiliary embedding 规模不能太大避免内存爆炸。
