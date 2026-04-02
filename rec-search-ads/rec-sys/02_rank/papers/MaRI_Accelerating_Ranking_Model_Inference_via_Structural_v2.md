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

## 模型架构详解

### 特征处理
- **稀疏特征**：百万级 ID 特征（用户ID/物品ID/类目ID）→ Embedding Lookup
- **稠密特征**：数值特征（年龄/价格/历史统计量）→ Batch Normalization
- **序列特征**：用户行为序列 → Target Attention / Transformer 编码

### 核心网络
- **特征交叉**：显式（DCN Cross Layer / CIN）+ 隐式（DNN）
- **注意力机制**：Multi-Head Self-Attention 建模特征间交互
- **预测层**：多目标输出（CTR/CVR/停留时长/互动率）

### 训练与部署
- **样本构建**：曝光日志 + 事件 Join（考虑归因窗口）
- **分布式训练**：参数服务器（PS）处理大规模稀疏 Embedding
- **在线推理**：模型蒸馏 + 量化 + TensorRT 加速，延迟 < 30ms

## 与相关工作对比

| 维度 | 本文方法 | 经典方法（DIN/DIEN） | Transformer方法 |
|------|---------|-------------------|----------------|
| 表达能力 | 强 | 中 | 强 |
| 训练效率 | 中高 | 高 | 中 |
| 推理延迟 | 可优化 | 低 | 中高 |
| 可解释性 | 中 | 中 | 低 |

## 面试深度追问

- **Q: 排序模型的特征交叉为什么重要？**
  A: 推荐/广告场景中，单特征信息不足以准确预测用户行为。例如"年轻女性"+"美妆品类"的组合效应远超两者独立效应之和。特征交叉捕获这种组合模式。

- **Q: DIN（Deep Interest Network）的核心创新？**
  A: 用 Target Attention 替代简单的 Pooling 聚合用户行为序列。对于不同目标物品，关注用户历史中不同的行为，实现用户兴趣的局部激活。

- **Q: 如何解决推荐系统中的 Position Bias？**
  A: 1) 浅层塔建模位置偏差（PAL）；2) IPW（逆倾向加权）用位置CTR加权；3) 无偏数据收集（随机打散部分流量）。

- **Q: GAUC vs AUC 的区别和适用场景？**
  A: GAUC 按用户分组计算 AUC 再取加权平均，更贴近线上排序效果（排序在用户维度进行）。AUC 在全局样本上计算，可能被高活跃用户主导。

## 总结与实战建议

本文在推荐系统领域提出了有价值的技术创新。该方法的核心思想可与现有系统组件互补，建议：
1. 先在离线数据集上复现核心实验结果
2. 评估在自身业务场景的适用性和改进空间
3. 通过 A/B Test 验证线上效果，关注 CTR/CVR/用户停留时长等核心指标
4. 监控模型长期表现，及时应对数据分布漂移

## 工业界实践参考

### 典型应用场景
- **电商推荐**：淘宝/京东/拼多多的商品推荐排序
- **内容推荐**：抖音/快手/B站的短视频推荐
- **广告系统**：百度/腾讯/字节的广告排序和创意优化
- **搜索引擎**：相关性排序和个性化搜索结果

### 关键技术指标
- 离线评估：AUC > 0.75, GAUC > 0.65, NDCG@10 > 0.5
- 在线效果：CTR 提升 1-5%, 用户时长提升 2-8%, GMV 提升 1-3%
- 系统延迟：端到端 < 200ms, 模型推理 < 30ms (P99)
- 模型规模：Embedding 参数 TB 级, Dense 参数 GB 级

### 部署与监控
1. **灰度发布**：新模型先在 1% 流量验证，确认无异常后逐步放量
2. **在线监控**：实时追踪 AUC/LogLoss/CTR 等核心指标的分钟级波动
3. **降级策略**：模型超时或异常时回退到上一版本或规则兜底
4. **增量更新**：小时级增量训练 + 天级全量训练的混合策略
5. **特征监控**：特征覆盖率、均值/方差漂移检测、缺失值报警

### 未来趋势
- 大模型与推荐的深度融合（LLM as Recommender / LLM-augmented Features）
- 生成式推荐（从判别式到生成式范式转变）
- 多模态统一表示（文本/图片/视频/行为的联合编码）
- 因果推断在推荐去偏中的更广泛应用
