# Meta Lattice: Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations

> 来源：arxiv | 领域：ads | 学习日期：20260328

## 问题定义

大型广告系统需要为不同广告主、不同场景（信息流、搜索、故事）部署大量推荐模型。每个场景维护独立模型导致：
1. **资源浪费**：每个模型独立训练，计算/存储开销线性增长
2. **数据碎片化**：每个场景数据量有限，模型质量受限
3. **维护复杂性**：N 个场景 = N 套独立模型和 pipeline

Meta Lattice 提出**模型空间重设计**：用一个统一的"Lattice 结构"覆盖所有场景，以更低成本实现多场景优化。

## 核心方法与创新点

### Lattice 模型空间

核心思想：将不同场景的模型视为一个**高维模型空间中的子空间**，用共享基底 + 场景专属调制来表示每个场景：

$$
\theta_{scene_k} = \theta_{shared} + \Delta\theta_k
$$

其中 $\Delta\theta_k$ 是场景 k 的专属调制参数（小，低秩）。

### 三层结构

1. **Global Backbone（全局主干）**：所有场景共享的 dense 网络，捕获通用特征交叉
2. **Scene-Specific Adapter（场景适配器）**：低秩矩阵 $A_k B_k^T$，专属于场景 k，类似 LoRA
3. **Task Head**：每个场景的输出层（CTR/CVR/各自目标）

$$
f_k(x) = TaskHead_k\left(Backbone(x) + Adapter_k(x)\right)
$$

### 联合训练策略

- 所有场景数据联合训练 Global Backbone
- 各场景的 Adapter 和 Head 独立更新
- 使用 gradient routing 控制哪些参数被哪些场景的梯度更新

### Cost-Effectiveness 量化

对比指标：**参数量 / AUC 提升** 的比值，Lattice 结构在同等参数量下显著优于独立多模型。

## 实验结论

- 与独立多模型相比，参数量减少 60%，AUC 持平或略优
- 跨场景知识共享对数据稀疏场景（新广告主、小场景）提升尤其明显
- 场景适配器（Adapter）的秩 r=16 即可捕获绝大部分场景差异
- 在 Meta 广告系统大规模部署，显著降低计算成本

## 工程落地要点

1. **Adapter 秩的选择**：r=8~32 是实践中的甜点区域，过小欠拟合，过大接近独立模型开销
2. **梯度路由实现**：用 mask 或 conditional backward 控制场景特定梯度流
3. **冷启动场景**：新场景可以从 Global Backbone 直接初始化，快速收敛
4. **部署统一性**：所有场景共享同一套推理图，只切换 Adapter 权重，部署简化
5. **增量学习**：新场景加入时只需训练新的 Adapter，不影响已有场景
6. **监控**：需要监控每个场景的 AUC 独立变化，防止 Backbone 更新损害某个场景

## 常见考点

**Q：Meta Lattice 和多任务学习（MMOE）有什么本质区别？**
A：MMOE 在同一数据上学习多个任务（CTR、CVR、etc.），任务间共享 expert；Meta Lattice 针对不同数据来源（不同场景/广告位）的模型参数共享，侧重参数效率和跨场景知识迁移，而非多任务联合优化。

**Q：场景专属 Adapter 为什么用低秩矩阵？**
A：场景间差异通常集中在少数几个语义维度（如广告格式、用户群体偏好），不需要全秩更新；低秩分解 $A_k B_k^T$（$rank=r \ll d$）用 $2dr$ 参数表示 $d^2$ 的调制，大幅降低开销，且类似 LoRA 的经验表明低秩表示对于 fine-tuning 已经足够。

**Q：联合训练时如何防止大场景数据"淹没"小场景的学习信号？**
A：1) 按场景对训练数据做上下采样（temperature sampling），平衡各场景比例；2) 对小场景的 Adapter gradient 施加更高权重；3) 用 meta-learning 策略（MAML 变体）让 Backbone 对所有场景保持同等适应性。
