# DeepMTL2R: A Library for Deep Multi-task Learning to Rank
> 来源：arXiv:2602.14519 | 领域：rec-sys | 学习日期：20260330

## 问题定义
工业推荐/搜索排序系统同时面临多个排序目标（相关性、CTR、满意度、多样性），且目标之间存在冲突。现有 LTR（Learning to Rank）库不支持多任务，深度多任务 LTR 缺乏统一框架。DeepMTL2R 开源了一套深度多任务排序库，集成了主流多任务和排序算法。

## 核心方法与创新点
1. **统一接口**：抽象 TaskHead（任务头）和 Backbone（共享特征提取），任意组合 MMOE/PLE/CrossStitch 作为 backbone，任意组合 Pointwise/Pairwise/Listwise 作为 loss。
2. **多目标权重调度**：内置 GradNorm、PCGrad、UncertaintyWeighting 三种梯度平衡策略，可插拔切换。
3. **生产就绪**：支持 ONNX 导出、TensorRT 优化、Triton Inference Server，直接对接工业部署。
4. **基准测试套件**：提供标准化的多任务排序数据集（Ali-CCP、KuaiRec、MIND）评估流程，便于方法比较。

## 实验结论
- 在 Ali-CCP 数据集：PLE + PCGrad 组合 AUC CTR+0.82%，AUC CVR+1.23%（对比单任务基线）
- GradNorm 在梯度量级差异大（>10x）时最有效，UncertaintyWeighting 在梯度方向冲突时最有效
- ONNX 导出后 TensorRT 推理加速 3-5x（对比 PyTorch eager mode）

## 工程落地要点
- 多任务 loss 权重初始化建议：根据各 loss 量级倒数初始化（自动平衡）
- PCGrad 计算成本比 GradNorm 低（不需要二阶项），工业推荐 PCGrad
- 多任务训练需要各任务数据量级均衡，否则小任务被大任务淹没
- Pairwise loss 需要在线负采样，建议 in-batch negative（高效且无 bias）

## 面试考点
- Q: 多任务学习中负迁移（Negative Transfer）的原因和解决方案？
  - A: 原因：任务梯度方向冲突，一个任务的更新损害另一个；解法：PCGrad（投影去除冲突分量）、MMOE（软路由隔离）、Gradient Surgery
- Q: Listwise 排序 loss（如 ListMLE、LambdaLoss）的训练难点？
  - A: ① 需要全 session 物品同时计算（batch 设计难）；② 梯度稀疏（只有 top-k 位置贡献）；③ 对 missing label 敏感
- Q: 如何选择多任务权重平衡策略？
  - A: 任务重要性明确 → 手动权重；梯度量级差异大 → GradNorm/UncertaintyWeighting；梯度方向冲突 → PCGrad

## 数学公式

$$
\mathcal{L}_\text{total} = \sum_{t=1}^T w_t \mathcal{L}_t, \quad w_t \propto \frac{1}{\sigma_t^2} \text{ (UncertaintyWeighting)}
$$

$$
g_i' = g_i - \frac{g_i \cdot g_j}{||g_j||^2} g_j \text{ (PCGrad, if } g_i \cdot g_j < 0)
$$
