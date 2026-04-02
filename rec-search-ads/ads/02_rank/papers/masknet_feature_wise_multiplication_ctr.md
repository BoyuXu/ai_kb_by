# MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask

> 来源：arxiv | 领域：ads | 学习日期：20260328

## 问题定义

传统 CTR 模型（DeepFM、DCN 等）通过加法操作（sum pooling、concatenation）聚合特征，忽略了不同样本（instance）对特征重要性的差异化需求。MaskNet 提出：
- 不同用户、不同上下文下，同一特征的重要性不同
- 需要一种能够根据输入实例动态调整特征权重的机制
- 乘法操作（multiplication）比加法更适合捕捉这种实例级别的特征调制

## 核心方法与创新点

### Instance-Guided Mask

核心思想：用 instance 的特征向量生成一个 mask，对所有特征做 element-wise 乘法：

$$
V_{mask} = LayerNorm(f(e) \odot V_{emb})
$$

其中：
- $e$ 是当前 instance 的 embedding 向量（通过 MLP 映射得到 mask）
- $V_{emb}$ 是所有特征的 embedding 拼接
- $f(e)$ 是由 instance 生成的 mask 向量
- LayerNorm 稳定训练

### MaskBlock

每个 MaskBlock 包含：
1. **Instance-guided mask**：动态调制特征
2. **隐藏层**：带激活函数的全连接
3. **LayerNorm**：归一化

$$
MaskBlock(V) = LayerNorm(Linear(V_{mask}))
$$

### 两种组合方式

1. **Serial MaskNet**：多个 MaskBlock 串联，逐层精炼
2. **Parallel MaskNet**：多个 MaskBlock 并联，输出聚合

### 与 SENet 的区别

SENet 做的是全局平均池化后的 channel-wise 权重；MaskNet 是 instance-specific 的 element-wise 调制，粒度更细。

## 实验结论

- 在 Criteo、Avazu、MovieLens 数据集上超越 DeepFM、DCN、xDeepFM、AutoInt
- Serial MaskNet 通常优于 Parallel MaskNet
- 乘法操作对于捕捉用户意图的细粒度差异效果显著
- 在微博广告系统线上 A/B 测试中，CTR 显著提升

## 工程落地要点

1. **LayerNorm 关键**：Mask 后的乘法结果需要 LayerNorm 稳定，否则训练不收敛
2. **Mask 生成网络**：建议 1-2 层 MLP，过深反而引入噪声
3. **串联深度**：3-4 个 MaskBlock 通常足够，更多收益递减
4. **与 FFM 结合**：MaskNet 可以看作动态 FFM 的一种实现
5. **计算开销**：相比普通 DNN 增加约 20-30% 计算，但收益显著
6. **冷启动**：新用户无足够历史，Mask 生成质量下降，需要回退策略

## 常见考点

**Q：MaskNet 与 SENet 的核心区别是什么？**
A：SENet 做全局 squeeze-excitation，权重对同一 batch 内所有样本相同（channel-wise）；MaskNet 的 mask 由每个 instance 的特征动态生成，每个样本有不同的 mask，粒度更细是 instance-specific 的。

**Q：为什么用乘法而不是加法？**
A：乘法可以实现"门控"效果，当 mask 值接近 0 时可以完全抑制某个特征维度；加法只能做线性偏移，无法实现这种强调/抑制的效果。

**Q：Serial 和 Parallel MaskNet 分别适合什么场景？**
A：Serial 适合需要逐步精炼特征表示的场景，每层 mask 基于上一层的输出；Parallel 适合希望捕获多视角特征表示的场景，适合特征多样性强的数据集。

## 模型架构详解

### 输入层
- **用户特征**：用户画像（年龄/性别/地域）、行为序列（点击/购买历史）、实时上下文（时间/设备/位置）
- **广告特征**：广告 ID Embedding、广告主类目、创意特征（标题/图片 Embedding）
- **上下文特征**：页面位置、请求时段、竞价上下文

### 特征交叉层
- 低阶交叉：FM/FFM 风格的二阶特征交叉
- 高阶交叉：Deep 网络或 Cross Network 捕获高阶非线性

### 预测层
- 多目标输出头：CTR Tower + CVR Tower（或联合 CTCVR）
- 校准层：Platt Scaling 或 Isotonic Regression 确保预测概率准确

### 训练策略
- 样本构建：曝光日志 + 点击/转化事件 Join
- 负采样：全量负样本或按比例降采样 + 权重修正
- 优化器：Adam / AdaGrad with learning rate warmup


## 与相关工作对比

| 维度 | 本文方法 | 传统方法 | 优势 |
|------|---------|---------|------|
| 特征交叉 | 自动化/高阶 | 手工/低阶 | 减少特征工程，捕获复杂模式 |
| 多任务学习 | 联合优化 | 独立模型 | 共享知识，缓解数据稀疏 |
| 在线适应 | 增量更新 | 全量重训 | 更快响应分布漂移 |
| 样本效率 | 全空间/去偏 | 有偏子集 | 更准确的概率估计 |


## 面试深度追问

- **Q: 这个方法如何处理特征稀疏问题？**
  A: 通过预训练 Embedding + 特征交叉网络，将稀疏 ID 特征映射到稠密空间。对于冷启动特征，可借助 side information（类目、文本描述）进行特征迁移。

- **Q: 在线服务的延迟优化方案？**
  A: 1) 模型蒸馏/量化减少参数量；2) 特征缓存避免重复计算；3) 异步特征获取 + 超时降级；4) GPU/TensorRT 推理加速。

- **Q: 如何评估模型的线上效果？**
  A: 离线指标（AUC/GAUC/LogLoss）+ 在线 A/B Test（CTR/CVR/RPM/GMV）。注意 GAUC 按用户分组计算 AUC 的均值，更贴近线上排序效果。

- **Q: 训练样本的时效性如何保证？**
  A: 1) 实时特征流（Flink）+ 小时级增量训练；2) 样本时间窗口控制（通常7~14天）；3) 特征 serving 与训练使用相同 pipeline 避免 train-serve skew。
