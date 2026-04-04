# ESMM：全空间多任务 CVR 预估

> 来源：工程实践 / SIGIR 2018 阿里巴巴 | 日期：20260317

## 问题定义

CVR（Conversion Rate）预估面临两大挑战：

1. **样本选择偏差（SSB）**：CVR 模型只用曝光中被点击的样本训练，但预测时需要对所有曝光（包含未点击的）预估。点击集合是曝光集合的非随机子集，导致训练分布与推理分布不一致。

2. **稀疏数据（Data Sparsity）**：转化样本（购买/下单）远少于点击样本（通常少 1~2 个数量级），传统 CVR 模型因数据稀疏而难以训练。

$$
\text{pCTCVR} = p(\text{click}|\text{impression}) \times p(\text{conversion}|\text{click}) = \text{pCTR} \times \text{pCVR}
$$

## 核心方法与创新点

1. **全空间建模（Entire Space Modeling）**
   - 在**曝光空间**（Impression Space）直接建模 pCTCVR
   - 避免在点击子空间训练 CVR，消除样本选择偏差

2. **多任务联合训练**
   - **Task 1**：pCTR = $P(\text{click}|\text{impression})$，在曝光空间训练
   - **Task 2**：pCTCVR = $P(\text{conversion}|\text{impression})$，在曝光空间训练（点击且转化为正样本）
   - 推导 pCVR：$\hat{pCVR} = \hat{pCTCVR} / \hat{pCTR}$（推理时）

3. **共享 Embedding**
   - CTR Tower 和 CVR Tower 共享底层 Embedding 参数
   - CVR Tower 借助 CTR 的丰富点击数据学习更好的特征表征（解决数据稀疏）

4. **训练目标**

$$
\mathcal{L} = \mathcal{L}_{CTR} + \mathcal{L}_{CTCVR} = \sum \text{BCE}(y_{click}, \hat{pCTR}) + \sum \text{BCE}(y_{convert}, \hat{pCTCVR})
$$

## 实验结论

- 淘宝电商场景 CVR 预估 AUC 提升约 1.0%
- CTCVR AUC 提升约 0.5%
- 相比仅用点击数据训练的 CVR 模型，全空间建模显著改善了尾部商品的 CVR 预估

## 工程落地要点

1. **标签构建**：正样本需要 impression → click → conversion 的完整链路，需要事件 join
2. **归因窗口**：转化通常有延迟（如 7 天归因），需要控制 label delay
3. **pCVR 计算**：在线推理时 $\hat{pCVR} = \hat{pCTCVR} / \hat{pCTR}$，需处理 pCTR 接近 0 的数值稳定性问题
4. **延伸**：ESM²（加入 wish/cart 中间转化节点）、AITM（自适应信息迁移）

## 常见考点

- **Q: CVR 样本选择偏差的根本原因？**
  A: CVR 模型在训练时只能观测到被点击的曝光（因为只有点击才可能发生转化），但推理时需要对所有曝光打分。点击 ≠ 曝光的随机样本，两者分布不同（高 CTR 物品被过度代表）。

- **Q: ESMM 如何解决数据稀疏？**
  A: 通过共享 Embedding，CVR Tower 可以间接利用 CTR 任务的大量点击数据更新 Embedding，而不局限于稀少的转化样本。CTR 数据量通常是 CVR 数据量的 10~100 倍。

- **Q: 为什么不直接训练 pCTCVR，而要拆分两个 tower？**
  A: 如果直接训练 pCTCVR，则不能分解出 pCVR 用于广告出价（广告主按转化出价时需要 pCVR 估计）。拆分使得 pCTR、pCVR 均可独立使用，满足不同下游需求（CTR 用于排序，CVR 用于转化出价）。

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
