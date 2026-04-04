# DCN V2: Improved Deep & Cross Network for Feature Cross Learning in Web-Scale Learning to Rank Systems

> 来源：arxiv | 领域：ads | 学习日期：20260328

## 问题定义

在大规模广告和推荐系统的 Learning-to-Rank 场景中，特征交叉（feature crossing）对于捕捉用户行为与商品/广告的非线性关系至关重要。DCN V1 虽然引入了 cross network，但其每层只能学习有限阶次的显式特征交叉，表达能力受限。DCN V2 旨在：
1. 提升 cross network 的表达能力，支持学习更丰富的特征交叉
2. 探索 Cross Network 与 Deep Network 的高效组合方式
3. 在真实工业级系统中验证效果

## 核心方法与创新点

### Cross Network 改进（DCN-V2 核心）

DCN V1 的 cross layer：

$$
x_{l+1} = x_0 \cdot x_l^T \cdot w_l + b_l + x_l
$$

DCN V2 将标量权重替换为矩阵权重：

$$
x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l
$$

其中 $W_l \in \mathbb{R}^{d \times d}$，大幅提升了交叉的表达能力，可以学习任意阶次的显式特征交叉。

### 两种组合结构

1. **Stacked（串联）**：Cross Network 输出接 Deep Network
   - 先学显式交叉，再学隐式高阶关系
2. **Parallel（并联）**：Cross Network 和 Deep Network 并行，输出拼接后接输出层
   - 两路同时进行，互补

### Low-Rank 近似优化

对 $W_l$ 做低秩分解：$W_l = U_l V_l^T$（$U_l, V_l \in \mathbb{R}^{d \times r}$，$r \ll d$），降低参数量和计算开销。

### Mixture of Experts (MoE) 扩展

多个 expert 矩阵 + gating 网络选择，进一步增强多样性。

## 实验结论

- 在 Google Ads 数据集和多个公开数据集（Criteo, MovieLens）上显著优于 DCN V1、DeepFM、xDeepFM
- Parallel 结构普遍优于 Stacked 结构
- Low-rank 版本在保持精度的同时大幅降低计算开销（参数减少 40%，延迟降低 35%）
- AUC 提升 0.1-0.3%，对工业级系统意义重大

## 工程落地要点

1. **维度选择**：Low-rank 的秩 `r` 通常选 `d/4` 左右，平衡精度与效率
2. **并联结构优先**：工业实践中 Parallel 更稳定，不易出现梯度消失
3. **Cross Layer 层数**：通常 2-4 层即可，过深收益递减
4. **特征预处理**：数值特征需要归一化，否则 cross 学习会被大值主导
5. **部署注意**：矩阵乘法计算图优化，可用 fused kernel 加速
6. **与其他模型结合**：常与 MMOE、SENet 等结合使用

## 常见考点

**Q：DCN V1 和 DCN V2 的核心区别是什么？**
A：V1 的 cross layer 用标量权重，每层只做有限阶显式交叉，表达能力弱；V2 改用矩阵权重 $W_l$，可以学习任意复杂的显式特征交叉，同时支持低秩近似降低开销。

**Q：为什么 Parallel 结构通常优于 Stacked？**
A：并联允许显式交叉和隐式深度特征并行学习，两路互补；串联时 Deep 网络可能覆盖 Cross Network 的学习，且梯度传播路径更长。

**Q：Low-rank 近似的原理是什么？**
A：$W_l = U_l V_l^T$，将 $d \times d$ 矩阵分解为两个 $d \times r$ 矩阵的乘积，参数从 $O(d^2)$ 降到 $O(2dr)$，当 $r \ll d$ 时大幅节省计算。

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
