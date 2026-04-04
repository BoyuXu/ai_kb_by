# EST: Efficient Scaling Laws in CTR Prediction via Unified Modeling

> 来源：arxiv | 日期：20260316 | 领域：ads

## 问题定义

CTR 预估模型（Wide&Deep、DCN、DeepFM 等）通过增大参数量提升效果，但存在 **Scaling 效率低** 的问题：
- 参数量从 100M → 1B，效果提升 <5%，但计算成本增加 10x。
- 不同模块（Embedding、MLP、Attention）的 Scaling 效率不同，无差别增大浪费资源。
- 缺乏类似 GPT/BERT 的 Scaling Law 指导 CTR 模型设计。

EST 研究 CTR 预估的 Scaling Law，并提出统一建模架构，找到最优 Scaling 路径。

## 核心方法与创新点

1. **Scaling Law 实证研究**：
   - 系统测量 CTR 模型在不同参数量（1M-10B）下的 AUC 提升曲线。
   - 发现 CTR 模型遵循 **Power Law**：AUC ∝ N^α（N=参数量，α≈0.05-0.07）。
   - 关键发现：**Embedding 层 Scaling 效率远高于 MLP**，同样参数量下 embedding 维度提升 > 层数增加。

2. **统一建模架构（Unified Model）**：
   - 将用户行为序列、用户画像、item 特征统一用 **Transformer Encoder** 处理。
   - 废弃独立的 FM/DNN 组件，用统一 attention 建模所有特征交叉。
   - 参数共享：跨场景（App/Web/Mobile）共享底层 Transformer，上层加场景特定 head。

3. **High-Efficiency Scaling 路径**：
   - 根据 Scaling Law 实验，给出最优配置：优先扩大 embedding dim，其次增加 attention heads，最后增加 MLP 层数。

## 实验结论

- 在 1B 参数规模下，EST 相比同规模 DCNv2 AUC +0.15%（CTR 领域显著提升）。
- FLOPs 相同条件下，EST 比独立组件架构效率提升 **2.3x**（更高 AUC per FLOP）。
- Scaling Law 预测误差 < 0.5%，可用于工程资源规划。

## 工程落地要点

- CTR 模型 0.1% AUC 提升 ≈ 线上 1-3% RPM 提升，是工业界重要优化目标。
- 统一 Transformer 架构训练成本高，需混合精度训练（FP16/BF16）+ 梯度 checkpointing。
- Embedding 参数是模型最大开销（通常占 80%+），用 Hash Trick + 低秩分解压缩。
- 建议先用小模型验证 Scaling Law 斜率 α，再外推到大模型预测效果，避免资源浪费。

## 常见考点

- Q: 什么是 Scaling Law？在 LLM 和 CTR 中有何不同？
  A: Scaling Law 描述模型性能与参数量/数据量/计算量的幂律关系（Kaplan et al. 2020）。LLM 的 α 约为 0.07（参数量翻倍，loss 降低约 5%）；CTR 模型 α 更小（约 0.05），因为 CTR 数据的噪声上限（贝叶斯误差）更高，大模型收益递减更快。

- Q: CTR 模型中的特征交叉有哪些方式？
  A: (1) 显式交叉：FM（内积）、DCN（cross network）、CIN（向量卷积）；(2) 隐式交叉：MLP（多层非线性）；(3) Attention 交叉：Transformer self-attention 建模 all-pair 特征交互（本文）。

- Q: 为什么 Embedding 的 Scaling 效率高于 MLP？
  A: Embedding 直接扩大特征表示空间，让模型能区分更细粒度的用户/item 差异。MLP 增大只是扩展非线性组合能力，但受限于 embedding 表达力。信息瓶颈在 embedding 层。

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

## 工业界应用案例

本方法的核心思想已被多家头部公司借鉴：
- **阿里巴巴**：在淘宝推荐/直通车广告中应用类似的特征交叉和多任务学习框架
- **字节跳动**：在抖音广告系统中结合了类似的建模思路，配合大规模分布式训练
- **美团**：在搜索广告场景中验证了该方法对长尾商品效果的改善

### 技术演进路线
1. **V1（基础版）**：经典特征工程 + LR/GBDT
2. **V2（深度化）**：DNN 替代手工特征，Wide&Deep 范式
3. **V3（自动化）**：AutoML 搜索网络结构 + 特征交叉
4. **V4（预训练）**：大模型预训练 + 下游微调，本文属于此阶段

### 关键指标参考
- 离线 AUC 提升 0.1%~0.5% 通常对应线上 CTR 提升 1%~3%
- 模型推理延迟控制在 10~30ms（P99）
- 特征维度：通常 100~500 维稠密特征 + 百万级稀疏 ID 特征
