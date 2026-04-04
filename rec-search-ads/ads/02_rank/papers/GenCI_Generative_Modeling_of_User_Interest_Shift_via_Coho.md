# GenCI: Generative Modeling of User Interest Shift via Cohort-based Intent Learning for CTR

> 来源：arxiv | 日期：20260316 | 领域：ads

## 问题定义

用户兴趣随时间动态变化（Interest Drift），传统 CTR 模型用静态 embedding 表示用户，无法捕捉兴趣漂移。DIN/DIEN 等方法用行为序列建模动态兴趣，但存在：
- **长序列计算昂贵**：attention over 1000 items 代价高。
- **个体噪声大**：单用户行为稀疏，信号嘈杂。
- **缺乏群体共性**：不同用户的兴趣转移有相似模式，未被利用。

GenCI 提出 **Cohort（群体）** 层面的生成式兴趣建模，降噪并提升泛化。

## 核心方法与创新点

1. **Cohort 发现（群体聚类）**：
   - 将用户按 **近期兴趣轨迹** 聚类（K-means on behavior sequence embedding）。
   - 同一 Cohort 的用户有相似的兴趣演化模式（如"从运动 → 健康食品"的用户群）。

2. **生成式 Intent 建模**：
   - 用 **VAE（Variational Autoencoder）** 对每个 Cohort 的兴趣转移模式建模，学习潜在意图分布 q(z|history)。
   - 给定用户当前序列，从 Cohort 分布采样潜在意图向量 z，作为用户当前 intent 的表示。
   - z 具有连续性：可以插值预测兴趣未来走向。

3. **Cohort-aware CTR 模型**：
   - 用户表示 = 个体行为 embedding + Cohort 意图向量 z 的加权融合。
   - Cohort 权重由用户与 Cohort 中心的相似度决定（soft assignment）。

4. **对抗训练去噪**：对 Cohort 分布添加小扰动，训练模型对噪声鲁棒（类似 VAT）。

## 实验结论

- 公开数据集 Criteo、Avazu：AUC 分别 +0.15%、+0.21%，显著优于 DIEN、SIM 等基线。
- 工业数据集（某电商广告）：CTR +1.8%，CVR +1.2%，RPM +2.1%。
- Cohort 数量 K=50-100 时效果最佳，K 过大（>500）反而退化（每个 Cohort 样本不足）。

## 工程落地要点

- Cohort 聚类是离线任务，每天重新聚类一次，用 FAISS 加速大规模 K-means。
- VAE 编码器轻量化：用 2 层 MLP 即可，不需要复杂架构。
- 推理时：查找用户所属 Cohort（近邻查找，O(1) 哈希实现），取 Cohort 意图向量拼接，无额外实时计算。
- Cohort 更新延迟问题：用户行为可能快速变化，建议 Cohort 每天更新，不适合每小时更新（计算太贵）。

## 常见考点

- Q: DIN（Deep Interest Network）的核心思想是什么？
  A: DIN 用 Attention 机制动态加权用户历史行为序列：query = 候选 item embedding，key/value = 历史行为 embedding，attention weight = sim(query, key)，输出用户动态兴趣向量。与候选 item 相关的历史行为权重更高。

- Q: DIEN 相比 DIN 的改进是什么？
  A: DIEN（Deep Interest Evolution Network）在 DIN 基础上建模兴趣随时间的演化：用 GRU 提取兴趣演化序列，用 Attention GRU（AUGRU）在更新时引入目标 item 的注意力，建模"用户为什么会对该 item 感兴趣"的因果链。

- Q: 什么是 Cohort-based 建模的优势？
  A: 个体行为稀疏导致信号噪声大，Cohort 将相似用户聚合，提供更稳定的统计信号。类比：在 A/B 实验中，小流量实验用 CUPED 利用历史同组数据降低方差——Cohort 建模是类似思路，用群体先验降低个体估计方差。

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
