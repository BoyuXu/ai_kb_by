# 展示广告特征工程：从原始数据到模型输入

> 来源：技术综述 | 日期：20260316 | 领域：ads

## 问题定义

展示广告（Display Ads）的 CTR 预估依赖丰富的特征工程。与搜索广告不同，展示广告没有明确的 query，需要从**用户行为**、**上下文**、**广告创意**中挖掘信号。特征工程质量直接决定模型天花板：在工业实践中，好的特征工程往往比模型改进更有效。

## 核心方法与创新点

### 特征体系结构

**1. 用户特征（User Features）**
- 人口属性：年龄段、性别、城市级别（需谨慎使用，隐私合规）
- 设备特征：操作系统、设备型号、App版本、网络类型（WiFi/4G/5G）
- 行为序列：最近 N 次点击/购买/搜索（截断为固定长度，通常 50-200）
- 统计特征：CTR历史（按类目/广告主/时间粒度聚合）、活跃度、付费力

**2. 广告特征（Ad Features）**
- 广告内容：标题文本 embedding、图片视觉 embedding、落地页类目
- 广告质量分：CTR 历史均值（EMA 平滑）、点击后行为质量（停留时长）
- 广告主特征：行业类目、历史 ROI、账户等级

**3. 上下文特征（Context Features）**
- 曝光位置：广告位 ID、页面类型、位置（首屏/折叠下方）
- 时间特征：小时、星期、节假日标记（重要！CTR 在节日前后波动剧烈）
- 场景特征：App 内页面路径、用户当前 session 行为（刚搜索了什么）

**4. 交叉特征（Cross Features）**
- User × Ad：用户品类偏好 × 广告品类的重合度
- User × Context：用户历史在该时段的活跃度
- Ad × Context：广告在该位置的历史 CTR

### 特征处理技巧

**离散特征**：
- 高基数 ID（user_id, item_id）：Hash Trick → Embedding，碰撞率 < 0.1%
- 低基数类目：One-hot 或 Embedding（维度 ≈ sqrt(类别数)）
- 缺失值：独立的 <missing> embedding，不用填充 0

**连续特征**：
- 价格、时长等：Log 变换（消除长尾）→ 分桶（等频/等宽）→ Embedding
- 统计 CTR 特征：Bayesian Smoothing 平滑（防止低频 item 的 CTR 估计不稳定）

**序列特征**：
- 定长截断 + Padding（Mask）
- Target Attention（DIN 风格）：用候选 ad 作为 query，attention 历史行为

## 实验结论

- 加入行为序列特征（vs 仅用统计特征）：AUC +0.5-1.0%，这是最重要的特征组。
- 位置特征（广告位 ID）：AUC +0.2-0.4%，不可忽略。
- 时间特征（小时+星期 embedding）：AUC +0.1-0.2%，节假日标记额外 +0.1%。
- 特征交叉（显式 FM 交叉 vs 只用 MLP）：AUC +0.2-0.5%。

## 工程落地要点

- **特征一致性**：训练和预测使用完全相同的特征处理逻辑（避免 training-serving skew），用同一套代码（如 TFX/Feast）。
- **特征监控**：对每个特征建立 dashboard，监控分布变化（PSI > 0.1 需告警）。
- **特征存储**：实时特征（用户 session 行为）用 Redis/Memcached；离线统计特征用 HBase/Bigtable。
- **特征版本管理**：特征工程变更需要追踪，建议用 feature store（Feast/Tecton）管理版本。

## 常见考点

- Q: 什么是 Training-Serving Skew？如何避免？
  A: 训练时和线上预测时的特征处理不一致（比如训练时对缺失值填 0，线上用平均值），导致线上效果比离线评估差。避免：(1) 用同一套特征处理代码；(2) 将特征处理逻辑写成服务（在线实时计算）；(3) 定期做线上/离线特征一致性检查（log 比对）。

- Q: Bayesian Smoothing 在 CTR 特征中怎么用？
  A: 对低频 item 的历史 CTR 用全局均值做平滑：CTR_smooth = (clicks + α × global_CTR) / (impressions + α)，α 是平滑系数（通常 100-1000）。频率越低，平滑后的值越接近全局均值，防止低频 item 的 CTR 估计方差过大。

- Q: 为什么高基数 ID 特征要用 Embedding 而不是 One-hot？
  A: One-hot 的维度 = 类别数量，对 user_id（亿级）完全不可行。Embedding 将高维 one-hot 压缩到低维稠密向量（通常 16-256 维），参数量从 O(N) 减小到 O(N×d)，d≪N，同时相似实体的 embedding 可以学习到相似性。
