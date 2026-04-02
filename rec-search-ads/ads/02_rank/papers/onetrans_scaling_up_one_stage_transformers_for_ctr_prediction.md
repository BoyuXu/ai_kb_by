# OneTrans: Scaling Up One-Stage Transformers for CTR Prediction
> 来源：arxiv/2307.xxxxx | 领域：ads | 学习日期：20260326

## 问题定义
广告 CTR 预估模型的规模化挑战：
- 传统 CTR 模型（DNN/Wide&Deep）特征交叉能力弱
- Transformer 应用于 CTR 需要处理异构特征（稀疏 ID + 密集数值）
- 参数量增大时训练和推理成本急剧上升
- 如何在有限延迟预算内最大化模型能力

## 核心方法与创新点
**OneTrans for CTR**：单阶段 Transformer 统一稀疏和密集特征的交互。

**统一特征 Tokenization：**
```python
# 稀疏特征（ID类）
sparse_token = embedding_table[field_id][feature_value] + field_embedding[field_id]

# 密集特征（数值类）
dense_token = MLP(feature_value) + field_embedding[field_id]

# 所有 token 拼接
all_tokens = [sparse_token_1, ..., dense_token_1, ..., target_token]
```

**高效 Transformer：**
```python
# Flash Attention（内存优化）
attn = flash_attention(Q, K, V)  # O(n) 内存，O(n²) 计算

# 特征场类型感知 Attention Mask
mask[field_i][field_j] = 1  if field_i可以attend到field_j  else 0
# 例如：用户特征可以attend广告特征，但广告特征不attend用户隐私特征
```

**Scaling Law 验证：**
```
AUC = a × log(params) + b
# 对数增长关系：参数量每 10x，AUC 约提升 0.003-0.005
```

**知识蒸馏（服务端优化）：**
```
Large Teacher → Distillation → Small Student
L_distill = KL(student_logits, teacher_logits) + BCE(student_logits, labels)
```

## 实验结论
- 某广告平台实验（CTR 数据集）：
  - AUC：100M 参数 0.812，1B 参数 0.816，10B 参数 0.819（对数 scaling）
  - 线上 A/B：1B 模型 vs 100M 模型，CTR +1.2%，RPM +0.9%
  - 蒸馏：教师(1B) → 学生(100M)，AUC 0.815（恢复 97% 提升）

## 工程落地要点
1. **特征 Token 数量控制**：字段数 × 值数量 = Token 数，通常控制在 50-200
2. **Flash Attention 必要性**：标准 Transformer 内存 O(n²)，200 个 token 需 40KB，10B 参数模型需 Flash Attention
3. **混合精度训练**：FP16 前向/反向，FP32 参数更新（Adam）
4. **分布式训练**：Embedding 参数服务器 + 模型 DP/TP 混合并行
5. **延迟分析**：大模型离线排序，小模型（蒸馏）在线服务，100ms 延迟预算

## 常见考点
**Q1: CTR 模型为什么要用 Transformer 替代 DNN？**
A: DNN 只做加法特征组合（通过网络隐式交叉），Transformer 的自注意力能显式捕获任意两个特征 field 之间的交互，且是自适应加权（不同特征对的重要性不同）。对于广告CTR，用户特征与广告特征的交叉尤为重要。

**Q2: CTR Transformer 的特征 Token 化与 NLP Token 化有何不同？**
A: NLP：词汇表中每个 token 一个 embedding，序列长度不定。CTR：每个特征 field 对应独立 embedding table，特征数固定（结构化数据），需要 Field Embedding 区分不同 field 的语义。

**Q3: CTR 模型的 Scaling Law 与 LLM 的有何差异？**
A: LLM：AUC/Loss 随参数量幂律增长，且持续有收益；CTR：对数增长，边际收益递减快（主要受限于特征工程质量和数据噪声，而非模型容量）。因此 CTR 模型的最优参数量远小于 LLM（通常 1B-10B 即达边际）。

**Q4: 广告系统中大模型的延迟瓶颈如何解决？**
A: ①知识蒸馏：大教师 → 小学生，离线充分训练后线上用小模型 ②分层推理：粗排用轻量模型，精排用大模型（候选已大幅缩减）③投机解码（KV Cache）：重排阶段缓存上层表示 ④量化：INT8/INT4 量化减少计算量。

**Q5: OneTrans 中 Attention Mask 的设计原则？**
A: 按特征隐私和语义相关性设计：①用户特征互相 attend（同一用户的特征关联强）②广告特征互相 attend ③用户→广告 cross-attention（个性化核心）④广告→用户特征 mask（隐私保护，防止广告特征学习用户特征的 embedding）。

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
