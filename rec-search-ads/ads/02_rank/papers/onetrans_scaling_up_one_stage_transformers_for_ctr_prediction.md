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
