# 表示学习：Embedding 到特征交互

> **创建日期**: 2026-04-13 | **合并来源**: Embedding学习_推荐系统表示基石, 推荐系统特征工程体系, 推荐系统ScalingLaw_Wukong
>
> **核心命题**: 推荐系统从手工特征工程走向端到端表示学习，Embedding 是一切的基石，Scaling Law 证明 embedding 扩展收益大于 DNN 深度扩展

---

## 一、Embedding 技术演进

```
One-hot Encoding (稀疏，无泛化)
  → Word2Vec / Skip-gram (语言模型思路)
    → Item2Vec (推荐场景迁移)
      → EGES (阿里，Side Information 图嵌入)
        → Contrastive Learning (对比学习)
          → Semantic ID / RQ-VAE (生成式推荐基础)
```

### 1.1 Word2Vec → Item2Vec

**Word2Vec 核心思想**：用上下文预测中心词（CBOW）或用中心词预测上下文（Skip-gram）。

$$
P(w_O | w_I) = \frac{\exp(\mathbf{v}_{w_O}^T \mathbf{v}_{w_I})}{\sum_{w=1}^{W} \exp(\mathbf{v}_w^T \mathbf{v}_{w_I})}
$$

**Item2Vec 迁移**：将用户行为序列类比为"句子"，物品类比为"词"：
- 用户点击序列 [item_A, item_B, item_C] → 类比 NLP 句子
- Skip-gram 目标：用 item_B 预测上下文 item_A, item_C
- 负采样近似 softmax，避免全量计算

### 1.2 EGES — 图嵌入 (阿里)

Enhanced Graph Embedding with Side Information：
- 将用户-物品交互构建为图
- 节点不仅有 ID，还有 Side Info（品类、品牌、价格区间等）
- 图上 Random Walk 生成序列 → Skip-gram 训练
- **冷启动优势**：新物品无行为数据，但有 Side Info，可以通过图邻居传播获得初始 embedding

### 1.3 对比学习 Embedding

$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j^+) / \tau)}{\sum_{k=1}^{N} \exp(\text{sim}(z_i, z_k) / \tau)}
$$

- 正样本：同一物品的不同增强视角（dropout、特征掩码）
- 负样本：batch 内其他物品
- 温度 $\tau$ 控制分布锐度

### 1.4 Semantic ID — Embedding 的终极形态

RQ-VAE 将 embedding 量化为离散 token 序列，使物品可以被自回归模型"生成"。详见 [[01_语义ID与生成式召回演进]]。

---

## 二、特征工程体系

### 2.1 特征工程演进

```
手工特征 (人工交叉，维度爆炸)
  → FM (自动二阶交叉)
    → DeepFM/Wide&Deep (低阶+高阶并行)
      → DCN/DCN-V2 (显式高阶交叉)
        → Transformer (端到端特征交叉+序列建模)
          → OneTrans (统一 tokenization)
```

### 2.2 特征分类

| 特征类型 | 示例 | 处理方式 |
|---------|------|---------|
| 用户画像 | 年龄、性别、城市 | Embedding lookup |
| 物品属性 | 类目、品牌、价格 | Embedding lookup |
| 统计特征 | 历史 CTR、曝光次数 | 分桶 + Embedding 或直接输入 |
| 上下文特征 | 时间、设备、位置 | Embedding lookup |
| 交叉特征 | 用户×物品、时间×类目 | FM/DCN 自动学习 |
| 序列特征 | 点击序列、购买序列 | DIN/Transformer |

### 2.3 FM 家族

**FM（Factorization Machine）**：
$$
\hat{y} = w_0 + \sum_i w_i x_i + \sum_i \sum_{j>i} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j
$$

复杂度从 O(n²) 优化到 O(nk)：
$$
\sum_i \sum_{j>i} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j = \frac{1}{2}\left[\left(\sum_i \mathbf{v}_i x_i\right)^2 - \sum_i \mathbf{v}_i^2 x_i^2\right]
$$

**DeepFM**：FM 处理低阶交叉 + DNN 处理高阶交叉，共享 embedding 层。

**DCN-V2**：
$$
x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l
$$

每层保留原始特征 $x_0$ 的乘法交叉，显式建模到 (l+1) 阶交叉。

### 2.4 Embedding 维度设计

**经验法则**：
- 高频特征（<1M distinct values）：embedding dim 16-64
- 中频特征（1M-100M）：embedding dim 64-128
- 低频特征/ID 特征（>100M）：embedding dim 128-256

**Wukong 发现**：embedding 维度扩展的收益大于 DNN 深度扩展。

---

## 三、Scaling Law 与表示学习

### 3.1 Wukong Scaling Law

$$
\mathcal{L}(N_e, N_d) = A \cdot N_e^{-\alpha} + B \cdot N_d^{-\beta} + C, \quad \alpha > \beta
$$

- $N_e$: embedding 参数量（稀疏参数）
- $N_d$: DNN 参数量（稠密参数）
- $\alpha > \beta$: **embedding 扩展效率 > DNN 扩展效率**

### 3.2 关键发现

| 发现 | 含义 |
|------|------|
| $\alpha > \beta$ | 增大 embedding 比增深 MLP 更有效 |
| Embedding 是性能主体 | 推荐的 scaling 体现在 embedding 维度/数量 |
| 与 LLM 不同 | LLM 的 dense 参数是主体，推荐的 sparse 参数是主体 |
| Compute-optimal 分配 | 固定算力下应优先扩大 embedding table |

### 3.3 HSTU Scaling Law

Meta HSTU 首次验证推荐领域类 LLM 的 scaling law：
$$
L(N, D) = \alpha N^{-\beta} + \gamma D^{-\delta} + \epsilon
$$

从 1.5B 到 1.5T 参数性能持续提升未饱和。

### 3.4 FuXi-Linear 的 Power-Law Scaling

在千级 token 长度下观察到线性注意力模型的 power-law scaling property，证明线性模型也有 scaling law。

---

## 四、分布式 Embedding 训练

### 4.1 TorchRec 三层切分策略

| 切分方式 | 原理 | 优劣 |
|---------|------|------|
| Table-wise | 不同 embedding 表分配到不同 GPU | 通信最少但负载均衡困难 |
| Row-wise | 按用户/物品 ID 范围分片 | 适合高维 embedding |
| Column-wise | 按 embedding 维度分片 | 更细粒度负载均衡，需全局通信 |

### 4.2 DeepRec Embedding Variable

- 动态维度：不同特征自适应 embedding 大小
- 多级存储：GPU → CPU → SSD
- 内存效率比 TF 原生高 3-5x

### 4.3 HugeCTR 跨节点扩展

All-to-All 通信模式 + 梯度累积，单 embedding 表跨多节点训练，突破单 GPU 内存限制。

---

## 五、面试高频考点

**Q1: Word2Vec 和 Item2Vec 的联系与区别？**
A: 联系：都用 Skip-gram 目标学习分布式表示。区别：(1) Item2Vec 的"句子"是用户行为序列，无固定语法结构；(2) 推荐场景的"词表"远大于 NLP（百万级物品 vs 十万级词汇）；(3) 物品共现关系比词共现更稀疏，需要更多负采样。

**Q2: EGES 如何解决冷启动？**
A: 新物品无行为数据但有 Side Info（品类、品牌等），图嵌入通过邻居传播将 Side Info 转化为初始 embedding。核心是图结构让信息在相似物品间流动。

**Q3: FM 的复杂度优化原理？**
A: 利用恒等式将 $O(n^2)$ 的二阶交叉展开为 $\frac{1}{2}[(\sum v_i x_i)^2 - \sum v_i^2 x_i^2]$，只需 O(nk) 计算。

**Q4: DCN-V2 和 FM 的本质区别？**
A: FM 只建模二阶交叉。DCN-V2 通过多层 Cross Network 显式建模任意高阶交叉，每层保留原始特征乘法交叉：$x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l$。

**Q5: 推荐系统的 Scaling Law 核心发现？**
A: embedding 扩展效率 > DNN 深度扩展效率。固定算力下应优先扩大 embedding table，而非增深 MLP。这与 LLM 的 scaling law 有本质区别。

**Q6: Embedding 维度怎么选？**
A: 经验法则：高频特征 16-64，中频 64-128，低频/ID 128-256。Wukong 发现增大 embedding 维度的 marginal return 高于增深 DNN。实践中常用超参搜索确定。

**Q7: 对比学习在推荐 embedding 中的作用？**
A: 通过正负样本对比学习更鲁棒的表示。正样本构造：同一物品的不同增强视角（dropout、特征掩码、随机裁剪）。温度 $\tau$ 控制分布锐度，较低温度让模型更关注困难负例。

---

## 相关概念

- [[concepts/embedding_everywhere|Embedding 技术全景]]
- [[concepts/attention_in_recsys|Attention 在搜广推中的演进]]
- [[concepts/sequence_modeling_evolution|序列建模演进]]
- [[concepts/generative_recsys|生成式推荐统一视角]]
