# 对比学习：从 InfoNCE 到工业级召回

> 标签：#对比学习 #InfoNCE #SimCLR #MoCo #BYOL #负样本 #召回 #双塔模型 #SentenceBERT

---

## 1. 对比学习的核心目标

### 1.1 直觉理解

对比学习的目标：**让语义相似的样本在向量空间中靠近，语义不相似的样本远离**。

类比：将空间中的数据点想象为磁铁：
- 相似样本之间有引力（被拉近）
- 不相似样本之间有斥力（被推远）

学习到的表示（Representation）不依赖特定的下游任务标签，而是捕获了数据内在的语义结构。

### 1.2 InfoNCE Loss 完整推导

**设定**：给定锚点样本 $q$（query）、正样本 $k^+$（同类/增强版本）、负样本 $k_1^-, k_2^-, \ldots, k_{N-1}^-$。

**InfoNCE Loss**：

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(z_i \cdot z_i^+ / \tau)}{\exp(z_i \cdot z_i^+ / \tau) + \sum_{j=1}^{N-1} \exp(z_i \cdot z_j^- / \tau)}$$

等价地：

$$\mathcal{L} = -\mathbb{E}_{(q, k^+, \{k_j^-\})} \left[\log \frac{e^{s(q, k^+)/\tau}}{e^{s(q, k^+)/\tau} + \sum_j e^{s(q, k_j^-)/\tau}}\right]$$

其中 $s(u, v) = u \cdot v$（点积相似度，如已 L2 归一化则等于余弦相似度）。

**与交叉熵的等价性**：

这就是一个 $(N-1)+1=N$ 类分类问题的交叉熵，正类是 $k^+$，负类是 $\{k_j^-\}$。

**梯度分析**：

对 $q$ 的梯度：

$$\frac{\partial \mathcal{L}}{\partial q} = \frac{1}{\tau}\left[\underbrace{(p^+ - 1) \cdot k^+}_{\text{正样本：拉近 q 和 k^+}} + \underbrace{\sum_j p_j \cdot k_j^-}_{\text{负样本：推开 q 和 k\_j^-}}\right]$$

其中 $p^+ = \text{softmax}(s(q, k^+)/\tau)$。

### 1.3 温度参数 $\tau$ 的深入分析

**$\tau$ 的信息论含义**：

$$\mathcal{L}_{\text{InfoNCE}} \geq -I(q; k^+)$$

InfoNCE Loss 是互信息 $I(q; k^+)$ 的下界（van den Oord et al. 2018），最大化互信息等价于最小化 InfoNCE。温度参数影响下界的紧凑程度：小 $\tau$ 使下界更紧（但梯度方差更大）。

**$\tau$ 对学习的影响**：

```python
import torch
import torch.nn.functional as F

def analyze_temperature_effect(similarity_scores, tau_values=[0.01, 0.07, 0.1, 0.5]):
    """
    分析温度对梯度信号分布的影响
    similarity_scores: [sim_pos, sim_neg1, sim_neg2, ...]
    """
    for tau in tau_values:
        scaled = torch.tensor(similarity_scores) / tau
        probs = F.softmax(scaled, dim=0)
        
        # 分析梯度集中度：越小的 tau，梯度越集中在最难的负样本
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        print(f"tau={tau:.2f}: 概率分布 = {probs.numpy().round(3)}, 熵 = {entropy:.3f}")
```

| $\tau$ | 概率集中度 | 训练信号 | 适用场景 |
|--------|-----------|---------|---------|
| 0.01 | 极端集中（hard max）| 几乎只从最难负样本学习 | 大 batch，负样本质量高 |
| 0.07 | 适中 | 均衡学习 | SimCLR 默认值 |
| 0.1-0.3 | 较分散 | 温和学习 | 推荐系统常用 |
| 1.0 | 均匀 | 几乎无梯度信号 | 过大，不用 |

---

## 2. 负样本策略

### 2.1 In-batch Negative（批内负样本）

**原理**：在一个 batch 内，样本 $i$ 的正例是 $k_i^+$，负例是其他所有样本的正例 $\{k_j^+\}_{j \neq i}$。

```python
def in_batch_contrastive_loss(q_embeddings, k_embeddings, tau=0.07):
    """
    q_embeddings: (N, d) 查询向量（如用户嵌入）
    k_embeddings: (N, d) 键向量（如物品嵌入，第 i 个是第 i 个查询的正例）
    """
    # L2 归一化
    q = F.normalize(q_embeddings, dim=-1)
    k = F.normalize(k_embeddings, dim=-1)
    
    # 计算所有对的相似度矩阵 (N, N)
    sim_matrix = torch.matmul(q, k.T) / tau
    
    # 对角线是正例，其余是负例
    labels = torch.arange(len(q), device=q.device)
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss
```

**优点**：完全无需额外存储负样本，实现简单高效。

**缺点：采样偏差（Sampling Bias）**：
- 热门物品出现在 batch 中的概率正比于其频率
- 热门物品更频繁地被当作负样本
- 模型学会压低热门物品得分，导致马太效应（冷门物品被过度召回）

**修正（Frequency-Based Correction）**：

$$\text{score}_{corrected}(q, k_j) = \text{score}(q, k_j) - \log P(k_j)$$

其中 $P(k_j)$ 是物品 $k_j$ 的采样概率（正比于出现频率）。

### 2.2 Hard Negative Mining（难负样本挖掘）

**直觉**：若负样本太"简单"（与正样本差距很大），梯度几乎为 0，学不到有用信息。难负样本（语义相近但实际上是负样本）提供更强的学习信号。

**策略1：ANN 检索挖掘**：

```python
def mine_hard_negatives(query_embs, item_embs, positive_ids, top_k=50, hard_n=5):
    """
    在每个 batch 训练前，用 ANN 找难负样本
    """
    hard_negatives = []
    
    for i, (q_emb, pos_id) in enumerate(zip(query_embs, positive_ids)):
        # ANN 检索最近的 top_k 个物品
        scores = item_embs @ q_emb
        top_k_indices = scores.argsort()[-top_k-1:][::-1]
        
        # 排除正例，取最近的 hard_n 个作为难负样本
        hard_neg_indices = [idx for idx in top_k_indices if idx != pos_id][:hard_n]
        hard_negatives.append(hard_neg_indices)
    
    return hard_negatives
```

**策略2：Debiased Contrastive Learning**（修正假负样本）：

假负样本（False Negative）：与 query 高度相关但被错误地当作负样本。

修正损失（Chuang et al. 2020）：

$$\mathcal{L}_{debiased} = -\log \frac{e^{s(q, k^+)/\tau}}{e^{s(q, k^+)/\tau} + N \cdot g(q, \{k_j^-\})}$$

$$g(q, \{k_j^-\}) = \max\left(\frac{1}{N}\sum_j e^{s(q, k_j^-)/\tau} - \tau^+ \cdot e^{1/\tau}, 0\right)$$

其中 $\tau^+$ 是正样本的先验概率，减去了负样本中可能混入的正样本贡献。

---

## 3. SimCLR / MoCo / BYOL 框架

### 3.1 SimCLR：大 batch + 数据增强

**核心思路**：同一张图片的两种随机增强视图互为正例，同 batch 内所有其他图片视图为负例。

```python
class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim=128):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(encoder.output_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, projection_dim)
        )
    
    def forward(self, x1, x2):
        # 两路增强视图
        h1, h2 = self.encoder(x1), self.encoder(x2)
        z1 = F.normalize(self.projector(h1), dim=-1)
        z2 = F.normalize(self.projector(h2), dim=-1)
        
        # 对称 NT-Xent 损失（SimCLR 的 InfoNCE 变体）
        N = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # (2N, d)
        sim = torch.matmul(z, z.T) / self.tau  # (2N, 2N)
        
        # 掩码对角线（自相似性）
        mask = torch.eye(2*N, dtype=bool)
        sim.masked_fill_(mask, float('-inf'))
        
        # 正例位置：z1[i] 和 z2[i] 互为正例（位置 i 和 i+N）
        labels = torch.cat([torch.arange(N, 2*N), torch.arange(N)])
        return F.cross_entropy(sim, labels)
```

**SimCLR 的关键发现**：
- Projection Head 很重要：加在 encoder 后的 2 层 MLP 显著提升效果
- 数据增强组合很重要：随机裁剪+颜色抖动是图像任务的关键组合
- 需要大 batch（4096-8192）：batch 内负样本数量直接影响效果

### 3.2 MoCo：动量编码器 + 队列

**解决 SimCLR 的大 batch 问题**：维护一个负样本队列（大小远超 batch size）。

```python
class MoCo(nn.Module):
    def __init__(self, encoder, K=65536, m=0.999, tau=0.07):
        super().__init__()
        self.K = K  # 队列大小
        self.m = m  # 动量系数
        self.tau = tau
        
        # 两个编码器：在线编码器 + 动量编码器
        self.encoder_q = encoder   # 正常反向传播更新
        self.encoder_k = copy.deepcopy(encoder)  # 动量更新，不反向传播
        
        # 冻结动量编码器参数
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        
        # 负样本队列
        self.register_buffer("queue", F.normalize(torch.randn(K, 128), dim=-1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def momentum_update(self):
        """指数滑动平均更新动量编码器"""
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = self.m * p_k.data + (1 - self.m) * p_q.data
    
    def forward(self, x_q, x_k):
        q = F.normalize(self.encoder_q(x_q), dim=-1)  # (N, d)
        
        with torch.no_grad():
            self.momentum_update()
            k = F.normalize(self.encoder_k(x_k), dim=-1)  # (N, d)
        
        # 正样本相似度
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # (N, 1)
        
        # 负样本相似度（队列中的历史样本）
        l_neg = torch.einsum('nc,kc->nk', [q, self.queue.clone().detach()])  # (N, K)
        
        logits = torch.cat([l_pos, l_neg], dim=1) / self.tau
        labels = torch.zeros(len(q), dtype=torch.long)
        
        # 更新队列
        self._dequeue_and_enqueue(k)
        
        return F.cross_entropy(logits, labels)
```

### 3.3 BYOL：无负样本

**惊人发现**：不需要负样本也能学好表示，只要避免模型坍塌（表示全部变成同一个向量）。

**关键设计**：
1. **动量编码器**（Target Network）：提供稳定的"目标"表示
2. **Predictor**（额外的预测头）：在线网络预测目标网络的输出
3. **不对称性**：在线网络有 predictor，目标网络没有，打破对称性防止坍塌

$$\mathcal{L}_{BYOL} = \left\| q_\theta(z_q) - z_k \right\|_2^2$$

其中 $q_\theta$ 是 predictor，$z_q$ 来自在线编码器，$z_k$ 来自动量编码器（stop gradient）。

---

## 4. 在推荐/广告召回中的应用

### 4.1 用户-物品双塔的对比训练

```python
class TwoTowerModel(nn.Module):
    def __init__(self, user_dim, item_dim, embed_dim=128):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(user_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
        self.item_tower = nn.Sequential(
            nn.Linear(item_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
    
    def forward(self, user_features, item_features):
        u = F.normalize(self.user_tower(user_features), dim=-1)
        v = F.normalize(self.item_tower(item_features), dim=-1)
        return u, v
    
    def compute_loss(self, u, v_pos, v_neg_batch, tau=0.1):
        """
        u: (N, d) 用户向量
        v_pos: (N, d) 正样本物品向量
        v_neg_batch: 来自 batch 内其他用户的正样本（in-batch negative）
        """
        # 拼接正负样本
        N = u.shape[0]
        all_v = torch.cat([v_pos, v_neg_batch], dim=0)  # 实际是同一个 v_pos
        
        sim = torch.matmul(u, all_v.T) / tau  # (N, N)
        labels = torch.arange(N, device=u.device)
        return F.cross_entropy(sim, labels)
```

### 4.2 为什么对比训练优于传统协同过滤

| 维度 | 传统协同过滤（矩阵分解）| 对比学习双塔 |
|------|----------------------|-----------|
| 特征利用 | 只用交互矩阵（ID 特征）| 可用侧信息（属性、文本、图像）|
| 冷启动 | 差（新用户/物品无交互）| 好（可用内容特征初始化）|
| 语义理解 | 无（纯协同信号）| 有（对比学习捕获语义）|
| 负样本设计 | 随机（可随机负）| 灵活（可用难负样本）|
| 扩展性 | 难以加入新类型特征 | 易于多模态扩展 |

### 4.3 采样偏差修正（频率加权负采样）

**问题**：In-batch 负采样中，热门物品被采样为负样本的概率与其出现频率成正比，模型学会压低热门物品得分。

**修正公式**（Yi et al. 2019，Google 双塔论文）：

修正后的 logit：

$$\text{logit}_{corrected}(u, v) = \text{logit}(u, v) - \log(\hat{p}(v))$$

其中 $\hat{p}(v)$ 是物品 $v$ 的采样概率（从训练数据频率估计）。

```python
def frequency_corrected_loss(u_embs, v_embs, item_frequencies, tau=0.07):
    """
    u_embs: (N, d) 用户向量
    v_embs: (N, d) 物品向量  
    item_frequencies: (N,) 每个物品的采样概率
    """
    u = F.normalize(u_embs, dim=-1)
    v = F.normalize(v_embs, dim=-1)
    
    sim = torch.matmul(u, v.T) / tau  # (N, N)
    
    # 频率修正：减去 log(采样概率)（负数，所以是增大负样本的 logit）
    freq_correction = torch.log(item_frequencies + 1e-8).unsqueeze(0)  # (1, N)
    sim = sim - freq_correction
    
    labels = torch.arange(len(u), device=u.device)
    return F.cross_entropy(sim, labels)
```

---

## 5. SentenceBERT 和语义召回

### 5.1 双编码器训练

SentenceBERT（SBERT）将 BERT 适配为高效的句子向量生成模型：

```python
class SentenceBERT(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        # 池化方式：mean pooling（比 [CLS] 更稳定）
    
    def mean_pooling(self, token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (token_embeddings * mask).sum(1) / mask.sum(1)
    
    def encode(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sentence_emb = self.mean_pooling(
            outputs.last_hidden_state, attention_mask
        )
        return F.normalize(sentence_emb, dim=-1)
    
    def forward(self, anchor_ids, pos_ids, neg_ids, anchor_mask, pos_mask, neg_mask):
        a = self.encode(anchor_ids, anchor_mask)
        p = self.encode(pos_ids, pos_mask)
        n = self.encode(neg_ids, neg_mask)
        
        # Triplet Loss
        loss = F.triplet_margin_loss(a, p, n, margin=0.5)
        
        # 或 InfoNCE Loss（效果通常更好）
        # loss = in_batch_contrastive_loss(a, p, tau=0.05)
        
        return loss
```

### 5.2 在广告创意检索中的应用

**场景**：广告主上传商品描述，系统需要为其匹配相关的历史创意广告（文案、图片）。

**流程**：
1. 离线：对所有历史广告创意用 SentenceBERT 编码，建 HNSW 索引
2. 在线：用户输入商品描述 → SentenceBERT 实时编码 → ANN 检索 Top-K 相关创意

**Fine-tuning 策略**：
- 正样本对：同一广告主、同类商品的创意对
- 负样本：不同品类的广告创意（easy negative）+ 同品类但不同商品（hard negative）
- 对比学习 + 分类损失（有商品类别标签时）联合训练

---

## 6. 面试考点

### Q1：InfoNCE 中 batch size 越大越好吗？有没有上限？

更大的 batch 意味着更多的负样本，理论上更接近真实的负样本分布，效果更好。但存在实际限制：(1) 显存瓶颈：batch_size=4096 时，如每个样本 d=768 维 FP32，仅 embedding 就占 4096×768×4×3（三路）≈ 36MB，加上模型权重接近显存上限；(2) 假负样本问题：batch 越大，同类样本碰到同类"负样本"的概率越高，引入噪声；(3) 收益递减：实验表明 batch_size 超过 8192 后提升趋于饱和。MoCo 通过队列（65536）解决 batch 限制，效果接近 SimCLR 大 batch。

### Q2：对比学习中的 Representation Collapse（表示坍塌）是什么？如何防止？

表示坍塌：所有样本的向量表示收敛到同一个点（或超低秩子空间），L2 归一化后都在单位球面的一个点，无法区分任何样本。原因：如果没有负样本，最小化正样本之间的距离的"捷径"解是让所有向量相同。防止方法：(1) 负样本（SimCLR、MoCo）：明确推开不相似样本；(2) BYOL 的不对称性：predictor + stop gradient 的非对称设计防止坍塌；(3) VICReg/Barlow Twins：通过协方差正则使不同维度的信息独立（反坍塌的正则化项）。

### Q3：负样本的质量如何影响对比学习效果？

难负样本（Hard Negative）提供更强的梯度信号，但存在假负样本风险：若将语义相近的样本错误标为负样本，会损害表示质量。最佳实践：混合使用随机负样本（70%）和难负样本（30%）；对于用户行为数据，可以排除用户历史点击过的物品（减少假负样本）；对于文本语义任务，同文档不同段落互为正例，不同文档为负例，通常假负样本较少。

### Q4：双塔模型的 embedding 维度如何选择？（另见 embedding_ann.md Q7）

权衡：维度高 → 表达能力强，但 ANN 索引内存大，检索速度慢。经验规律：百万量级物品库，64-256 维足够；千万量级，128-512 维；亿级，考虑 PQ 压缩（允许更高原始维度）。另一角度：若双塔输出的 embedding 后续用于 ANN 检索，维度应使 HNSW 或 IVF 索引的内存 < 可用 GPU 内存。推荐从 128 维开始，通过实验评估召回率 vs 维度的关系曲线。

### Q5：对比学习和普通分类损失有什么本质区别？

分类损失（交叉熵）：每个样本映射到固定的 C 个类别，类别之间无相似性概念，"猫"和"狗"的向量关系没有约束。对比损失：没有固定类别，直接在 embedding 空间中定义相似性关系，同类样本的向量距离比不同类更小。对比学习的 embedding 更适合以下任务：语义搜索（余弦相似度有意义）、零样本分类（新类别无需重训）、聚类（相似样本自然成簇）。本质上，对比学习是一种度量学习（Metric Learning），而分类学习是判别式学习。

### Q6：MoCo 的动量系数 m 如何影响训练？

m 控制动量编码器（Key Encoder）的更新速度：$\theta_k \leftarrow m \theta_k + (1-m) \theta_q$。m 接近 1（如 0.999）：Key Encoder 更新极慢，提供非常稳定的负样本 key，队列内的负样本高度一致（Key Encoder 变化很小），对比损失的"学习目标"稳定。m 较小（如 0.9）：Key Encoder 更新较快，队列内的负样本不一致（同一物品在队列不同位置的表示可能差异大），训练噪声大。实践中 m=0.999 是标准配置，可以从这个值开始微调。

### Q7：如何在推荐系统中处理多模态对比学习？

多模态场景（图文广告召回）：用户行为（用户塔）+ 广告图文（广告塔），正例是用户点击的广告，需要跨模态对齐。技术路线：(1) Late Fusion：图像 CLIP encoder + 文本 BERT encoder 分别编码，concat 后投影到统一空间；(2) Cross-Modal Alignment：图文 embedding 做对比学习（图文正例对）+ 用户行为对比学习（用户-广告正例对），多目标联合训练；(3) CLIP 风格：直接将图文广告作为 key，用户行为作为 query，图文匹配作为自监督信号。

---

## 参考资料

- van den Oord et al. "Representation Learning with Contrastive Predictive Coding" (InfoNCE, 2018)
- Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR, 2020)
- He et al. "Momentum Contrast for Unsupervised Visual Representation Learning" (MoCo, 2020)
- Grill et al. "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning" (BYOL, 2020)
- Yi et al. "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations" (频率修正, 2019)
- Reimers & Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (2019)
- Chuang et al. "Debiased Contrastive Learning" (假负样本修正, 2020)
