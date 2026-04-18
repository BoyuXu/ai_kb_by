## 多模态推荐模型与应用

本文深入展开多模态序列建模、GNN+多模态、冷启动多模态解决方案等核心面试考点。

---

## 1. 多模态序列兴趣建模

### 核心思路

传统序列模型（DIN/SASRec）用 ID Embedding 表示物品，多模态序列模型用多模态融合 Embedding 替代，捕捉更丰富的兴趣演化。

### 架构设计

```
输入：用户行为序列 [item_1, item_2, ..., item_n]

第1层 - 商品多模态编码：
  每个 item_i：
    h_img_i = ViT(image_i)           # 图像编码，768维
    h_txt_i = BERT(title_i)          # 文本编码，768维
    h_id_i  = Embedding(id_i)        # ID嵌入，64维
    h_item_i = MLP(Concat(h_img_i, h_txt_i, h_id_i))  # 融合，128维

第2层 - 序列编码：
  H = Transformer_Encoder([h_item_1, ..., h_item_n])
  # 多头自注意力 + 位置编码 + FFN
  # 输出：每个位置的上下文感知表示

第3层 - 兴趣提取（Target Attention）：
  # 用候选商品 query 用户序列
  alpha_i = softmax(h_candidate^T * W * H_i)
  u = sum(alpha_i * H_i)             # 用户兴趣向量

第4层 - 预测：
  p = sigmoid(u^T * h_candidate)     # 点击概率
```

### 双通道序列模型

分别建模视觉兴趣和文本兴趣序列：

```
视觉通道：[img_1, img_2, ..., img_n] → ViT → Transformer → v_user
文本通道：[txt_1, txt_2, ..., txt_n] → BERT → Transformer → t_user

跨通道交互：
  v_user_enhanced = CrossAttn(Q=v_user, KV=t_user)
  t_user_enhanced = CrossAttn(Q=t_user, KV=v_user)

融合：
  u = MLP(Concat(v_user_enhanced, t_user_enhanced))
```

优势：视觉和文本兴趣可能不同步变化（如文字偏好稳定但视觉风格在变），分通道建模更灵活。

### 时间衰减机制

```
# 在注意力计算中引入时间衰减
time_decay = exp(-lambda * (t_now - t_i))  # lambda 为衰减率
alpha_i = softmax(score_i * time_decay_i)
```

作用：近期行为权重更高，处理兴趣漂移（比如用户最近从运动鞋转向皮鞋）。

### 多粒度兴趣建模

```
# 短期兴趣：最近 K 个行为的多模态序列
short_interest = Transformer(recent_K_items)

# 长期兴趣：全量行为的多模态聚类中心
long_interest = KMeans_Centers(all_item_embeddings)

# 融合
user_interest = Gate(short_interest, long_interest)
# Gate 根据候选物品动态分配权重
```

---

## 2. GNN + 多模态推荐

### 异构图多模态网络

构建包含多种节点和边的异构图：

```
节点类型：
  - 用户节点：用户 ID Embedding
  - 物品节点：多模态融合 Embedding（图像+文本+ID）
  - 属性节点：品牌、类目、标签等

边类型：
  - 用户-物品：点击/购买/收藏
  - 物品-物品：共现/相似
  - 物品-属性：归属关系

消息传递（多模态增强）：
  h_i^(l+1) = AGG({h_j^(l) | j in N(i)})
  # 聚合邻居信息时，物品节点的 h 包含多模态信息
```

### MMGCN（Multi-Modal Graph Convolution Network）

核心创新：为每个模态构建独立的用户-物品图，分别做图卷积后融合。

```
# 视觉模态图
H_visual = GCN_visual(A, X_visual)    # A: 交互矩阵

# 文本模态图
H_text = GCN_text(A, X_text)

# ID 模态图
H_id = GCN_id(A, X_id)

# 融合
H_item = H_visual + H_text + H_id     # 逐元素加
# 或用注意力加权
alpha = softmax(MLP(Concat(H_visual, H_text, H_id)))
H_item = alpha_v * H_visual + alpha_t * H_text + alpha_id * H_id
```

优势：不同模态的图结构信息传播路径不同，避免噪声模态污染。

### LATTICE（挖掘模态间潜在结构）

核心思路：利用多模态特征构建物品-物品相似图（潜在结构），增强 GNN 的消息传递。

```
# 构建模态感知的物品关系图
S_visual(i,j) = cos(x_visual_i, x_visual_j)
S_text(i,j) = cos(x_text_i, x_text_j)

# 融合为统一的物品关系图（取 top-k 邻居）
A_latent = TopK(w_v * S_visual + w_t * S_text)

# 在原始交互图 + 潜在结构图上做 GCN
H = GCN(A_interaction + A_latent, X)
```

### 图对比学习 + 多模态

```
# 视图1：原始交互图上的 GNN 表示
z1 = GNN(A_original, X_multimodal)

# 视图2：增强图（边 dropout / 节点 dropout）
z2 = GNN(A_augmented, X_multimodal)

# 对比损失
L_cl = -log(exp(sim(z1_i, z2_i)/tau) / sum_j(exp(sim(z1_i, z2_j)/tau)))

# 总损失
L = L_rec + lambda * L_cl
```

作用：提升多模态 GNN 表示的鲁棒性，缓解数据稀疏。

---

## 3. 冷启动的多模态解决方案

### 冷启动问题本质

新物品/新用户无历史交互数据，纯协同过滤失效。多模态信息是天然的冷启动缓解手段。

### 新物品冷启动

```
策略1：多模态 Embedding 直接入库
  - 新物品上架时，用预训练模型（CLIP/BERT/ViT）提取多模态特征
  - 投影到与协同过滤空间对齐的共享空间
  - 直接参与向量召回

策略2：多模态相似物品推荐
  - 在多模态特征空间找 K 个最相似的已有物品
  - 用这些物品的交互数据作为新物品的先验
  - warm_embedding = mean([emb(similar_item_1), ..., emb(similar_item_K)])

策略3：内容特征 + 元学习
  - MAML/Prototypical Network 在少样本场景下快速适配
  - 输入：物品多模态特征
  - 目标：用极少交互数据快速学会新物品的用户偏好
```

### 新用户冷启动

```
策略1：多模态兴趣探索
  - 展示多样化的多模态内容（不同风格的图片/视频）
  - 用户选择偏好后，用多模态特征初始化用户画像
  - 比传统的类目选择更直观、信息量更大

策略2：跨平台多模态迁移
  - 用户在其他平台的多模态交互数据（如社交媒体发帖）
  - 提取视觉/文本偏好迁移到推荐系统

策略3：人口统计 + 多模态群体画像
  - 相同年龄/性别/地区用户群的多模态偏好中心
  - 作为新用户初始画像
```

### 跨域冷启动

```
# 源域（有数据）→ 目标域（冷启动）
# 多模态作为跨域桥梁

# 物品在源域的协同过滤 Embedding
h_source_cf = CF_Encoder(item_source)

# 物品的多模态 Embedding（跨域通用）
h_mm = CLIP(image, text)

# 对齐网络：多模态 → 目标域协同过滤空间
h_target_cf = AlignNet(h_mm)

# 训练：在源域数据上学习 AlignNet
# 推理：目标域新物品通过 h_mm → AlignNet → h_target_cf 获得初始表示
```

---

## 4. 多模态特征工程实践

### 特征提取模型选择

```
模态     | 轻量（在线）         | 重型（离线）
图像     | MobileNet-V3 (3ms)  | ViT-L/14 (50ms)
文本     | DistilBERT (5ms)    | BERT-Large (30ms)
视频     | 抽帧+MobileNet      | SlowFast (200ms)
音频     | Mel+1D CNN (2ms)    | Whisper (100ms)
```

### 特征归一化与对齐

```
# 1. L2 归一化（必做）
x_norm = x / ||x||_2

# 2. 维度对齐
# 图像 2048维 → MLP → 128维
# 文本 768维 → MLP → 128维
# 统一到同一维度后才能融合

# 3. 分布对齐（进阶）
# BatchNorm / LayerNorm 统一各模态的均值和方差
x_aligned = LayerNorm(MLP(x_norm))
```

### 实时特征管道

```
新物品上架
  → Kafka 消息
  → Flink 调用 GPU 推理服务
  → 提取图像/文本/视频特征
  → 写入 Feature Store + 向量数据库
  → 可被召回/排序服务查询

延迟要求：
  - 图文特征：物品上架后 < 5分钟可被召回
  - 视频特征：< 30分钟（需转码+抽帧+推理）
```

### 离线 vs 在线特征

```
离线预计算（物品侧）：
  - 物品多模态 Embedding（CLIP/ViT/BERT 提取）
  - 物品间多模态相似度矩阵
  - 物品多模态聚类标签

在线实时计算（用户侧）：
  - 用户近期行为序列的多模态兴趣向量
  - 用户 Query 的文本编码
  - 用户-候选物品的跨模态注意力分数
```

---

## 5. 评估与消融实验

### 多模态消融实验设计

```
实验组                | 说明
ID only              | 纯协同过滤基线
ID + Text            | 加文本模态
ID + Image           | 加图像模态
ID + Text + Image    | 双模态
Full (+ Video/Audio) | 全模态

每个实验组保持模型结构一致，仅改变输入特征
关注指标：AUC, NDCG@K, Recall@K, 冷启动物品的 Hit Rate
```

### 模态贡献分析

```
# 注意力权重分析
attention_weights = model.get_modal_attention()
# 按场景统计
服饰类：视觉 0.6, 文本 0.3, ID 0.1
书籍类：视觉 0.1, 文本 0.7, ID 0.2
3C 类：视觉 0.4, 文本 0.4, ID 0.2
```

### 冷启动专项评估

```
# 按物品交互量分组评估
cold_items  = items with < 10 interactions
warm_items  = items with 10-100 interactions
hot_items   = items with > 100 interactions

# 多模态模型在 cold_items 上的提升通常最大（+20-40%）
# 在 hot_items 上提升较小（+3-8%），因为 CF 信号已充分
```

---

## 6. 面试高频追问

Q: 多模态序列模型和普通序列模型的本质区别？
A: 物品表示从 ID Embedding 升级为多模态融合 Embedding。ID Embedding 只编码协同过滤信号，多模态 Embedding 还编码了物品内容语义，对冷启动和跨域泛化更强。

Q: GNN + 多模态的优势在哪？
A: GNN 传播邻居信息时，多模态特征让物品节点携带内容语义。两个从未共现但视觉相似的物品，通过多模态 GNN 可以互相增强表示。纯 ID-based GNN 做不到这点。

Q: 多模态特征是否总能提升效果？
A: 不一定。低质量模态（模糊图片、无意义描述）可能引入噪声。需要：1) 模态质量过滤 2) 门控网络自动降权低质量模态 3) 消融实验验证每个模态的净贡献。

Q: 视频特征处理的工程难点？
A: 1) 视频转码和抽帧的计算量大 2) 时序信息的有效压缩（取关键帧 vs 全帧3D CNN） 3) 特征维度远高于图文 4) 延迟要求下通常只抽 5-10 帧用图像模型处理。

Q: 多模态推荐系统的主要瓶颈？
A: 1) 特征提取的 GPU 成本（大模型提取一张图 ~50ms） 2) 特征存储（每个物品 5-10KB 的向量 * 亿级物品） 3) 在线推理延迟（跨模态注意力计算） 4) 多模态训练数据的标注成本。解决：轻量模型 + 知识蒸馏 + 离线预计算 + 异步更新。
