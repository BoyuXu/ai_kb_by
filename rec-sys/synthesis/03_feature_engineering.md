# 🍈 MelonEggLearn 推荐系统特征工程深度笔记

> 📚 参考文献
> - [A-Unified-Language-Model-For-Large-Scale-Search...](../../rec-sys/papers/20260321_a-unified-language-model-for-large-scale-search-recommendation-and-reasoning-at-spotify.md) — A Unified Language Model for Large Scale Search, Recommen...
> - [Linear-Item-Item-Session-Rec](../../rec-sys/papers/20260319_linear-item-item-session-rec.md) — Linear Item-Item Model with Neural Knowledge for Session-...
> - [Din-Deep-Interest-Network](../../rec-sys/papers/20260317_din-deep-interest-network.md) — DIN：深度兴趣网络（Deep Interest Network）
> - [Etegrec Generative Recommender With End-To-End Lea](../../rec-sys/papers/20260323_etegrec_generative_recommender_with_end-to-end_lea.md) — ETEGRec: Generative Recommender with End-to-End Learnable...
> - [A Generative Re-Ranking Model For List-Level Multi](../../rec-sys/papers/20260323_a_generative_re-ranking_model_for_list-level_multi.md) — A Generative Re-ranking Model for List-level Multi-object...
> - [Act-With-Think Chunk Auto-Regressive Modeling For ](../../rec-sys/papers/20260323_act-with-think_chunk_auto-regressive_modeling_for_.md) — Act-With-Think: Chunk Auto-Regressive Modeling for Genera...
> - [Multi-Behavior-Rec-Survey](../../rec-sys/papers/20260319_multi-behavior-rec-survey.md) — Multi-behavior Recommender Systems: A Survey
> - [Deploying-Semantic-Id-Based-Generative-Retrieva...](../../rec-sys/papers/20260321_deploying-semantic-id-based-generative-retrieval-for-large-scale-podcast-discovery-at-spotify.md) — Deploying Semantic ID-based Generative Retrieval for Larg...


> **核心认知**：数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限。

---

## 目录
1. [特征类型与处理](#1-特征类型与处理)
2. [用户行为序列特征](#2-用户行为序列特征)
3. [高基数类别特征](#3-高基数类别特征)
4. [特征穿越（Data Leakage）](#4-特征穿越data-leakage)
5. [实战经验](#5-实战经验)

---

## 1. 特征类型与处理

### 1.1 数值特征（Numerical Features）

#### 归一化 vs 标准化

| 方法 | 公式 | 适用场景 | 优点 | 缺点 |
|------|------|----------|------|------|
| **Min-Max归一化** | $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$ | 有界数据、神经网络输入 | 保留原始分布、映射到[0,1] | 受异常值影响大 |
| **Z-Score标准化** | $x' = \frac{x - \mu}{\sigma}$ | 服从正态分布的数据、异常值较多 | 不受异常值影响 | 改变数据分布 |
| **MaxAbs归一化** | $x' = \frac{x}{|x_{max}|}$ | 稀疏数据 | 保持稀疏性 | 受异常值影响 |

**推荐系统实践建议**：
- 深度模型（DNN/DeepFM）：建议归一化/标准化，加速收敛
- 树模型（GBDT/XGBoost）：不需要，天然处理不同量纲
- 交叉特征构造前：建议标准化，避免数值差异过大

#### 分桶（Bucketization / Binning）

```python
# 等频分桶 - 保持每个桶样本数相等
pd.qcut(age, q=10, labels=False)

# 等距分桶 - 按数值范围均匀划分
pd.cut(age, bins=10, labels=False)

# 基于业务规则分桶
bins = [0, 18, 25, 35, 50, 100]
labels = ['未成年', '青年', '中青年', '中年', '老年']
```

**什么时候用分桶？**
- 年龄、收入等连续值 → 转换为离散年龄段
- 处理长尾分布，将高频区和低频区分开
- 作为类别特征输入Embedding层

#### 对数变换 / 幂变换

```python
# 对数变换 - 处理右偏分布（如曝光数、点击数）
log_feature = np.log1p(raw_feature)  # log(1+x) 处理0值

# Box-Cox变换 - 自动选择最优lambda
from scipy.stats import boxcox
transformed, lambda_param = boxcox(raw_feature)
```

**适用场景**：
- 用户历史点击数（长尾分布）
- 商品价格（跨度大）
- 视频观看时长

---

### 1.2 类别特征（Categorical Features）

#### One-Hot Encoding

```python
# 原生实现
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=True)
encoded = encoder.fit_transform(categories)

# Pandas简化版
pd.get_dummies(df, columns=['category'], prefix='cat')
```

**缺点**：
- 高基数特征 → 维度爆炸（千万级用户ID）
- 无法表达类别间的相似性
- 稀疏性高，存储和计算效率低

#### Label Encoding

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category_id'] = le.fit_transform(df['category'])
```

**注意事项**：
- 会引入虚假的有序关系（类别1 < 类别2）
- 只适用于树模型或配合Embedding使用

#### Target Encoding（目标编码）

```python
def target_encoding(df, cat_col, target_col, smoothing=10):
    """
    用目标变量的均值对类别特征进行编码
    smoothing: 平滑参数，防止过拟合
    """
    global_mean = df[target_col].mean()
    agg = df.groupby(cat_col)[target_col].agg(['mean', 'count'])
    
    # 贝叶斯平滑
    smoothed_mean = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
    
    return df[cat_col].map(smoothed_mean)
```

**使用要点**：
- ⚠️ **必须配合交叉验证使用**，防止数据穿越
- 适用于高基数类别特征（如城市、商品类目）
- 对低频类别做平滑处理，防止过拟合

#### Embedding（嵌入）

**本质**：将高维稀疏的类别特征映射到低维密集的向量空间，保留语义关系。

```python
import torch.nn as nn

# PyTorch实现
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, x):
        return self.embedding(x)  # [batch_size, embed_dim]
```

**推荐系统中的Embedding维度选择**：

| 特征类型 | 典型基数 | 推荐维度 | 说明 |
|---------|---------|---------|------|
| 用户ID | 千万级 | 32-128 | 用户信息丰富，维度可较高 |
| 商品ID | 百万级 | 32-64 | 中等维度 |
| 类目ID | 千级 | 8-16 | 基数小，维度不宜过大 |
| 品牌ID | 万级 | 16-32 | 中等基数 |

---

### 1.3 时序特征（Temporal Features）

#### 时间离散化特征

```python
def extract_time_features(df, timestamp_col):
    """提取基础时间特征"""
    df['hour'] = df[timestamp_col].dt.hour           # 小时（0-23）
    df['day_of_week'] = df[timestamp_col].dt.dayofweek  # 星期（0-6）
    df['day_of_month'] = df[timestamp_col].dt.day    # 日期（1-31）
    df['month'] = df[timestamp_col].dt.month         # 月份（1-12）
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # 是否周末
    
    # 时间段编码（早/中/晚/深夜）
    df['time_period'] = pd.cut(df['hour'], 
                                bins=[0, 6, 12, 18, 24], 
                                labels=['night', 'morning', 'afternoon', 'evening'])
    return df
```

#### 滑窗统计特征（Rolling Statistics）

```python
def rolling_features(df, user_col, item_col, timestamp_col):
    """
    用户历史行为滑窗统计
    """
    # 按用户和时间排序
    df = df.sort_values([user_col, timestamp_col])
    
    # 过去7天的行为统计
    df['user_click_7d'] = df.groupby(user_col)[item_col].transform(
        lambda x: x.rolling('7D', on=df[timestamp_col]).count()
    )
    
    # 过去30天的品类偏好分布熵
    df['category_entropy_30d'] = df.groupby(user_col)['category'].transform(
        lambda x: x.rolling('30D').apply(calculate_entropy)
    )
    
    return df
```

**常用滑窗统计**：
- 近N天点击率、购买率
- 近N天活跃天数占比
- 近N天消费金额均值/方差
- 近N天品类多样性（熵值）

#### 趋势特征

```python
def trend_features(df, user_col, metric_col):
    """
    用户行为趋势特征
    """
    # 近期 vs 远期活跃度对比
    recent = df.groupby(user_col)[metric_col].rolling('7D').mean()
    distant = df.groupby(user_col)[metric_col].rolling('30D').mean()
    
    # 活跃度趋势（正值表示上升）
    df['activity_trend'] = (recent - distant) / (distant + 1e-8)
    
    return df
```

#### 周期性编码

```python
def cyclical_encoding(df, col, period):
    """
    周期性特征编码（保留周期性关系）
    例如：23点和0点其实是相近的
    """
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
    return df

# 对小时进行周期性编码
df = cyclical_encoding(df, 'hour', period=24)
```

---

### 1.4 交叉特征（Cross Features）

#### 手动交叉

```python
def manual_cross_features(df):
    """领域知识驱动的手动交叉"""
    # 用户-时间交叉：用户在不同时间段的偏好
    df['user_hour_cross'] = df['user_id'].astype(str) + '_' + df['hour'].astype(str)
    
    # 类目-价格带交叉
    df['price_bucket'] = pd.qcut(df['price'], q=5, labels=['very_low', 'low', 'mid', 'high', 'very_high'])
    df['category_price_cross'] = df['category_id'].astype(str) + '_' + df['price_bucket'].astype(str)
    
    # 设备-地理位置交叉
    df['device_location'] = df['device_type'] + '_' + df['province']
    
    return df
```

#### 模型自动交叉

| 模型 | 交叉方式 | 特点 |
|------|---------|------|
| **FM** | 二阶特征交叉 | $x_i \cdot x_j$，自动学习交叉权重 |
| **FFM** | 域感知交叉 | 每个特征对每个field有独立隐向量 |
| **DeepFM** | FM + DNN | 浅层交叉 + 深层非线性 |
| **DCN** | Cross Network | 显式高阶交叉，每层交叉度+1 |
| **xDeepFM** | CIN压缩感知 | 显式高阶交叉，避免参数爆炸 |
| **AutoInt** | Multi-head Attention | 自注意力机制自动学习特征间关系 |

**DCN Cross Layer 原理**：

$$x_{l+1} = x_0 x_l^T w_l + b_l + x_l = f(x_l, w_l, b_l) + x_l$$

- $x_0$：原始输入
- $x_l$：第l层输出
- $w_l, b_l$：可学习参数
- 每一层增加一阶交叉，L层可实现L+1阶交叉

---

## 2. 用户行为序列特征

### 2.1 行为序列截断策略

#### 最近N条截断（Last-N）

```python
def truncate_last_n(behavior_seq, n=50, padding_idx=0):
    """
    保留最近N条行为，早期行为截断
    适用于：实时性强的场景（新闻、短视频）
    """
    if len(behavior_seq) >= n:
        return behavior_seq[-n:]
    else:
        # 前置填充
        return [padding_idx] * (n - len(behavior_seq)) + behavior_seq
```

**特点**：
- 简单高效，计算复杂度固定
- 可能丢失长期兴趣信号
- 适用于行为快速变化的场景

#### SIM 超长序列处理

**思想**：通过两阶段检索，从超长历史（如10000条）中筛选相关行为

```
┌─────────────────────────────────────────────────────────┐
│                    SIM Framework                        │
├─────────────────────────────────────────────────────────┤
│  Stage 1: Hard Search / Soft Search                     │
│  ┌─────────────┐     ┌─────────────┐                   │
│  │  超长序列    │ --> │  相关性筛选  │                   │
│  │  (10K条)    │     │  (Top-K条)  │                   │
│  └─────────────┘     └─────────────┘                   │
│         │                      │                        │
│         ▼                      ▼                        │
│  Hard: 相同类目匹配      Soft: 向量相似度检索             │
│                                                          │
│  Stage 2: Target Attention                              │
│  ┌─────────────┐     ┌─────────────┐                   │
│  │  候选物品    │ --> │  注意力加权  │ --> 用户兴趣表示   │
│  │  (Target)   │     │  序列建模   │                   │
│  └─────────────┘     └─────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

**Soft Search（阿里巴巴实践）**：
- 用泛化类目（如一级类目）代替精确ID匹配
- 计算候选物品与历史行为的向量相似度
- 保留Top-K最相关的行为

### 2.2 位置编码（Positional Encoding）

#### 绝对位置编码

```python
def absolute_positional_encoding(max_len, d_model):
    """
    Transformer原版的正弦位置编码
    """
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe  # [max_len, d_model]
```

**为什么需要位置编码？**
- 自注意力机制本身是无序的（permutation invariant）
- 行为序列的时间顺序包含重要信息（最近行为更重要）
- 位置编码让模型感知"第几个位置"

#### 相对位置编码

```python
class RelativePositionBias(nn.Module):
    """
    T5/BERT中的相对位置偏置
    学习相对位置的偏置向量
    """
    def __init__(self, num_buckets, max_distance, n_heads):
        super().__init__()
        self.relative_attention_bias = nn.Embedding(num_buckets, n_heads)
        
    def forward(self, query_len, key_len):
        # 计算相对位置并映射到bucket
        relative_position = torch.arange(query_len)[:, None] - torch.arange(key_len)[None, :]
        rp_bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(rp_bucket)
        return values.permute(2, 0, 1).unsqueeze(0)  # [1, n_heads, q_len, k_len]
```

#### 时间衰减位置编码

```python
class TimeDecayPosition(nn.Module):
    """
    基于时间间隔的位置编码（推荐系统常用）
    行为越近，权重越高
    """
    def __init__(self, d_model, max_time_diff=3600*24*30):
        super().__init__()
        self.time_embed = nn.Embedding(max_time_diff, d_model)
        
    def forward(self, timestamps):
        # timestamps: [batch, seq_len]
        # 相对于当前时间的间隔（秒）
        return self.time_embed(timestamps)
```

### 2.3 Target Attention vs Self-Attention

| 对比维度 | Self-Attention | Target Attention |
|---------|----------------|------------------|
| **注意力来源** | 序列内部元素互相关注 | 候选物品(Target)关注序列 |
| **计算方式** | $Q,K,V$ 都来自序列 | $Q$来自候选，$K,V$来自序列 |
| **适用场景** | 学习序列内部模式 | 候选物品与历史兴趣匹配 |
| **计算复杂度** | $O(n^2)$ | $O(n)$ |
| **推荐系统应用** | 序列预训练（BERT4Rec） | 精排模型（DIN/DIEN/SIM） |

#### DIN（Deep Interest Network）- Target Attention

```python
class DINAttention(nn.Module):
    """
    DIN中的Activation Unit（Target Attention）
    """
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        # 输入：[target_emb, hist_emb, target_emb*hist_emb, target_emb-hist_emb]
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, target_emb, hist_emb, hist_mask):
        """
        target_emb: [batch, embed_dim]
        hist_emb: [batch, seq_len, embed_dim]
        hist_mask: [batch, seq_len]
        """
        # 扩展target到序列长度
        target_expanded = target_emb.unsqueeze(1).expand(-1, hist_emb.size(1), -1)
        
        # 构造输入特征
        concat = torch.cat([
            target_expanded,
            hist_emb,
            target_expanded * hist_emb,  # 逐元素乘
            target_expanded - hist_emb   # 差值
        ], dim=-1)  # [batch, seq_len, embed_dim*4]
        
        # 计算注意力权重
        attention = self.mlp(concat).squeeze(-1)  # [batch, seq_len]
        
        # Mask并softmax
        attention = attention.masked_fill(~hist_mask.bool(), float('-inf'))
        attention = F.softmax(attention, dim=1)
        
        # 加权求和
        output = torch.bmm(attention.unsqueeze(1), hist_emb).squeeze(1)
        return output
```

#### DIEN（Deep Interest Evolution Network）- 兴趣演化

```
┌─────────────────────────────────────────────────────────┐
│                   DIEN Architecture                     │
├─────────────────────────────────────────────────────────┤
│  Behavior Layer: 行为序列Embedding                       │
│         ↓                                               │
│  GRU Layer: 学习兴趣状态变化                             │
│         ↓                                               │
│  Attentional Update Gate (AUGRU):                       │
│    - 用候选物品指导GRU的更新门                           │
│    - 筛选与候选相关的兴趣演化路径                         │
│         ↓                                               │
│  Final Interest State → MLP预测                          │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 高基数类别特征

### 3.1 问题定义

**高基数特征**：类别数量极大的特征（如用户ID、商品ID、视频ID）

| 特征 | 典型基数 | 挑战 |
|------|---------|------|
| 用户ID | 千万-亿级 | 内存爆炸、冷启动 |
| 商品ID | 百万-千万级 | 长尾分布、Embedding稀疏 |
| 视频ID | 亿级 | 实时性要求高 |
| 查询词 | 无限增长 | OOV问题 |

### 3.2 Hash Trick（特征哈希）

```python
class FeatureHashing:
    """
    特征哈希：将高维稀疏特征映射到固定低维空间
    """
    def __init__(self, hash_size=100000):
        self.hash_size = hash_size
        
    def hash_feature(self, feature_value):
        """哈希到固定维度"""
        hash_val = hash(str(feature_value)) % self.hash_size
        return hash_val
    
    def hash_vector(self, features, values):
        """
        哈希编码为向量
        处理冲突：正负号抵消
        """
        vector = np.zeros(self.hash_size)
        for f, v in zip(features, values):
            idx = self.hash_feature(f)
            # 根据哈希值的符号决定加减，减少冲突影响
            sign = 1 if hash(f) % 2 == 0 else -1
            vector[idx] += sign * v
        return vector
```

**优缺点**：
- ✅ 内存固定，与特征基数无关
- ✅ 无需维护词汇表，天然支持新值
- ❌ 哈希冲突导致信息损失
- ❌ 不可解释，无法恢复原始特征

### 3.3 Embedding降维策略

#### 共享Embedding（Shared Embedding）

```python
class SharedEmbedding(nn.Module):
    """
    多个相关特征共享同一个Embedding表
    例：用户点击过的商品 & 候选商品 共享商品Embedding
    """
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, user_hist_items, candidate_item):
        # 历史行为序列和候选物品使用同一组Embedding
        hist_emb = self.embedding(user_hist_items)  # [batch, seq, dim]
        cand_emb = self.embedding(candidate_item)    # [batch, dim]
        return hist_emb, cand_emb
```

#### 聚类降维

```python
def cluster_embedding_reduction(ids, features, n_clusters=10000):
    """
    对高基数特征聚类，用聚类中心代表一组ID
    """
    from sklearn.cluster import MiniBatchKMeans
    
    # 基于特征聚类
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000)
    cluster_labels = kmeans.fit_predict(features)
    
    # 建立ID到聚类的映射
    id_to_cluster = dict(zip(ids, cluster_labels))
    
    return id_to_cluster, kmeans.cluster_centers_
```

#### 层次化Embedding

```
商品Embedding = 类目Embedding + 品牌Embedding + 独立残差Embedding

┌────────────────────────────────────────┐
│         商品ID Embedding (128d)         │
├────────────────────────────────────────┤
│  类目Embedding (32d)                   │
│  品牌Embedding (16d)                   │  
│  店铺Embedding (16d)                   │
│  商品独有Embedding (64d) ← 高频商品才有  │
└────────────────────────────────────────┘
```

### 3.4 频次过滤阈值

```python
def frequency_filter(df, col, min_freq=5, max_vocab_size=1000000):
    """
    低频过滤 + 高频截断
    """
    # 统计频次
    freq = df[col].value_counts()
    
    # 低频过滤：出现次数<min_freq的转为UNK
    frequent_items = freq[freq >= min_freq].index.tolist()
    
    # 高频截断：保留Top-K高频
    if len(frequent_items) > max_vocab_size:
        frequent_items = frequent_items[:max_vocab_size]
    
    # 建立映射
    item_to_idx = {item: idx + 1 for idx, item in enumerate(frequent_items)}
    item_to_idx['<UNK>'] = 0  # 0保留给未知/填充
    
    return item_to_idx
```

**阈值选择经验**：
- 用户ID：不建议过滤，每个用户都有价值
- 商品ID：出现<3次的可以归入"长尾商品"桶
- 搜索词：出现<5次的用`<UNK>`或字符级编码

---

## 4. 特征穿越（Data Leakage）

### 4.1 常见穿越场景

#### 场景1：标签信息泄露

```python
# ❌ 错误示例：用目标变量的统计作为特征
def wrong_feature_engineering(df):
    # 用户当天平均点击率 - 包含了当前样本的标签！
    df['user_ctr_today'] = df.groupby('user_id')['label'].transform('mean')
    return df

# ✅ 正确做法：只用历史数据计算统计特征
def correct_feature_engineering(df):
    df = df.sort_values('timestamp')
    # 用户历史点击率（不包含当前样本）
    df['user_ctr_history'] = df.groupby('user_id')['label'].expanding().mean().shift(1).reset_index(0, drop=True)
    return df
```

#### 场景2：时间穿越

```python
# ❌ 错误：特征计算用了"未来"数据
def wrong_time_feature(df):
    # 计算商品未来7天的销量作为特征
    df['item_future_sales_7d'] = df.groupby('item_id')['sales'].shift(-7).rolling(7).sum()
    return df

# ✅ 正确：特征只能用当前时刻之前的数据
def correct_time_feature(df):
    df = df.sort_values('timestamp')
    # 商品过去7天的销量
    df['item_past_sales_7d'] = df.groupby('item_id')['sales'].shift(1).rolling(7).sum()
    return df
```

#### 场景3：全局统计穿越

```python
# ❌ 错误：全局统计包含了测试集信息
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2)

# 用全量数据计算target encoding - 信息泄露！
global_mean = df['label'].mean()  # 包含了测试集的标签！

# ✅ 正确：每个fold的统计只基于训练部分
from sklearn.model_selection import KFold

def safe_target_encoding(df, col, target, n_folds=5):
    """交叉验证方式计算target encoding"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    encoded = np.zeros(len(df))
    
    for train_idx, val_idx in kf.split(df):
        # 只在训练集上计算统计
        mapping = df.iloc[train_idx].groupby(col)[target].mean()
        # 应用到验证集
        encoded[val_idx] = df.iloc[val_idx][col].map(mapping)
    
    return encoded
```

### 4.2 防穿越的数据切分方式

#### 时间序列切分（Time-based Split）

```
Timeline: |---- Train ----|---- Val ----|---- Test ----|
           t0           t1           t2           t3

- 训练集：t0 ~ t1
- 验证集：t1 ~ t2  
- 测试集：t2 ~ t3

所有特征只能用该样本timestamp之前的数据计算！
```

```python
def time_based_split(df, time_col, train_ratio=0.7, val_ratio=0.15):
    """按时间顺序切分"""
    df = df.sort_values(time_col)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    return train, val, test
```

#### 用户级切分（User-based Split）

```python
def user_based_split(df, user_col, test_user_ratio=0.1):
    """
    冷启动评估：测试集用户不出现在训练集
    """
    users = df[user_col].unique()
    test_users = np.random.choice(users, size=int(len(users) * test_user_ratio), replace=False)
    
    train = df[~df[user_col].isin(test_users)]
    test = df[df[user_col].isin(test_users)]
    
    return train, test
```

### 4.3 线上线下一致性问题

#### 问题根源

```
┌─────────────────────────────────────────────────────────┐
│                  特征不一致常见原因                       │
├─────────────────────────────────────────────────────────┤
│  1. 数据口径不一致                                       │
│     - 离线：T+1批量数据                                   │
│     - 在线：实时流数据                                    │
│                                                          │
│  2. 计算逻辑不一致                                       │
│     - 离线：Spark SQL / Pandas                            │
│     - 在线：Java/Go/Python服务代码                        │
│                                                          │
│  3. 时间窗口不一致                                       │
│     - 离线：整点批量计算                                  │
│     - 在线：请求时刻实时计算                              │
│                                                          │
│  4. 特征回填 vs 实时计算                                 │
│     - 离线：可以"偷看"未来数据回填                        │
│     - 在线：只能用过去数据                                │
└─────────────────────────────────────────────────────────┘
```

#### 解决方案：特征平台架构

```
┌─────────────────────────────────────────────────────────┐
│                   Feature Store                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   ┌──────────────┐        ┌──────────────┐             │
│   │ Offline Store │        │ Online Store │             │
│   │  (Hive/HDFS)  │        │  (Redis/Flink)│             │
│   │              │        │              │             │
│   │  T+1特征计算  │───────>│  实时同步    │             │
│   │  离线训练    │        │  在线服务    │             │
│   └──────────────┘        └──────────────┘             │
│          │                       ▲                      │
│          │                       │                      │
│          ▼                       │                      │
│   ┌──────────────┐               │                      │
│   │ Feature SDK  │───────────────┘                      │
│   │              │                                      │
│   │ - 统一计算逻辑│                                      │
│   │ - 版本管理   │                                      │
│   │ - 一致性校验│                                      │
│   └──────────────┘                                      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### 一致性校验 checklist

```python
def feature_consistency_check(online_feature, offline_feature, tolerance=1e-5):
    """
    线上线下特征一致性校验
    """
    # 1. 样本对齐
    merged = online_feature.merge(offline_feature, on=['user_id', 'item_id'], suffixes=('_online', '_offline'))
    
    # 2. 缺失值检查
    online_null = merged['feature_online'].isnull().sum()
    offline_null = merged['feature_offline'].isnull().sum()
    
    # 3. 数值差异检查
    diff = np.abs(merged['feature_online'] - merged['feature_offline'])
    inconsistent_ratio = (diff > tolerance).mean()
    
    # 4. 统计量对比
    online_stats = merged['feature_online'].describe()
    offline_stats = merged['feature_offline'].describe()
    
    return {
        'online_null_rate': online_null / len(merged),
        'offline_null_rate': offline_null / len(merged),
        'inconsistent_ratio': inconsistent_ratio,
        'online_stats': online_stats,
        'offline_stats': offline_stats
    }
```

---

## 5. 实战经验

### 5.1 特征重要性分析

#### SHAP值分析

```python
import shap

def shap_feature_importance(model, X_sample, feature_names):
    """
    SHAP值特征重要性分析
    """
    explainer = shap.TreeExplainer(model)  # 树模型
    # explainer = shap.DeepExplainer(model, X_sample)  # 深度学习
    
    shap_values = explainer.shap_values(X_sample)
    
    # 全局重要性：绝对值的均值
    importance = np.abs(shap_values).mean(axis=0)
    
    # 可视化
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names)
    
    return dict(zip(feature_names, importance))
```

**SHAP解读要点**：
- 正SHAP值：推动预测为正类（如点击）
- 负SHAP值：推动预测为负类（如不点击）
- 绝对值大小：特征影响力

#### 特征消融实验（Ablation Study）

```python
def ablation_study(model, X_val, y_val, feature_groups):
    """
    特征消融实验：逐步移除特征组，观察性能变化
    
    feature_groups = {
        'user_profile': ['age', 'gender', 'city'],
        'item_stat': ['item_ctr', 'item_pv'],
        'cross_feature': ['user_category_prefer']
    }
    """
    results = []
    
    # 全量特征 baseline
    baseline_auc = evaluate(model, X_val, y_val)
    results.append(('all_features', baseline_auc))
    
    # 逐个移除
    for group_name, features in feature_groups.items():
        X_ablation = X_val.drop(columns=features)
        # 重新训练或mask特征
        auc = evaluate_with_masked_features(model, X_val, y_val, features)
        auc_drop = baseline_auc - auc
        results.append((f'w/o {group_name}', auc, auc_drop))
    
    return sorted(results, key=lambda x: x[2] if len(x) > 2 else 0, reverse=True)
```

### 5.2 特征监控与异常检测

#### PSI（Population Stability Index）

```python
def calculate_psi(expected, actual, buckets=10):
    """
    PSI: 衡量特征分布变化
    PSI < 0.1: 变化很小
    0.1 <= PSI < 0.25: 轻微变化
    PSI >= 0.25: 显著变化，需要关注
    """
    def scale_range(input, min_val, max_val):
        return (input - min_val) / (max_val - min_val)
    
    # 分桶
    breakpoints = np.linspace(0, 1, buckets + 1)
    breakpoints = np.percentile(expected, breakpoints * 100)
    
    expected_counts = np.histogram(expected, breakpoints)[0]
    actual_counts = np.histogram(actual, breakpoints)[0]
    
    # 计算百分比
    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)
    
    # 避免除0
    expected_percents = np.clip(expected_percents, 0.0001, 1)
    actual_percents = np.clip(actual_percents, 0.0001, 1)
    
    # 计算PSI
    psi = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
    
    return psi

# 监控特征漂移
for feature in features:
    psi = calculate_psi(train[feature], online[feature])
    if psi > 0.25:
        alert(f"Feature {feature} has significant drift: PSI={psi:.3f}")
```

#### 特征异常检测

```python
class FeatureMonitor:
    """特征监控告警系统"""
    
    def __init__(self):
        self.baselines = {}
        
    def fit_baseline(self, df, features):
        """建立基线统计"""
        for feat in features:
            self.baselines[feat] = {
                'mean': df[feat].mean(),
                'std': df[feat].std(),
                'min': df[feat].min(),
                'max': df[feat].max(),
                'null_rate': df[feat].isnull().mean()
            }
    
    def check_anomaly(self, df, features, n_std=3):
        """检测异常"""
        alerts = []
        
        for feat in features:
            baseline = self.baselines[feat]
            
            # 1. 均值漂移检测
            current_mean = df[feat].mean()
            if abs(current_mean - baseline['mean']) > n_std * baseline['std']:
                alerts.append({
                    'feature': feat,
                    'type': 'mean_drift',
                    'baseline': baseline['mean'],
                    'current': current_mean
                })
            
            # 2. 空值率检测
            current_null_rate = df[feat].isnull().mean()
            if current_null_rate > baseline['null_rate'] * 2 + 0.01:
                alerts.append({
                    'feature': feat,
                    'type': 'null_spike',
                    'baseline': baseline['null_rate'],
                    'current': current_null_rate
                })
            
            # 3. 取值范围检测
            if df[feat].min() < baseline['min'] or df[feat].max() > baseline['max']:
                alerts.append({
                    'feature': feat,
                    'type': 'out_of_range',
                    'expected': (baseline['min'], baseline['max']),
                    'actual': (df[feat].min(), df[feat].max())
                })
        
        return alerts
```

### 5.3 特征存储：实时特征 vs 离线特征

#### 特征分类体系

```
┌─────────────────────────────────────────────────────────────────┐
│                     特征分类与存储策略                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │   离线特征        │  │   实时特征        │                     │
│  │   (T+1)          │  │   (近实时)        │                     │
│  ├──────────────────┤  ├──────────────────┤                     │
│  │ • 用户画像        │  │ • 实时行为序列    │                     │
│  │ • 商品属性        │  │ • 实时统计特征    │                     │
│  │ • 历史统计特征    │  │ • 上下文特征      │                     │
│  │ • 长期兴趣标签    │  │ • 实时交叉特征    │                     │
│  │                  │  │                  │                     │
│  │ 存储: Hive/HDFS  │  │ 存储: Redis/Flink │                     │
│  │ 计算: Spark      │  │ 计算: Flink/Kafka │                     │
│  └──────────────────┘  └──────────────────┘                     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    近线特征 (Near-realtime)                │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │ • 近1小时用户行为统计                                      │  │
│  │ • 近24小时商品热度                                         │  │
│  │                                                          │  │
│  │ 存储: HBase/Cassandra (支持高并发点查)                      │  │
│  │ 更新: 流式计算批量更新                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 实时特征计算架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     实时特征计算链路                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   User Action          Kafka            Flink          Redis    │
│       │                  │                 │             │      │
│       │  click/purchase  │                 │             │      │
│       ▼                  ▼                 ▼             ▼      │
│   ┌────────┐      ┌──────────┐      ┌──────────┐   ┌────────┐  │
│   │ APP/Web │─────>│  Kafka   │─────>│  Flink   │──>│ Redis  │  │
│   └────────┘      └──────────┘      │ Cluster  │   └────────┘  │
│                                      └──────────┘               │
│                                          │                      │
│                                          ▼                      │
│                                    ┌──────────┐                 │
│                                    │ 滑窗统计  │                 │
│                                    │ • 近1h点击│                 │
│                                    │ • 近24h购买│                │
│                                    └──────────┘                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 特征服务设计

```python
class FeatureService:
    """
    统一特征服务接口
    """
    def __init__(self):
        self.offline_store = HiveFeatureStore()
        self.online_store = RedisFeatureStore()
        self.nearline_store = HBaseFeatureStore()
        
    def get_features(self, user_id, item_id, context):
        """
        聚合各存储的特征
        """
        features = {}
        
        # 1. 获取离线特征（用户画像、商品属性）
        offline_feats = self.offline_store.get(
            user_id=user_id, 
            item_id=item_id
        )
        features.update(offline_feats)
        
        # 2. 获取近线特征（近24小时统计）
        nearline_feats = self.nearline_store.get(
            user_id=user_id,
            item_id=item_id
        )
        features.update(nearline_feats)
        
        # 3. 实时计算特征（上下文相关）
        realtime_feats = self._compute_realtime_features(
            user_id=user_id,
            item_id=item_id,
            context=context
        )
        features.update(realtime_feats)
        
        return features
    
    def _compute_realtime_features(self, user_id, item_id, context):
        """实时计算特征"""
        # 实时行为序列
        recent_behavior = self.online_store.get_recent_behavior(user_id, n=50)
        
        # 实时交叉特征
        cross_features = {
            'realtime_category_match': self._check_category_match(recent_behavior, item_id),
            'realtime_price_prefer': self._calc_price_prefer(recent_behavior, item_id),
            'time_since_last_click': context['timestamp'] - recent_behavior[0]['timestamp']
        }
        
        return cross_features
```

---

## 附录：特征工程 checklist

### 建模前检查

- [ ] 是否存在特征穿越？（时间、标签、全局统计）
- [ ] 训练/验证/测试集划分是否合理？
- [ ] 数值特征是否需要归一化/标准化？
- [ ] 类别特征基数是否过高？如何处理？
- [ ] 缺失值处理策略是否统一？

### 建模后检查

- [ ] 线上线下特征计算逻辑是否一致？
- [ ] 特征分布是否发生漂移？（PSI监控）
- [ ] 特征重要性是否合理？有无冗余？
- [ ] 冷启动场景特征覆盖率如何？

### 上线前检查

- [ ] 实时特征延迟是否满足要求？
- [ ] 特征服务QPS和RT是否达标？
- [ ] 特征降级预案是否到位？

---

> 📌 **核心要点总结**：
> 1. 特征工程的核心是信息表达，而非复杂变换
> 2. 警惕特征穿越，保证线上线下一致性
> 3. 高基数特征用Embedding，低频过滤要谨慎
> 4. 行为序列用Target Attention捕捉动态兴趣
> 5. 建立完善的特征监控体系

---

*Author: MelonEggLearn*  
*Last Updated: 2026-03-11*
