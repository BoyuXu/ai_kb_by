# 项目4：推荐系统冷启动优化

## 项目概览

这个项目核心解决推荐系统中的"冷启动"问题——新用户和新内容因为数据稀缺而推荐效果差。通过元学习（Meta-Learning）、多臂老虎机（Contextual Bandit）和内容理解（NLP+CV），我将新用户的 7 日留存率从 28% 提升到 35%（+25%），这是推荐系统的黄金指标。项目周期 12 个月，涉及算法、工程、数据管道三个维度。

---

## 一、冷启动问题的本质

### 1.1 现象与数据

```
用户生命周期：
├─ 新用户（Day 1-7）
│   ├─ 推荐效果：⭐⭐（糟糕）
│   ├─ CTR：1.8%（老用户是 4.2%）
│   ├─ 7 日留存率：28%（老用户 70%+）
│   └─ 问题：无历史行为，无法进行协同过滤
│
├─ 成长用户（Day 8-30）
│   ├─ 推荐效果逐步改善
│   ├─ CTR：3.2%
│   ├─ 留存率：45%
│   └─ 开始积累行为数据
│
└─ 成熟用户（Day 30+）
    ├─ 推荐效果：⭐⭐⭐⭐⭐
    ├─ CTR：4.2%
    ├─ 留存率：70%+
    └─ 行为数据丰富，协同过滤有效
```

### 1.2 根本原因

冷启动的根本不是"数据少"，而是**分布偏移（Distribution Shift）**：

```
新用户的行为分布 ≠ 老用户的行为分布

例子：
  新用户（18-25岁）：
    ├─ 偏好短视频、娱乐内容
    ├─ 活跃时间：晚上 20-24点
    └─ 内容消费频率：3-5条/次
  
  老用户（25-40岁）：
    ├─ 偏好长视频、教育内容
    ├─ 活跃时间：午休 12-13点
    └─ 内容消费频率：8-12条/次

用老用户的模型推荐给新用户 → 分布完全不匹配 ✗
```

### 1.3 商业影响

```
新用户 7 日流失 ≈ 获新成本损失

假设：
  - 广告获新成本：5 元/用户
  - 新用户 7 日流失率：72%（28% 留存）
  - 月新增用户：100 万

损失：100 万 × 72% × 5 元 = 3600 万 元/月
```

目标：通过优化推荐策略，将新用户 7 日留存率从 28% 提升到 35%+（行业 top 水平是 36-38%）。

---

## 二、算法方案

### 2.1 新用户冷启动

#### 2.1.1 用户相似度匹配（基线方案）

```python
def get_similar_users(new_user, all_users):
    """
    基于用户属性，找到相似的老用户，迁移他们的行为偏好
    """
    # 新用户的属性
    new_user_features = {
        'age': new_user.age,
        'gender': new_user.gender,
        'location': new_user.location,
        'signup_source': new_user.signup_source,  # 来自哪个渠道
        'device_type': new_user.device_type,
        'network_type': new_user.network_type,
    }
    
    # 计算新用户与所有老用户的相似度
    similarities = []
    for old_user in all_users:
        if old_user.days_active < 30:  # 排除太新的用户
            continue
        
        # 属性相似度
        attr_sim = compute_attribute_similarity(new_user_features, old_user.features)
        
        # 迁移这个老用户的行为数据
        if attr_sim > threshold:
            similarities.append((old_user, attr_sim))
    
    # 选择 top-k 相似的用户
    similar_users = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
    return similar_users

def transfer_user_preferences(new_user, similar_users):
    """
    迁移相似用户的内容偏好
    """
    transferred_preferences = {}
    
    for old_user, similarity in similar_users:
        # 加权平均
        for content_id, preference in old_user.content_preferences.items():
            if content_id not in transferred_preferences:
                transferred_preferences[content_id] = 0
            
            # 用相似度作为权重
            transferred_preferences[content_id] += similarity * preference
    
    # 归一化
    if transferred_preferences:
        max_pref = max(transferred_preferences.values())
        for key in transferred_preferences:
            transferred_preferences[key] /= max_pref
    
    return transferred_preferences
```

**效果**：CTR 从 1.8% → 2.3%（+28%），但只是基础改善。

#### 2.1.2 图结构学习（高级方案）

用户属性之间的关系可以用图来表示：

```
用户属性图：
  ├─ 节点：用户、年龄段、地域、兴趣标签
  ├─ 边：关系（用户属于某年龄段、来自某地域、有某兴趣）
  └─ 目标：通过图神经网络学习用户的隐表示
```

用 GraphSAGE（Graph Sample and Aggregate）来学习：

```python
import tensorflow as tf
from spektral.layers import GraphAttention

class UserGraphLearning(tf.keras.Model):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # User embedding：初始化
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)
        
        # Graph Attention layers：学习图结构上的信息流
        self.graph_att_1 = GraphAttention(embedding_dim, activation='relu')
        self.graph_att_2 = GraphAttention(embedding_dim, activation='relu')
        
        # 输出层：预测用户对内容的偏好
        self.output_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    
    def call(self, user_ids, adjacency_matrix, training=False):
        """
        输入：
          user_ids: 用户 ID
          adjacency_matrix: 用户属性图的邻接矩阵
        """
        # 用户初始嵌入
        user_emb = self.user_embedding(user_ids)
        
        # 图注意力聚合
        # 通过图的邻接关系，聚集相似用户的信息
        user_emb = self.graph_att_1(user_emb, adjacency_matrix, training=training)
        user_emb = self.graph_att_2(user_emb, adjacency_matrix, training=training)
        
        # 预测偏好
        preferences = self.output_layer(user_emb, training=training)
        
        return preferences
```

**优势**：
- 不仅考虑直接的属性相似性，还考虑间接的图结构
- 例如：两个用户虽然属性不同，但都喜欢某个兴趣标签，图学习会发现这个共同点

**效果**：CTR 从 2.3% → 2.8%（进一步 +22%）

#### 2.1.3 元学习（Meta-Learning）

最高级的方案：学习"如何快速学习"。

**核心思想**：用元学习，训练一个"初始化器"。当新用户到来时，从这个初始化开始，只需几步梯度下降就能适应新用户的偏好。

```python
class MetaLearningBidding:
    def __init__(self):
        # 初始化一个"通用用户"的偏好向量
        self.universal_preference = tf.Variable(
            tf.random.normal((num_contents,), stddev=0.1)
        )
    
    def adapt_to_new_user(self, new_user, observed_interactions, num_steps=5):
        """
        Fast Adaptation：从通用偏好向量出发，快速适应新用户
        """
        # 复制通用偏好，作为新用户的初始化
        user_preference = self.universal_preference.numpy().copy()
        user_preference = tf.Variable(user_preference)
        
        # 用新用户的观察数据（几次点击）进行梯度更新
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        
        for step in range(num_steps):
            with tf.GradientTape() as tape:
                # 预测新用户对其他内容的点击概率
                pred = tf.nn.sigmoid(user_preference)
                
                # 损失：与观察数据的差异
                loss = tf.keras.losses.binary_crossentropy(
                    observed_interactions,
                    pred
                )
            
            # 梯度更新
            grads = tape.gradient(loss, [user_preference])
            optimizer.apply_gradients(zip(grads, [user_preference]))
        
        return user_preference
```

**效果**：CTR 从 2.8% → 3.1%（+11%）

**总体冷启动效果链**：
```
基线（无优化）：CTR 1.8%
+ 用户相似度匹配：1.8% → 2.3%（+28%）
+ 图学习：2.3% → 2.8%（+22%）
+ 元学习：2.8% → 3.1%（+11%）
= 总体提升：1.8% → 3.1%（+72%，接近老用户的 70%）
```

---

### 2.2 新内容冷启动

#### 2.2.1 内容理解（NLP + CV）

用自然语言处理和计算机视觉来理解新内容，而不是等待人工标签或用户反馈。

```python
class ContentUnderstanding:
    def __init__(self):
        # NLP 模型：提取文本特征
        self.text_encoder = load_pretrained_bert()
        
        # CV 模型：提取视觉特征
        self.image_encoder = load_pretrained_resnet()
    
    def understand_content(self, content):
        """
        自动理解内容的各个维度
        """
        features = {}
        
        # 文本特征
        if content.text:
            text_emb = self.text_encoder.encode(content.text)
            
            # 从文本提取关键信息
            topics = extract_topics_from_text(content.text)  # ["美食", "旅游"]
            sentiment = extract_sentiment(content.text)  # [0.8]（积极）
            entities = extract_entities(content.text)  # ["北京", "餐厅"]
            
            features['text_embedding'] = text_emb
            features['topics'] = topics
            features['sentiment'] = sentiment
            features['entities'] = entities
        
        # 视觉特征
        if content.image_url:
            image = load_image(content.image_url)
            image_emb = self.image_encoder.encode(image)
            
            # 视觉属性
            objects = detect_objects(image)  # ["人", "食物", "室内"]
            color_histogram = extract_color_distribution(image)
            
            features['image_embedding'] = image_emb
            features['objects'] = objects
            features['color_histogram'] = color_histogram
        
        return features

def initial_rating_estimation(content, similar_contents):
    """
    基于相似内容的反馈，估计新内容的初始评分
    """
    # 找到与新内容相似的老内容
    similarities = [
        compute_similarity(content, old_content)
        for old_content in similar_contents
    ]
    
    # 加权平均
    initial_rating = 0
    for old_content, similarity in zip(similar_contents, similarities):
        initial_rating += similarity * old_content.avg_rating
    
    initial_rating /= len(similar_contents)
    return initial_rating
```

#### 2.2.2 探索策略：Thompson Sampling

对于新内容，需要平衡两个目标：
1. **推荐已验证的内容**（点击率高）
2. **探索新内容**（可能更好，但不确定）

用 Thompson Sampling：

```python
class ExplorationStrategy:
    def __init__(self):
        # 对每个内容，维护一个 Beta 分布
        # 表示该内容的"点击率"的不确定性
        self.content_beta_params = {}
    
    def update_distribution(self, content_id, clicked):
        """
        用新的反馈数据，更新 Beta 分布
        """
        if content_id not in self.content_beta_params:
            # 初始化：先验相信点击率是 5%（Beta(1, 19)）
            self.content_beta_params[content_id] = {'alpha': 1, 'beta': 19}
        
        # 更新后验
        if clicked:
            self.content_beta_params[content_id]['alpha'] += 1
        else:
            self.content_beta_params[content_id]['beta'] += 1
    
    def sample_and_rank(self, candidate_contents, k=10):
        """
        采样，并根据采样的点击率排名
        """
        sampled_scores = []
        
        for content in candidate_contents:
            if content.id not in self.content_beta_params:
                # 新内容，使用初始评分
                sampled_score = self.initial_rating(content)
            else:
                # 从 Beta 分布采样这个内容的点击率
                alpha = self.content_beta_params[content.id]['alpha']
                beta_param = self.content_beta_params[content.id]['beta']
                
                sampled_ctr = np.random.beta(alpha, beta_param)
                sampled_score = sampled_ctr
            
            sampled_scores.append((content, sampled_score))
        
        # 根据采样分数排名
        ranked = sorted(sampled_scores, key=lambda x: x[1], reverse=True)
        return [content for content, score in ranked[:k]]
```

**Thompson Sampling 的妙处**：
- **初期**：不确定性大，采样差异大，自然地探索不同内容
- **后期**：观察数据多，采样聚集到高点击率的内容，自动利用

```
采样过程举例：

新内容：
  Beta(1, 19) → 采样出的点击率波动大，可能 2% 也可能 8%
  → 给机会让它跟老内容竞争

3 天后，新内容获得 30 次点击，其中 10 次转化：
  Beta(11, 39) → 采样点击率集中在 20-30%
  → 自动排序靠前，获得更多曝光

最终，如果新内容实际点击率确实高：
  Beta(100, 50) → 采样点击率稳定在 65% 左右
  → 成为热点内容
```

#### 2.2.3 联合优化：Contextual Bandit

新用户冷启动和新内容冷启动是**耦合的问题**。不能分别优化：

```
错误做法：
  ├─ 用用户冷启动方案推荐给新用户
  └─ 推荐的是老内容（因为老内容有反馈数据）
     → 新内容永远没有机会

正确做法：
  ├─ 用 Contextual Bandit 框架
  ├─ 状态 = （新用户特征 + 候选内容特征）
  ├─ 行为 = 推荐排名
  └─ 奖励 = 点击 / 转化 / 留存
```

```python
class ContextualBandit:
    def __init__(self):
        # 策略网络：根据（用户, 内容）的特征，输出推荐分数
        self.policy_network = PolicyNetwork(user_dim=128, content_dim=128)
    
    def recommend(self, new_user, candidate_contents):
        """
        推荐：联合考虑新用户和新内容的特征
        """
        recommendations = []
        
        for content in candidate_contents:
            # 提取用户和内容特征
            user_features = extract_user_features(new_user)
            content_features = extract_content_features(content)
            
            # 用策略网络评分
            score = self.policy_network(user_features, content_features)
            
            # 加入探索噪声（ε-贪心或 softmax）
            if np.random.random() < 0.1:  # 10% 的概率随机探索
                score = np.random.random()
            
            recommendations.append((content, score))
        
        # 排名
        ranked = sorted(recommendations, key=lambda x: x[1], reverse=True)
        return [content for content, score in ranked[:10]]
    
    def update_policy(self, user, recommended_content, clicked):
        """
        用反馈数据，更新策略网络
        """
        # 计算奖励信号
        reward = 1.0 if clicked else 0.0
        
        # 梯度更新
        loss = self.policy_network.compute_loss(
            user, recommended_content, reward
        )
        self.policy_network.optimize(loss)
```

---

## 三、特征工程

### 3.1 用户侧特征

```
显式特征（用户提供的）：
  ├─ 年龄、性别、位置
  ├─ 兴趣标签（用户注册时选择）
  └─ 设备、网络类型

行为特征（新用户的行为）：
  ├─ 注册来源（搜索、广告、推荐等）
  ├─ 首次打开到注册的时间间隔
  ├─ 注册后的首次行为（搜索 / 点赞 / 分享）
  └─ 前 N 次点击的内容类型分布

图特征（通过用户属性图）：
  ├─ 相邻用户的平均偏好
  ├─ 所在社区（社区检测）的特征
  └─ 与相似用户的距离
```

### 3.2 内容侧特征

```
元数据特征：
  ├─ 创作者历史 CTR / 转化率
  ├─ 内容创建时间（新鲜度）
  ├─ 字数 / 图片数 / 视频时长

NLP 特征：
  ├─ 文本主题（Topic Modeling）
  ├─ 情感（Sentiment Analysis）
  ├─ 实体（NER：人名、地点、品牌）
  └─ 词频-逆文档频率（TF-IDF）

CV 特征：
  ├─ 物体检测（人、食物、室内等）
  ├─ 颜色分布（主色调）
  ├─ 图像美学评分（清晰度、对比度）
  └─ 人脸检测（有人脸的内容 CTR 更高）

社交特征：
  ├─ 评论数 / 分享数 / 赞数
  ├─ 内容话题的热度
  └─ 相关话题的流行趋势
```

---

## 四、AB 测试与效果

### 4.1 测试设计

```
对照组（旧冷启动策略）：50% 新用户
实验组（新方案：图学习 + Thompson Sampling）：50% 新用户

时长：2 个月（足够观察 7 日和 30 日留存）
样本量：50 万新用户
```

### 4.2 主要指标

| 指标 | 对照 | 实验 | 提升 | 显著性 |
|------|------|------|------|--------|
| Day 1 CTR | 1.8% | 2.9% | **+61%** | <0.001 |
| Day 7 留存 | 28% | 35% | **+25%** | <0.001 |
| Day 30 留存 | 42% | 48% | **+14%** | <0.001 |
| 内容多样性（Entropy） | 2.1 | 2.4 | +14% | <0.001 |
| 7 日总交互 | 8.5 | 11.2 | **+32%** | <0.001 |

最关键的是 **7 日留存 +25%**。在互联网产品中，这是个巨大的改进。

### 4.3 分层分析

不同渠道的新用户反应：

```python
organic_users (自然搜索来源):
  Day 7 留存：28% → 36% (+28%)
  → 有动力的用户，推荐改善效果最好
  
paid_users (付费广告渠道):
  Day 7 留存：25% → 32% (+28%)
  → 质量略低，但也有改善

referral_users (推荐渠道):
  Day 7 留存：32% → 38% (+19%)
  → 本身质量高，改善空间有限
```

发现：**付费用户从 25% 留存提升到 32%，ROI 巨大**。这是因为：
- 付费用户质量较低（可能是被广告吸引来的）
- 好的推荐能显著提升他们的体验
- 从 25% 留存提升 7 pp，相当于回收了 28% 的获新成本

---

## 五、关键洞察

### 5.1 冷启动的根本是"分布偏移"

很多人一开始以为冷启动就是"数据不足"。但实验表明：

```
新用户分布 ≠ 老用户分布

即使我们有新用户的"历史数据"（比如他们注册时填的兴趣），
如果用老用户的模型，仍然效果不好。

解决方案：
  ├─ 不是给新用户"更多数据"
  └─ 而是用"适配新分布的模型"（元学习、迁移学习）
```

### 5.2 探索-利用的平衡很微妙

设定太高的探索率（如 50%）：
- 新用户体验差（经常推荐垃圾内容）
- 留存率反而下降

设定太低的探索率（如 1%）：
- 新用户只能看老内容
- 新内容没有机会，生态停滞

**最优的探索率**：10-15%（通过网格搜索找到）

### 5.3 内容理解比用户历史更重要

```
对比实验：

方案 A：用用户的历史点击预测
  - 新用户没历史 → 无法预测
  - Day 1 CTR：1.8%
  
方案 B：用内容的文本/图像特征
  - 新内容也有特征（即使没人点过）
  - Day 1 CTR：2.4%（+33%）
  
方案 C：混合（用户特征 + 内容特征）
  - Day 1 CTR：2.9%（再 +21%）
```

发现：**对于冷启动，内容理解（NLP+CV）比用户历史更重要**。这是直觉之外的：
- 推荐系统通常依赖用户相似性（协同过滤）
- 但在冷启动阶段，用户数据稀缺，内容特征更可靠

---

## 六、讲故事要点

### 6.1 30 秒电梯演讲

> "我优化了推荐系统的冷启动问题。新用户因为没有历史行为数据，系统推不了好的内容，导致 7 日留存率只有 28%。我用三个方案：（1）图神经网络学习用户属性之间的关系，（2）NLP+CV 自动理解新内容的特征，（3）Thompson Sampling 在推荐已验证内容和探索新内容之间平衡。结果，新用户 Day 1 CTR 从 1.8% 提升到 2.9%，7 日留存率从 28% 提升到 35%（+25%），达到业界领先水平。"

### 6.2 完整讲述

**问题**：新用户在平台上的推荐效果很差。因为他们没有历史点击数据，推荐系统无法用协同过滤来预测他们的偏好。结果，新用户的 7 日留存率只有 28%，而老用户是 70%+。这直接导致获新成本浪费。

**解决方案**：我用三层递进式的方案：
1. **用户层**：用图神经网络（GraphSAGE）学习用户属性之间的关系。比如，两个用户虽然来自不同城市，但都喜欢"美食"话题，图学习会发现这个共同点。
2. **内容层**：用 NLP 和 CV 自动理解新内容。不用等人工标签或用户反馈，直接从文本和图像提取主题、情感、对象等信息。
3. **推荐策略**：用 Thompson Sampling 来平衡"推荐已验证的热门内容"和"探索新颖内容"。初期，新内容因为不确定性大，会被多次推荐；随着数据积累，好内容自动浮上来。

**结果**：
- Day 1 CTR：1.8% → 2.9%（+61%）
- 7 日留存率：28% → 35%（+25%，业界 top 水平）
- 新用户在 7 天内的总交互：8.5 条 → 11.2 条（+32%）

**学到的**：最大的洞察是理解了冷启动的本质——**分布偏移而非数据稀缺**。即使给新用户更多数据，如果用的模型是针对老用户分布设计的，也没用。我们需要用迁移学习和元学习这样的技术来适应新分布。

---

## 七、总结

这个项目让我学到：

1. **问题的诊断比解决更重要**：花了 2 周分析"冷启动为什么这么难"，发现是分布偏移而不是数据少。诊断对了，解决方案就水到渠成。

2. **多模态特征的力量**：在冷启动阶段，文本+图像的内容理解，比单一的用户行为特征更有用。这颠覆了传统推荐系统的思路（总是以用户为中心）。

3. **Thompson Sampling 的优雅**：简单的贝叶斯方法，自动在探索和利用之间找平衡，不需要手调超参数。

4. **系统化思维**：不能单独优化新用户或新内容，要用 Contextual Bandit 框架，在联合的作用空间中优化。
