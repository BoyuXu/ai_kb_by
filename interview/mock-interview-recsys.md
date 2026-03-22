# 推荐算法岗模拟面试（字节/美团/快手风格）

> **面试时长**：60分钟  
> **目标岗位**：推荐算法工程师  
> **面试风格**：大厂标准流程，注重工程实践与理论结合

---

## 开场白（2分钟）

**面试官**：你好，请先做个自我介绍吧，重点介绍你的项目经历和推荐算法相关的经验。

**候选人（参考答案）**：
> 您好，我是XXX，X年推荐算法工作经验。目前在XX公司负责主Feed流推荐，主要工作包括：
> 1. **召回层优化**：实现了双塔模型+DSSM多路召回，召回覆盖率提升15%
> 2. **精排模型**：主导从GBDT+LR迁移到DeepFM，再到自研多目标模型，CTR提升8%
> 3. **工程优化**：引入FAISS向量检索、模型蒸馏，服务P99延迟从120ms降到60ms
> 
> 今天很高兴能参加这次面试，期待和您交流。

---

## 一、基础概念（10题，20分钟）

### Q1: 请解释召回层和排序层的职责分别是什么？为什么需要分层设计？

**面试官**：先从基础开始，说一下召回和排序的职责，以及为什么要分层？

**候选人（参考答案）**：
- **召回层**：从海量候选集（百万/千万级）中快速筛选出千级别相关item，追求高覆盖率和低延迟
- **排序层**：对召回结果精细排序，输出Top-N给用户，追求精准度和业务目标最大化
- **分层原因**：
  1. **计算复杂度**：全库排序计算量太大，O(N)不可接受
  2. **不同优化目标**：召回重覆盖，排序重精准
  3. **工程解耦**：召回可并行多路，排序专注模型精度

---

### Q2: 协同过滤的User-CF和Item-CF有什么区别？各自的适用场景是什么？

**面试官**：User-CF和Item-CF的区别是什么？什么时候用哪个？

**候选人（参考答案）**：
| 维度 | User-CF | Item-CF |
|------|---------|---------|
| 核心思想 | 找相似用户，推荐他们喜欢的 | 找相似物品，推荐用户历史相似的 |
| 相似度计算 | 用户向量相似度 | Item向量相似度 |
| 适用场景 | 用户少、社交属性强（新闻） | 物品少、用户兴趣稳定（电商） |
| 冷启动 | 新用户需要积累行为 | 新物品需要被点击积累 |
| 实时性 | 弱，用户相似度更新慢 | 强，item相似度可预计算 |

**适用场景**：
- User-CF：新闻/短视频（兴趣多变，热点驱动）
- Item-CF：电商/音乐（兴趣稳定，复购场景多）

---

### Q3: 什么是双塔模型？相比矩阵分解有什么优势？

**面试官**：双塔模型了解吗？相比传统的MF有什么改进？

**候选人（参考答案）**：
**双塔模型结构**：
```
User塔：用户特征 → DNN → User Embedding
Item塔：物品特征 → DNN → Item Embedding
Score：dot(User_emb, Item_emb)
```

**相比MF的优势**：
1. **特征扩展性**：MF只能用ID特征，双塔可融合画像、上下文等多类特征
2. **表达能力**：深度网络 > 内积操作
3. **灵活性**：两侧可独立优化（Item塔可离线预计算）
4. **负采样**：双塔可方便地做in-batch负采样

**工程优势**：Item embedding可离线生成，线上用FAISS毫秒级检索。

---

### Q4: 请解释AUC和GAUC的区别，为什么推荐系统常用GAUC？

**面试官**：AUC和GAUC的区别？为什么推荐系统要用GAUC？

**候选人（参考答案）**：
- **AUC**：全局排序能力，反映模型把正样本排在负样本前面的概率
- **GAUC（Group AUC）**：分用户计算AUC后加权平均

$$GAUC = \frac{\sum_{u} w_u \cdot AUC_u}{\sum_u w_u}$$

**为什么用GAUC**：
1. **用户差异大**：全局AUC会被活跃用户主导，无法反映长尾用户效果
2. **业务公平性**：推荐系统需要每个用户都有好体验，不只是头部用户
3. **线上相关性高**：GAUC与线上AB实验效果相关性更强

**注意**：分桶大小要合适，避免单个用户样本过少导致AUC波动大。

---

### Q5: 什么是位置偏差（Position Bias）？如何消除？

**面试官**：位置偏差了解吗？怎么解决？

**候选人（参考答案）**：
**定义**：用户点击受物品展示位置影响，前排物品天然CTR高，不一定因为更相关。

**消除方法**：

| 方法 | 原理 | 实践 |
|------|------|------|
| **IPW** | 逆概率加权，降低前排样本权重 | 需要估计位置CTR，可用随机实验数据 |
| **Pal** | 位置作为特征输入，预测时置0 | 简单有效，工程常用 |
| **DLCM** | 用Attention建模位置影响 | 模型复杂，收益边际递减 |
| **随机实验** | 随机打乱展示位置 | 成本最高，用于离线评估 |

**字节/美团常用做法**：位置作为特征，同时结合IPW做样本加权。

---

### Q6: 深度学习推荐模型中，如何处理高维稀疏特征？

**面试官**：高维稀疏特征（如用户ID、Item ID）怎么处理？

**候选人（参考答案）**：
**处理方法**：

1. **Embedding层**
   - 将高维one-hot映射到低维稠密向量
   - 参数量 = 特征取值数 × embedding_dim

2. **Hash Trick**
   - 对超高维特征（如URL）做Hash取模，控制embedding表大小
   - 冲突可控，内存友好

3. **特征共享**
   - 相关特征共享embedding（如城市和省）
   - 减少参数，提升泛化

4. **Vocabulary管理**
   - 高频特征独立ID，低频特征映射到UNK
   - 定期清理过期特征，控制表大小

5. **Embedding优化**
   - 使用HashEmbedding或Compositional Embedding减少参数量
   - 采用自适应embedding维度（AutoDim）

---

### Q7: 推荐系统中常用的损失函数有哪些？各自适用场景？

**面试官**：说说推荐系统常用的损失函数及适用场景。

**候选人（参考答案）**：

| 损失函数 | 形式 | 适用场景 |
|----------|------|----------|
| **BCE Loss** | $-[y\log(p) + (1-y)\log(1-p)]$ | 二分类（CTR/CVR预估），最常用 |
| **Softmax Cross Entropy** | $-\log\frac{e^{s_+}}{\sum e^{s_i}}$ | 采样多分类，对比学习 |
| **Hinge Loss** | $\max(0, 1 - s_+ + s_-)$ | pairwise学习，如BPR |
| **BPR Loss** | $-\log\sigma(s_+ - s_-)$ | 隐式反馈，排序优化 |
| **Focal Loss** | $-\alpha(1-p)^\gamma \log(p)$ | 解决正负样本不均衡 |
| **Listwise Loss** | Softmax over list | 精排直接优化NDCG |

**实践建议**：
- 精排：BCE Loss + Focal Loss（处理不平衡）
- 召回/粗排：Sampled Softmax或BPR（学习相对序）
- 多目标：各目标独立Loss加权或PLE动态加权

---

### Q8: 什么是多目标优化？常用方法有哪些？

**面试官**：多目标优化了解吗？有哪些方法？

**候选人（参考答案）**：
**背景**：推荐需同时优化点击、收藏、加购、购买等多个目标。

**方法演进**：

1. **共享底层 + 独立塔（Shared-Bottom）**
   - 底层共享，顶层各目标独立
   - 简单，但容易负迁移

2. **MMoE（Multi-gate Mixture-of-Experts）**
   - 多Expert + Gate机制动态加权
   - 解决负迁移，效果较好

3. **PLE（Progressive Layered Extraction）**
   - 显式分离共享组件和任务特定组件
   - 防止跷跷板现象，工业界SOTA

4. **多目标加权**
   - 手动加权：$Loss = \sum w_i L_i$
   - 自动加权：Uncertainty Weighting、GradNorm、UWL

**工程落地**：MMoE最常用，PLE在任务差异大时更好。

---

### Q9: 模型线上Serving有哪些优化手段？

**面试官**：模型上线后怎么保证低延迟高吞吐？

**候选人（参考答案）**：

**计算优化**：
- **向量检索**：FAISS/HNSW加速召回，毫秒级响应
- **模型蒸馏**：大模型蒸馏到小模型，精度损失<1%，延迟降50%
- **量化推理**：FP32→INT8，TensorRT加速
- **特征缓存**：热点用户特征预计算缓存

**系统架构**：
- **并行化**：召回多路并行，排序Batch推理
- **异步加载**：实时特征异步获取，不阻塞主链路
- **局部敏感哈希（LSH）**：近似最近邻搜索

**工程实践**：
- **服务分级**：核心链路P99<50ms，非核心可放宽
- **降级策略**：超时用简单模型或规则兜底
- **预热机制**：模型上线前预热，避免冷启动延迟

---

### Q10: 冷启动问题有哪些解决思路？

**面试官**：新用户/新物品冷启动怎么处理？

**候选人（参考答案）**：

**新用户冷启动**：
1. **基于规则**：热门推荐、编辑推荐、运营策略
2. **基于画像**：年龄/性别/地域推荐同类用户喜欢的
3. **探索机制**：EE问题（Bandit、Thompson Sampling）
4. **少样本学习**：元学习（MAML）快速适应新用户

**新物品冷启动**：
1. **内容相似**：基于内容特征找相似老物品
2. **流量扶持**：新物品保量曝光（如1000次曝光）
3. **冷启通道**：独立召回通路，避免被老物品淹没
4. **探索利用**：UCB/Thompson Sampling平衡探索与利用

**系统层面**：
- 冷启动用户走独立模型（模型更小、特征更少）
- 实时反馈：快速捕捉用户首点行为，5分钟内更新推荐

---

## 二、系统设计题：设计一个直播推荐系统（15分钟）

**面试官**：现在进入系统设计环节。请你设计一个直播推荐系统，类似于抖音直播或快手直播。考虑召回、排序、实时性、冷启动等方面。

**候选人（参考答案）**：

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         直播推荐系统架构                          │
├─────────────────────────────────────────────────────────────────┤
│  客户端                                                           │
│     ↓                                                           │
│  API Gateway → 推荐服务（RecServer）                              │
│     ↓                                                           │
│  ┌─────────────┬─────────────┬─────────────┬─────────────────┐  │
│  │  多路召回    │   粗排      │   精排      │   重排/混排      │  │
│  │  • 实时行为  │  • 双塔筛选 │  • DeepFM   │  • 多样性策略   │  │
│  │  • 关注召回  │  • 轻量模型 │  • MMOE     │  •  freshness   │  │
│  │  • 向量召回  │             │  • 多目标   │  • 业务规则     │  │
│  │  • 热门召回  │             │             │  • 疲劳度控制   │  │
│  └─────────────┴─────────────┴─────────────┴─────────────────┘  │
│     ↓                                                           │
│  结果返回 + 曝光/点击实时上报                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 1. 召回层设计（多路召回）

**为什么要多路**：直播场景用户意图多样（追关注主播、发现新内容、看热门），单一路召回覆盖不足。

| 召回通路 | 实现方式 | 占比 | 作用 |
|----------|----------|------|------|
| **关注召回** | 用户关注主播列表 | 30% | 满足社交需求，留存核心 |
| **向量召回（I2I）** | Item2Vec/DSSM，找相似直播 | 25% | 兴趣扩展 |
| **向量召回（U2I）** | 双塔模型，用户实时兴趣 | 25% | 个性化主通路 |
| **热门召回** | 当前在线人数/热度排序 | 15% | 解决冷启动，保证内容质量 |
| **探索召回** | 新主播/长尾内容 | 5% | EE探索，内容生态 |

**直播特有考虑**：
- **实时状态过滤**：召回后过滤已下播直播（用Redis维护在线状态）
- **实时行为召回**：最近点击/观看的相似直播（FAISS实时更新）
- **地理位置召回**：附近直播（LBS特征）

### 2. 粗排层

**作用**：召回千级 → 粗排筛选百级，减轻精排压力

**模型选择**：
- 轻量级双塔模型，预计算item向量
- 或者简化版DeepFM，减少层数和特征

**特征**：
- 用户画像（User ID、历史行为统计）
- 直播画像（主播ID、类目、热度）
- 交叉特征（用户-类目偏好）

### 3. 精排层

**模型选择**：MMoE多目标模型

**目标设计**：
```python
# 多目标定义
objectives = {
    'click': 0.3,      # 点击率
    'view': 0.25,      # 观看时长
    'like': 0.15,      # 点赞
    'comment': 0.15,   # 评论
    'gift': 0.1,       # 打赏
    'follow': 0.05     # 关注主播
}
```

**特征工程**：

| 特征类别 | 具体特征 | 处理方式 |
|----------|----------|----------|
| 用户特征 | User ID、年龄、性别、活跃时段、消费能力 | Embedding |
| 行为特征 | 最近观看直播、观看时长、互动行为 | Sequence建模（Transformer）|
| 直播特征 | 主播ID、类目、当前热度、开播时长 | Embedding + 连续值 |
| 上下文 | 时间、设备、网络 | 离散化/归一化 |
| 交叉特征 | 用户-类目偏好、主播-用户历史交互 | Attention自动学习 |

**实时特征**：
- 最近5分钟观看序列（Kafka+Flink实时计算）
- 当前直播间实时热度（在线人数、互动率）

### 4. 重排层

**多样性策略**：
```python
# MMR（Maximal Marginal Relevance）
score = lambda * rel_i - (1-lambda) * max_sim(i, selected)
```
- 避免同一主播连续出现
- 类目打散（最多连续3个同类直播）

**新鲜度**：
- 新开播直播加权（开播<30分钟）
- 热度衰减：Score = pred_score × exp(-decay × time)

**业务规则**：
- 疲劳度控制：用户拒绝过的主播24小时内不再推
- 流量扶持：新主播/签约主播保底曝光
- 风控过滤：低俗/违规内容拦截

### 5. 实时性保障

```
用户行为 → Kafka → Flink实时特征计算 → Redis → 召回/排序使用
                ↓
            模型实时更新（Online Learning）
```

- **延迟要求**：从用户行为到特征更新 < 1分钟
- **在线学习**：FTRL或增量更新，小时级模型迭代

### 6. 冷启动策略

**新用户**：
- 前3次请求：热门+编辑精选+新手专属内容
- 快速收集反馈：优先展示有明显特征的内容（明确类目）
- 3次后：基于初步画像走个性化

**新主播**：
- 冷启流量池：保底曝光500次
- 内容理解：用直播标题/封面图像特征，找相似老主播
- 快速反馈：前30分钟互动率决定后续流量分配

### 7. 评估体系

| 指标类型 | 具体指标 | 说明 |
|----------|----------|------|
| 点击率 | CTR | 点击/曝光 |
| 互动率 | 点赞率、评论率、分享率 | 深度互动 |
| 消费深度 | 平均观看时长、完播率 | 内容质量 |
| 商业价值 | 打赏率、关注转化率 | 营收/留存 |
| 生态健康 | 长尾主播曝光占比、新主播成长速度 | 内容生态 |

**面试官追问**：如果某个主播通过刷量提升热度，怎么识别？

**候选人补充**：
1. **异常检测**：统计检测（同IP大量进入、观看时长极短）
2. **特征层面**：热度特征中加入"真实互动率"（去除机器行为）
3. **模型层面**：热度分与内容质量分分离，排序时综合
4. **事后惩罚**：确认刷量后降权/封禁，并回溯修正历史数据

---

## 三、代码题：实现MMoE的PyTorch代码（15分钟）

**面试官**：请你用PyTorch实现MMoE（Multi-gate Mixture-of-Experts）模型，支持3个任务。

**候选人（参考答案）**：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Expert(nn.Module):
    """单个Expert网络"""
    def __init__(self, input_dim: int, expert_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_dim, expert_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MMOELayer(nn.Module):
    """MMoE核心层：多Expert + Gate机制"""
    def __init__(self, input_dim: int, expert_dim: int, num_experts: int, num_tasks: int):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        
        # 多个Expert
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_dim) for _ in range(num_experts)
        ])
        
        # 每个任务有一个Gate网络
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=-1)
            ) for _ in range(num_tasks)
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [batch_size, input_dim]
        Returns:
            list of [batch_size, expert_dim], length=num_tasks
        """
        # 每个Expert的输出 [batch_size, expert_dim]
        expert_outputs = [expert(x) for expert in self.experts]  # list of (B, D)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, expert_dim]
        
        # 每个任务的输出
        task_outputs = []
        for i in range(self.num_tasks):
            # Gate权重 [batch_size, num_experts]
            gate_weights = self.gates[i](x)  # [B, num_experts]
            
            # 加权融合Expert输出 [batch_size, expert_dim]
            weighted_output = torch.bmm(
                gate_weights.unsqueeze(1),  # [B, 1, num_experts]
                expert_outputs               # [B, num_experts, expert_dim]
            ).squeeze(1)  # [B, expert_dim]
            
            task_outputs.append(weighted_output)
        
        return task_outputs


class MMOE(nn.Module):
    """
    完整MMoE模型，支持多任务学习
    任务示例：CTR、CVR、观看时长
    """
    def __init__(
        self,
        feature_dims: dict,  # {'user_id': 10000, 'item_id': 5000, ...}
        embedding_dim: int = 64,
        expert_dim: int = 128,
        num_experts: int = 4,
        num_tasks: int = 3,
        task_hidden_dims: List[int] = [128, 64]
    ):
        super().__init__()
        self.num_tasks = num_tasks
        
        # Embedding层
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embedding_dim)
            for name, dim in feature_dims.items()
        })
        
        # 计算输入维度（假设所有embedding拼接）
        input_dim = embedding_dim * len(feature_dims)
        
        # MMOE层
        self.mmoe_layer = MMOELayer(input_dim, expert_dim, num_experts, num_tasks)
        
        # 每个任务的Tower网络
        self.task_towers = nn.ModuleList()
        for _ in range(num_tasks):
            layers = []
            prev_dim = expert_dim
            for hidden_dim in task_hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                prev_dim = hidden_dim
            # 输出层
            layers.append(nn.Linear(prev_dim, 1))
            self.task_towers.append(nn.Sequential(*layers))
        
        # 任务输出类型：sigmoid、sigmoid、linear（时长）
        self.task_output_types = ['sigmoid', 'sigmoid', 'linear']
    
    def forward(self, features: dict) -> List[torch.Tensor]:
        """
        Args:
            features: {'user_id': [B], 'item_id': [B], 'category': [B], ...}
        Returns:
            list of [batch_size, 1], 每个任务的预测值
        """
        # Embedding查找并拼接
        embeds = []
        for name, ids in features.items():
            if name in self.embeddings:
                embeds.append(self.embeddings[name](ids))
        x = torch.cat(embeds, dim=-1)  # [B, input_dim]
        
        # MMOE层输出每个任务的表示
        task_representations = self.mmoe_layer(x)  # list of [B, expert_dim]
        
        # 每个任务独立Tower预测
        predictions = []
        for i in range(self.num_tasks):
            pred = self.task_towers[i](task_representations[i])  # [B, 1]
            
            # 根据任务类型应用输出函数
            if self.task_output_types[i] == 'sigmoid':
                pred = torch.sigmoid(pred)
            
            predictions.append(pred)
        
        return predictions


def test_mmoe():
    """测试MMoE模型"""
    # 配置
    batch_size = 32
    feature_dims = {
        'user_id': 10000,
        'item_id': 5000,
        'category': 100,
        'hour': 24
    }
    
    # 创建模型
    model = MMOE(
        feature_dims=feature_dims,
        embedding_dim=64,
        expert_dim=128,
        num_experts=4,
        num_tasks=3
    )
    
    # 模拟输入数据
    features = {
        'user_id': torch.randint(0, 10000, (batch_size,)),
        'item_id': torch.randint(0, 5000, (batch_size,)),
        'category': torch.randint(0, 100, (batch_size,)),
        'hour': torch.randint(0, 24, (batch_size,))
    }
    
    # 前向传播
    predictions = model(features)
    
    print("MMoE模型测试")
    print(f"输入batch size: {batch_size}")
    print(f"输出任务数: {len(predictions)}")
    for i, pred in enumerate(predictions):
        print(f"  任务{i+1}输出shape: {pred.shape}, 范围: [{pred.min():.4f}, {pred.max():.4f}]")
    
    # 计算损失示例
    labels = [
        torch.randint(0, 2, (batch_size, 1)).float(),  # CTR (0/1)
        torch.randint(0, 2, (batch_size, 1)).float(),  # CVR (0/1)
        torch.randn(batch_size, 1).abs()              # 时长 (连续值)
    ]
    
    losses = []
    loss_fns = [nn.BCELoss(), nn.BCELoss(), nn.MSELoss()]
    
    for i in range(3):
        loss = loss_fns[i](predictions[i], labels[i])
        losses.append(loss)
        print(f"  任务{i+1}损失: {loss.item():.4f}")
    
    total_loss = sum(losses)
    print(f"总损失: {total_loss.item():.4f}")
    
    return model


if __name__ == "__main__":
    test_mmoe()
```

**面试官追问**：如果两个任务相关性很低（比如点击率和举报率），MMoE能处理好么？怎么改进？

**候选人补充**：

MMoE对于相关性低的任务可能出现**跷跷板现象**（一个任务提升，另一个下降）。改进方案：

1. **PLE（Progressive Layered Extraction）**
```python
# 核心思想：显式分离共享和任务特定组件
class PLELayer(nn.Module):
    def __init__(self, input_dim, expert_dim, num_shared_experts, num_task_experts, num_tasks):
        # 共享Experts（所有任务共用）
        self.shared_experts = nn.ModuleList([...])
        # 任务特定Experts（每个任务独享）
        self.task_specific_experts = nn.ModuleList([
            nn.ModuleList([...]) for _ in range(num_tasks)
        ])
        # 每个任务有自己的Gate，选择共享+独享Expert
```

2. **动态加权**：根据任务难度动态调整Loss权重
```python
# Uncertainty Weighting
log_vars = torch.nn.Parameter(torch.zeros(num_tasks))
loss = torch.sum(0.5 * torch.exp(-log_vars) * task_losses + log_vars)
```

3. **先验知识**：已知不相关的任务，完全独立建模，不走MMoE。

---

## 四、场景题：如何解决推荐系统的流行度偏差？（5分钟）

**面试官**：推荐系统中热门物品（流行度高的）总是被过度推荐，导致马太效应，怎么解决？

**候选人（参考答案）**：

### 问题分析
**流行度偏差（Popularity Bias）**：热门物品因为曝光多→点击多→被模型学得更强→继续被多曝光，形成正反馈，长尾内容得不到展示。

### 解决方案

#### 1. 数据层面

**负采样修正**：
- 传统：全局均匀负采样
- 改进：按 popularity^α 采样（α=0.5~0.75），让热门item有更多机会成为负样本，降低其分数

**Propensity Score（IPS）**：
```python
# 训练时加权：冷门样本权重高，热门样本权重低
weight = 1.0 / propensity_score(item)
loss = weight * bce_loss(pred, label)
```

#### 2. 模型层面

**去偏特征**：
- 不直接使用点击数作为特征，而使用"点击/曝光比"（CTR）
- 或者引入流行度作为特征，让模型学习流行度的影响

**解耦表示学习**：
- 物品表示 = 质量表示 + 流行度表示
- 排序时只用质量表示，去除流行度影响

#### 3. 训练策略

**对比学习去偏**：
```python
# 拉近同质量但不同流行度物品的表示
# 推远不同质量但同流行度物品的表示
loss = bce_loss + λ * contrastive_loss
```

#### 4. 线上干预

**多样性重排**：
- MMR算法控制热门物品占比，强制插入长尾内容
- 规则：每页最多50%热门内容

**流量扶持**：
- 长尾物品保底曝光（如每天每个物品至少100次曝光）
- 探索机制：ε-贪心，10%流量用于随机探索

#### 5. 评估修正

** beyond-accuracy 指标**：
- Coverage（覆盖率）：推荐系统覆盖的item占比
- Gini系数：衡量推荐分布不平等程度
- 长尾准确率：单独看冷门物品的推荐效果

### 实际落地建议

1. **短**：训练时IPS加权，降低热门item分数
2. **中**：重排层多样性控制，MMR+流量扶持
3. **长**：建立内容生态评估体系，看新作者成长速度

---

## 五、反问环节建议（3分钟）

**面试官**：今天的面试差不多结束了，你有什么问题想问我吗？

### 推荐反问问题（按优先级）

#### 1. 团队/业务相关（推荐必问）
- "咱们团队目前主要负责哪个业务线的推荐？是Feed流、直播还是其他？"
- "团队目前的规模大概是怎样的？算法和工程的比例如何？"
- "推荐系统目前的最大挑战是什么？是模型效果、工程性能还是业务目标定义？"

#### 2. 技术栈/发展相关
- "咱们推荐系统的技术栈是怎样的？用的是自研框架还是TensorFlow/PyTorch？"
- "模型的迭代频率是怎样的？在线学习还是离线批量更新？"
- "对于新人，团队有哪些培养机制？比如mentor制度、内部技术分享等"

#### 3. 工作相关
- "这个岗位的日常工作是怎样的？模型研发占比多少，工程开发占比多少？"
- "团队目前最缺哪方面的人才？是模型优化、系统工程还是数据分析？"

#### 4. 避免问的问题
n- ❌ "我这次面试表现怎么样？"（让面试官为难）
- ❌ "加班多吗？"（可以问"工作节奏是怎样的"）
- ❌ "薪资是多少？"（HR面再问）

---

## 面试总结与评分参考

### 考察点权重

| 考察维度 | 权重 | 关键指标 |
|----------|------|----------|
| 基础理论 | 25% | 概念清晰、能举一反三 |
| 系统设计 | 30% | 架构合理、考虑全面、有取舍 |
| 工程能力 | 25% | 代码规范、边界处理、优化意识 |
| 业务理解 | 15% | 懂业务指标、有产品思维 |
| 沟通表达 | 5% | 逻辑清晰、态度积极 |

### 面试官评分卡

```
候选人：________  日期：________  面试官：________

基础概念 (10题)  □ 优秀(9-10) □ 良好(7-8) □ 及格(5-6) □ 不及格(<5)
直播系统设计    □ 优秀       □ 良好      □ 及格      □ 不及格
MMoE代码实现    □ 优秀       □ 良好      □ 及格      □ 不及格
流行度偏差场景  □ 优秀       □ 良好      □ 及格      □ 不及格

综合评价：__________________________________________

是否进入下一轮：□ 是  □ 否
```

---

> **提示**：本模拟面试涵盖了大厂推荐算法岗的核心考察点。建议候选人结合自己的项目经历，对每道题准备1-2个实际案例，增强说服力。
