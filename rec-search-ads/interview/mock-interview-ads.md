# 🎯 广告算法岗模拟面试

> **面试官**：字节跳动/腾讯/阿里妈妈 - 广告算法组  
> **候选人**：MelonEggLearn  
> **岗位**：高级算法工程师 - 广告方向

---

## 一、基础概念（5题）

### Q1: CTR/CVR预估的差异及样本空间问题

**参考答案：**

| 维度 | CTR预估 | CVR预估 |
|------|---------|---------|
| **目标** | p(click=1\|impression) | p(convert=1\|click) |
| **正样本** | 点击 | 转化 |
| **样本空间** | 全量曝光 | 有点击的子集 |
| **延迟** | 实时反馈 | 分钟/小时/天级延迟 |
| **稀疏度** | ~1-5% | ~0.1-1% |

**样本空间问题（Sample Selection Bias, SSB）：**
```
传统做法：在点击样本上训练 CVR 模型
问题：训练分布 p(x|click=1) ≠ 推理分布 p(x|impression)
      导致模型在曝光空间预测不准

解决方案：
1. ESMM - 在全量曝光样本上训练
2. 倾向性得分（IPS）- 加权校正
3. 因果推断方法 - 反事实估计
```

**追问：为什么CVR延迟问题比CTR严重？**
- 转化路径长：点击 → 落地页 → 填写信息 → 支付 → 确认
- 不同场景延迟差异大：游戏激活（小时）vs 电商下单（天）
- 归因窗口通常7-30天

---

### Q2: ESMM模型如何解决CVR样本稀疏？

**参考答案：**

**核心思想：** 多任务学习 + 概率链式法则

```
pCVR = p(conversion|impression, click=1)
     = p(conversion|impression) / p(click|impression)
     = pCTCVR / pCTR

ESMM 同时学习 CTR 和 CTCVR，间接得到 CVR
```

**网络结构：**
```
┌─────────────────────────────────────────┐
│           Shared Embedding               │
│  [User特征] [Item特征] [Context特征]      │
└─────────────┬───────────────────────────┘
              │
    ┌─────────┴─────────┐
    ▼                   ▼
┌─────────┐       ┌─────────┐
│  CTR塔   │       │ CTCVR塔 │
│ (隐层)   │       │ (隐层)   │
└────┬────┘       └────┬────┘
     │                 │
     ▼                 ▼
   pCTR              pCTCVR
     │                 │
     └────────┬────────┘
              ▼
           pCVR = pCTCVR / pCTR
```

**关键优势：**
1. **全量样本训练**：CTR和CTCVR都在曝光样本上学习
2. **缓解SSB**：训练分布和推理分布一致
3. **缓解DS（Data Sparsity）**：CTR塔为CVR塔提供迁移学习
4. **避免除零风险**：实际实现用 exp(log pCTCVR - log pCTR)

**损失函数：**
```python
L = L_CTR + L_CTCVR
  = -Σ[y_click * log(pCTR) + (1-y_click)*log(1-pCTR)]
    -Σ[y_conv * log(pCTCVR) + (1-y_conv)*log(1-pCTCVR)]
```

---

### Q3: 广告竞价：GSP vs VCG机制

**参考答案：**

| 维度 | GSP (Generalized Second Price) | VCG (Vickrey-Clarke-Groves) |
|------|-------------------------------|-----------------------------|
| **计费公式** | bid_{i+1} * quality_{i+1} / quality_i | externality_i = Σ(bid_j * x_j) - Σ(bid_j * x_j without i) |
| **核心思想** | 按下一个广告的价格付费 | 按给其他广告主带来的损失付费 |
| **占优策略** | 不是真实报价（需bid shading）| 真实报价是占优策略 |
| **计算复杂度** | O(n log n) | O(n²) 或更高 |
| **收入** | 较高 | 较低（理论下界）|
| **应用** | Google/Bing/Facebook/字节/腾讯 | 学术界、部分RTB场景 |

**GSP计费公式（质量度模式）：**
```
排名得分 = bid * quality_score
实际扣费 = (下一名bid * 下一名quality) / 当前quality + 0.01

假设：
  广告主A: bid=10, quality=0.8 → 得分=8
  广告主B: bid=8, quality=0.7 → 得分=5.6
  广告主C: bid=5, quality=0.6 → 得分=3

A的扣费 = (8 * 0.7) / 0.8 + 0.01 = 7.01
B的扣费 = (5 * 0.6) / 0.7 + 0.01 = 4.29
```

**为什么工业界主要用GSP？**
1. 简单易实现，计算高效
2. 收入通常高于VCG
3. 广告主已习惯bid shading策略
4. 平台可通过quality_score调节生态

---

### Q4: 探索与利用：ε-greedy/UCB/Thompson Sampling

**参考答案：**

**问题定义（Multi-Armed Bandit）：**
- K个拉杆（候选广告），每个有未知奖励分布
- 目标：最大化T轮后的累积奖励
- 核心矛盾：探索（explore）vs 利用（exploit）

| 算法 | 核心思想 | 公式 | 特点 |
|------|---------|------|------|
| **ε-greedy** | 以ε概率随机探索，1-ε选当前最优 | a = random() if rand<ε else argmax(Q) | 简单，ε需调参，探索效率低 |
| **UCB** | 上界置信区间，考虑不确定性 | Q(a) + c√(ln t / N(a)) | 有理论bound，确定性策略 |
| **Thompson** | 采样后验分布选动作 | a ~ π(θ), θ ~ Posterior | 天然随机，效果好，计算复杂 |

**UCB详解：**
```
UCB(a) = μ̂_a + √(2lnT / N_a)

μ̂_a: 当前平均奖励估计
N_a: 动作a的尝试次数
T: 总轮数

第一项：利用（估计值高）
第二项：探索（尝试次数少）
```

**Thompson Sampling详解：**
```
1. 假设每个臂的奖励服从Beta(α, β)分布
2. 从每个臂的后验中采样一个值
3. 选择采样值最大的臂
4. 观测奖励后更新后验参数

Beta分布共轭先验：
   点击 ~ Bernoulli(θ)
   θ ~ Beta(α, β)
   后验: θ|data ~ Beta(α+clicks, β+impressions-clicks)
```

**广告场景应用：**
- 冷启动：新广告需要探索
- 位置分配：上下文相关的bandit（Contextual Bandit/LinUCB）
- 出价探索：在预估pCTR附近探索最优bid

---

### Q5: 预算分配与Pacing算法

**参考答案：**

**问题背景：**
- 广告主设定日预算 B，目标是在一天内平滑消耗
- 挑战：流量波动、竞价环境变化、效果最大化

**经典Pacing算法：**

**1. 基于PID控制的Pacing**
```
当前 pacing_rate = 1 + Kp*e(t) + Ki*∫e(t)dt + Kd*de(t)/dt

其中 e(t) = 目标消耗率 - 实际消耗率

应用：调节bid或参与竞价概率
  bid_adjusted = bid_raw * pacing_rate
  或
  prob_participate = min(1, pacing_rate)
```

**2. 预算消耗曲线（Pacing Curve）**
```
目标消耗比例 = f(时间比例)

常见曲线：
- 匀速：f(t) = t
- 前慢后快：f(t) = t²
- 前快后慢：f(t) = √t
- 基于流量质量：f(t) = ∫quality(τ)dτ / ∫quality(τ)dτ_total
```

**3. 多层级预算控制**
```
账户预算
  ├── 计划A预算
  │     ├── 单元A1预算
  │     └── 单元A2预算
  └── 计划B预算
        └── ...

控制策略：
- 上层剩余预算影响下层bid调整
- 保证层级间不超额
```

**效果与消耗的平衡（Dual Problem）：**
```
max Σ value(a)
s.t. Σ cost(a) ≤ B

拉格朗日对偶：
max Σ [value(a) - λ*cost(a)]

λ是shadow price，决定当前是否值得竞价
```

---

## 二、系统设计：实时CTR预估系统（10w QPS）

### 题目要求

设计一个支持10万QPS的实时广告CTR预估系统，支持特征实时更新、模型在线学习、完整的实验和监控体系。

### 参考答案

#### 1. 系统整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                           流量入口层                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐              │
│  │ Gateway │  │  Load   │  │   Rate  │  │  Circuit│              │
│  │         │  │Balancer │  │ Limiter │  │ Breaker │              │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘              │
└───────┼────────────┼────────────┼────────────┼───────────────────┘
        │            │            │            │
        └────────────┴────────────┴────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          预估服务层                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  预估服务集群 (10w QPS / 单机5k QPS = 20+ 实例)           │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │    │
│  │  │ Instance1│  │ Instance2│  │   ...    │  │ InstanceN│  │    │
│  │  │ - Model  │  │ - Model  │  │          │  │ - Model  │  │    │
│  │  │ - Feature│  │ - Feature│  │          │  │ - Feature│  │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│   特征服务     │      │   模型服务     │      │   日志回流     │
│ Feature Store  │      │ Model Store   │      │ Kafka/Flume   │
│ - Redis Cluster│      │ - 内存热加载   │      │               │
│ - Local Cache  │      │ - 版本管理    │      │               │
└───────────────┘      └───────────────┘      └───────────────┘
```

#### 2. 特征工程架构

**特征分层：**

| 层级 | 类型 | 特征示例 | 更新延迟 | 存储 |
|------|------|---------|---------|------|
| L1 | 实时上下文 | 时间、位置、设备 | 实时 | 请求携带 |
| L2 | 用户实时特征 | 最近点击/曝光序列 | <5分钟 | Redis |
| L3 | 用户短期特征 | 近1天兴趣标签 | ~15分钟 | Redis + Local Cache |
| L4 | 用户长期特征 | 历史偏好统计 | 天级 | KV存储 + Local Cache |
| L5 | 广告实时特征 | 当前ctr、消耗速率 | <1分钟 | Redis |
| L6 | 广告静态特征 | 广告主行业、素材 | 离线 | 本地内存 |

**特征实时计算链路：**
```
用户行为流:
  Client → Kafka → Flink → Feature Store (Redis)
                      ↓
                 窗口聚合(滑动窗口)
                 - 近30分钟点击数
                 - 近1小时曝光类目分布
                 - 实时序列特征(last N items)

模型回流特征:
  预估日志 → Kafka → Flink → 特征拼接 → 
  正负样本生成 → 训练样本队列
```

**特征存储设计：**
```
Redis Cluster 分片策略:
  - 用户特征：hash(user_id) % 1024
  - 广告特征：hash(ad_id) % 256
  
Local Cache (LRU):
  - 热点用户特征：命中率>95%
  - 广告静态特征：全量加载
  
序列特征存储:
  - Redis List: [item1, item2, ..., itemN]
  - 维护最近50个交互item
```

#### 3. 模型架构选型

**选型对比：**

| 模型 | 推理延迟 | AUC增益 | 复杂度 | 适用场景 |
|------|---------|---------|--------|---------|
| LR | <1ms | Baseline | 低 | 极致性能要求 |
| GBDT | 2-5ms | +3-5% | 中 | 中小规模 |
| Wide&Deep | 5-10ms | +5-8% | 中 | 通用场景 |
| DeepFM | 5-10ms | +6-9% | 中 | 自动特征交叉 |
| DIN | 10-15ms | +8-12% | 高 | 序列建模强 |
| DIEN | 15-25ms | +10-15% | 很高 | 兴趣演化 |
| MMOE | 10-20ms | +7-10% | 高 | 多任务 |

**推荐架构：Wide&Deep + DIN（混合部署）**

```
基础流量 → Wide&Deep（快速过滤）
优质流量 → DIN（精细预估）

或
粗排阶段：双塔模型（向量检索）
精排阶段：DIN/MMOE
```

**模型轻量化方案：**
```python
# 知识蒸馏
Teacher: DIN-大模型 (AUC=0.82)
Student: 浅层DIN (AUC=0.805, 延迟-60%)

# 量化
FP32 → INT8: 模型体积-75%, 延迟-40%

# 算子优化
- TensorRT/FasterTransformer
- 自定义CUDA kernel
- Batch推理优化
```

#### 4. 在线学习体系

**实时训练流程：**
```
曝光/点击/转化 事件流
       │
       ▼
┌──────────────┐
│ Kafka Topic  │
│ (样本流)      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Flink Join   │ ← 关联特征、标签延迟处理
│ (窗口: 30min)│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 实时训练     │ ← TF/PyTorch Streaming
│ - FTRL       │    每5分钟一个mini-batch
│ - Online SGD │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 模型验证     │ ← AUC检验、异常检测
│ & 发布       │
└──────────────┘
```

**更新频率策略：**
| 模块 | 更新频率 | 方式 |
|------|---------|------|
| Embedding层 | 5分钟 | 在线学习 |
| DNN权重 | 1小时 | 增量训练 |
| 全量模型 | 天级 | 全量训练 |
| 特征词典 | 小时级 | 动态扩缩容 |

**在线学习算法：**
```python
# FTRL (Follow The Regularized Leader)
# 适合大规模稀疏特征

w_{t+1} = argmin_w { g_{1:t}·w + λ1||w||_1 + λ2||w||_2^2 
                      + 1/2 Σ_{s=1}^t σ_s||w-w_s||^2 }

# 优点:
# - 支持L1正则，产生稀疏解
# - 自适应学习率
# - 适合实时流式训练
```

#### 5. A/B实验体系

**分层实验架构：**
```
总流量 (100%)
  ├── 层1: 召回策略 (桶A/B/C)
  ├── 层2: 粗排模型 (桶D/E)
  ├── 层3: 精排模型 (桶F/G/H) ← 你的实验在这里
  ├── 层4: 出价策略 (桶I/J)
  └── 层5: 创意优选 (桶K/L)

正交分层：每层独立哈希，实验互不影响
```

**实验指标：**

| 类型 | 指标 | 说明 |
|------|------|------|
| **核心指标** | CTR | 点击率，直接反映预估质量 |
| | CPM | 千次展示收益 |
| | RPM | 千次请求收益 |
| **辅助指标** | AUC | 离线排序能力 |
| | PCOC | 预估校准度 (预测CTR/真实CTR) |
| | 覆盖率 | 有预估结果的比例 |
| **观测指标** | 延迟P99 | 服务稳定性 |
| | 模型新鲜度 | 最后一次更新时间 |

**实验设计要点：**
```
样本量计算：
  n = 16 * σ² / δ²
  假设CTR=0.02，希望检测相对提升5% (δ=0.001)
  n ≈ 16 * 0.02 * 0.98 / (0.001)² ≈ 310万曝光
  
实验时长：至少覆盖一个完整周期（7天），避免工作日/周末偏差
```

#### 6. 核心指标监控

**监控Dashboard：**
```
┌────────────────────────────────────────────────────────┐
│                    实时监控大屏                         │
├────────────────────────────────────────────────────────┤
│  QPS: ████████████████████ 95k/100k (95%)              │
│  延迟P99: 12ms (SLA: 20ms) ✅                          │
│  错误率: 0.001% ✅                                      │
├────────────────────────────────────────────────────────┤
│  今日CTR: 2.85% (基准: 2.80% ↑1.8%)                    │
│  今日CPM: ¥45.2 (基准: ¥44.0 ↑2.7%)                    │
├────────────────────────────────────────────────────────┤
│  AUC: 0.823 (小时级)                                    │
│  PCOC: 1.02 (理想: 1.0)                                 │
├────────────────────────────────────────────────────────┤
│  特征缺失率: 0.5%                                        │
│  模型版本: v2.3.1 (发布于 2小时前)                       │
└────────────────────────────────────────────────────────┘
```

**告警规则：**
| 指标 | 阈值 | 级别 |
|------|------|------|
| P99延迟 > 50ms | 持续1分钟 | P0-紧急 |
| 错误率 > 0.1% | 持续2分钟 | P0-紧急 |
| AUC下降 > 2% | 单次 | P1-高 |
| PCOC偏离 [0.9, 1.1] | 持续10分钟 | P1-高 |
| QPS突降 > 30% | 单次 | P0-紧急 |

**模型效果监控：**
```python
# 实时AUC计算（近似）
# 使用 Reservoir Sampling 保存正负样本对

class StreamingAUC:
    def __init__(self, reservoir_size=10000):
        self.positive_scores = []  # 正例分数蓄水池
        self.negative_scores = []  # 负例分数蓄水池
        
    def update(self, pred, label):
        if label == 1:
            self._reservoir_update(self.positive_scores, pred)
        else:
            self._reservoir_update(self.negative_scores, pred)
    
    def get_auc(self):
        # 计算蓄水池中样本对的AUC
        return self._compute_auc_from_samples()
```

---

## 三、代码题：PyTorch实现ESMM

### 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class ESMM(nn.Module):
    """
    Entire Space Multi-Task Model
    同时学习CTR和CVR，解决样本选择偏差和数据稀疏问题
    
    Reference: "Entire Space Multi-Task Model: An Effective Approach 
                for Estimating Post-Click Conversion Rate"
    """
    
    def __init__(
        self,
        feature_dims: Dict[str, int],  # 各特征域的维度
        embed_dim: int = 64,
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rate: float = 0.2,
        use_bn: bool = True
    ):
        """
        Args:
            feature_dims: 特征词典，如 {'user_id': 100000, 'item_id': 50000, ...}
            embed_dim: 嵌入维度
            hidden_dims: MLP隐层维度
            dropout_rate: Dropout比例
            use_bn: 是否使用BatchNorm
        """
        super(ESMM, self).__init__()
        
        self.feature_dims = feature_dims
        self.embed_dim = embed_dim
        self.num_features = len(feature_dims)
        
        # ========== 共享Embedding层 ==========
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embed_dim)
            for name, dim in feature_dims.items()
        })
        
        # 对embedding做初始化
        for emb in self.embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        
        # 输入维度 = 特征数 * 嵌入维度
        input_dim = self.num_features * embed_dim
        
        # ========== CTR Tower ==========
        self.ctr_tower = self._build_mlp(
            input_dim, hidden_dims, dropout_rate, use_bn
        )
        self.ctr_head = nn.Linear(hidden_dims[-1], 1)
        
        # ========== CTCVR Tower ==========
        # ESMM的创新：预测pCTCVR = pCTR * pCVR
        self.ctcvr_tower = self._build_mlp(
            input_dim, hidden_dims, dropout_rate, use_bn
        )
        self.ctcvr_head = nn.Linear(hidden_dims[-1], 1)
        
    def _build_mlp(
        self, 
        input_dim: int, 
        hidden_dims: List[int],
        dropout_rate: float,
        use_bn: bool
    ) -> nn.Sequential:
        """构建MLP网络"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            
        return nn.Sequential(*layers)
    
    def forward(
        self, 
        features: Dict[str, torch.Tensor],
        return_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            features: 输入特征，{特征名: 特征值}
            return_logits: 是否返回原始logits
            
        Returns:
            {
                'p_ctr': CTR概率,
                'p_cvr': CVR概率 (推导得到),
                'p_ctcvr': CTCVR概率,
                'logits_ctr': CTR logits (可选),
                'logits_ctcvr': CTCVR logits (可选)
            }
        """
        # ========== Embedding Lookup ==========
        embed_list = []
        for name in self.feature_dims.keys():
            idx = features[name]  # [batch_size]
            emb = self.embeddings[name](idx)  # [batch_size, embed_dim]
            embed_list.append(emb)
        
        # 拼接所有embedding: [batch_size, num_features * embed_dim]
        embed_concat = torch.cat(embed_list, dim=1)
        
        # ========== CTR Tower ==========
        ctr_hidden = self.ctr_tower(embed_concat)
        logits_ctr = self.ctr_head(ctr_hidden).squeeze(-1)  # [batch_size]
        p_ctr = torch.sigmoid(logits_ctr)
        
        # ========== CTCVR Tower ==========
        ctcvr_hidden = self.ctcvr_tower(embed_concat)
        logits_ctcvr = self.ctcvr_head(ctcvr_hidden).squeeze(-1)  # [batch_size]
        p_ctcvr = torch.sigmoid(logits_ctcvr)
        
        # ========== 推导pCVR ==========
        # pCVR = pCTCVR / pCTR
        # 为避免除零和数值不稳定，使用log空间计算
        # log pCVR = log pCTCVR - log pCTR
        # pCVR = exp(log pCTCVR - log pCTR)
        
        epsilon = 1e-7
        log_p_ctr = torch.log(p_ctr + epsilon)
        log_p_ctcvr = torch.log(p_ctcvr + epsilon)
        
        log_p_cvr = log_p_ctcvr - log_p_ctr
        p_cvr = torch.exp(log_p_cvr)
        
        # 裁剪到[0, 1]范围
        p_cvr = torch.clamp(p_cvr, 0.0, 1.0)
        
        result = {
            'p_ctr': p_ctr,
            'p_cvr': p_cvr,
            'p_ctcvr': p_ctcvr
        }
        
        if return_logits:
            result['logits_ctr'] = logits_ctr
            result['logits_ctcvr'] = logits_ctcvr
            
        return result


class ESMMLoss(nn.Module):
    """ESMM联合损失函数"""
    
    def __init__(self, ctr_weight: float = 1.0, ctcvr_weight: float = 1.0):
        super(ESMMLoss, self).__init__()
        self.ctr_weight = ctr_weight
        self.ctcvr_weight = ctcvr_weight
        
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor],
        labels_ctr: torch.Tensor,
        labels_ctcvr: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算联合损失
        
        Args:
            predictions: 模型输出
            labels_ctr: 点击标签 (0/1)，所有曝光样本都有
            labels_ctcvr: 转化标签 (0/1)，只有点击样本有
            mask: 有效样本mask（用于处理CTCVR标签缺失）
        """
        p_ctr = predictions['p_ctr']
        p_ctcvr = predictions['p_ctcvr']
        
        # CTR损失：所有曝光样本
        loss_ctr = F.binary_cross_entropy(p_ctr, labels_ctr.float(), reduction='mean')
        
        # CTCVR损失：所有曝光样本（转化标签对未点击样本为0）
        if mask is not None:
            # 只计算有效样本的CTCVR损失
            loss_ctcvr = F.binary_cross_entropy(
                p_ctcvr[mask], 
                labels_ctcvr[mask].float(),
                reduction='mean'
            )
        else:
            loss_ctcvr = F.binary_cross_entropy(
                p_ctcvr, 
                labels_ctcvr.float(),
                reduction='mean'
            )
        
        # 联合损失
        total_loss = self.ctr_weight * loss_ctr + self.ctcvr_weight * loss_ctcvr
        
        return total_loss, {
            'loss_ctr': loss_ctr.item(),
            'loss_ctcvr': loss_ctcvr.item(),
            'loss_total': total_loss.item()
        }


# ========== 使用示例 ==========

def demo():
    """ESMM使用示例"""
    
    # 配置
    feature_dims = {
        'user_id': 100000,
        'item_id': 50000,
        'category_id': 1000,
        'hour': 24,
        'device_type': 10
    }
    
    batch_size = 32
    
    # 初始化模型
    model = ESMM(
        feature_dims=feature_dims,
        embed_dim=64,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.2
    )
    
    # 模拟输入数据
    features = {
        'user_id': torch.randint(0, 100000, (batch_size,)),
        'item_id': torch.randint(0, 50000, (batch_size,)),
        'category_id': torch.randint(0, 1000, (batch_size,)),
        'hour': torch.randint(0, 24, (batch_size,)),
        'device_type': torch.randint(0, 10, (batch_size,))
    }
    
    # 模拟标签
    labels_ctr = torch.randint(0, 2, (batch_size,)).float()  # 点击标签
    labels_ctcvr = torch.zeros(batch_size)
    # 只有点击的样本才可能有转化
    for i in range(batch_size):
        if labels_ctr[i] == 1:
            labels_ctcvr[i] = torch.randint(0, 2, (1,)).item() * 0.1  # 转化稀疏
    
    # 前向传播
    predictions = model(features)
    
    print("=== ESMM 预测结果 ===")
    print(f"CTR概率范围: [{predictions['p_ctr'].min():.4f}, {predictions['p_ctr'].max():.4f}]")
    print(f"CVR概率范围: [{predictions['p_cvr'].min():.4f}, {predictions['p_cvr'].max():.4f}]")
    print(f"CTCVR概率范围: [{predictions['p_ctcvr'].min():.4f}, {predictions['p_ctcvr'].max():.4f}]")
    
    # 验证关系: pCTR * pCVR ≈ pCTCVR
    p_ctr = predictions['p_ctr']
    p_cvr = predictions['p_cvr']
    p_ctcvr = predictions['p_ctcvr']
    computed_ctcvr = p_ctr * p_cvr
    print(f"\n验证 pCTR * pCVR ≈ pCTCVR:")
    print(f"平均偏差: {(computed_ctcvr - p_ctcvr).abs().mean():.6f}")
    
    # 计算损失
    criterion = ESMMLoss(ctr_weight=1.0, ctcvr_weight=1.0)
    loss, loss_dict = criterion(predictions, labels_ctr, labels_ctcvr)
    
    print(f"\n=== 损失 ===")
    print(f"CTR Loss: {loss_dict['loss_ctr']:.4f}")
    print(f"CTCVR Loss: {loss_dict['loss_ctcvr']:.4f}")
    print(f"Total Loss: {loss_dict['loss_total']:.4f}")
    
    # 统计
    print(f"\n=== 数据统计 ===")
    print(f"曝光样本数: {batch_size}")
    print(f"点击样本数: {labels_ctr.sum().item()}")
    print(f"转化样本数: {labels_ctcvr.sum().item()}")
    print(f"点击率: {labels_ctr.mean():.4f}")
    print(f"转化率: {labels_ctcvr.sum() / labels_ctr.sum():.4f}" if labels_ctr.sum() > 0 else "N/A")
    
    return model, predictions


if __name__ == "__main__":
    demo()
```

---

## 四、场景题：如何处理位置偏差（Position Bias）

### 问题背景

位置偏差指广告的点击率受展示位置影响（通常越靠前CTR越高），这会导致：
1. 模型学习到"位置"而非"真实相关性"
2. 新广告/尾部广告难以获得公平展示机会
3. 排序结果偏离真实用户偏好

### 解决方案对比

| 方案 | 核心思想 | 优缺点 | 适用场景 |
|------|---------|--------|---------|
| **位置特征** | 训练时加入位置，推理时统一设为0 | 简单，但破坏概率解释性 | 快速迭代 |
| **Pal/DLCM** | 分离检验因子（examination）和相关性 | 可解释性强，需位置随机化数据 | 学术/实验 |
| **IPW/IPS** | 倾向性得分加权校正 | 需要位置倾向估计，方差大 | 已有位置分布数据 |
| **UnbiasCTR** | 多任务学习分解位置和点击 | 端到端训练，效果较好 | 工业界主流 |
| **点击模型** | Cascade/DBN用户浏览模型 | 假设强，参数难估计 | 搜索场景 |
| **随机实验** | 随机位置展示收集无偏数据 | 成本高，样本少 | 校准/验证 |

### 详细方案：PAL (Position-bias Aware Learning)

```
模型结构：
┌─────────────────────────────────────────┐
│              输入特征                    │
│    [用户] [广告] [上下文]                │
└─────────────┬───────────────────────────┘
              │
    ┌─────────┴─────────┐
    ▼                   ▼
┌─────────┐       ┌─────────┐
│ 相关性塔 │       │ 位置偏置塔│
│   MLP    │       │   MLP    │
└────┬────┘       └────┬────┘
     │                 │
     ▼                 ▼
  pRelevance        pExamination
     │                 │
     └────────┬────────┘
              ▼
        pCTR = pRelevance * pExamination
```

**关键实现：**
```python
class PositionBiasModel(nn.Module):
    def __init__(self, user_dim, item_dim, num_positions=10):
        super().__init__()
        # 相关性网络（与位置无关）
        self.relevance_net = nn.Sequential(
            nn.Linear(user_dim + item_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # 检验网络（仅依赖位置）
        self.examination_net = nn.Embedding(num_positions, 1)
        nn.init.uniform_(self.examination_net.weight, -0.1, 0.1)
        
    def forward(self, user_feat, item_feat, position, training=True):
        # 相关性得分（位置无关）
        relevance = self.relevance_net(
            torch.cat([user_feat, item_feat], dim=1)
        )
        
        if training:
            # 训练时使用真实位置
            examination = torch.sigmoid(
                self.examination_net(position)
            )
            ctr = relevance * examination
        else:
            # 推理时假设位置1（或平均位置）
            ctr = relevance  # 或使用 examination(position=1)
        
        return {
            'ctr': ctr,
            'relevance': relevance,
            'examination': examination if training else None
        }
```

### 工业界实践：联合训练方案

```python
class PositionDebiasModel(nn.Module):
    """
    工业界常用方案：
    1. 训练时同时使用User/Item/Position特征
    2. 推理时Position设为固定值(如1)或去掉
    3. 多任务学习辅助去偏
    """
    
    def __init__(self, feature_dims, num_positions=20):
        super().__init__()
        
        # 普通特征Embedding
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, 64) 
            for name, dim in feature_dims.items()
        })
        
        # 位置Embedding
        self.position_emb = nn.Embedding(num_positions, 16)
        
        # 主预测网络
        input_dim = len(feature_dims) * 64 + 16
        self.main_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 辅助网络：预测位置（去偏监督）
        self.position_predictor = nn.Sequential(
            nn.Linear(len(feature_dims) * 64, 64),
            nn.ReLU(),
            nn.Linear(64, num_positions)
        )
        
    def forward(self, features, position, training=True):
        # Embedding lookup
        embs = [self.embeddings[f](features[f]) for f in features]
        feat_concat = torch.cat(embs, dim=1)
        
        # 辅助任务：位置预测（强制模型从User/Item中学习位置无关特征）
        pos_logits = self.position_predictor(feat_concat.detach())
        
        if training:
            # 训练时使用真实位置
            pos_emb = self.position_emb(position)
        else:
            # 推理时固定位置（如首位）
            fixed_pos = torch.zeros_like(position)
            pos_emb = self.position_emb(fixed_pos)
        
        # 主预测
        concat = torch.cat([feat_concat, pos_emb], dim=1)
        ctr = self.main_net(concat)
        
        return {
            'ctr': ctr,
            'position_logits': pos_logits
        }
    
    def compute_loss(self, output, labels, position):
        """联合损失"""
        # CTR预测损失
        ctr_loss = F.binary_cross_entropy(
            output['ctr'].squeeze(), 
            labels.float()
        )
        
        # 位置预测损失（对抗训练思想）
        # 模型应该无法从User/Item特征预测位置（位置无关）
        pos_loss = F.cross_entropy(
            output['position_logits'],
            position
        )
        
        # 总损失：让主网络利用好位置，但辅助网络难以预测位置
        # 实际上常用方案是：推理时固定位置，或完全去掉位置特征
        total_loss = ctr_loss - 0.1 * pos_loss  # 对抗
        
        return total_loss
```

### 无偏数据收集：随机化实验

```
实施方法：
1. 随机位置实验：小流量(如1%)随机打乱排序结果的位置
2. 结果：获得 position vs click 的真实关系
3. 应用：
   - 校准位置偏置系数
   - 作为去偏模型训练数据
   - 评估其他去偏方案效果

数据分析示例：
┌──────────┬────────────┬────────────┐
│ 位置     │ 原始CTR    │ 随机化CTR  │
├──────────┼────────────┼────────────┤
│ 1        │ 8.5%       │ 3.2%       │
│ 2        │ 5.2%       │ 3.0%       │
│ 3        │ 3.8%       │ 2.9%       │
│ ...      │ ...        │ ...        │
│ 10       │ 0.8%       │ 2.5%       │
└──────────┴────────────┴────────────┘

原始CTR差异大 → 位置影响严重
随机化后差异小 → 真实相关性差异
```

### 总结

**选择建议：**

1. **快速上线**：推理时位置特征置为固定值（如位置1的值）
2. **中长期优化**：采用PAL/ UnbiasCTR 多任务分解架构
3. **数据驱动**：定期做随机位置实验，校准偏置强度
4. **系统配合**：
   - 新广告冷启动：随机探索获取不同位置数据
   - 效果评估：对比不同位置下的CTR公平性
   - 长期价值：避免位置偏差导致的"马太效应"

---

## 五、面试官点评

### 总体评价

| 考察点 | 评分 | 点评 |
|--------|------|------|
| 理论基础 | A | 对核心概念理解扎实，能举一反三 |
| 系统设计 | A- | 架构完整，细节考虑周全，可补充容灾设计 |
| 代码能力 | A | ESMM实现规范，考虑数值稳定性 |
| 业务洞察 | A- | 对位置偏差有深入思考，可补充更多A/B测试经验 |

### 后续建议

1. **深入方向**：因果推断在广告中的应用（CMAB、Uplift Model）
2. **工程强化**：模型 Serving 性能优化（TensorRT、量化压缩）
3. **业务扩展**：竞价策略、预算分配算法的联动优化

---

*祝面试顺利！🎉*
