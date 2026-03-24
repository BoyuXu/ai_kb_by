# 推荐系统中的因果推断 (Causal Inference in Recommendation Systems)

> 📚 参考文献
> - [Etegrec Generative Recommender With End-To-End Lea](../../rec-sys/papers/20260323_etegrec_generative_recommender_with_end-to-end_lea.md) — ETEGRec: Generative Recommender with End-to-End Learnable...
> - [Linear-Item-Item-Session-Rec](../../rec-sys/papers/20260319_linear-item-item-session-rec.md) — Linear Item-Item Model with Neural Knowledge for Session-...
> - [Gems-Breaking-The-Long-Sequence-Barrier-In-Gene...](../../rec-sys/papers/20260321_gems-breaking-the-long-sequence-barrier-in-generative-recommendation-with-a-multi-stream-decoder.md) — GEMs: Breaking the Long-Sequence Barrier in Generative Re...
> - [A-Unified-Language-Model-For-Large-Scale-Search...](../../rec-sys/papers/20260321_a-unified-language-model-for-large-scale-search-recommendation-and-reasoning-at-spotify.md) — A Unified Language Model for Large Scale Search, Recommen...
> - [A Generative Re-Ranking Model For List-Level Multi](../../rec-sys/papers/20260323_a_generative_re-ranking_model_for_list-level_multi.md) — A Generative Re-ranking Model for List-level Multi-object...
> - [Variable-Length-Semantic-Ids-For-Recommender-Sy...](../../rec-sys/papers/20260321_variable-length-semantic-ids-for-recommender-systems.md) — Variable-Length Semantic IDs for Recommender Systems
> - [Interplay-Training-Independent-Simulators-For-R...](../../rec-sys/papers/20260321_interplay-training-independent-simulators-for-reference-free-conversational-recommendation.md) — Interplay: Training Independent Simulators for Reference-...
> - [Mmoe-Multi-Task-Learning](../../rec-sys/papers/20260317_mmoe-multi-task-learning.md) — MMoE：多门控混合专家（Multi-gate Mixture-of-Experts）


> MelonEggLearn 专题整理 | 推荐系统进阶

---

## 一、为什么需要因果推断？

### 1.1 推荐系统的核心问题：从相关到因果

传统推荐系统基于**关联学习**（Correlation-based Learning），目标是拟合用户历史行为：

```
P(点击|曝光) → 最大化预测准确率
```

但用户行为由两部分组成：
- **因果效应 (Treatment Effect)**：用户真正喜欢导致的点击
- **混杂偏差 (Confounding Bias)**：位置、流行度等外部因素导致的点击

```
观察到的点击 = f(真实兴趣) + f(位置偏见) + f(流行度偏见) + f(选择偏见) + ε
```

### 1.2 三大核心偏差

#### 1.2.1 混淆偏差 (Confounding Bias / Selection Bias)

**问题描述**：训练数据来自已有推荐系统的曝光，用户只能看到被推荐的物品，形成**MNAR (Missing Not At Random)** 数据。

```
用户 ←→ 物品
  ↓      ↓
  被观测到的交互 (有偏样本)
  
未观测到的交互 → 无法学习
```

**后果**：
- 模型只在热门物品上表现好
- 新物品/长尾物品得不到曝光机会
- 形成「马太效应」

#### 1.2.2 曝光偏差 / 位置偏差 (Exposure Bias / Position Bias)

**问题描述**：用户更倾向于点击位置靠前的物品，而非因为真正喜欢。

| 位置 | 相对点击率 |
|------|-----------|
| 1    | 100% (baseline) |
| 2    | ~70% |
| 3    | ~50% |
| 5    | ~30% |
| 10   | ~10% |

**数学建模**：
```
P(点击|曝光在位置k, 真实相关性r) = P(检查位置k) × P(点击|检查, r)
                                = θ_k × r
```

其中 θ_k 是位置k的「检查概率」(examination probability)。

#### 1.2.3 流行度偏差 (Popularity Bias)

**问题描述**：热门物品获得更多曝光和点击，模型倾向于推荐热门物品。

```
物品流行度 → 更多曝光 → 更多点击 → 模型认为用户喜欢 → 更多推荐 → 更热门
     ↑_____________________________________________________|
```

**量化分析**：
- 头部20%的物品可能占据80%的交互
- 长尾物品即使相关也难以被发现

### 1.3 因果推断的解决思路

```
┌─────────────────────────────────────────────────────────────┐
│                    因果推断框架                              │
├─────────────────────────────────────────────────────────────┤
│  潜在结果框架 (Potential Outcomes)                           │
│  Y(1): 如果曝光物品，用户的反馈                              │
│  Y(0): 如果不曝光，用户的反馈                                │
│  因果效应: τ = E[Y(1) - Y(0)]                               │
├─────────────────────────────────────────────────────────────┤
│  结构因果模型 (Structural Causal Model)                      │
│  显式建模变量间的因果关系                                     │
│  X → Z → Y                                                  │
│  └──→^                                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、倾向分 (Propensity Score) 去偏

### 2.1 什么是倾向分

**定义**：倾向分是给定协变量X条件下，接受处理（曝光）的概率。

```
e(X) = P(T=1 | X)

其中：
- T=1: 物品被曝光/被点击
- X: 用户特征、物品特征、上下文特征
```

### 2.2 倾向分的性质

**关键定理（Rosenbaum & Rubin, 1983）**：

如果满足**无混淆性假设 (Unconfoundedness)**，则：
```
{T=1, T=0} ⊥ Y(1), Y(0) | e(X)
```

即：给定倾向分后，处理分配与潜在结果独立。

### 2.3 倾向分估计方法

#### 方法1：逻辑回归
```python
import numpy as np
from sklearn.linear_model import LogisticRegression

def estimate_propensity_lr(features, treatment):
    """
    features: [N, D] - 用户/物品特征
    treatment: [N] - 是否曝光 (0/1)
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(features, treatment)
    propensity = model.predict_proba(features)[:, 1]
    return propensity
```

#### 方法2：神经网络 (更常用)
```python
import torch
import torch.nn as nn

class PropensityNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)
```

#### 方法3：基于日志的估计 (工业界常用)
```python
def estimate_propensity_from_logs(log_data, position):
    """
    基于位置点击率估计倾向分
    """
    # P(点击|位置k) / P(点击|位置1)
    position_ctr = log_data.groupby('position')['click'].mean()
    propensity = position_ctr / position_ctr.iloc[0]
    return propensity
```

### 2.4 倾向分的裁剪与平滑

```python
def clip_propensity(propensity, lower=0.01, upper=0.99):
    """
    裁剪极端倾向分，防止权重过大
    """
    return np.clip(propensity, lower, upper)

def smooth_propensity(propensity, alpha=0.1):
    """
    添加平滑项，处理估计误差
    """
    return (propensity + alpha) / (1 + 2*alpha)
```

---

## 三、IPS / SNIPS 加权校正

### 3.1 IPS (Inverse Propensity Scoring)

**核心思想**：用逆倾向分加权，让被低估的样本（低曝光概率）获得更高权重。

#### 3.1.1 数学原理

**标准损失（有偏）**：
```
L_standard = -1/N Σ [Y_i · log(Ŷ_i)]
```

**IPS校正损失（无偏）**：
```
L_IPS = -1/N Σ [ (T_i · Y_i · log(Ŷ_i)) / e(X_i) ]

其中：
- T_i: 是否曝光 (0/1)
- Y_i: 观测到的反馈 (点击/评分)
- e(X_i): 倾向分
```

**无偏性证明**：
```
E[L_IPS] = E[ (T·Y·log(Ŷ)) / e(X) ]
         = E[ E[ (T·Y·log(Ŷ)) / e(X) | X ] ]
         = E[ (E[T|X] · E[Y(1)|X] · log(Ŷ)) / e(X) ]  (给定X，T⊥Y)
         = E[ (e(X) · E[Y(1)|X] · log(Ŷ)) / e(X) ]
         = E[ E[Y(1)|X] · log(Ŷ) ]
         = 真实期望损失 ✓
```

#### 3.1.2 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class IPSLoss(nn.Module):
    def __init__(self, propensity_estimator=None):
        super().__init__()
        self.propensity_estimator = propensity_estimator
        self.eps = 1e-6
    
    def forward(self, predictions, labels, features, exposure_mask):
        """
        predictions: [B] - 模型预测概率
        labels: [B] - 真实标签 (0/1)
        features: [B, D] - 特征用于估计倾向分
        exposure_mask: [B] - 是否曝光 (0/1)
        """
        # 估计倾向分
        if self.propensity_estimator is not None:
            with torch.no_grad():
                propensity = self.propensity_estimator(features).squeeze()
                propensity = torch.clamp(propensity, min=0.01, max=0.99)
        else:
            # 使用预计算的倾向分
            propensity = features['propensity'].squeeze()
        
        # 计算加权损失
        ce_loss = F.binary_cross_entropy(predictions, labels, reduction='none')
        
        # IPS加权: 只对曝光的样本加权
        ips_weights = exposure_mask / (propensity + self.eps)
        
        # 归一化权重（可选，对应SNIPS）
        # ips_weights = ips_weights / ips_weights.sum() * len(ips_weights)
        
        weighted_loss = (ce_loss * ips_weights).mean()
        
        return weighted_loss
```

### 3.2 SNIPS (Self-Normalized IPS)

**问题**：IPS方差大，当倾向分很小时权重爆炸。

**解决方案**：自归一化

```
L_SNIPS = - Σ [ (T_i · Y_i · log(Ŷ_i)) / e(X_i) ] / Σ [ T_i / e(X_i) ]
```

**优势**：
1. 权重和归一化为样本数，方差更小
2. 对倾向分估计误差更鲁棒

```python
class SNIPSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6
    
    def forward(self, predictions, labels, propensity, exposure_mask):
        ce_loss = F.binary_cross_entropy(predictions, labels, reduction='none')
        
        # 计算权重
        ips_weights = exposure_mask / (propensity + self.eps)
        
        # 自归一化
        weighted_loss = (ce_loss * ips_weights).sum() / (ips_weights.sum() + self.eps)
        
        return weighted_loss
```

### 3.3 倾向分估计的联合训练

```python
class JointEstimator(nn.Module):
    """
    同时训练推荐模型和倾向分估计器
    """
    def __init__(self, user_dim, item_dim, hidden_dim=64):
        super().__init__()
        # 推荐模型
        self.recommender = nn.Sequential(
            nn.Linear(user_dim + item_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 倾向分估计器
        self.propensity_net = nn.Sequential(
            nn.Linear(user_dim + item_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, user_feat, item_feat):
        concat = torch.cat([user_feat, item_feat], dim=-1)
        
        # 推荐分数
        score = self.recommender(concat)
        
        # 倾向分
        propensity = self.propensity_net(concat.detach())  # 阻断梯度
        
        return score, propensity
```

---

## 四、Counterfactual Learning (反事实学习)

### 4.1 核心概念

**反事实问题**：「如果用户看到了物品A（实际没看），他会点击吗？」

```
事实世界 (Factual):    曝光 → 点击/未点击
反事实世界 (Counterfactual): 未曝光 → ？？？
```

### 4.2 基于反事实学习的推荐框架

#### 4.2.1 问题建模

```
目标：估计每个用户-物品对的因果效应

τ(u, i) = E[Y(u, i) | do(expose(u, i))]

其中 do(·) 表示干预操作
```

#### 4.2.2 反事实风险最小化 (CRM)

```
R(θ) = E[ δ(O, Y, Ŷ) / π(O|X) ]

其中：
- O: 观测行为
- Y: 真实标签
- Ŷ: 预测
- π: 日志策略（产生数据的旧模型）
- δ: 损失函数
```

### 4.3 Doubly Robust (DR) 估计器

**动机**：IPS方差大，直接模型(DM)有偏，DR结合两者优势。

**公式**：
```
Ŷ_DR = Ŷ_DM + (Y - Ŷ_DM) · T / e(X)

其中：
- Ŷ_DM: 直接模型的预测
- Y: 观测到的标签
- T: 处理指示
- e(X): 倾向分
```

**双重鲁棒性**：只要倾向分估计或直接模型有一个正确，估计就是无偏的。

```python
class DoublyRobustLoss(nn.Module):
    """
    双重鲁棒损失
    """
    def __init__(self, imputation_model):
        super().__init__()
        self.imputation_model = imputation_model  # 直接模型
        self.eps = 1e-6
    
    def forward(self, pred, label, propensity, exposure, user_feat, item_feat):
        # 直接模型预测
        imputed_label = self.imputation_model(user_feat, item_feat)
        
        # 误差项
        error = label - imputed_label
        
        # DR估计
        dr_estimate = imputed_label + (exposure * error) / (propensity + self.eps)
        
        # 使用DR估计计算损失
        loss = F.binary_cross_entropy(pred, dr_estimate.detach())
        
        return loss
```

### 4.4 基于多任务的反事实学习

```python
class CounterfactualMultiTask(nn.Module):
    """
    同时学习观测到的交互和反事实预测
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 主任务：预测点击率
        self.click_pred = nn.Linear(hidden_dim, 1)
        
        # 辅助任务：预测曝光概率（去偏）
        self.exposure_pred = nn.Linear(hidden_dim, 1)
        
        # 反事实预测：估计未曝光物品的潜在兴趣
        self.counterfactual_pred = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        h = self.encoder(x)
        
        click_score = torch.sigmoid(self.click_pred(h))
        exposure_prob = torch.sigmoid(self.exposure_pred(h))
        cf_score = torch.sigmoid(self.counterfactual_pred(h))
        
        return {
            'click': click_score,
            'exposure': exposure_prob,
            'counterfactual': cf_score
        }
```

### 4.5 反事实数据增强

```python
def generate_counterfactual_samples(user_history, all_items, n_negative=5):
    """
    生成反事实负样本
    """
    cf_samples = []
    
    for user, interacted in user_history.items():
        # 用户未交互的物品
        non_interacted = list(set(all_items) - set(interacted))
        
        # 采样作为反事实样本
        sampled = np.random.choice(
            non_interacted, 
            size=min(n_negative, len(non_interacted)),
            replace=False
        )
        
        for item in sampled:
            cf_samples.append({
                'user': user,
                'item': item,
                'label': 0,  # 假设未点击
                'is_counterfactual': True
            })
    
    return cf_samples
```

---

## 五、DML (Double Machine Learning, 双重机器学习)

### 5.1 DML 核心思想

**背景**：传统方法在估计因果效应时，需要正确指定模型形式，否则会产生正则化偏差。

**DML解决方案**：
1. 使用机器学习模型灵活估计干扰参数
2. 通过Neyman正交化消除正则化偏差

### 5.2 数学框架

**部分线性模型 (Partially Linear Model)**：
```
Y = θ · T + g(X) + ε    (结果方程)
T = m(X) + η            (处理方程)

其中：
- Y: 结果 (点击/评分)
- T: 处理 (曝光)
- X: 协变量
- θ: 目标因果效应
- g(X), m(X): 任意复杂的 nuisance 函数
```

**估计步骤**：

```
Step 1: 样本分割 (Sample Splitting)
        D → D1 ∪ D2

Step 2: 在D1上估计 nuisance 函数
        ĝ(·) ≈ argmin E[(Y - g(X))²]
        ṁ(·) ≈ argmin E[(T - m(X))²]

Step 3: 在D2上估计因果效应
        θ̂ = argmin E[(Y - ĝ(X) - θ·(T - ṁ(X)))²]
```

### 5.3 DML for Recommendation

```python
import torch
import torch.nn as nn

class DML4Rec(nn.Module):
    """
    双重机器学习用于推荐系统
    """
    def __init__(self, user_dim, item_dim, hidden_dim=128):
        super().__init__()
        
        # Nuisance function 1: 结果模型 g(X)
        self.outcome_model = nn.Sequential(
            nn.Linear(user_dim + item_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Nuisance function 2: 倾向分模型 m(X)
        self.propensity_model = nn.Sequential(
            nn.Linear(user_dim + item_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 因果效应估计器
        self.treatment_effect = nn.Linear(1, 1)
    
    def forward(self, user_feat, item_feat, treatment):
        x = torch.cat([user_feat, item_feat], dim=-1)
        
        # Step 1: 估计 nuisance 函数
        g_x = self.outcome_model(x)  # E[Y|X]
        m_x = self.propensity_model(x)  # E[T|X]
        
        # Step 2: 计算残差
        # (这里假设Y和T已知，实际训练中传入)
        
        return g_x, m_x
    
    def estimate_causal_effect(self, user_feat, item_feat, y, t):
        """
        估计因果效应 θ
        """
        x = torch.cat([user_feat, item_feat], dim=-1)
        
        # Neyman正交评分函数
        g_x = self.outcome_model(x).squeeze()
        m_x = self.propensity_model(x).squeeze()
        
        # 正交化残差
        residual_y = y - g_x
        residual_t = t - m_x
        
        # 估计 θ
        # θ = E[(Y - g(X))(T - m(X))] / E[(T - m(X))²]
        theta = (residual_y * residual_t).mean() / (residual_t ** 2).mean()
        
        return theta
```

### 5.4 Cross-Fitting DML

```python
class CrossFittingDML:
    """
    交叉拟合DML，提高样本效率
    """
    def __init__(self, model, n_folds=2):
        self.model = model
        self.n_folds = n_folds
        self.models = [model for _ in range(n_folds)]
    
    def fit(self, data):
        n = len(data)
        fold_size = n // self.n_folds
        
        for i in range(self.n_folds):
            # 分割数据
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < self.n_folds - 1 else n
            
            train_data = data[:val_start] + data[val_end:]
            val_data = data[val_start:val_end]
            
            # 在训练集上拟合 nuisance
            self.models[i].fit_nuisance(train_data)
            
            # 在验证集上估计 θ
            theta_i = self.models[i].estimate_theta(val_data)
        
        # 聚合结果
        self.theta = np.mean([m.theta for m in self.models])
        return self.theta
```

### 5.5 DML vs 传统方法对比

| 方法 | 模型灵活性 | 偏差控制 | 计算成本 | 适用场景 |
|-----|----------|---------|---------|---------|
| IPS | 低 | 依赖倾向分准确 | 低 | 倾向分易估计 |
| DR | 中 | 双重保障 | 中 | 需要额外建模 |
| DML | 高 | 正交化消除 | 高 | 复杂nuisance |

---

## 六、工业界案例

### 6.1 快手 D2Q (Deconfounded Deep Q-learning)

**论文**: "Deconfounded Recommendation for Alleviating Bias Amplification" (KDD 2021)

#### 6.1.1 背景问题

快手短视频推荐面临：
1. **位置偏差**：用户倾向滑到后面的视频
2. **选择偏差**：只能观测到被推荐的视频反馈
3. **反馈循环**：偏差随时间放大

#### 6.1.2 核心创新

```
D2Q框架 = Deconfounded + Deep Q-learning

         ┌─────────────────────────────────────┐
         │           D2Q 架构                  │
         ├─────────────────────────────────────┤
         │  State: 用户历史观看序列             │
         │  Action: 选择哪个视频推荐            │
         │  Reward: 观看时长 / 互动            │
         │  Confounder: 视频曝光历史           │
         └─────────────────────────────────────┘
```

#### 6.1.3 技术方案

**1. 因果图建模**：
```
用户兴趣(U) → 曝光(Z) → 反馈(Y)
     ↓______________↑
     
U是混杂因子，影响曝光和反馈
```

**2. Deconfounded Q-network**：
```
Q(s, a) = Q_base(s, a) + correction_term

correction_term 使用 IPS 或 DR 方法计算
```

**3. 实现细节**：
```python
class D2QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # 状态编码
        self.state_encoder = TransformerEncoder(state_dim, hidden_dim)
        
        # 基础Q值
        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 倾向分估计 (用于去偏)
        self.propensity_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state, action, action_features):
        state_emb = self.state_encoder(state)
        
        # 基础Q值
        q_input = torch.cat([state_emb, action_features], dim=-1)
        q_base = self.q_network(q_input)
        
        # 估计倾向分
        propensity = self.propensity_net(state)
        
        # IPS校正
        ips_weight = 1.0 / (propensity[:, action] + 1e-6)
        
        return q_base, ips_weight
```

**4. 离线训练流程**：
```
1. 从日志中提取 (s, a, r, s')
2. 估计每个action的倾向分
3. 计算 IPS 加权 TD error
4. 更新Q-network
```

#### 6.1.4 实验效果

| 指标 | 基线 | D2Q | 提升 |
|-----|-----|-----|-----|
| 观看时长 | 100% | 115% | +15% |
| 多样性 | 100% | 130% | +30% |
| 长尾视频曝光 | 100% | 180% | +80% |

### 6.2 抖音去偏实践

#### 6.2.1 推荐系统中的偏差问题

抖音面临的核心挑战：
1. **创作者冷启动**：新视频难以获得初始曝光
2. **兴趣漂移**：用户兴趣变化快，历史行为不代表未来
3. **位置效应复杂**：不只是位置，还有上下文视频的影响

#### 6.2.2 整体去偏架构

```
┌─────────────────────────────────────────────────────────────┐
│                    抖音去偏架构                              │
├─────────────────────────────────────────────────────────────┤
│  数据层                                                      │
│  ├── 曝光日志 (带位置信息)                                    │
│  ├── 点击/完播日志                                           │
│  └── 随机流量 (小流量随机推荐用于估计倾向分)                   │
├─────────────────────────────────────────────────────────────┤
│  模型层                                                      │
│  ├── 主排序模型 (双塔/DIN)                                   │
│  ├── 倾向分估计器 (独立网络)                                  │
│  └── 反事实评估器 (预估如果曝光会怎样)                         │
├─────────────────────────────────────────────────────────────┤
│  训练层                                                      │
│  ├── IPS/SNIPS 加权                                         │
│  ├── 多任务学习 (曝光预测 + 点击预测)                          │
│  └── 对抗去偏 (Adversarial Debiasing)                        │
├─────────────────────────────────────────────────────────────┤
│  评估层                                                      │
│  ├── 随机流量评估 (无偏估计)                                  │
│  ├── 反事实A/B测试                                           │
│  └── 长期用户满意度指标                                       │
└─────────────────────────────────────────────────────────────┘
```

#### 6.2.3 位置偏差处理

**方案1：位置作为特征 (Position as Feature)**
```python
class PositionAwareModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_tower = UserTower()
        self.item_tower = ItemTower()
        self.position_embedding = nn.Embedding(100, 32)
        
    def forward(self, user_feat, item_feat, position, training=True):
        user_emb = self.user_tower(user_feat)
        item_emb = self.item_tower(item_feat)
        
        # 位置嵌入
        pos_emb = self.position_embedding(position)
        
        # 训练时使用位置，推理时不使用
        if training:
            score = torch.dot(user_emb, item_emb) + torch.dot(pos_emb, item_emb)
        else:
            score = torch.dot(user_emb, item_emb)
        
        return torch.sigmoid(score)
```

**方案2：PAL (Position-bias Aware Learning)**
```python
class PALModel(nn.Module):
    """
    显式建模位置效应
    score = relevance(user, item) × examination(position)
    """
    def __init__(self):
        super().__init__()
        self.relevance_net = nn.Sequential(...)  # 用户-物品相关性
        self.examination_net = nn.Sequential(...)  # 位置检查概率
    
    def forward(self, user_feat, item_feat, position):
        relevance = self.relevance_net(torch.cat([user_feat, item_feat]))
        examination = self.examination_net(position)
        
        # 乘法模型
        score = relevance * examination
        return score, relevance, examination
```

#### 6.2.4 流行度去偏

```python
class PopularityDebiasedLoss(nn.Module):
    def __init__(self, item_popularity, temperature=0.5):
        super().__init__()
        self.popularity = item_popularity
        self.t = temperature
    
    def forward(self, logits, labels, item_ids):
        # 获取物品流行度
        pop = self.popularity[item_ids]
        
        # 流行度惩罚权重
        # 热门物品权重低，冷门物品权重高
        weights = 1.0 / (pop ** self.t + 1e-6)
        weights = weights / weights.mean()  # 归一化
        
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        weighted_loss = (ce_loss * weights).mean()
        
        return weighted_loss
```

#### 6.2.5 工业级实践要点

**1. 倾向分估计的稳定性**
```python
def stable_propensity_estimation():
    """
    工业界倾向分估计的最佳实践
    """
    # 1. 使用多天的数据平滑估计
    # 2. 分层估计（按用户活跃度、物品类别等）
    # 3. 定期更新，但不要太频繁
    # 4. 异常值处理（top位置CTR可能超过100%需要校准）
    pass
```

**2. 在线A/B测试设计**
```
反事实实验设计：
- 对照组：现有模型
- 实验组1：IPS加权训练
- 实验组2：位置去偏模型
- 实验组3：流行度去偏

评估指标：
- 短期：CTR、完播率、互动率
- 中期：用户留存、会话时长
- 长期：创作者生态、内容多样性
```

### 6.3 其他工业案例

| 公司 | 方法 | 场景 | 效果 |
|-----|-----|-----|-----|
| 快手 | D2Q | 短视频推荐 | 观看时长+15% |
| 抖音 | PAL + IPS | 短视频推荐 | 长尾曝光+80% |
| 阿里 | DBias | 商品推荐 | 新商品CTR+20% |
| 美团 | CVR去偏 | 转化率预估 | 预估偏差-30% |
| YouTube | Position Debias | 视频推荐 | 多样性+25% |

---

## 七、面试回答指南

### 7.1 高频问题：「如何处理推荐系统的偏差问题？」

#### 回答框架（STAR法则变体）

```
S - Situation: 推荐系统中存在哪些偏差
T - Types: 偏差分类（曝光/位置/流行度/选择）
A - Approaches: 解决方案分层
R - Results: 如何评估效果
```

#### 标准回答模板

> **【开场】识别偏差类型**
>
> 推荐系统主要有三类偏差：
> 1. **选择偏差/曝光偏差**：用户只能看到被推荐的物品，导致观测数据MNAR
> 2. **位置偏差**：用户倾向点击靠前的物品，不反映真实兴趣
> 3. **流行度偏差**：热门物品获得更多曝光和点击，形成马太效应
>
> **【展开】解决方案**
>
> 针对不同偏差，我了解的解决方案包括：
>
> **1. 数据层面**
> - 随机曝光：小流量随机推荐，获得无偏数据估计倾向分
> - 数据增强：采样未曝光样本作为反事实负样本
>
> **2. 模型层面**
> - **IPS/SNIPS加权**：用逆倾向分加权，让低曝光样本获得更高权重
>   ```
>   L = Σ (y · log(ŷ)) / e(x)
>   ```
> - **位置建模**：显式建模位置效应，如PAL模型 score = relevance × examination(pos)
> - **流行度惩罚**：损失函数中对热门物品降权
> - **反事实学习**：使用DR估计器结合直接模型和IPS
>
> **3. 训练层面**
> - 多任务学习：同时预测曝光概率和点击率
> - 对抗去偏：用对抗网络学习去偏表示
>
> **【深入】技术细节（可选，根据面试官反应）**
>
> 以IPS为例，核心挑战是**倾向分估计**：
> - 可以用历史CTR估计位置倾向分
> - 需要注意倾向分裁剪（clipping），防止极端权重
> - SNIPS自归一化可以降低方差
>
> **【评估】效果验证**
>
> - 离线：随机流量数据上的无偏评估
> - 在线：长期用户留存、内容多样性指标
> - 反事实评估：预估如果改变推荐策略会怎样
>
> **【总结】**
>
> 去偏是一个系统工程，需要数据、模型、评估三管齐下。实践中通常从简单的位置建模开始，逐步引入IPS等更复杂的方法。

### 7.2 进阶问题与回答

#### Q1: IPS的方差问题如何解决？

> **回答要点**：
> 1. **裁剪（Clipping）**：将倾向分限制在[0.01, 0.99]范围
> 2. **SNIPS**：自归一化降低方差
> 3. **DR估计器**：双重鲁棒，即使倾向分估计不准也能保持无偏
> 4. **方差正则化**：在损失中加入方差惩罚项

#### Q2: 如何估计倾向分？

> **回答要点**：
> 1. **基于位置**：统计每个位置的平均CTR
> 2. **基于模型**：用逻辑回归或神经网络估计 P(expose|X)
> 3. **随机流量**：小流量随机推荐获得无偏估计
> 4. **分层估计**：按用户群体、物品类别分别估计

#### Q3: 反事实学习和传统多任务学习的区别？

> **回答要点**：
> - 多任务学习同时预测曝光和点击，但任务间关系不明确
> - 反事实学习显式建模因果关系：曝光 → 潜在结果 → 观测结果
> - 反事实学习关注「如果曝光会怎样」，多任务学习关注「曝光的概率是多少」

### 7.3 面试禁忌 ❌

| 错误说法 | 为什么错 |
|---------|---------|
| "增加负样本采样可以解决选择偏差" | 负采样不能解决MNAR问题，只是增加训练数据 |
| "直接用位置作为特征然后推理时设0" | 这样学出的模型依赖位置特征，设0后效果不好 |
| "IPS就是给冷门物品加权重" | IPS是给低曝光概率的样本加权，不等于冷门物品 |
| "去偏会降低CTR" | 短期可能降，但长期提升用户满意度和留存 |

### 7.4 加分项 💡

```
✓ 提到「随机流量」是估计倾向分的黄金标准
✓ 区分「训练时去偏」和「推理时去偏」
✓ 讨论去偏与探索（Exploration）的关系
✓ 提到工业界的具体案例（快手D2Q、抖音PAL）
✓ 讨论长期效果 vs 短期指标的权衡
```

---

## 八、总结与延伸阅读

### 8.1 方法选择决策树

```
面临什么偏差问题？
    │
    ├── 位置偏差？
    │   ├── 位置少且固定 → 位置特征/embedding
    │   └── 位置多变 → PAL模型 / IPS
    │
    ├── 选择偏差？
    │   ├── 有随机流量 → 估计准确倾向分 → IPS
    │   └── 无随机流量 → 启发式倾向分 / 隐式反馈建模
    │
    ├── 流行度偏差？
    │   ├── 损失重加权
    │   └── 因果嵌入 (Causal Embedding)
    │
    └── 多种偏差共存？
        └── DR估计器 / DML
```

### 8.2 核心论文推荐

| 论文 | 会议 | 核心贡献 |
|-----|-----|---------|
| Unbiased Learning-to-Rank with Biased Feedback | WWW'18 | IPS应用于排序，开山之作 |
| Addressing Unmeasured Confounders | SIGIR'19 | 处理未观测混杂因子 |
| Doubly Robust Joint Learning | IJCAI'19 | DR估计器用于推荐 |
| Model-Agnostic Counterfactual Estimators | KDD'20 | 模型无关的反事实估计 |
| Deconfounded Recommendation (D2Q) | KDD'21 | 快手的强化学习+去偏 |
| Position Bias Estimation | SIGIR'21 | 位置偏差估计综述 |

### 8.3 关键公式速查

```
┌─────────────────────────────────────────────────────────────┐
│  IPS损失:  L = -Σ (T·Y·log(Ŷ)) / e(X)                      │
│  SNIPS损失: L = -Σ (T·Y·log(Ŷ)) / e(X) / Σ(T/e(X))         │
│  DR估计:   Ŷ = g(X) + (Y-g(X))·T/e(X)                      │
│  PAL模型:  score = rel(u,i) × exam(pos)                    │
└─────────────────────────────────────────────────────────────┘
```

---

*文档版本: v1.0 | 最后更新: 2026-03-12 | MelonEggLearn*
