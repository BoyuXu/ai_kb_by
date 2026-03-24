# Project 3：从 LTR 到 RL 的排序学习演进

> 项目类型：算法深度 | 日期：20260324 | 领域：ads × machine learning

## 导言

广告排序中的"学习"经历了演进：
```
【Phase 1】离线学习：LTR (Learning To Rank)
  用离线标注数据训练排序模型
  
【Phase 2】在线学习：RL (Reinforcement Learning)
  用实时用户反馈学习排序策略
  
【Phase 3】混合方案：LTR + RL + Bandit
  结合离线和在线，更新的行业方案
```

本项目覆盖这三个阶段的核心算法和工程实现。

---

## Part A：LTR（Learning To Rank）基础

### A1. 三大范式对比

| 范式 | 输入 | 输出 | 损失函数 | 适用场景 |
|-----|------|------|---------|---------|
| **Pointwise** | 单个广告 | 相关度评分 | MSE / Cross-Entropy | 分类问题转排序 |
| **Pairwise** | 广告对 (ad_i, ad_j) | i > j 的概率 | Pairwise 排序损失 | 经典、历史悠久 |
| **Listwise** | 广告列表 | 最优排列 | NDCG / MAP 损失 | 直接优化排序指标 |

### A2. LambdaMART：工业标准

**为什么是标准**：
- ✅ 梯度增强树天然支持
- ✅ 推理速度快（<10ms）
- ✅ 容易解释和调试
- ✅ 已有成熟库（LightGBM Rank, XGBoost Rank）

**核心思想**：
```
【普通 GBDT】
梯度 = ∂Loss/∂pred_i

【LambdaMART】
梯度 = λᵢⱼ = |NDCG_ij| × σ / (1 + exp(σ × (pᵢ - pⱼ)))

λ_ij 的含义：
- 当排序错误时（i 应排在 j 前，但实际反了），λ 很大
- 当排序正确时，λ 很小
- 考虑了排序的"重要性"
```

**实战参数**：
```python
# LightGBM Rank 示例
params = {
    'task': 'rank',
    'metric': 'ndcg',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
}

# 关键：指定 group（每个 query 有多少个候选）
# group = [50, 50, 50, ...]  # 每个 query 50 个候选
```

### A3. 特征工程：排序特征的设计

```
【广告特征】
- 初始得分（精排模型输出）
- 广告质量分
- 话题 embedding
- 创意特征（图片、文案等）

【用户特征】
- 用户 ID embedding
- 人口属性（年龄、性别、地区）
- 历史兴趣（最近看过哪些话题）
- 活跃度（日活跃天数、月消费）

【上下文特征】
- 时间（小时、星期几）
- 设备类型（手机 vs PC）
- 位置
- 网络质量

【交互特征】（最关键）
- 用户与广告主的历史交互
- 用户与话题的历史交互
- 新鲜度（广告发布距离现在的时间）

【排序特征】（用于重排）
- 与已选择广告的相似度
- 位置偏差（当前是第几个位置）
- 频率特征（用户看过多少次这个话题）
```

---

## Part B：强化学习排序（RL Ranking）

### B1. MDP 定义

```
【状态】S_t
- 已展示的广告列表（序列编码）
- 用户交互历史（点击、停留时间）
- 当前候选池
- 用户上下文

【动作】A_t
- 从候选中选择下一个广告展示

【奖励】R_t
- 用户点击：+1.0
- 用户停留 > 3 秒：+0.5
- 用户看到但没点击：+0.1（至少产生了展示）
- 用户立即跳过：0

【目标】
maximize E[Σ γ^t × R_t]  // 折扣累计奖励
γ = 0.99（衰减因子，后续奖励权重较低）
```

### B2. Policy Gradient（A3C）

**基本思路**：用神经网络学习策略（概率分布）

```python
# 伪代码
def policy_network(state):
    embedding = encode_state(state)
    # 为每个候选广告输出选择概率
    logits = mlp(embedding)  
    probs = softmax(logits)  # [0.3, 0.2, 0.15, ..., 0.05]
    return probs

def train_step():
    # 采样轨迹（根据当前策略收集数据）
    trajectory = sample_from_policy(policy_network)
    
    # 计算优势函数（相对价值）
    advantage = compute_advantage(trajectory)
    
    # 策略梯度更新
    loss = -log_prob(action) * advantage  # REINFORCE
    optimizer.step(loss)
```

**关键挑战**：
1. **样本效率**：需要大量线上交互才能学会
2. **稳定性**：策略梯度的方差很大
3. **离线数据**：如何用历史数据初始化？

### B3. Contextual Bandit（简化 RL）

**简化假设**：不考虑序列效应，只预测"下一个"最好的广告

```
状态 S = 当前用户上下文（不包含已展示历史）
动作 A = 选择一个广告
奖励 R = 用户是否点击（0 或 1）

算法：Thompson Sampling
1. 对每个广告维护"转化率分布"（例如 Beta 分布）
2. 每次从分布采样，选择 sampled CTR 最高的广告
3. 看到用户反馈后更新分布

优点：
- 简单，易于实现
- 自动平衡探索和利用
- 对新广告友好（不确定性高，更容易被选中）
```

### B4. 离线 RL：从历史数据学习

**问题**：如何从历史点击数据中学习排序策略？

```
【行为克隆】
BC(Behavior Cloning)
- 从历史数据中，看用户点击了哪个广告
- 直接学习一个监督模型：P(click | state)
- 缺点：只学到历史政策，无法改进

【逆强化学习】
IRL(Inverse Reinforcement Learning)
- 从用户的选择反推"用户的奖励函数"
- 然后基于推断的奖励学习新策略
- 复杂，但可以改进历史政策

【离线策略评估】
OPE(Offline Policy Evaluation)
- 不实际运行新策略，而是用历史数据估计其效果
- 关键技术：Importance Weighting、Doubly Robust
```

---

## Part C：Online Learning & Exploration

### C1. Exploration vs Exploitation 权衡

**问题**：
```
已知：广告 A 的 CTR = 0.05
未知：广告 B 的 CTR 是多少（可能 0.03，也可能 0.07）

应该选哪个？
- Exploitation：选 A（贪心最优）
- Exploration：选 B（了解其潜力）
```

**方案**：
1. **ε-Greedy**：以 ε 概率随机探索，1-ε 概率贪心
2. **UCB（Upper Confidence Bound）**：加入不确定性
   - 分数 = 预估 CTR + uncertainty
   - 新广告 uncertainty 高，更容易被选中
3. **Thompson Sampling**：基于贝叶斯的探索

### C2. Regret Minimization

**定义**：
```
Regret = Σ (最优奖励 - 实际奖励)
目标：最小化累积 regret
```

**算法**：
- **Multiplicative Weights Update**：调整权重比例
- **Regret Matching**：基于过往 regret 调整策略
- **FTRL（Follow The Regularized Leader）**：加正则化的在线学习

---

## Part D：工业融合方案

### D1. LTR + Bandit 融合

```
【架构】
1. 离线 LTR：用历史数据训练排序模型
2. 在线 Bandit：覆盖 LTR 的"探索"部分
3. 反馈循环：用线上数据重训 LTR，周期 1 周

【流程】
用户请求
  ↓
候选广告池（来自召回和粗排）
  ↓
[LTR 排序] 获得初步排序
  ↓
[Bandit 探索] 
  - 以 80% 概率用 LTR 结果
  - 以 20% 概率随机探索
  ↓
展示给用户，收集反馈
  ↓
[反馈循环]
  - 积累数据
  - 每周重训 LTR
```

**优点**：
- LTR 保证基础质量（80%）
- Bandit 保证探索（20%）
- 低风险（大部分流量仍用 LTR）

### D2. RL + Policy Distillation

```
【问题】
直接用 RL 训练排序策略很不稳定，收敛慢

【解决方案】
1. 先用 LTR 训练一个教师模型
2. 用教师模型初始化 RL agent 的策略网络
3. 在此基础上用 RL 微调

【好处】
- 初始化点更好，收敛快
- 教师网络作为正则，防止过度探索
```

---

## Part E：评估与诊断

### E1. 排序质量指标

```
【NDCG（Normalized Discounted Cumulative Gain）】
衡量排序与理想排列的匹配度

NDCG@10 = (DCG@10) / (Ideal_DCG@10)
其中 DCG = Σ (2^rel_i - 1) / log2(i+1)

rel_i = 第 i 个广告的相关度（0-5 分）

示例：
用户点击了第 3 位广告
rel = [0, 0, 5, 0, 0, 0, 0, 0, 0, 0]
DCG = (2^0-1)/1 + ... + (2^5-1)/4 + ... 

NDCG 接近 1：排序完美
NDCG 接近 0：排序很差
```

### E2. 线上离线 Gap

**问题**：离线指标好，线上效果差

**原因**：
- 数据分布不同（离线是历史数据，线上是实时）
- 新广告冷启动问题
- 模型与真实 online 行为的mismatch

**诊断**：
- 按用户类型分层对比
- 按时间分段对比
- 检查新广告的表现
- 对比不同模型版本

---

## Part F：实战路线图

### F1. 第 1 阶段：启动 LTR（2-4 周）

- [ ] 定义排序标签（例如：点击=2 分，无点击=1 分）
- [ ] 特征工程（10-50 个特征）
- [ ] 训练 LambdaMART 基线
- [ ] 小流量 A/B 测试

### F2. 第 2 阶段：加入 Bandit 探索（4-8 周）

- [ ] 实现 Thompson Sampling 或 UCB
- [ ] 融合 LTR + Bandit
- [ ] 监控探索的质量
- [ ] 全量 A/B 测试

### F3. 第 3 阶段：尝试 RL（2-3 个月）

- [ ] 用 LTR 初始化 RL agent
- [ ] 离线模拟评估
- [ ] 小流量在线 RL 实验
- [ ] 调参与优化

---

## 下一步阅读

- `ads_ranking_evolution.md` — 完整演进路线
- `p1_reranking_fundamentals.md` — 重排算法
- `ctr-prediction-comprehensive-survey.md` — CTR 预估基础

---

**维护者**：Boyu | 2026-03-24
