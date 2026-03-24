# 抖音 RL 混排探索：自适应多样性策略

> 深度案例研究 | 强化学习与在线学习 | 字数：450+ 行

---

## 前言

抖音日活超 7 亿（2023），混排系统处理**超 100 万 QPS**。在这个规模下：
- 传统 A/B 测试反馈周期太长（2 周）
- 用户兴趣变化很快（热点话题周期短）
- 需要能**实时自适应**的混排策略

强化学习（RL）的优势：
1. 能处理**长期目标**（不只是点击，还有留存）
2. 能**在线学习**（用户来一个，学一个）
3. 能**自动平衡**多个目标（无需手调权重）

---

## Part A：为什么需要强化学习

### A.1 有监督学习的局限

**传统方法**：
```
数据：用户看过什么，点击了什么
标签：点击 = 1，未点击 = 0
模型：学习预测"点击概率"
排序：选择最可能被点击的 K 个

问题：
1. 点击信号很弱（大多数视频不会被点击）
2. 忽视了"链式反应"（第 t 个视频影响第 t+1 个视频的点击概率）
3. 反馈延迟（点击是即时的，但留存/转化是延迟的）
4. 无法处理"探索"（推荐新视频需要先投放，但收益未知）
```

**强化学习的优势**：
```
显式建模：
- 状态 S：用户当前状态
- 动作 A：选择哪个视频推荐
- 奖励 R：用户反应（点击、停留、分享）
- 目标：最大化累积奖励（长期价值）

能处理：
✓ 长期反馈（7 日留存比点击更重要）
✓ 链式效应（考虑推荐序列的整体效果）
✓ 多目标（CTR、点赞、分享、转发的权衡）
✓ 探索（主动测试新视频，学习潜力）
```

### A.2 混排问题的 MDP 建模

```
【状态 S】

用户特征向量（Embedding）：
- 年龄、地区、性别
- 最近 7 天的兴趣（话题分布、创作者偏好）
- 观看时长、点赞率
→ 用户 embedding（128 维）

已推荐视频状态：
- 前 t-1 个视频的特征
- 话题分布、创作者分布、视频类型分布
→ 上下文 embedding（128 维）

环境状态：
- 时间（小时、星期几）
- 平台热点话题
- 网络状况
→ 环境 embedding（32 维）

总状态向量：288 维

【动作 A】

候选集合：精排后的 Top-50 视频
动作：选择第 t 个位置推荐的视频（50 个选择）
→ 离散动作空间，大小 50

【奖励 R】

设计关键：奖励要体现商业目标

多目标加权：
R(t) = w₁ × click(t) + w₂ × watch_time(t) + w₃ × like(t) + 
       w₄ × share(t) + w₅ × conversion(t) + w₆ × retention(t)

其中：
- click(t) ∈ {0, 1}：是否点击
- watch_time(t) ∈ [0, 60]：观看时长（秒）
- like(t) ∈ {0, 1}：是否点赞
- share(t) ∈ {0, 1}：是否分享
- conversion(t) ∈ {0, 1}：是否转化（点击商品）
- retention(t) ∈ [0, 1]：7 日回访概率

权重示例：
w₁ = 1    （点击基础奖励）
w₂ = 0.05 × watch_time（观看时长奖励）
w₃ = 5    （点赞权重高，说明内容好）
w₄ = 10   （分享权重最高，传播价值大）
w₅ = 20   （转化权重最高，直接收益）
w₆ = 50 × retention（留存权重最高，长期价值）

【转移概率 P】

简化假设：用户反应是确定的（给定状态和动作）
P(s' | s, a) = 1 如果 a 导致推荐视频 v
用户观看 v → 得到反应（点击/点赞/...)
→ 转移到 s'

实际中：用历史数据估计转移概率
```

### A.3 MDP 的求解

```
目标：找到最优策略 π*
π*(a|s) = 推荐这个视频的概率（给定用户状态 s）

方法 1：价值迭代（Value Iteration）
V(s) = max_a { R(s, a) + γ × E[V(s')] }
迭代求解，直到收敛

问题：状态空间太大（288 维 embedding），无法枚举

方法 2：策略梯度（Policy Gradient）
用神经网络逼近最优策略：
π_θ(a|s) = softmax(NN_θ(s))
通过梯度上升优化 θ

方法 3：Actor-Critic（最常用）
- Actor：学习策略 π_θ(a|s)
- Critic：学习价值 V_φ(s)
- 用 Critic 的预测作为 Actor 的"反馈"
```

---

## Part B：Actor-Critic 架构实现

### B.1 网络架构

```
【输入层】288 维状态向量
├─ 用户 embedding（128）
├─ 上下文 embedding（128）
└─ 环境 embedding（32）

【共享底层】Transformer 编码器
输入：288 维状态向量
处理：自注意力（学习状态各部分的关联）
输出：256 维编码向量

【Actor 分支】策略网络
输入：256 维
隐层：[256] → ReLU
输出：softmax([50])  （50 个候选的概率）
π_θ(a|s) = Pr{选择视频 a}

【Critic 分支】价值网络
输入：256 维
隐层：[256] → ReLU
输出：[1] 标量  （当前状态的价值）
V_φ(s) = 累积奖励期望

【优势函数】
Advantage(s, a) = R(s, a) + γ × V_φ(s') - V_φ(s)
                = 当前动作的"超额收益"
```

### B.2 训练算法

```python
class MixingRLAgent:
    def __init__(self, state_dim=288, action_dim=50):
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        self.actor_opt = Adam(lr=1e-4)
        self.critic_opt = Adam(lr=1e-3)
        self.gamma = 0.99  # 折扣因子
    
    def select_action(self, state, explore=True):
        """
        给定用户状态，选择一个视频推荐
        
        explore=True：从分布中采样（探索）
        explore=False：贪心选择（利用）
        """
        # Actor 输出每个候选视频的概率
        action_probs = self.actor(state)  # [50,]
        
        if explore:
            # 从概率分布中采样（探索）
            action = np.random.choice(50, p=action_probs.numpy())
        else:
            # 选择概率最高的视频（利用）
            action = np.argmax(action_probs.numpy())
        
        return action, action_probs
    
    def train_step(self, trajectory):
        """
        从一个用户的完整 Feed 浏览轨迹学习
        
        trajectory = [
            {'state': s0, 'action': a0, 'reward': r0, 'next_state': s1},
            {'state': s1, 'action': a1, 'reward': r1, 'next_state': s2},
            ...
        ]
        """
        for t in range(len(trajectory)):
            state = trajectory[t]['state']
            action = trajectory[t]['action']
            reward = trajectory[t]['reward']
            next_state = trajectory[t]['next_state']
            done = (t == len(trajectory) - 1)
            
            # ============ Critic 更新 ============
            with tf.GradientTape() as tape:
                # 当前状态的价值估计
                v_current = self.critic(state)
                
                # 目标价值（Bellman 目标）
                v_next = self.critic(next_state) if not done else 0
                target_v = reward + self.gamma * v_next
                
                # 损失：均方误差
                critic_loss = tf.square(v_current - target_v)
            
            # 梯度下降更新 Critic
            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_opt.apply_gradients(zip(critic_grad, self.critic.trainable_variables))
            
            # ============ Actor 更新 ============
            with tf.GradientTape() as tape:
                # Actor 输出的动作概率
                action_probs = self.actor(state)
                log_prob = tf.math.log(action_probs[action] + 1e-8)
                
                # 优势（用 Critic 计算）
                advantage = target_v - v_current
                
                # 策略梯度损失（REINFORCE with baseline）
                actor_loss = -log_prob * advantage
            
            # 梯度上升更新 Actor（注意：-loss，所以是上升）
            actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_opt.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
    
    def collect_trajectory(self, user_context, num_steps=20):
        """
        与用户互动，收集一个轨迹（20 个推荐）
        """
        trajectory = []
        state = encode_user_context(user_context)  # 编码为向量
        
        for step in range(num_steps):
            # 选择动作
            action, probs = self.select_action(state, explore=True)
            video = candidates[action]
            
            # 推荐给用户，获得奖励
            user_response = show_video(user_context, video)
            reward = compute_reward(user_response)
            
            # 更新状态
            next_state = update_state(state, video, user_response)
            
            # 保存轨迹
            trajectory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state
            })
            
            state = next_state
        
        return trajectory
```

### B.3 线上部署：Off-Policy 学习

**问题**：不能用线上用户不断试错训练模型（用户体验会变差）。

**解决**：从历史日志学习（Off-Policy）。

```python
class OffPolicyMixingAgent:
    """
    用历史用户行为日志（可能来自旧模型）训练新模型
    """
    
    def train_from_historical_data(self, historical_log, batch_size=1024):
        """
        historical_log: [
            {
                'user': user_id,
                'feed': [video_id_1, video_id_2, ...],
                'rewards': [r1, r2, ...],
                'old_policy_probs': [p1, p2, ...],  # 旧模型的动作概率
            },
            ...
        ]
        """
        
        for batch in iterate_batches(historical_log, batch_size):
            for example in batch:
                user_embedding = encode_user(example['user'])
                
                for t, (video_id, reward, old_prob) in enumerate(
                    zip(example['feed'], example['rewards'], example['old_policy_probs'])
                ):
                    state = compute_state(user_embedding, t)
                    action = video_id
                    
                    # ======== Off-Policy 修正 ========
                    # 新模型可能与旧模型不同，需要用"重要性采样"修正
                    new_probs = self.actor(state)
                    new_prob = new_probs[action]
                    
                    # 重要性权重：new_prob / old_prob
                    # 如果比值太大，说明新模型在"改变"历史发生的动作
                    # 需要裁剪（clipping）避免方差爆炸
                    importance_weight = tf.clip_by_value(
                        new_prob / (old_prob + 1e-8),
                        0.5, 2.0  # 裁剪到 [0.5, 2.0]
                    )
                    
                    # 使用加权奖励
                    weighted_reward = reward * importance_weight
                    
                    # 继续 Actor-Critic 训练
                    self.train_step_with_weight(
                        state, action, weighted_reward, importance_weight
                    )
```

---

## Part C：多样性的自适应控制

### C.1 多样性作为动作空间的约束

**问题**：RL 可能学到"总是推荐高 eCPM 的视频"（经济收益最大，但单调）。

**解决**：在动作空间中加入多样性约束。

```python
def create_valid_action_set(candidates, recent_videos, diversity_config):
    """
    根据多样性配置，过滤出"有效"的候选
    
    只有满足多样性约束的候选才能被推荐
    """
    valid_actions = []
    
    for idx, video in enumerate(candidates):
        valid = True
        
        # 约束 1：话题多样性
        recent_topics = [v['topic'] for v in recent_videos[-3:]]
        if video['topic'] in recent_topics:
            topic_count = recent_topics.count(video['topic'])
            if topic_count >= diversity_config['max_same_topic']:
                valid = False  # 同话题超过限制
        
        # 约束 2：创作者多样性
        recent_creators = [v['creator_id'] for v in recent_videos[-5:]]
        if video['creator_id'] in recent_creators:
            if recent_creators.count(video['creator_id']) >= 2:
                valid = False  # 同创作者超过 2 个
        
        # 约束 3：内容类型多样性
        recent_types = [v['type'] for v in recent_videos[-3:]]
        if all(t == video['type'] for t in recent_types):
            valid = False  # 全是同类型内容
        
        if valid:
            valid_actions.append(idx)
    
    return valid_actions

def select_action_with_constraints(self, state, recent_videos):
    """
    在满足多样性约束的动作空间中选择
    """
    # 获得有效动作集合
    valid_actions = create_valid_action_set(candidates, recent_videos, self.diversity_config)
    
    if not valid_actions:
        # 降级：如果没有有效动作，放松约束
        valid_actions = list(range(len(candidates)))
    
    # 获得所有动作的概率
    all_probs = self.actor(state)
    
    # 将无效动作的概率设为 0
    valid_probs = tf.scatter_nd(
        [[i] for i in valid_actions],
        tf.gather_nd(all_probs, [[i] for i in valid_actions]),
        [len(candidates)]
    )
    
    # 归一化
    valid_probs = valid_probs / tf.reduce_sum(valid_probs)
    
    # 从有效分布中采样
    action = np.random.choice(len(candidates), p=valid_probs.numpy())
    
    return action
```

### C.2 动态权重调整

**灵活性**：不同用户群体的多样性偏好不同。

```python
class AdaptiveWeightingAgent:
    """
    根据用户特征动态调整多样性权重
    """
    
    def compute_diversity_weight(self, user_context):
        """
        新用户 → 高多样性权重（保护体验）
        活跃用户 → 中多样性权重（平衡）
        留存风险用户 → 低多样性权重（保证质量）
        """
        
        # 用户活跃度评分
        activity_score = (
            0.3 * user_context['dau_ratio'] +  # DAU 频率
            0.3 * user_context['avg_watch_time'] +  # 平均观看时长
            0.4 * user_context['like_ratio']  # 点赞率
        )
        
        # 用户新旧程度
        user_age = (now - user_context['create_time']).days
        if user_age < 7:
            is_new = 1.0
        elif user_age > 365:
            is_new = 0.0
        else:
            is_new = 1.0 - (user_age / 365)
        
        # 动态权重
        # 新用户：w_diversity = 0.4（保护体验）
        # 老活跃用户：w_diversity = 0.2（他们喜欢高质量）
        # 老不活跃用户：w_diversity = 0.35（需要激活）
        
        w_diversity = (
            0.4 * is_new +
            0.2 * (1 - is_new) * activity_score +
            0.35 * (1 - is_new) * (1 - activity_score)
        )
        
        return w_diversity
    
    def compute_reward(self, user_context, user_response):
        """
        根据用户特征调整奖励函数
        """
        w_diversity = self.compute_diversity_weight(user_context)
        w_quality = 1.0 - w_diversity
        
        # 多样性奖励（代理指标：停留时长）
        diversity_reward = user_response['watch_time'] / 30.0  # 30 秒为满分
        
        # 质量奖励（点击 + 点赞 + 分享）
        quality_reward = (
            user_response['click'] * 1.0 +
            user_response['like'] * 2.0 +
            user_response['share'] * 5.0
        )
        
        # 综合奖励
        total_reward = (
            w_diversity * diversity_reward +
            w_quality * quality_reward
        )
        
        return total_reward
```

---

## Part D：效果评估与迭代

### D.1 关键指标

```
【离线指标】（用历史日志评估）
- Normalized NDCG（混排排序质量）
- Diversity Score（话题/创作者多样性）
- Prediction Error（模型预测准确度）

【在线指标】（A/B 测试）
- CTR：广告/视频点击率
- 停留时长：用户在 App 中停留多久
- 完播率：用户是否看完整个视频
- DAU：日活跃用户
- 7 日留存：用户是否在 7 天后回访
- 点赞率、分享率、转化率

【RL 特有指标】
- Exploration Rate：探索新视频的比例
- Regret：相比最优策略的损失
- Convergence Speed：学习速度（收敛到最优需要多少样本）
```

### D.2 实战案例（推测）

```
【对照组】DNN LTR 混排（无 RL）
【实验组】RL 混排（Actor-Critic）
【运行时间】4 周（RL 需要更长的测试时间）

【结果】

指标                 │ 对照    │ 实验    │ 变化
─────────────────────┼─────────┼─────────┼──────
视频点击率           │ 4.2%    │ 4.1%    │ -2.4%
视频完播率           │ 68%     │ 71%     │ +4.4%
用户停留时长         │ 12m30s  │ 13m50s  │ +10.4%
视频多样性分数       │ 0.72    │ 0.81    │ +12.5%
创作者覆盖率         │ 0.58    │ 0.67    │ +15.5%
DAU                  │ 600M    │ 612M    │ +2.0%
7 日留存             │ 52%     │ 54.3%   │ +4.4%
广告 eCPM（千次）   │ ¥4.5    │ ¥4.35   │ -3.3%
新创作者曝光提升     │ --      │ +45%    │ +45%

【整体评价】
✓ 留存 +4.4%（核心指标）
✓ 完播率 +4.4%（内容质量指标）
✓ 多样性 +12.5%（系统目标）
✓ DAU +2%（商业影响）
✗ 短期 eCPM -3.3%（因为探索了低 eCPM 视频）
✓ 新创作者机会 +45%（生态价值）
→ 全量上线，长期价值 > 短期收益

【6 个月后评估】
- 新创作者粘性提升（留存 +8%）
- eCPM 反弹到 ¥4.6（+ 2.2%）
- DAU 进一步增长到 650M（+8.3%）
→ RL 混排成为核心策略
```

---

## Part E：工程难点与优化

### E.1 在线学习的稳定性

```
【问题】
每个用户来一个，模型学一个。
如果用户数据有噪声，模型可能被"带偏"。

【解决方案】

1. 经验回放（Experience Replay）
   - 不是立即用一个用户的轨迹训练
   - 而是积累 1000+ 个用户轨迹，然后批量训练
   - 打破时间相关性

2. 目标网络（Target Network）
   - 维护两个网络：主网络 + 目标网络
   - 目标网络延迟同步（每 1000 步同步一次）
   - 避免训练目标不稳定

3. 梯度裁剪（Gradient Clipping）
   - 限制梯度大小，避免梯度爆炸
   - 梯度范数 > 1.0 时裁剪到 1.0

实现示例：
```

```python
class StableRLAgent:
    def __init__(self, state_dim, action_dim, batch_size=256):
        self.actor = ActorNetwork(state_dim, action_dim)
        self.target_actor = ActorNetwork(state_dim, action_dim)  # 目标网络
        
        self.replay_buffer = []  # 经验回放缓冲
        self.batch_size = batch_size
        self.update_freq = 1000  # 每 1000 步同步一次目标网络
        self.step_count = 0
    
    def add_trajectory(self, trajectory):
        """添加轨迹到回放缓冲"""
        self.replay_buffer.extend(trajectory)
        
        # 只保持最近 100K 条经验（内存限制）
        if len(self.replay_buffer) > 100000:
            self.replay_buffer = self.replay_buffer[-100000:]
        
        # 当缓冲足够大时，开始训练
        if len(self.replay_buffer) >= self.batch_size:
            self.train()
    
    def train(self):
        """从回放缓冲批量训练"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 随机采样一个 batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        for exp in batch:
            state = exp['state']
            action = exp['action']
            reward = exp['reward']
            next_state = exp['next_state']
            
            # 用目标网络计算 TD 目标
            target_value = reward + 0.99 * self.target_critic(next_state)
            
            # 训练 Critic 和 Actor（带梯度裁剪）
            with tf.GradientTape() as tape:
                loss = self.compute_loss(state, action, reward, target_value)
            
            grads = tape.gradient(loss, self.actor.trainable_variables)
            
            # 梯度裁剪
            clipped_grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
            
            self.actor_opt.apply_gradients(zip(clipped_grads, self.actor.trainable_variables))
        
        self.step_count += 1
        
        # 定期同步目标网络
        if self.step_count % self.update_freq == 0:
            self.target_actor.set_weights(self.actor.get_weights())
```

### E.2 延迟反馈处理

```
【问题】
- 点击反馈：立即（用户点击时知道）
- 点赞反馈：延迟 5-10 秒
- 7 日留存反馈：延迟 7 天

不能等 7 天才更新模型，用户已经离开了。

【解决方案】

1. 分层反馈
   - 立即用"点击"反馈更新模型
   - 延迟的反馈（点赞、分享）定期批量更新

2. 预测反馈
   - 用快速指标（观看时长）预测延迟指标（点赞）
   - 基于预测的反馈更新模型

3. 多目标学习
   - 同时预测多个指标（CTR + 点赞率 + 7 日留存）
   - 用可得到的反馈训练各个目标

实现示例：
```

```python
def process_delayed_feedback(user_id, video_id, feedback_type):
    """
    处理延迟反馈
    """
    
    if feedback_type == 'click':
        # 立即反馈，直接用于训练
        immediate_update(user_id, video_id, reward=1.0)
    
    elif feedback_type == 'like':
        # 延迟反馈，加入到"待处理"队列
        delayed_queue.append({
            'user_id': user_id,
            'video_id': video_id,
            'reward': 2.0,
            'timestamp': now
        })
    
    elif feedback_type == 'retention_7d':
        # 极端延迟，用历史数据建立预测模型
        # 根据"7 日内的其他信号"预测"7 日留存"
        predicted_retention = predict_7d_retention(user_id, video_id)
        
        # 用预测反馈更新（权重降低）
        delayed_update(user_id, video_id, reward=predicted_retention, weight=0.5)

# 定期批处理延迟反馈
def process_delayed_batch():
    """每小时处理一次延迟反馈"""
    
    for exp in delayed_queue:
        if time_since(exp['timestamp']) > delay_threshold:
            # 足够的时间已经过了，反馈应该准确
            user_id = exp['user_id']
            actual_feedback = get_actual_feedback(user_id, exp['video_id'])
            
            if actual_feedback:
                # 用实际反馈（而不是预测）更新
                batch_update(exp['user_id'], exp['video_id'], actual_feedback)
            
            delayed_queue.remove(exp)
```

---

## 总结

**RL 混排的核心价值**：
1. **自动学习**：无需手调权重，自适应不同用户群体
2. **长期目标优化**：显式优化留存/DAU，而非短期点击
3. **在线学习**：用户来一个，学一个，快速适应趋势
4. **多样性自动平衡**：通过奖励函数自动权衡质量和多样性

**工程挑战**：
- 模型稳定性：经验回放、目标网络
- 反馈延迟：分层反馈、预测反馈
- 线上可靠性：A/B 测试周期长（4+ 周）

**适用场景**：
✓ 大规模平台（日活 1 亿+）
✓ 内容更新快速（热点话题频繁变化）
✓ 多目标优化需求强（CTR + 留存 + 转化）
✓ 工程资源充足（需要 RL 专家 + 基础设施）

---

**维护者**：Boyu | 2026-03-24 | 字数：480+ 行
