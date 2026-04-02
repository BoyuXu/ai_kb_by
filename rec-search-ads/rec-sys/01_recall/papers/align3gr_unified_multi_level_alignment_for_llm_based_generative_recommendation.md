# Align³GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation
> 来源：arxiv/2405.xxxxx | 领域：rec-sys | 学习日期：20260326

## 问题定义
基于 LLM 的生成式推荐系统存在多层次对齐问题：
- **语义对齐**：LLM 的语言语义 ≠ 推荐系统的协同语义
- **任务对齐**：语言建模目标 ≠ 推荐排序目标（准确率/多样性）
- **偏好对齐**：模型输出 ≠ 用户真实偏好（点击不等于满意）
- 三层对齐问题相互纠缠，难以联合优化

## 核心方法与创新点
**Align³GR（Triple Alignment for Generative Recommendation）**：三级对齐框架。

**Level 1 - 语义对齐（Semantic Alignment）：**
```python
# 将协同过滤信号注入 LLM 表示
L_semantic = InfoNCE(LLM_emb(item), CF_emb(item))
# 使 LLM item 表示与协同过滤 embedding 对齐
```

**Level 2 - 任务对齐（Task Alignment）：**
```python
# SFT：在推荐格式数据上微调
L_sft = CrossEntropy(LLM(history) → next_item)

# 偏好学习（DPO-style）：
L_dpo = -log σ(log π(y_w|x) - log π(y_l|x) - β(log π_ref(y_w|x) - log π_ref(y_l|x)))
# y_w: 用户点击的 item，y_l: 曝光未点击的 item
```

**Level 3 - 偏好对齐（Preference Alignment）：**
```python
# 结合显式反馈（评分）和隐式反馈（点击）
reward(item) = α·explicit_rating + β·click_prob + γ·dwell_time
L_rl = PPO(reward) with KL constraint
```

**统一训练流程：**
```
Phase 1: L_semantic 预热
Phase 2: L_sft + L_semantic 联合微调
Phase 3: L_dpo / PPO 偏好对齐
```

## 实验结论
- Amazon Review 数据集：
  - NDCG@10：+8.3%（vs LLM+SFT 无对齐）
  - HR@10：+6.7%
- 消融实验：三级对齐缺一不可，各贡献约 2-3% 提升
- 多样性指标（ILD）：+12%（对齐后减少了马太效应）

## 工程落地要点
1. **对齐顺序**：务必先 Semantic → Task → Preference 顺序，反序会导致退化
2. **DPO 数据构建**：正样本=点击，负样本=曝光未点击（需注意曝光偏差校正）
3. **KL 散度上限**：PPO 阶段 KL ≤ 0.2（防止推荐能力退化为聊天模型）
4. **显式/隐式反馈融合**：用时间衰减权重，近期行为权重更高
5. **增量对齐**：每日用新增用户反馈做增量 DPO，保持偏好新鲜度

## 常见考点
**Q1: 为什么 LLM 用于推荐时需要三级对齐？**
A: LLM 预训练在语言语料上，与推荐的协同语义不同（Level 1）；语言建模目标是 token 预测，推荐目标是排序准确性（Level 2）；用户偏好复杂，不能只用点击信号（Level 3）。三者缺一则模型在对应维度上表现差。

**Q2: DPO 相比 PPO 在推荐场景的优劣？**
A: DPO 优势：无需奖励模型，直接用偏好数据优化，训练稳定；劣势：只用二元偏好，无法利用连续奖励信号（如停留时长）。推荐场景奖励信号丰富，PPO 理论上更强，但实践中 DPO 稳定性更好。

**Q3: Semantic Alignment 的 InfoNCE 损失如何构建？**
A: 正样本对：同一 item 的 LLM 语义表示和 CF 协同表示；负样本对：不同 item 的跨模态表示。通过对比学习让 LLM 的语义空间与 CF 的协同空间对齐，使 LLM 理解"用户喜欢相似品类"这类协同规律。

**Q4: 如何防止对齐过程中推荐能力退化？**
A: ①KL 约束：限制每轮更新与参考模型的 KL 距离 ②多任务平衡：L_total = L_rec + λ·L_language（保留语言能力）③Early Stopping：监控推荐验证集指标，不过度对齐。

**Q5: 偏好对齐中如何处理曝光偏差？**
A: 使用 IPS（逆倾向加权）校正：reward = actual_reward / propensity_score，其中 propensity_score 是曝光概率。避免模型过拟合热门 item（因为热门 item 曝光多、点击绝对值大）。
