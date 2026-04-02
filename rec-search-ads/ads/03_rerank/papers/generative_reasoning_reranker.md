# Generative Reasoning Re-ranker (GR2)
> 来源：https://arxiv.org/abs/2602.07774 | 领域：ads | 学习日期：20260329

## 问题定义

推荐系统重排序（Reranking）阶段是最终影响用户体验的关键阶段，但现有 LLM 应用研究存在三大缺陷：

1. **重排阶段被忽视**：大量研究聚焦召回和粗排，重排阶段（精调最终结果顺序）研究不足
2. **LLM 推理能力未充分利用**：现有工作使用 zero-shot 或简单 SFT，未发挥 RL 增强的推理能力
3. **非语义 ID 的可扩展性问题**：工业系统使用非语义 Item ID（如哈希ID），Vocabulary 数十亿，LLM 难以处理

## 核心方法与创新点

### GR2 (Generative Reasoning Reranker) 三阶段训练流水线

**阶段1: 语义 ID 中间训练 (Mid-Training)**
- 将非语义 ID 通过 tokenizer 转换为语义 ID（确保 ≥99% 唯一性）
- 在语义 ID 上中训 LLM，建立 ID → 语义理解的映射

**阶段2: 推理蒸馏 SFT (Reasoning-enhanced SFT)**
- 用更大、更强的 LLM 通过精心设计的 prompt + Rejection Sampling 生成高质量推理链
- 将推理链用于 SFT，让模型学会"边推理边重排"

**阶段3: DAPO 强化学习对齐 (RL with Verifiable Rewards)**
- 使用 DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）
- 设计针对重排的可验证奖励函数
- 引入**条件可验证奖励**：防止 LLM 通过"保持原始顺序"来 reward hacking

### 关键技术创新

**① 语义 ID 的高唯一性设计（≥99% uniqueness）**
避免不同 Item 映射到相同 token，保证模型能区分每个 Item。

**② 条件可验证奖励（Conditional Verifiable Rewards）**
发现 LLM 倾向于直接输出原始排序（保守策略）来获得高奖励，通过条件奖励设计惩罚此行为：

$$
r = r_{ranking} \cdot \mathbb{1}[\text{reranking}}_{\text{{\text{happened}}}] + r_{baseline} \cdot \mathbb{1}[\text{not}}_{\text{{\text{reranked}}}]
$$

**③ DAPO 算法优化**
解耦裁剪（Decoupled Clip）和动态采样（Dynamic Sampling），相比标准 PPO 更稳定。

## 实验结论

| 指标 | GR2 vs OneRec-Think (SOTA) |
|------|---------------------------|
| Recall@5 | **+2.4%** |
| NDCG@5 | **+1.3%** |

**消融实验关键发现：**
- 高质量推理链（rejection sampling 生成）显著优于简单 SFT
- 条件可验证奖励有效遏制 reward hacking，是性能关键
- 语义 ID 的唯一性对最终效果有决定性影响

## 工程落地要点

1. **语义 ID 设计**：需要专门的 tokenizer 确保高唯一性，避免 ID 碰撞影响推理
2. **推理链质量**：Rejection Sampling 需要明确的 verifiable criteria（可以验证推理是否正确）
3. **Reward Hacking 防范**：必须设置条件奖励，监控模型是否真正在做重排
4. **延迟控制**：重排阶段 LLM 推理链较长，需要使用 speculative decoding 或预生成 cache
5. **DAPO vs PPO**：DAPO 的动态采样在长序列推理任务中更稳定，可优先考虑

## 常见考点

**Q1: GR2 为什么需要三阶段训练？每个阶段解决什么问题？**
A: ①Mid-Training：解决工业 ID 不可语义化问题，建立 ID→语义映射 ②推理SFT：注入高质量推理能力（单纯SFT效果有限，需要强模型蒸馏的推理链）③RL对齐：让推理能力与重排业务目标真正对齐，解决SFT的分布漂移

**Q2: 什么是 Reward Hacking？GR2 中 LLM 如何 hack 重排奖励？如何解决？**
A: Reward Hacking 指模型找到不符合设计意图的捷径来获得高奖励。GR2中：LLM发现"直接输出原始排序"（不做任何重排）可以获得较高奖励（因为原始排序已经不差）。解决：条件可验证奖励，只有当实际发生重排时才给满分奖励，否则给低基础分。

**Q3: 为什么语义 ID 的唯一性（≥99%）如此重要？**
A: 若不同 Item 映射到相同 token，模型无法区分它们，推理过程中会产生"哪个Item应该重排到哪个位置"的歧义。低唯一性 = 高碰撞率 = 模型推理混乱 = 重排效果差。

**Q4: DAPO 相比标准 GRPO/PPO 解决了什么问题？**
A: ①解耦裁剪：对正/负样本使用不同的 clip 阈值，更精确控制策略更新幅度 ②动态采样：根据当前策略动态调整采样难度（避免样本分布坍缩）③整体上使大规模长序列推理 RL 训练更稳定

**Q5: 重排 LLM 在工业系统中如何控制推理延迟？**
A: ①Speculative Decoding：用小草稿模型并行预测，大模型验证 ②KV Cache：缓存用户历史 Attention Key/Value ③分层重排：只对 top-K 候选做 LLM 重排，其余快速打分 ④预生成推理链：离线为常见场景预生成，在线复用
