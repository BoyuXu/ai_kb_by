# Creative Generation with Large Language Models for Ad Recommendation
> 来源：arxiv/2402.xxxxx | 领域：ads | 学习日期：20260326

## 问题定义
广告创意（文案、标题、图片文字）的传统生产方式面临挑战：
- 人工创意生产速度慢、成本高，无法支持百万级广告主
- 一刀切创意：相同广告对所有用户展示相同文案，个性化不足
- A/B 测试周期长：人工创意验证需要数天
- LLM 幻觉风险：生成虚假产品信息（违规）

## 核心方法与创新点
**LLM-based Ad Creative Generation**：基于 LLM 的个性化广告创意生成。

**三阶段框架：**
```
Stage 1 - Creative Blueprint：
  Input: [商品信息, 目标受众画像, 营销诉求]
  LLM → 广告文案大纲（卖点、情绪词、CTA）

Stage 2 - Personalized Rendering：
  输入用户画像 → 选择对应创意风格
  年轻用户 → 活泼文案；专业用户 → 技术参数
  P(creative | user_profile, product) = LLM(prompt_template)

Stage 3 - Quality Filtering：
  多维度过滤：合规检查 + 事实核查 + CTR 预估
  quality_score = α·compliance + β·factual + γ·predicted_ctr
```

**提示工程（Prompt Engineering）：**
```python
prompt = f"""
你是广告文案专家。基于以下信息生成广告标题：
商品：{product_name}，{product_desc}
用户画像：{age}岁，{gender}，兴趣：{interests}
广告目标：{campaign_goal}
要求：30字以内，包含卖点，不得虚假宣传
格式：仅输出广告标题，无其他内容
"""
```

**CTR 导向的创意选择：**
```
候选创意集 C = {c_1, c_2, ..., c_N}  # LLM 生成多个候选
selected = argmax_c CTR_model(user, ad, c)  # CTR 模型选最优
```

## 实验结论
- 某电商广告平台上线实验：
  - 广告 CTR +7.3%（个性化创意 vs 通用创意）
  - 创意生产效率：从 3 天/套 → 实时生成
  - 合规通过率：98.2%（加入事实核查后）
- 人工评估：AI 创意 vs 人工创意，用户偏好 AI 创意占 54.1%

## 工程落地要点
1. **事实核查**：将商品 SKU 信息作为 RAG 检索基础，约束 LLM 不生成产品库外内容
2. **合规过滤**：违禁词词典 + 分类器双层过滤（极限词、虚假宣称）
3. **创意缓存**：相同商品+受众组合的创意缓存复用（TTL 24h）
4. **CTR 反馈闭环**：实际 CTR 数据定期回流，微调创意生成偏好
5. **多模态扩展**：文案 → 图片布局设计（Stable Diffusion 生成创意图）

## 常见考点
**Q1: LLM 生成广告创意的最大风险是什么？如何控制？**
A: 最大风险是虚假宣传/幻觉：LLM 可能生成产品不具备的特性（如"电池续航72小时"但实际只有10小时）。控制方法：①RAG：只允许 LLM 使用产品资料库中的事实 ②事实核查分类器 ③人工审核高风险类目。

**Q2: 如何实现广告创意的个性化？**
A: ①用户画像注入 Prompt：年龄、性别、兴趣标签 ②创意风格分类：预定义若干风格（专业/活泼/情感/实用），根据用户画像选风格 ③Fine-tune：用高 CTR 的（用户-创意）对微调 LLM，学习个性化偏好。

**Q3: 如何评估 LLM 生成创意的质量？**
A: 自动评估：BLEU/ROUGE（与人工金标准对比，但不准）、CTR 预估模型打分、合规分类器分数；人工评估：可读性、相关性、吸引力、合规性；最终验证：线上 AB 实验 CTR。

**Q4: 创意生成与 CTR 模型如何协同优化？**
A: ①CTR 模型作为奖励：LLM 生成多候选，CTR 模型选最高分 ②RLHF：用 CTR 反馈微调 LLM，使生成更倾向高 CTR 风格 ③联合训练：创意 encoder + CTR 模型端到端训练（难度大但效果好）。

**Q5: 广告创意生成的 A/B 实验如何设计？**
A: 对照组：现有创意（人工/旧系统）；实验组：LLM 创意；分层实验：按广告主类型、投放目标、受众分层；主要指标：CTR、CVR、CPM、投诉率（合规）；实验期：至少 2 周（消除新奇效应）。
