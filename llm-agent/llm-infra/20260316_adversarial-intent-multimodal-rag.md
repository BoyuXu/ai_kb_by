# Adversarial Intent as Latent Variable: Stateful Trust Inference for Multimodal Agentic RAG

> 来源：arxiv | 日期：20260316 | 领域：llm-infra

## 问题定义

RAG 系统假设检索结果是可信的，但在对抗环境中（如社交媒体、不可信数据源）：
1. **虚假信息注入**：恶意用户或数据质量差导致检索到错误/误导信息。
2. **多模态混淆**：文本和图像可能不匹配（如使用过时图片配新闻文本）。
3. **隐藏意图**：对手试图通过特定 query 让 RAG 检索到某些信息并被 LLM 相信。

传统 RAG 无法应对这些对抗，本文提议引入 **Trust Inference**。

## 核心方法与创新点

1. **对抗意图建模为隐变量**：
   将用户意图分解为两部分：
   - **Manifest Intent**：用户明确要求（"告诉我 X 的信息"）。
   - **Adversarial Intent**：隐藏的恶意意图（"试图让我相信错误信息"），建模为隐变量 z_adv。

2. **多模态可信度评估**：
   ```
   trust_score(doc, img, query) = 
       semantic_alignment(text, img) × 
       domain_confidence(doc_source) × 
       temporal_freshness(doc_date) ×
       fact_consistency(doc vs knowledge_graph)
   ```
   - 语义对齐：文本和图像 embedding 一致性。
   - 域信誉：根据来源（NYT vs random blog）设定基础信任。
   - 时间鲜度：旧文档对某类问题（新闻、科技）不可信。
   - 事实一致性：与知识图谱/事实数据库交叉验证。

3. **有状态 Trust 推理（Stateful Trust Inference）**：
   系统记忆用户与检索结果的交互历史，更新信任评估：
   ```
   # Session state
   user_feedback = [doc_A: "correct", doc_B: "misleading", ...]
   
   # 更新信任
   source_reliability[source_of_doc_A] += 0.1
   source_reliability[source_of_doc_B] -= 0.2
   
   # 后续查询时应用
   trust_score_new = trust_base × source_reliability
   ```

4. **对抗意图检测**：
   识别以下模式：
   - 用户 query 试图强化某个虚假信念（重复问相同问题）。
   - 多模态矛盾（文字说"天晴"但图片显示"下雨"）。
   - 来源信誉度突降（平时可信的源突然发布异常内容）。

## 实验结论

- 对抗性虚假信息注入实验：未加防护的 RAG 信了 47% 虚假信息；加入 Trust Inference 降至 8%。
- 多模态混淆检测：准确率 92%（识别文图不匹配）。
- 有状态学习：一个用户交互 10 次后，系统对其偏好源的信任度学习准确率 > 88%。

## 工程落地要点

- **信任评分维护**：需要维护每个数据源的可信度评分，定期 audit（人工审核）。
- **知识图谱集成**：事实检验需要接入 Wikidata、Snopes、PolitiFact 等，实时查询成本高，建议离线预处理。
- **隐私考量**：记住用户反馈（有状态）涉及隐私，需要用户同意并提供遗忘选项。
- **多语言挑战**：虚假信息在低资源语言上难以检测（没有充足的事实库），需要翻译后检验。

## 面试考点

- Q: RAG 系统为什么容易被对抗？
  A: RAG 将检索结果视为真实，直接拼接进 prompt，大模型往往倾向相信 context 中的内容（context bias）。恶意用户可以通过污染数据源或精心设计 query 让 RAG 检索到虚假信息，而 LLM 会信以为真。

- Q: 事实检验（Fact Checking）和信任评估的区别？
  A: 事实检验是逐句检查内容是否准确（二值判断）。信任评估是对数据源和内容的综合评分（连续分数），考虑多个维度（来源可靠性、时间鲜度、用户反馈历史）。信任评估成本低但不精确，事实检验精确但成本高（需要知识库或人工）。

- Q: 用户反馈能否被滥用？
  A: 可能。恶意用户可以故意给出错误反馈，污染有状态学习。防护方法：(1) 权重反馈（信息用户的反馈权重高）；(2) 异常检测（离群反馈被标记）；(3) 多源确认（单一用户反馈不够信服）。
