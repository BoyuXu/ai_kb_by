# Reason4Rec: LLMs for Recommendation with Deliberative User Preference Alignment

**ArXiv:** 2502.02061 | **Date:** 2025-02 | **Code:** github.com/Peter-Fy/Reason4Rec

## 核心问题
当前 LLM 推荐对齐方法聚焦于直接生成用户反馈，缺乏**deliberation**（深思熟虑），在复杂推荐场景中表现不足。

## 核心贡献
**Deliberative Recommendation 任务**：将显式用户偏好推理作为额外对齐目标。

## 框架：Reason4Rec (R4R)

### 三大核心能力

1. **Preference Distillation（偏好蒸馏）**
   - 分析用户历史，识别方面级（aspect-level）用户偏好
   - 研究已有的言语化用户反馈

2. **Preference Matching（偏好匹配）**
   - 将蒸馏出的用户偏好与 item 特征匹配

3. **Feedback Prediction（反馈预测）**
   - 在生成的理由基础上预测用户反馈

## 实验结果
3 个真实数据集上：预测准确性和推理质量均显著优于基线。

## 面试考点
- Deliberative alignment vs 标准 RLHF 的区别？
- 为什么推理过程对推荐质量有帮助？
- 如何评估推荐推理的质量？

**Tags:** #rec-sys #llm #recommendation #alignment #deliberation #reasoning
