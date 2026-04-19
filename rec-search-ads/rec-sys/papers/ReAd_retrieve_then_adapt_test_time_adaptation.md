# Retrieve-then-Adapt: Retrieval-Augmented Test-Time Adaptation for Sequential Recommendation
> 来源：arXiv:2604.05379 | 领域：rec-sys | 学习日期：20260419

## 问题定义
序列推荐模型在训练数据上学习用户偏好模式，但部署后面临实时偏好漂移（distributional divergence），导致预测准确率下降。现有方法（测试时训练TTT、测试时增强TTA、检索增强微调）存在计算开销大、随机增强策略不稳定、需复杂两阶段训练等问题。

## 核心方法与创新点
1. **Retrieve-then-Adapt (ReAd) 框架**：两阶段——先检索相关用户偏好信号，再动态适配已部署模型
2. **有效增强（Effective Augmentation）**：从用户历史和相似用户中检索高质量偏好信号，替代随机增强
3. **高效适配（Efficient Adaptation）**：测试时仅更新少量参数（adapter层），避免全模型微调的开销
4. **实时偏好捕捉**：通过检索最新交互数据，动态感知用户短期兴趣变化

## 实验结论
- 在多个benchmark上显著超越TTT4Rec等基线
- 推理时间开销可控，适合在线部署
- 检索质量是关键——高质量偏好信号 > 随机增强

## 工程落地要点
- 检索索引需实时更新（近线更新 + 在线补充）
- Adapter参数量需控制，避免测试时适配延迟过高
- 适合偏好漂移明显的场景（如新闻推荐、短视频）

## 面试考点
- Q: Test-Time Adaptation 在推荐中的挑战？
  - A: ①实时性要求高（ms级）；②无标签（只有用户行为信号）；③需平衡适配程度与灾难性遗忘
