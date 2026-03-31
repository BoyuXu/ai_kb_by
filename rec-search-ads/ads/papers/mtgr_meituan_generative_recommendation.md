# MTGR: Industrial-Scale Generative Recommendation Framework in Meituan

> 来源：https://arxiv.org/abs/2505.18654 | 领域：计算广告 | 学习日期：20260331

## 问题定义

美团多场景（外卖、到店、酒旅）统一建模挑战，每场景独立训练维护成本高且无法充分利用跨场景数据。

## 核心方法与创新点

1. **生成式统一框架**：推荐建模为序列生成任务，decoder-only架构服务多场景
2. **场景Prompt**：不同场景通过Prompt区分

$$\text{Output} = \text{Decoder}([\text{Scene}; \text{User}; \text{Context}; \text{Candidates}])$$

3. **多粒度Token化**：行为序列、候选item、上下文特征统一token化
4. **知识蒸馏**：从场景专有模型蒸馏到统一模型

## 实验结论

美团线上：外卖CTR持平（-0.1%），到店CVR提升2.1%，酒旅提升3.4%。模型数从12降到1。

## 工程落地要点

- 统一模型大幅降低运维成本
- 新场景接入只需设计Prompt
- 需大规模GPU资源训练
- 分阶段上线，先小场景验证

## 面试考点

1. **生成式推荐优势？** 统一框架、跨场景迁移、灵活输出
2. **多场景统一风险？** 大场景可能被小场景拖累
3. **美团为什么decoder-only？** 与LLM架构一致，可复用预训练权重
