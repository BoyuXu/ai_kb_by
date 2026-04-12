# LLM 基础设施论文/资源笔记 — 2026-04-13

## 1. smolagents: HuggingFace 轻量级 Agent 框架

**来源：** https://github.com/huggingface/smolagents
**领域：** LLM Agent 框架
**核心定位：** 极简 Agent 构建库，核心逻辑约 1000 行代码

**关键创新：**
- **Code-First Agent：** Agent 用代码（而非 JSON）执行工具调用，天然支持函数嵌套、循环、条件
- **安全沙箱：** 支持 Modal、Blaxel、E2B、Docker 沙箱执行
- **LLM 无关：** 支持任意推理提供商
- **多模态：** 支持视觉、视频、音频输入
- **工具灵活：** 支持 MCP server、LangChain 工具、Hub Space 工具

**面试考点：** Code Agent vs JSON Tool-calling Agent 的优劣、Agent 安全执行方案

---

## 2. RefAgent: 多智能体 LLM 自动软件重构框架

**来源：** arXiv:2511.03153 (2025)
**领域：** LLM + 软件工程
**核心贡献：** 多智能体框架实现端到端软件重构

**关键结果：**
- 单元测试通过率中位数 90%
- 代码坏味道减少中位数 52.5%
- 关键质量属性（如可复用性）提升中位数 8.6%
- 相比单智能体：单元测试通过率提升 64.7%，编译成功率提升 40.1%
- 与开发者重构和搜索工具的 F1 分数达 79.15% 和 72.7%

**面试考点：** 多智能体协作架构设计、LLM 在代码质量改进中的应用

---

## 3. LLM 强化学习综述（2025）

**来源：** 多篇综述论文
**领域：** LLM 对齐 / 强化学习

**核心技术演进：**
- **RLHF + PPO：** 仍是 LLM 对齐的基石，PPO 是事实标准算法
- **DPO（直接偏好优化）：** 可匹配或超越 PPO-based RLHF，大幅降低复杂度
- **GRPO（组相对策略优化）：** 新兴高效方法
- **RLVR（可验证奖励的 RL）：** 提供逐步推理反馈

**持续挑战：** 奖励黑客、计算成本、可扩展反馈收集
**前沿方向：** 混合 RL 算法、验证器引导训练、多目标对齐框架

**面试考点：** PPO vs DPO vs GRPO 对比、RLHF 的奖励模型设计、reward hacking 问题

---

## 4. IR-RS-with-LLM Survey: LLM 与 IR/RS 共生演进综述

**来源：** https://github.com/sisinflab/IR-RS-with-LLM-survey
**领域：** LLM + 信息检索/推荐
**规模：** 收集 2356 篇论文，筛选后 1862 篇
**核心价值：** 提供发表趋势、引用速度、子课题分布等分析数据

---

## 5. Twitter/X 推荐算法开源

**来源：** https://github.com/twitter/the-algorithm
**领域：** 工业推荐系统 / LLM 基础设施
**核心架构（三阶段）：**
1. **候选生成：** 从约 5 亿日推文中选出 1500 条候选（关注内 + 关注外）
2. **排序：** 基于 MaskNet 的 48M 参数神经网络模型
3. **过滤：** 启发式规则（屏蔽用户、NSFW、已见推文等）

**开源限制：** 不含广告推荐代码、不含隐私相关代码、仅含模型训练管线（不含权重）

**面试考点：** 工业推荐系统三阶段架构、MaskNet 模型设计、候选生成的 in-network vs out-of-network 策略
