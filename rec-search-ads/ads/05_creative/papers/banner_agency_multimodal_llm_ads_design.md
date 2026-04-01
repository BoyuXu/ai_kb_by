# BannerAgency: Advertising Banner Design with Multimodal LLM Agents

> 来源：arxiv | 领域：ads | 学习日期：20260328

## 问题定义

广告 Banner 设计是一项需要多种能力协同的任务：
1. **理解营销目标**：根据广告主需求（产品、受众、诉求）生成合适设计
2. **视觉-文字协调**：标题文字、产品图片、背景、CTA 按钮的布局与配色
3. **品牌一致性**：保持广告主品牌风格
4. **自动化挑战**：现有 AI 图像生成工具缺乏对广告构图和营销逻辑的理解

BannerAgency 提出多智能体（Multi-Agent）框架，由不同角色的 LLM Agent 协同完成 Banner 设计全流程。

## 核心方法与创新点

### 多智能体角色设计

**Agent 1：Creative Director（创意总监）**
- 分析广告主 Brief（产品信息、目标受众、核心诉求）
- 制定设计策略（视觉风格、情感基调、色彩方向）
- 输出：设计规格书（Design Brief）

**Agent 2：Copywriter（文案师）**
- 基于 Design Brief 生成标题、副标题、CTA 文案
- 控制文字长度和视觉层级

**Agent 3：Visual Designer（视觉设计师）**
- 将 Design Brief 转化为具体布局方案（JSON 格式的坐标、颜色、字体）
- 调用 T2I 模型（DALL-E/Stable Diffusion）生成背景图

**Agent 4：QA Reviewer（质检员）**
- 检查生成结果是否符合品牌规范、可读性、合规性
- 反馈修改建议，触发重新生成（最多 N 轮）

### 工作流：顺序 + 反馈循环

$$
Brief \xrightarrow{Director} Strategy \xrightarrow{Copy+Visual} Draft \xrightarrow{QA} Revision \rightarrow Final
$$

QA Reviewer 的反馈可以触发部分 Agent 的重新生成（而非全流程重做），提升效率。

### 结构化中间表示

各 Agent 通过 JSON Schema 规范的"设计状态"传递信息，而非自由文本，减少信息丢失和误解。

## 实验结论

- 在人工评估（美观度、相关性、品牌一致性）上显著优于单 LLM 直接生成
- QA 反馈循环将"可用率"（通过合规检查的比例）从 61% 提升到 89%
- 多轮修订通常 2-3 轮收敛，极少超过 5 轮
- 与人工设计师生成的 Banner 相比，60% 场景下 AI 生成质量被评为"可接受"

## 工程落地要点

1. **角色职责边界清晰**：每个 Agent 只负责一个模块，避免职责重叠导致不一致
2. **JSON Schema 约束**：用严格 schema 规范 Agent 输出，便于下游 Agent 解析
3. **渐进式生成**：先确定布局 → 再生成图像 → 最后叠加文字，顺序不可逆
4. **模板库辅助**：预设一批品牌模板，Visual Designer 在模板上修改而非从零生成，提升质量
5. **并行化**：Copy 和初步 Visual 规划可以并行，节省时间
6. **成本控制**：GPT-4V 调用成本高，QA 轮次设上限（3-5 轮），超出转人工审核

## 面试考点

**Q：为什么广告 Banner 设计需要多个专门化 Agent 而不是单一 LLM？**
A：单一 LLM 难以同时兼顾营销策略、文案写作、视觉构图等专业能力；多 Agent 允许每个角色专注于自己的专业领域，通过明确的接口传递中间结果，减少单点的认知负担，同时 QA Agent 提供独立的质量把关。

**Q：多智能体系统中如何处理 Agent 间的信息不一致问题？**
A：1) 使用结构化 JSON 作为 Agent 间的通信格式，而非自由文本；2) 设置 Schema 验证，不符合格式的输出触发重生成；3) QA Agent 作为最终把关，检测一致性问题；4) 关键信息（品牌色、字体）全程传递不丢失。

**Q：如何评估 AI 生成的广告 Banner 质量？**
A：1) 自动化指标：可读性分数（文字对比度）、合规性检查（合法声明、禁用词）、品牌色彩一致性；2) 人工评估：美观度（1-5分）、相关性（与产品的匹配度）、情感传达（是否符合品牌调性）；3) 线上指标：实际投放后的 CTR 和转化率。
