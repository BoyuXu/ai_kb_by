# Agent Harness：模型以外还应验证什么

核验日期：2026-11-08。阅读层级明确分开：官方工程文章相关章节已读；新论文仅摘要初读，未复现实验。

## 新研究线索：复杂组件不必然更好

[Hu 等，Beyond the Model（2026-09-26，arXiv:2609.32459v1）](https://arxiv.org/abs/2609.32459v1) 对不同模型、任务和 Harness 组件进行比较。摘要报告，效果依赖模型能力与任务；某些结构化工具和专用组件有益，压缩或通用子代理在部分任务中可能产生负效应。这是作者在其实验范围内的结论，不足以推广为所有 Agent 的设计定律。

待读正文：任务划分、预算控制、重复试验、代码快照、组件消融和失败案例。摘要数字暂不作为选型依据。

## 官方工程经验：分离可恢复状态

Anthropic 将 session 事件记录、执行循环和执行环境分开，降低某个执行环境故障导致整个任务丢失的风险。这里可迁移的是边界设计，不能据此声称本地部署具有同等可靠性。[Scaling Managed Agents，2026-04-08](https://www.anthropic.com/research/managed-agents)

## 验收必须看最终产物

“模型说完成”与环境里确实形成结果是两件事。Agent 评估应检查任务执行后状态，而不只读最终答复。[Demystifying evals，2026-01-09](https://www.anthropic.com/research/demystifying-evals-for-ai-agents)

本库维护的验收建议（本次工程推断）：

1. 运行层：进程退出码与结构化完成/失败事件；不要用日志中任意出现的401/403判定认证失败。
2. 内容层：正文获取范围、来源、未核验主张与笔记是否存在。
3. 交付层：文件哈希、实际提交及可打开的 PR。完成生成不等于完成发布。
4. 恢复层：任务去重键与已完成阶段落盘，失败后从缺失阶段继续。

关联：[Agent失败模式与解法](Agent失败模式与解法.md)。这套建议首先适用于知识整理流程，性能收益仍需自身任务的测量。

## 四种循环：触发、执行、验收与改进

完整阅读 [Agent智能体深度研究院，2026-07-05，BV1tUMA61ES1](https://www.bilibili.com/video/BV1tUMA61ES1) P1的307段字幕，未逐帧核对。对照 [LangChain，The art of loop engineering，Sydney Runkle，2026-06-16](https://www.langchain.com/blog/the-art-of-loop-engineering) 正文相关章节：agent loop执行动作，verification loop检查结果，event loop决定何时触发，hill-climbing loop利用轨迹调整Harness。它是该文的工程分类，不是全行业强制四层标准。

把失败任务原样再跑一次属于重试，不能称为系统已经自我改进。改进需要明确变更了提示、工具或评估标准，并在固定任务上比较成功率、耗时与成本；还需保留人工检查与回退边界。这补充了前述产物验收，而不替代验收。

## Dynamic Workflows：状态外置的适用边界

完整阅读 [AI超元域，2026-06-04，BV1a2EA6VEeX](https://www.bilibili.com/video/BV1a2EA6VEeX) P1全部298段字幕。视频中的代码图谱、跨工具演示未复现，性能宣称不入库为实测结论。当前 [Claude Code官方 Workflows 文档](https://code.claude.com/docs/en/workflows)（核验2026-11-08）说明脚本可维护规划和中间状态，适合复杂审计或迁移等工作。由此可以借鉴显式计划、持久化状态、停止条件；不能推出每次简短摘要都应增加多代理和跨工具桥接。

建议先在固定任务上对比简单单次流程与脚本编排，再决定复杂度是否值得。视频发布日期与当前产品版本不同，指令和能力应按当前官方文档复核，尤其不能推定读者已有额外订阅。
