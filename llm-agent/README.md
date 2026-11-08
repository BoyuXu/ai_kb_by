# LLM / Agent 知识地图

维护入口更新：2026-11-08。先按问题找主题，再看原始来源；视频作为发现入口，关键结论回到论文、官方文档或可复现实验。

| 学习问题 | 现有入口 | 下一步补充 |
|---|---|---|
| 模型怎样训练、对齐与推理 | [LLM Infra 导航](llm-infra/INDEX.md) | 对照模型版本、训练设定与评测范围 |
| 推理速度、显存与部署成本 | [KV Cache 全景](llm-infra/synthesis/KVCache与LLM推理优化全景.md) | 性能数字补硬件、精度、并发与上下文长度 |
| 怎样组织检索和外部知识 | [RAG 全景](llm-infra/synthesis/RAG系统全景.md) | 区分检索、生成与端到端评估 |
| Agent 怎样稳定完成长任务 | [失败模式](llm-infra/synthesis/Agent失败模式与解法.md) | [Harness 的证据与验收](llm-infra/synthesis/20261108_harness_evidence.md) |
| 工具、Skills 和上下文如何配合 | [上下文工程](agent-context-management/README.md) | [Skills 与 MCP 的边界](llm-infra/synthesis/20261108_skills_mcp.md) |
| 研究如何变成自己的工作改进 | 先形成知识卡，再写应用假设 | 本机配置与个人运行日志放独立本地工作流知识库 |

## 当前学习队列

重点 UP：AI探索喵、AI超元域、Agent智能体深度研究院、五道口纳什。历史范围从 **2026-06-01** 起，并持续纳入新视频。目录发现、读到字幕、总结、来源复核、入库分别计数，不能把搜到的视频数称为已学习数。

## 更新规则

见 [维护规范](../_meta/maintenance/大肥鱼维护规范.md)。优先修订已有主题并记录增量；不重复生成同题综述，不把未经复核的新观点覆盖成既有事实。
