# LLMOrbit: A Circular Taxonomy of LLMs from Scaling Walls to Agentic AI Systems

> 来源：arxiv | 日期：20260322 | 领域：LLM工程

## 问题定义

LLM 领域发展迅猛，技术路线多样（Scaling Laws、MoE、RLHF、Agent、Multimodal等），缺乏系统性的分类框架帮助研究者和工程师理解全貌。LLMOrbit 提出循环分类体系，组织 LLM 的技术演进路径。

## 核心方法与创新点

- **循环分类框架（Circular Taxonomy）**：
  - **内核（Core）**：基础 Transformer 架构、预训练范式
  - **第一轨道（Orbit 1）**：规模化阶段（Scaling Laws、GPT-3、PaLM）
  - **第二轨道（Orbit 2）**：对齐阶段（RLHF、InstructGPT、Constitutional AI）
  - **第三轨道（Orbit 3）**：效率优化（MoE、量化、蒸馏、PEFT）
  - **第四轨道（Orbit 4）**：能力扩展（Multimodal、长上下文、代码/数学推理）
  - **第五轨道（Orbit 5）**：Agentic AI（Tool Use、RAG、Multi-Agent、World Models）
- **"Scaling Wall" 分析**：探讨数据、计算、架构各维度的扩展壁垒
- **技术演进路径**：识别从各轨道向下一轨道的关键突破点

## 实验结论

（本文为综述/分类论文，无实验）

- 梳理了 2017-2025 年超过 400 篇 LLM 核心论文
- 识别出 5 个主要"Scaling Wall"：数据质量墙、计算成本墙、对齐税（alignment tax）、上下文长度墙、工具使用可靠性墙
- Agentic AI 是当前最活跃的研究方向（2024-2025 年论文占比 35%）

## 工程落地要点

- **技术选型参考**：根据业务阶段选择对应轨道的技术
  - 初期：选成熟预训练模型 + PEFT 微调
  - 中期：引入 RAG/Tool Use 扩展能力
  - 成熟期：考虑 Multi-Agent 架构、自定义对齐
- **避免跳轨道**：先做好基础对齐（Orbit 2）再上 Agent（Orbit 5），否则 Agent 行为不可控
- **MoE 权衡**：MoE 提升参数效率但增加工程复杂度（专家路由、负载均衡、通信开销）

## 面试考点

1. **Q：LLM 的主要 Scaling Wall 有哪些？如何突破？**
   A：数据质量墙（合成数据、数据飞轮）；计算成本墙（MoE、量化、蒸馏）；对齐税（Constitutional AI、DPO 降低对齐成本）；上下文长度墙（位置编码改进 RoPE/YARN、KV cache 压缩）

2. **Q：从 Dense LLM 到 MoE LLM 的主要技术挑战？**
   A：专家路由的负载均衡（避免 expert collapse）；分布式训练中专家参数的通信开销（All-to-All）；推理时的 expert activation 缓存管理

3. **Q：Agentic AI 和传统 LLM 应用的核心区别？**
   A：Agentic AI 需要规划（Planning）、工具调用（Tool Use）、记忆（Memory）、多步推理；传统应用是单轮问答。核心挑战是长程任务的可靠性和错误恢复
