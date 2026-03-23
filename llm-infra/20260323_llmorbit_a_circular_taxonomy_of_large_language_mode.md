# LLMOrbit: A Circular Taxonomy of Large Language Models from Scaling Walls to Agentic AI Systems
> 来源：https://arxiv.org/search/?query=LLMOrbit+taxonomy+large+language+models+agentic&searchtype=all | 领域：llm-infra | 日期：20260323

## 问题定义
LLM领域发展迅速，缺乏系统性的分类框架。LLMOrbit提出环形分类法（Circular Taxonomy），从Scaling Walls到Agentic AI系统，系统梳理LLM的技术演进路径和各类方法的关系。

## 核心方法与创新点
- 环形分类法：以Scaling为核心，向外扩展到对齐/高效/推理/Agent等维度
- 技术谱系：梳理预训练→SFT→RLHF→推理→Agent的完整演进链
- Scaling Wall分析：分析参数/数据/计算三维度的scaling上限
- Agentic AI：从单模型到多Agent系统的能力跃升分析

## 实验结论
分类框架综述了2024-2025年约200篇LLM论文，识别出5个核心技术维度：Efficient Training、Alignment、Reasoning、Efficiency、Agentic；Reasoning和Agentic是当前最活跃的研究方向。

## 工程落地要点
- 技术选型时参考分类框架，避免重复造轮子
- Scaling Wall促使研究从参数扩展转向数据效率和架构创新
- Agentic系统的工程挑战：工具调用可靠性、多步推理一致性、安全对齐

## 面试考点
1. **Q: LLM的三个Scaling维度及各自的上限？** A: 参数（内存墙）、数据（互联网数据已接近上限）、计算（电力/芯片产能）
2. **Q: 从预训练到Agent的完整技术栈？** A: 预训练→SFT（指令跟随）→RLHF（对齐）→工具使用→多步规划→多Agent协作
3. **Q: Scaling Law的基本形式？** A: L(N,D) = A/N^α + B/D^β + C，损失随参数N和数据D幂律降低
4. **Q: 为什么Reasoning能力成为当前研究重点？** A: 纯Scaling效果边际递减，推理能力（o1/R1/Chain-of-Thought）是下一个突破口
5. **Q: Agentic AI系统面临的核心技术挑战？** A: 长程规划（多步不出错）、工具使用（API调用可靠性）、记忆管理（长上下文）
