# MA-RAG: Multi-Agent Retrieval-Augmented Generation via Collaborative Chain-of-Thought Reasoning
> 来源：https://arxiv.org/abs/2505.20096 | 日期：20260319

## 问题定义
复杂问答任务中，单一LLM+RAG存在局限：单次检索难以覆盖所有必要信息，Chain-of-Thought推理可能因缺乏及时信息注入而出错。MA-RAG提出多Agent协作框架，不同Agent负责推理链的不同步骤，动态调用检索，实现更准确的复杂推理。

## 核心方法与创新点
1. **多Agent分工**：
   - Planner Agent：分解复杂问题为子任务序列
   - Retriever Agent：根据当前推理状态执行针对性检索
   - Reasoner Agent：结合检索结果推进推理链
   - Critic Agent：评估当前答案可信度，决定是否继续
2. **协作CoT**：推理链在多个Agent间传递，每步推理后触发相关检索，避免推理脱轨
3. **动态检索时机**：Critic Agent判断当前推理的不确定性，高不确定性时触发Retriever
4. **跨Agent记忆共享**：维护共享推理上下文（Shared Working Memory），所有Agent读写
5. **冲突解决机制**：当多个Agent结论冲突时，投票或置信度加权决策

## 实验结论
- 在MuSiQue、2WikiMultiHop、HotpotQA上，F1比单Agent RAG提升8%-13%
- 多Agent协作在需要≥3跳推理的问题上优势最显著（+15%-20%）
- 平均检索次数从3次增加到5次，延迟增加约40%，但准确率提升显著

## 工程落地要点
- Agent间通信设计关键：使用结构化JSON格式传递推理状态，而非自由文本
- 设置Agent数量上限和推理步数上限，防止无限循环
- 使用异步并行检索，Retriever和Reasoner并发执行减少延迟
- 生产环境建议缓存中间检索结果，避免重复检索相同子问题

## 面试考点
**Q: Multi-Agent系统与Single-Agent系统的核心区别？**
A: 分工专业化（每个Agent专注特定能力）、并行处理（多Agent同时工作）、角色分离（规划/执行/验证解耦）。代价是协调复杂度增加和通信开销。

**Q: RAG中的Hallucination如何减少？**
A: （1）提高检索质量（相关性更强的文档）；（2）添加引用机制，要求LLM引用来源；（3）事实验证Agent（Critic）；（4）RLHF对齐减少幻觉；（5）温度参数降低。

**Q: Chain-of-Thought推理如何与检索结合？**
A: （1）Interleaved RAG：每步推理后检索；（2）FLARE：预测下一句，若置信度低则检索；（3）ReAct：交替执行推理（Thought）和行动（Action=检索），本文MA-RAG是ReAct的多Agent扩展。
