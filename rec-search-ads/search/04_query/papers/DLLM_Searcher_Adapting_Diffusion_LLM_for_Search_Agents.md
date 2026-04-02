# DLLM-Searcher: Adapting Diffusion LLM for Search Agents

> 来源：arxiv | 日期：20260322 | 领域：搜索系统

## 问题定义

传统搜索 Agent 依赖自回归 LLM 进行 query 规划和结果综合，存在：①逐 token 生成延迟高；②无法并行推理多个搜索子目标；③曝光偏差（exposure bias）影响检索决策质量。DLLM-Searcher 引入扩散 LLM 解决上述问题。

## 核心方法与创新点

- **扩散 LLM（DLLM）**：基于 MDLM（Masked Diffusion Language Model），在掩码扩散过程中并行去噪所有 token
  - 与 BERT 类似但支持生成；与 GPT 类似但非自回归
  - 允许双向上下文，全局规划能力更强
- **搜索 Agent 框架**：
  - **规划阶段**：DLLM 并行生成多个搜索子 query（无需逐步串行）
  - **检索阶段**：并行执行多 query 检索，结果聚合
  - **综合阶段**：DLLM 基于检索结果生成最终答案
- **多轮搜索**：根据检索质量自适应决定是否需要追加检索（confidence-based stopping）
- **训练策略**：用 DPO（Direct Preference Optimization）对 DLLM 进行对齐，让其偏好高召回的 query 策略

## 实验结论

- 在 HotpotQA、2WikiMultiHopQA 等多跳推理基准上，F1 提升 6.3%（vs 自回归 LLM Agent）
- 延迟降低 2.1×（并行生成多子 query vs 串行生成）
- 多跳问答中的子 query 质量：NDCG 比 CoT-based 自回归方法提升 8.7%
- 消融：双向上下文（全局规划）贡献约 60% 的性能提升

## 工程落地要点

- **扩散步数**：推理时建议 20-50 步（vs 训练 1000 步），使用 DDIM/DDPM 加速
- **并行子 query 数量**：建议 3-5 个，过多增加检索开销和结果融合复杂度
- **Confidence-based stopping**：用检索结果的相关性分数作为置信度指标，阈值可调
- **与 BM25/稀疏检索集成**：DLLM 生成的扩展 query 可直接用于 BM25 检索，无需 dense retrieval

## 常见考点

1. **Q：扩散 LLM 和自回归 LLM 在搜索 Agent 场景的核心区别？**
   A：自回归串行生成 token（延迟高，无法并行规划）；扩散 LLM 并行去噪所有 token（延迟低），且双向上下文使全局规划更好

2. **Q：多跳搜索中如何决定何时停止检索？**
   A：置信度估计：当前检索结果的相关性分数超过阈值；或答案覆盖率指标（证据链完整性）；或设置最大轮数上限

3. **Q：扩散 LLM 在实际搜索系统部署的障碍？**
   A：扩散模型推理仍比自回归慢（需多步去噪）；DLLM 的生成质量在长文本上不及 GPT-4 级自回归模型；工程生态不成熟（TensorRT等优化工具不完善）
