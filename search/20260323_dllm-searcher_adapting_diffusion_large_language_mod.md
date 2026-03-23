# DLLM-Searcher: Adapting Diffusion Large Language Model for Search Agents
> 来源：https://arxiv.org/search/?query=DLLM+Searcher+diffusion+LLM+search&searchtype=all | 领域：search | 日期：20260323

## 问题定义
将扩散式大语言模型（Diffusion LLM）适配到搜索Agent任务，解决自回归LLM在搜索中的推理效率问题。扩散LLM可并行生成，天然适合搜索的迭代精化过程。

## 核心方法与创新点
- 扩散LLM用于搜索：利用扩散模型的非自回归特性实现并行搜索规划
- 迭代去噪搜索：搜索过程类比扩散去噪，从噪声查询逐步精化到精确查询
- 多步规划：Agent在多轮搜索中用扩散模型规划下一步查询策略
- 并行搜索候选：同时生成多个搜索策略，选取最优路径

## 实验结论
在搜索Agent benchmark（HotpotQA、2WikiQA等），DLLM-Searcher相比自回归LLM搜索Agent，推理速度快约3x，在相同时间预算下完成更多搜索步骤，准确率提升约5%。

## 工程落地要点
- 扩散LLM的训练比自回归LLM更复杂，需要specialized的DDPM训练框架
- 并行生成多个搜索策略需要有效的搜索结果聚合方法
- 目前扩散LLM规模较小，需要蒸馏或与自回归LLM协作

## 面试考点
1. **Q: 扩散LLM（MDLM/PLAID等）与自回归LLM的本质区别？** A: 扩散LLM非自回归，掩码扩散方式并行生成；自回归逐token生成
2. **Q: 搜索Agent的核心组件？** A: 查询规划模块、检索模块、结果处理模块、多轮迭代控制
3. **Q: 为什么搜索需要多步迭代？** A: 复杂问题需要分解（先找背景，再找细节）；单次查询信息不完整
4. **Q: DLLM在搜索中的"迭代精化"如何理解？** A: 初始查询含噪声（模糊意图），通过扩散去噪过程产生更精确的查询
5. **Q: 搜索Agent与RAG的主要区别？** A: Agent主动规划多轮搜索策略；RAG通常单次检索，被动增强生成
