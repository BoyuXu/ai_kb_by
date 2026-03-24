# RankLLM: Reranking with Large Language Models
> 来源：https://arxiv.org/abs/2309.15088 | 领域：search | 日期：20260323

## 问题定义
RankLLM提出用大语言模型（GPT-4、LLaMA等）进行搜索结果重排序，通过sliding window listwise排序范式，克服LLM context length限制，实现高质量的搜索重排序。

## 核心方法与创新点
- Sliding Window Listwise：将文档列表按窗口分组，LLM对每个窗口内文档排序
- 多种LLM对比：GPT-4、GPT-3.5、开源LLaMA等不同模型的重排效果对比
- 排序提示工程：设计高效的prompt引导LLM进行相关性判断
- 级联重排：先快速召回→RankLLM精排，效率与质量平衡

## 实验结论
在TREC Deep Learning等benchmark，RankLLM(GPT-4) NDCG@10达到0.75+，超越传统重排序约8%；GPT-3.5效果约80%的GPT-4；开源LLaMA微调版接近GPT-3.5；LLM重排是目前SOTA方案之一。

## 工程落地要点
- GPT-4 API重排成本高（约$0.01/query），大规模应用需要开源模型替代
- Sliding window步长设为文档数的50%，确保边界文档被充分评估
- 可以用RankLLM生成训练数据，蒸馏到cross-encoder进一步降低成本

## 面试考点
1. **Q: LLM重排的sliding window方法如何工作？** A: 取窗口大小N的文档，LLM排序后移动半个窗口继续，覆盖所有文档
2. **Q: LLM重排为何使用listwise而非pointwise？** A: LLM一次性考虑文档间的相对关系，比独立打分更准确；pointwise缺乏比较视角
3. **Q: 如何降低LLM重排的API成本？** A: 开源模型替代（Mistral/LLaMA）、蒸馏到cross-encoder、只对TopK文档重排
4. **Q: RankLLM的position bias如何缓解？** A: 多次shuffle取平均、从中间开始排序而非从头
5. **Q: 搜索重排序的完整链路？** A: BM25/dense召回（1000个）→bi-encoder rerank（100个）→LLM rerank（10个）
