# SGLang: Efficient Execution of Structured Language Model Programs
> 来源：https://arxiv.org/abs/2312.07104 | 领域：llm-infra | 日期：20260323

## 问题定义
构建LLM应用时，多轮对话、结构化输出、并行采样等常见模式缺乏高效的编程框架。SGLang提出结构化语言模型编程，通过RadixAttention等技术大幅提升复杂LLM应用的执行效率。

## 核心方法与创新点
- RadixAttention：基于Radix Tree的KV cache共享，自动复用共同prefix
- 结构化控制流：fork（并行）、join（同步）等原语，易写复杂LLM程序
- Constrained decoding：基于有限状态自动机（FSM）的JSON/正则表达式约束生成
- 推测执行：并行执行多个可能的生成路径

## 实验结论
SGLang相比原生HuggingFace，JSON生成吞吐量提升约4x；多轮对话吞吐量提升约6x；RadixAttention使prefix cache命中率从<10%提升至>80%；目前是最快的LLM serving框架之一。

## 工程落地要点
- SGLang的RadixAttention对长system prompt场景效果极好（100%复用）
- 结构化输出（JSON Schema）在tool calling和数据抽取场景必备
- SGLang在多GPU serving和张量并行上有优化，适合生产部署

## 面试考点
1. **Q: RadixAttention如何实现prefix cache？** A: 用Radix Tree存储所有历史KV，相同prefix的请求自动共享，无需重新计算
2. **Q: Constrained decoding（约束解码）如何保证JSON格式？** A: 在生成每个token时，只允许满足当前grammar状态的token（用FSM跟踪状态）
3. **Q: SGLang和vLLM的主要差异？** A: SGLang在结构化输出和prefix sharing上更优；vLLM通用性和生态更成熟
4. **Q: Prefix sharing对multi-turn对话的意义？** A: system prompt只计算一次，无论多少并发请求都复用同一KV cache
5. **Q: FSM（有限状态自动机）在约束解码中如何工作？** A: FSM的状态对应grammar进度，每个token选择必须是FSM的合法转移
