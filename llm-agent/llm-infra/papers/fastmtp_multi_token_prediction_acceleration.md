# FastMTP: Enhanced Multi-Token Prediction for Inference Acceleration

> 来源：https://arxiv.org/abs/2509.18362 | 领域：LLM基础设施 | 学习日期：20260331

## 问题定义

标准自回归LLM每步只生成一个token，推理效率受限。MTP能并行生成但质量下降。

## 核心方法与创新点

1. **增强型MTP头**：LLM最后一层后接多个并行预测头

$$
P(t_{i+k} | t_{\leq i}) = \text{Head}}_{\text{k(h}}_{\text{i + \text{PE}}(k))
$$

2. **位置编码增强**：每个预测头引入相对位置偏移编码
3. **渐进式训练**：先训练单token预测，再逐步增加头数
4. **验证-接受机制**：原始模型logits验证MTP输出质量

## 实验结论

代码和文本生成2-3x加速，质量损失<1%。

## 工程落地要点

- MTP头轻量（LM head复制），额外参数小
- 可与KV Cache无缝集成
- 适合batch=1低延迟场景
- 需修改serving框架（如vLLM）

## 常见考点

1. **MTP与Speculative Decoding区别？** MTP同一模型多头并行，SD是不同模型draft-verify
2. **需要验证机制？** MTP预测的后续token质量不如自回归
3. **MTP适合什么场景？** batch size小、延迟敏感的在线服务
