# LLaMA 3: The Llama 3 Herd of Models
> 来源：https://arxiv.org/abs/2407.21783 | 领域：llm-infra | 日期：20260323

## 问题定义
Meta发布的LLaMA 3系列模型技术报告，覆盖8B到405B参数规模，详述了数据、架构、训练和安全对齐的完整工程实践，是开源LLM领域最重要的工程参考之一。

## 核心方法与创新点
- 数据质量：15T+ token高质量预训练数据，严格的去重和过滤
- 架构改进：GQA（Grouped Query Attention）减少KV cache，RoPE位置编码
- 多阶段后训练：SFT→拒绝采样→DPO→PPO完整对齐pipeline
- 多模态：视觉/语音能力集成（LLaMA 3.2）
- 安全机制：Llama Guard、Prompt Guard、Code Shield

## 实验结论
LLaMA 3.1 70B在大多数benchmark接近GPT-4；405B模型在推理任务（MATH/MMLU）达到GPT-4级别；8B模型在同参数规模中SOTA；多语言能力覆盖8种语言。

## 工程落地要点
- LLaMA 3使用BF16训练，推理可用FP16或INT8量化
- context length 8K（base）到128K（128K版本），需要RoPE外推
- 开源权重可以用vLLM/SGLang等框架直接部署

## 常见考点
1. **Q: GQA（Grouped Query Attention）如何减少KV cache？** A: 多个Query head共享一个KV head，将KV head数从H降到H/G，显存减少G倍
2. **Q: RoPE（旋转位置编码）相比绝对位置编码的优势？** A: 相对位置感知、长上下文外推能力强（RoPE Scaling）、无需特殊位置token
3. **Q: LLaMA 3的后训练pipeline？** A: SFT（指令跟随）→拒绝采样（RS）精选高质量数据→DPO/PPO（偏好对齐）
4. **Q: 预训练数据的质量控制方法？** A: 基于规则过滤（去重/去噪）、基于模型的质量评分、领域权重调整
5. **Q: LLaMA开源对AI生态的影响？** A: 推动PEFT研究、降低研究门槛、加速多模态/Agent应用、促进安全研究
