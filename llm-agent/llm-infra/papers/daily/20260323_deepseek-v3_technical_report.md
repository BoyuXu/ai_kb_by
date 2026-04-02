# DeepSeek-V3 Technical Report
> 来源：https://arxiv.org/abs/2412.19437 | 领域：llm-infra | 日期：20260323

## 问题定义
DeepSeek-V3是深度求索发布的671B MoE模型技术报告，以极低成本（~600万美元训练）达到GPT-4级别效果，详述了FP8训练、MLA注意力、MoE负载均衡等创新技术。

## 核心方法与创新点
- MLA（Multi-head Latent Attention）：低秩KV压缩，减少KV cache约60%
- Fine-grained MoE：256个专家，每次激活37个，细粒度路由
- FP8训练：首次实现FP8精度的大规模LLM稳定训练，成本降低约40%
- 辅助-free负载均衡：通过bias调整代替auxiliary loss，避免性能损失
- 多token prediction：同时预测多个未来token，提升训练效率

## 实验结论
DeepSeek-V3使用2.8M H800 GPU-hours（约600万美元）训练671B模型；在代码（Codeforces）和数学（MATH）超越GPT-4o；中文能力大幅领先西方模型；MoE激活参数37B但效果媲美密集70B+。

## 工程落地要点
- FP8训练需要特殊的精度管理（某些层仍需BF16）
- MoE的细粒度专家需要高效的专家并行通信框架
- 开源权重可以用SGLang/vLLM部署，但需要多GPU（FP8量化后约400GB）

## 常见考点
1. **Q: MLA（Multi-head Latent Attention）如何减少KV cache？** A: 将K和V投影到低维latent space存储，推理时解压；类似LoRA应用到注意力
2. **Q: DeepSeek-V3的成本优势来源？** A: FP8训练节省显存和带宽、MoE减少激活计算、多token prediction提升数据效率
3. **Q: Auxiliary-free负载均衡如何工作？** A: 动态调整每个专家的bias，频繁使用的专家降bias（更难被选中），代替硬性loss
4. **Q: 多token prediction（MTP）在训练中的作用？** A: 同时预测下一个和下下个token，等效增加监督密度，提升训练效率约15%
5. **Q: 为什么FP8训练有难度？** A: FP8数值范围小（6位），激活值outlier容易溢出；需要per-channel量化和loss scaling
