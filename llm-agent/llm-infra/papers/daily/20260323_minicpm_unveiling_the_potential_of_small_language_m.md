# MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies
> 来源：https://arxiv.org/abs/2404.06395 | 领域：llm-infra | 日期：20260323

## 问题定义
清华/面壁提出MiniCPM，研究如何用可扩展的训练策略充分释放小语言模型（SLM）的潜力，使2B参数模型达到7B级别的效果，适用于端侧部署。

## 核心方法与创新点
- WSD学习率调度：Warmup-Stable-Decay，在stable阶段持续训练，decay阶段快速适配
- 模型扩缩规律：使用proxy模型快速搜索最优超参（学习率/batch size），再扩展到大模型
- 持续训练：支持不重新初始化地继续添加数据训练
- 高质量数据：课程学习（从简单到复杂），数据配比精心设计

## 实验结论
MiniCPM-2B在多数benchmark超越LLaMA-7B；端侧推理速度达到30+ tokens/sec（手机上）；WSD调度相比cosine调度最终loss降低约5%；小模型在代码任务上仍有明显差距。

## 工程落地要点
- 2B模型可以在手机（Qualcomm 888+）上实现实时推理，端侧应用首选
- WSD调度支持继续训练（不需要reset），适合持续学习场景
- MiniCPM的量化（GGUF格式）与llama.cpp完全兼容

## 常见考点
1. **Q: 为什么小模型（SLM）研究重要？** A: 端侧部署（隐私/低延迟）、降低推理成本、资源受限环境（边缘计算）
2. **Q: WSD学习率调度的原理？** A: Warmup快速升到峰值→Stable长期稳定训练（充分拟合数据）→Decay快速降至收敛
3. **Q: 如何用代理模型（proxy model）搜索超参？** A: 用小10-100x的代理模型做超参搜索（快速便宜），然后假设最优超参可迁移到大模型
4. **Q: 课程学习（Curriculum Learning）在LLM预训练中的应用？** A: 先训练简单（短句/基础知识），再引入复杂（长上下文/专业知识），提升学习效率
5. **Q: llama.cpp/GGUF格式在端侧部署中的作用？** A: 纯CPU推理、INT4量化大幅减少内存（2B模型约1.5GB）、跨平台（iOS/Android/PC）
