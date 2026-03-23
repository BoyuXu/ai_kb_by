# LoRA: Low-Rank Adaptation of Large Language Models
> 来源：https://arxiv.org/abs/2106.09685 | 领域：llm-infra | 日期：20260323

## 问题定义
微软提出LoRA，解决大型语言模型全量微调（Full Fine-tuning）的计算和内存成本问题。核心假设：预训练权重的更新矩阵具有低内在秩（intrinsic rank），可以用低秩分解近似。

## 核心方法与创新点
- 低秩分解：ΔW = BA，其中B∈R^{d×r}，A∈R^{r×k}，r≪min(d,k)
- 冻结原始权重：只训练B和A，原始W不更新（serving时无额外latency）
- 合并权重：推理时W' = W + BA，完全等效，无额外计算
- 超参设计：rank r（推荐4-64）和scaling factor α

## 实验结论
LoRA在GPT-3的4个下游任务上，用0.1%的可训练参数达到全量微调的效果；推理时与全量微调完全一致（权重合并）；支持多任务快速切换（不同LoRA adapter）。

## 工程落地要点
- LoRA几乎是所有开源LLM微调的标准方案（Alpaca/Vicuna均用LoRA）
- r=16，α=32是常用默认配置，适合大多数任务
- 多个LoRA可以线性叠加（LoRA merging），实现多技能融合

## 面试考点
1. **Q: LoRA为什么假设更新矩阵是低秩的？** A: 预训练模型已有强语言能力，微调只需调整方向（低维空间），不需要改变整个权重空间
2. **Q: LoRA的参数数量计算？** A: 每个被适配的矩阵：r×(d+k)；通常只适配Q和V矩阵（约6-10%原始参数）
3. **Q: rank r的选择原则？** A: 简单任务（分类/翻译）：r=4-8；复杂任务（代码/推理）：r=16-64
4. **Q: LoRA和Adapter的区别？** A: Adapter在层间添加额外模块，推理有额外延迟；LoRA合并权重，推理零额外开销
5. **Q: LoRA-XS/DoRA/VeRA等变体各自的改进？** A: DoRA：分解幅度和方向；VeRA：共享随机矩阵；LoRA-XS：更小rank的进一步压缩
