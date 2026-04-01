# RankMixer: Scaling Up Ranking Models in Industrial Recommenders

> 来源：arxiv 2507.15551 | 领域：rec-sys | 学习日期：20260328

## 问题定义

大语言模型（LLM）的Scaling Law在NLP领域已被充分验证：模型越大、数据越多，性能越好。但**推荐系统排序模型是否也遵循类似规律**？现有工业排序模型存在：

1. 参数量停滞在百亿以内，进一步扩展效益不明显
2. 传统MLP/Transformer结构在处理海量稀疏特征时效率低下
3. 缺乏专门针对推荐排序任务的高效大模型架构

**核心问题**：设计一种可高效扩展（scale up）的工业排序模型架构，验证推荐排序的Scaling Law。

## 核心方法与创新点

RankMixer提出**面向推荐排序的混合专家（MoE）Scaling架构**：

### 1. Mixer Block 设计

$$
\mathbf{h}^{(l+1)} = \text{FFN}(\text{MHA}(\mathbf{h}^{(l)})) + \text{MoE-FFN}(\mathbf{h}^{(l)})
$$

核心创新：
- **Dense路径（MHA + FFN）**：处理全局特征交叉，参数共享，保证基础表达能力
- **Sparse MoE路径**：每个token只激活top-k个专家，参数量随专家数线性增长但计算量不变
- 双路径设计允许在保持推理延迟不变的情况下大幅扩展总参数量

### 2. 稀疏特征友好的Embedding Scaling

$$
\mathbf{E}_{item} = \text{Concat}[\mathbf{e}_{id}, \mathbf{e}_{text}, \mathbf{e}_{stat}]
$$

- 区分ID embedding、文本语义embedding和统计特征embedding，分别设计不同的scale策略
- ID embedding维度随item规模对数增长；文本embedding复用预训练LLM，固定参数

### 3. Scaling实验设计
验证了以下Scaling维度的效益：
- 模型深度（层数）
- 专家数量（MoE专家）
- 每专家容量（FFN隐层维度）
- 训练数据量

## 实验结论

在工业推荐平台（字节系平台）的大规模实验：
- 模型参数从百亿扩展到千亿量级，AUC持续提升（无plateau）
- MoE架构相比Dense架构，相同计算预算下AUC提升**约0.3-0.5%**（工业场景极显著）
- 数据量Scaling：增加训练数据1×，AUC提升约0.1-0.2%
- 验证了推荐排序存在**类似LLM的Scaling Law**

## 工程落地要点

1. **MoE负载均衡**：需要auxiliary loss防止Expert Collapse（少数专家被过度使用）
   ```
   L_balance = alpha * sum_e(f_e * P_e)  # f_e=实际负载, P_e=路由概率
   ```
2. **推理延迟控制**：MoE sparse激活保证FLOPS与Base模型相当，但内存带宽是瓶颈，需expert并行+流水线
3. **Embedding规模管理**：ID embedding table随item/user规模增长，需定期shrink冷item embedding，防止内存爆炸
4. **分布式训练**：千亿参数需Expert Parallelism + Tensor Parallelism，常用框架：Megatron-LM, DeepSpeed
5. **增量更新**：工业场景需支持每日增量训练，MoE结构支持只更新激活的专家，效率更高

## 面试考点

**Q1：MoE（Mixture of Experts）在推荐排序中的优势是什么？**
A：MoE允许模型参数量（容量）与计算量解耦。通过稀疏激活，千亿参数模型的单次推理FLOPs可以和百亿Dense模型相当，但模型容量（记忆能力、表达能力）更强，特别适合推荐系统中的长尾特征学习。

**Q2：推荐排序的Scaling Law和LLM的Scaling Law有何不同？**
A：LLM的Scaling Law体现在语言理解能力上（perplexity），推荐排序的Scaling Law体现在AUC/GAUC等业务指标上。推荐场景中稀疏特征（ID特征）的embedding table是重要的Scaling维度，LLM中没有类似概念。此外，推荐系统的数据分布随时间变化（概念漂移），需要持续更新。

**Q3：Expert Collapse是什么？如何解决？**
A：Expert Collapse是指MoE中大部分训练样本只激活少数几个专家，其他专家几乎不被使用（梯度稀少导致欠拟合）。解决方案：(1) 辅助负载均衡损失；(2) Expert Dropout；(3) 随机路由初始化；(4) 专家容量限制（capacity factor）。

**Q4：工业排序模型从百亿扩展到千亿的主要工程挑战？**
A：(1) 显存：需要多机多卡并行（Expert/Tensor Parallel）；(2) 通信：Expert Parallel引入All-to-All通信，延迟敏感；(3) 推理：线上serving需要模型蒸馏或量化；(4) 训练稳定性：大模型更容易梯度爆炸/消失，需要精细的学习率调度。

**Q5：如何验证推荐模型的Scaling Law（实验设计）？**
A：控制变量法：固定数据量，分别扩展模型深度/宽度/专家数，记录各维度的AUC增益；固定模型，扩展训练数据，绘制data-AUC曲线。如果拟合$\Delta AUC \propto C^\alpha$（C为计算量），且α在不同规模段稳定，则说明存在Scaling Law。
