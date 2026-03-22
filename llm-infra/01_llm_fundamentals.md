# 大模型基础面试考点

> 来源：AIGC-Interview-Book 大模型基础章节整理
> 更新：2026-03-12

---

## 一、Transformer 核心架构

### 1.1 整体结构

```
Input → Embedding + Position Encoding
  → N × (Multi-Head Self-Attention + Add&Norm + FFN + Add&Norm)
  → Output
```

- **Encoder-only**（BERT）：双向注意力，适合NLU任务（分类、NER）
- **Decoder-only**（GPT/LLaMA）：单向因果注意力，适合生成任务
- **Encoder-Decoder**（T5）：Encoder处理输入，Decoder生成输出

### 1.2 Multi-Head Attention（MHA）

**核心公式：**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $Q, K, V$ 来自输入的线性变换：$Q=XW_Q, K=XW_K, V=XW_V$
- $\sqrt{d_k}$ 缩放防止点积过大导致梯度消失
- Multi-Head：并行多组注意力，最后拼接：$\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h)W_O$
- 每个head学习不同的注意力模式

**面试常问：**
- Q：为什么要除以 $\sqrt{d_k}$？
  A：防止点积值过大，softmax进入饱和区（梯度接近0），导致梯度消失
- Q：Self-Attention 时间复杂度？
  A：$O(n^2 d)$，n为序列长度，是长序列处理的瓶颈

### 1.3 变体：MQA 和 GQA

- **Multi-Query Attention（MQA）**：多个Q共享单组K、V → 减少KV缓存显存
- **Grouped-Query Attention（GQA）**：将Query分组，每组共享一对K/V → 平衡效率与性能
- 应用：LLaMA2、Mistral 使用 GQA

### 1.4 FFN（前馈神经网络）

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

- 两层线性变换 + ReLU（或SwiGLU/GELU）
- 中间维度通常是 $4d_{model}$
- **SwiGLU（LLaMA/Qwen/DeepSeek使用）**：$\text{SwiGLU}(x) = (xW_1) \odot \sigma(xW_2) \cdot xW_3$

### 1.5 Layer Normalization

| 类型 | 位置 | 公式 | 代表模型 |
|------|------|------|---------|
| Post-LN | 残差之后 | 原始Transformer | BERT |
| Pre-LN | 残差之前 | 更稳定 | GPT-3, LLaMA |
| RMS-Norm | 只做均方归一化 | 不减均值 | LLaMA, Qwen |

**RMS-Norm：** $\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma$，省去均值计算，更快

### 1.6 位置编码

| 方法 | 说明 | 代表 |
|------|------|------|
| 绝对位置编码（sin/cos）| 固定，不需要学习 | 原始Transformer |
| 可学习位置编码 | 嵌入向量形式 | BERT, GPT |
| RoPE（旋转位置编码）| 在Q/K上施加旋转，天然外推 | LLaMA, Qwen, DeepSeek |
| ALiBi | 在Attention分数加线性偏置 | BLOOM |

**RoPE核心思想：** 对 $q_m$ 和 $k_n$ 施加旋转矩阵 $R_m, R_n$，使得内积 $q_m^T k_n = f(q,k,m-n)$ 只依赖相对位置。

---

## 二、预训练技术

### 2.1 预训练目标对比

| 目标 | 典型模型 | 核心思想 |
|------|---------|---------|
| MLM（Masked Language Model）| BERT | 遮盖15%的token，预测被遮盖的词 |
| CLM（Causal Language Model）| GPT系列 | 自回归预测下一个token |
| Prefix LM | T5 | 前缀双向，后缀单向 |
| Span Corruption | T5 | 遮盖连续span |

### 2.2 SFT（监督微调）

**步骤：**
1. 准备指令格式数据（prompt + response）
2. 在预训练基座上做有监督训练
3. 计算response部分的交叉熵损失（prompt不计算loss）

**基座选择：** Base还是Chat？
- 对话任务 → Chat模型（已具备对话格式）
- 专业领域微调 → Base模型（灵活性更高）

**问题：** SFT后LLM变傻？
- 原因：灾难性遗忘（模型过度拟合SFT数据，遗忘预训练知识）
- 解决：保留通用数据混合、EWC（弹性权重巩固）、小学习率

### 2.3 RLHF（基于人类反馈的强化学习）

**完整流程：**
1. **SFT阶段**：在高质量对话数据上做监督微调
2. **奖励模型训练**：人类对比评分，训练Reward Model
3. **PPO强化学习**：用Reward Model的分数作为奖励，通过PPO算法优化策略模型

**KL散度约束：** $L = -\mathbb{E}[r(x,y)] + \beta \cdot D_{KL}(\pi_\theta || \pi_{ref})$

**GRPO（DeepSeek-R1使用）：**
- 无需价值函数（Value Function），通过组内采样估计基线
- 减少内存消耗，更适合大规模训练

### 2.4 LoRA（低秩微调）

**核心思想：** 参数更新具有低秩特性，将 $\Delta W$ 分解为 $\Delta W = AB$

$$W' = W_0 + \frac{\alpha}{r} AB$$

- $A \in \mathbb{R}^{d \times r}$（随机初始化），$B \in \mathbb{R}^{r \times k}$（全零初始化）
- $r$（rank）越小，训练参数越少
- $\alpha/r$ 是缩放系数

**面试要点：**
- 为什么B初始化为0？→ 保证初始时LoRA层不影响原模型输出
- LoRA vs 全参微调：LoRA节省显存，但上限低于全参微调

---

## 三、推理优化

### 3.1 KV Cache

**问题：** 自回归推理中，每次生成新token都要重新计算所有历史token的K、V

**解决：** 缓存历史token的K和V矩阵，只计算新token的注意力

**收益：** 推理时间从 $O(n^2)$ 降为 $O(n)$（对已生成部分）

**代价：** 显存占用 = `2 × n_layers × seq_len × n_heads × head_dim × dtype_bytes`

### 3.2 FlashAttention

**问题：** 标准Attention需要将 $n \times n$ 的注意力矩阵写入HBM，IO成本高

**解决：** 分块计算（Tiling），在SRAM中完成softmax和加权，避免频繁HBM访问

**版本：**
- FA1：分块计算，IO复杂度降到 $O(n^2/B)$
- FA2：进一步优化并行和工作划分
- FA3：针对Hopper架构优化

### 3.3 模型量化

| 格式 | 位宽 | 典型精度损失 | 应用 |
|------|------|------------|------|
| FP32 | 32bit | 基准 | 训练 |
| BF16 | 16bit | 极小 | 主流训练/推理 |
| FP16 | 16bit | 极小 | 推理 |
| INT8 | 8bit | 小 | 量化推理 |
| INT4 | 4bit | 中 | 极限压缩 |

**量化类型：**
- **PTQ（训练后量化）**：无需重训，直接量化权重（GPTQ, AWQ）
- **QAT（量化感知训练）**：训练时模拟量化效果

**LLM.int8()：** 对异常值（outlier）保持FP16，其余INT8，混合精度推理

### 3.4 投机采样（Speculative Decoding）

**思路：**
1. 用小草稿模型（Draft Model）快速生成k个候选token
2. 用大目标模型（Target Model）并行验证这k个token
3. 接受前m个合法token，丢弃不合法的

**收益：** 在小模型高命中率时，大幅提升吞吐量（Target Model处理一批token vs 逐个推理）

### 3.5 分布式推理

| 并行方式 | 说明 |
|---------|------|
| 张量并行（TP）| 矩阵按列/行切分，每卡存一部分权重 |
| 流水线并行（PP）| 不同层分配到不同卡 |
| 数据并行（DP）| 每卡存完整模型，数据不同 |

---

## 四、主流模型架构对比

| 模型 | 架构 | 特色 |
|------|------|------|
| GPT-3/4 | Decoder-only | 海量参数，RLHF对齐 |
| LLaMA2/3 | Decoder-only | RoPE + GQA + SwiGLU + RMSNorm |
| Qwen3 | Decoder-only | MoE版本，Thinking模式切换 |
| DeepSeek-V3 | MoE + Decoder | MLA（Multi-head Latent Attention），大幅降低KV Cache显存 |
| DeepSeek-R1 | 推理模型 | GRPO强化学习，Long-CoT，冷启动+RL |
| BERT | Encoder-only | MLM + NSP预训练，双向注意力 |
| T5 | Encoder-Decoder | Text-to-Text统一框架 |

### Decoder-only 为什么成主流？

1. 自回归生成任务天然适合
2. 没有双向注意力的"低秩问题"（Encoder双向Attention信息冗余）
3. 训练效率高（单次前向可同时训练所有位置）
4. 在同等参数量下，Decoder-only优于Encoder-Decoder

### DeepSeek 核心创新

**MLA（Multi-head Latent Attention）：**
- 将K/V压缩到低维潜空间（Latent）
- 大幅减少KV Cache显存（从 $n_{kv}=n_q$ 降到极小维度）

**MoE设计：**
- 每个token只激活少量专家（稀疏激活）
- 计算量远小于Dense同规模模型

**训练流程（R1）：**
1. DeepSeek-V3（SFT + RLHF）
2. DeepSeek-R1-Zero（纯RL，无SFT冷启动）
3. DeepSeek-R1（冷启动SFT数据 + RL + 拒绝采样 + SFT + RL）

---

## 五、高频面试题

1. **Transformer中self-attention为什么要scaled？**
   → 防止d_k过大时点积过大，softmax饱和，梯度消失

2. **KV Cache是什么？对显存的影响？**
   → 缓存历史K/V避免重复计算；显存随序列长度线性增长

3. **LoRA的rank如何选取？**
   → 通常4~64；任务越复杂/微调变化越大，rank越大；alpha/rank固定缩放

4. **为什么现在LLM大多用Decoder-only？**
   → 生成任务天然适合；训练效率高；同参数量性能更好

5. **RLHF的GRPO vs PPO的区别？**
   → GRPO无需Value Network，组内采样估计baseline，节省显存；PPO需要额外的Critic Model

6. **FlashAttention解决了什么问题？**
   → IO bound问题：避免将n²注意力矩阵写到HBM，在SRAM分块计算

7. **INT4量化的trade-off？**
   → 显存/计算量大幅降低，但精度下降明显；适合推理不适合训练

8. **DeepSeek-R1的训练过程？**
   → 冷启动数据SFT → GRPO强化推理 → 拒绝采样+SFT → 全场景RL

9. **RoPE的优势？**
   → 外推能力强，相对位置感知，无需额外参数

10. **SFT之后LLM傻了怎么办？**
    → 混合通用数据、减小学习率、EWC正则、多轮渐进微调
