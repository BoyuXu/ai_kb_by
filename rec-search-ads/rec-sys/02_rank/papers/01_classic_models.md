# 经典模型常见考点

> 来源：AIGC-Interview-Book 经典模型章节
> 更新：2026-03-12

---

## 一、BERT

### 核心架构

- **类型：** Encoder-only（双向注意力）
- **预训练任务：**
  1. **MLM（Masked Language Model）：** 随机遮盖15%的token，预测被遮盖的词（80%用[MASK]，10%随机词，10%原词）
  2. **NSP（Next Sentence Prediction）：** 判断两句话是否相邻（后来被证明意义不大）
- **核心创新：** 双向上下文（vs GPT单向）

### BERT的变体

| 模型 | 改进点 |
|------|--------|
| RoBERTa | 去掉NSP，更大batch，更长训练 |
| ALBERT | 参数共享，Sentence Order Prediction替代NSP |
| DistilBERT | 知识蒸馏，BERT的60%参数，95%性能 |
| DeBERTa | 解耦位置和内容注意力 |

### 面试要点

- Q：BERT为什么不适合生成任务？
  → 双向注意力导致信息泄露；MLM目标和生成目标不匹配

- Q：BERT的[CLS] token作用？
  → 全局表示，用于分类任务的池化层输入

- Q：BERT中Masked比例为什么是15%？
  → 太高影响上下文理解，太低信号稀疏；实验最优

---

## 二、GPT 系列

### 核心架构

- **类型：** Decoder-only（单向因果注意力）
- **预训练：** CLM（Causal Language Modeling）—— 预测下一个token
- **训练流程：** 预训练 → SFT → RLHF（GPT-3.5/4）

### GPT系列演进

| 版本 | 参数量 | 关键特点 |
|------|--------|---------|
| GPT-1 | 117M | Decoder-only，零样本迁移 |
| GPT-2 | 1.5B | 拒绝公开，强大生成能力 |
| GPT-3 | 175B | 少样本提示，In-Context Learning |
| InstructGPT | 175B | RLHF对齐，减少幻觉 |
| GPT-4 | ~1T+ | 多模态，更强推理 |

### In-Context Learning（ICL）

- 不更新参数，通过示例在上下文中"学习"
- Zero-shot / Few-shot
- 原理：元学习，大模型在预训练时隐式学习了任务结构

---

## 三、T5

### 核心架构

- **类型：** Encoder-Decoder
- **框架：** Text-to-Text —— 所有任务统一为"输入文本→输出文本"
  - 翻译："translate English to German: The house is wonderful."
  - 分类："sentiment: This movie is great."

### 预训练任务

- **Span Corruption：** 遮盖连续span，用哨兵token替代，预测被遮盖的span
- 数据：C4（Colossal Clean Crawled Corpus）

---

## 四、LLaMA 系列

### LLaMA2 架构特点

| 特性 | 说明 |
|------|------|
| RoPE | 旋转位置编码，外推性好 |
| GQA | Grouped Query Attention，降低KV Cache |
| SwiGLU | 门控激活函数，比ReLU效果好 |
| RMSNorm | Pre-Norm，不需要减均值 |
| 更大上下文 | 4096 → 8192（LLaMA2） |

### 面试常问

- Q：LLaMA和BERT的主要区别？
  → LLaMA是Decoder-only用于生成，BERT是Encoder-only用于理解

- Q：为什么LLaMA用Pre-Norm而非Post-Norm？
  → Pre-Norm训练更稳定（梯度流更好），Post-Norm需要更精细的初始化

---

## 五、ViT（Vision Transformer）

### 核心思想

- 将图像分块（patch），每个patch展平后做线性投影，作为"视觉token"
- 加入[CLS] token用于分类
- 标准Transformer Encoder处理

**流程：**
```
图像 → 分块(16×16) → 线性投影 → + 位置编码 → Transformer → [CLS] → 分类
```

### 关键对比：CNN vs ViT

| 对比点 | CNN | ViT |
|--------|-----|-----|
| 归纳偏置 | 强（局部性、平移不变性）| 弱（纯数据驱动）|
| 小数据 | 更好 | 容易过拟合 |
| 大数据 | 好 | 更好（超越CNN）|
| 特征局部性 | 局部感受野逐步扩大 | 全局自注意力 |

### 面试要点

- Q：ViT为什么需要大量数据？
  → 缺少CNN的归纳偏置（局部性假设），需要更多数据学习这些先验

- Q：ViT的patch size对性能影响？
  → Patch越小，序列越长，计算量$O(n^2)$；Patch越大，细粒度信息丢失

---

## 六、强化学习关键算法（面试补充）

### PPO（Proximal Policy Optimization）

$$
L^{CLIP} = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
$$

- $r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$
- Clip限制更新步长，避免策略崩塌

### GRPO（DeepSeek-R1使用）

- 无需Critic Model（Value Function）
- 对同一问题采样多个输出，组内奖励归一化作为优势估计
- 公式：$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$

---

## 七、高频常见题

1. **BERT和GPT的最大区别？**
   → BERT双向Encoder（理解），GPT单向Decoder（生成）

2. **T5相比BERT的优势？**
   → 统一text-to-text框架，支持生成；Encoder-Decoder可处理变长输出

3. **ViT和CNN的区别？**
   → ViT无归纳偏置，大数据下更强；CNN局部特征提取好，小数据表现好

4. **LLaMA为什么用GQA而非MHA？**
   → 减少KV Cache显存，多个Query组共享Key/Value，推理更高效

5. **RLHF中的Reward Model是什么？**
   → 用人类偏好数据训练的模型，输入（prompt, response），输出标量奖励

6. **DeepSeek-R1为什么不需要SFT冷启动就能推理（R1-Zero）？**
   → 纯RL从Base模型出发，奖励函数覆盖格式+正确性；但R1用冷启动SFT提升稳定性

7. **ViT的位置编码用什么？**
   → 可学习的1D位置编码（标准ViT），2D位置编码变体也有

8. **GPT的In-Context Learning为什么不需要更新参数？**
   → 大规模预训练中隐式学习了Meta-Learning能力，上下文示例提供任务归纳
