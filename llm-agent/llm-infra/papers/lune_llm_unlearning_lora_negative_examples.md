# LUNE: Efficient LLM Unlearning via LoRA Fine-Tuning with Negative Examples

> 来源：https://arxiv.org/abs/2512.07375 | 领域：llm-infra | 学习日期：20260401

---

## 问题定义

### 机器遗忘（Machine Unlearning）的必要性

LLM 在训练过程中从海量语料中吸收了大量知识，但这带来了严重的现实问题：

1. **隐私合规（GDPR 等）**：用户有权要求删除其个人信息（Right to be Forgotten）
2. **偏见与有害内容清除**：模型可能习得不当偏见，需定向消除
3. **知识纠错（Factual Correction）**：旧知识过时或错误，需要修正

### 现有方案的局限

| 方法 | 问题 |
|------|------|
| 全量重训（从头训练）| 计算成本极高（百亿参数模型需数周）|
| 梯度上升（Gradient Ascent）| 不稳定，容易导致模型整体退化 |
| 直接权重编辑（Weight Editing）| 如 ROME/MEMIT，精度有限，可能破坏相关知识 |
| 全量微调（Full Fine-Tuning）| 计算代价大，灾难性遗忘风险高 |

**核心需求**：需要一种**轻量、局部、有效**的遗忘方法，既能精准消除目标知识，又不影响模型整体能力。

### LUNE 的定位

LUNE（**L**oRA-based **U**nlearning with **N**egative **E**xamples）：使用 LoRA 框架，通过负样本微调实现高效 LLM 遗忘，计算量比全量微调/权重编辑低约**一个数量级**。

---

## 核心方法与创新点

### 1. Negative-Only LoRA 微调策略

**核心设计原则**：仅使用**负样本（Negative Examples）**进行 LoRA 微调，不引入正样本。

**什么是负样本**：对于需要遗忘的知识 K，负样本是将 K 替换为错误答案或空白的训练样本。

例如：
- 遗忘目标："{人物名} 的出生地是 {城市}"
- 负样本："{人物名} 的出生地是 [MASK]" 或 "我不知道 {人物名} 的出生地"

**为什么仅用负样本有效**：
- 正向微调（学习正确知识）→ 知识强化（反向）
- 负向微调（在目标知识上产生错误输出）→ 中间表征被干扰，目标知识被压制
- LoRA 的低秩约束自然限制了更新范围，防止过度遗忘

### 2. 局部化编辑（Localized Editing via LoRA）

**LoRA 遗忘的优势**：
- **主干网络冻结（Backbone Frozen）**：原始权重不变，只更新 A、B 矩阵
- **局部化改变**：低秩约束 r 限制了编辑的影响半径
- **可逆性**：移除 LoRA adapter 即可恢复原始模型

遗忘 LoRA 的数学形式：

$$
W'_{\text{unlearn}} = W_0 + \Delta W_{\text{unlearn}} = W_0 + B_{\text{unlearn}} \cdot A_{\text{unlearn}}
$$

其中 $A_{\text{unlearn}}, B_{\text{unlearn}}$ 通过负样本优化得到，目标是最大化在目标知识上的预测损失。

### 3. 中间表征定向抑制

LUNE 的特殊之处在于作用于**中间层表征（intermediate representations）**而非仅输出层：

- 选择 Transformer 中特定层（通常是中间 MLP 层）插入遗忘 LoRA
- 目标：在信息流中"拦截"目标知识的表达，而非只改变输出 token

**训练目标**：

$$
\mathcal{L}_{\text{unlearn}} = -\mathbb{E}_{(x, y^-) \sim \mathcal{D}_{\text{forget}}} \left[ \log p_\theta(y^- | x) \right]
$$

通过最大化模型在遗忘数据上生成负样本的概率（即让模型"学会说错"），从而压制原知识。

### 4. 稳定性保障：Retain Set 约束

为防止遗忘扩散影响无关知识，引入保留集（Retain Set）约束：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{unlearn}} + \lambda \cdot \mathcal{L}_{\text{retain}}
$$

$\mathcal{L}_{\text{retain}}$ 是模型在保留知识上的交叉熵损失，$\lambda$ 控制遗忘强度与知识保留的权衡。

---

## 实验结论

**对比基线**：全量微调遗忘、ROME、MEMIT（直接权重编辑）

**遗忘有效性**：LUNE 在多个事实遗忘任务上达到与全量微调**相当的遗忘效果**

**计算成本**（关键结论）：
| 方法 | 相对计算成本 |
|------|------------|
| 全量微调 | 1.0× (基准) |
| ROME/MEMIT | ~0.5× |
| **LUNE** | **~0.1×（约1/10）** |

**两大核心结论**：
1. **有效性**：LUNE 的遗忘精度媲美全量微调和 MEMIT
2. **效率**：计算与内存成本降低约**一个数量级**

**多任务泛化**：在事实遗忘、关系遗忘、序列知识遗忘多类任务上一致有效。

---

## 工程落地要点

### 1. 遗忘 Pipeline 设计

```python
# LUNE 实现框架
class LUNEUnlearner:
    def __init__(self, base_model, lora_rank=8, target_layers=None):
        # 为选定层添加遗忘 LoRA adapter
        self.model = add_lora_adapters(base_model, lora_rank, target_layers)
        self.freeze_backbone()  # 冻结基础权重
    
    def prepare_negative_examples(self, forget_targets):
        """将遗忘目标转化为负样本训练数据"""
        negative_data = []
        for (query, answer) in forget_targets:
            # 替换为"我不知道"或随机错误答案
            negative_data.append((query, "I don't have information about that."))
        return negative_data
    
    def unlearn(self, forget_data, retain_data, epochs=3, lambda_retain=0.5):
        optimizer = AdamW(self.model.lora_parameters(), lr=1e-4)
        for epoch in range(epochs):
            # 负样本损失（遗忘）
            neg_loss = compute_negative_loss(forget_data)
            # 保留集损失（防止过度遗忘）
            retain_loss = compute_retain_loss(retain_data)
            # 组合损失
            total_loss = neg_loss + lambda_retain * retain_loss
            optimizer.step(total_loss)
    
    def verify_unlearning(self, forget_queries, threshold=0.1):
        """验证遗忘效果"""
        for query in forget_queries:
            prob = model.get_probability(query, original_answer)
            assert prob < threshold, f"遗忘不彻底：{query}"
```

### 2. 遗忘验证指标

关键评估维度：
- **遗忘率（Forget Rate）**：目标知识在模型输出中的消失程度
- **保留率（Retain Rate）**：无关知识是否受到影响
- **模型流畅性（Fluency）**：遗忘后模型生成质量
- **泛化遗忘（Generalization Forget）**：仅遗忘少量样本，是否能泛化到类似知识

### 3. 合规场景部署

**GDPR Right to be Forgotten 流程**：
1. 用户提交遗忘请求（提供其个人信息相关的训练样本）
2. 构建负样本数据集（将其个人信息替换为空白/错误）
3. 运行 LUNE（约 1/10 全量微调成本）
4. 验证遗忘有效性（使用多组探测问题）
5. 记录遗忘操作日志（合规存档）

### 4. 关键参数调优

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| LoRA rank (r) | 8-32 | 越大遗忘能力越强，但影响范围越广 |
| 目标层数 | 中间 1/3 层 | 通常选 MLP 层，避免最后几层（输出相关）|
| λ_retain | 0.3-0.8 | 越大保留效果越好，遗忘可能不彻底 |
| 负样本类型 | "不知道" 式 | 比随机错误答案更稳定 |
| 训练 epoch | 3-5 | 过多会过度干扰，过少效果不足 |

### 5. 与 RAG 遗忘的对比

另一种"遗忘"方案是 RAG 层面的过滤（如 HippoRAG 2 删除图节点）：
- RAG 过滤：适合知识库管理，不修改模型权重，但无法处理模型"记住"的知识
- LUNE：处理模型参数中编码的知识，覆盖更彻底但成本更高

---

## 常见考点

**Q1：什么是机器遗忘（Machine Unlearning）？为什么对 LLM 特别重要？**

A：机器遗忘是让模型"忘记"特定训练数据或知识的技术。对 LLM 特别重要原因：(1) GDPR/隐私法规要求删除个人数据；(2) 模型可能记住并泄露训练集中的私人信息（Membership Inference Attack 风险）；(3) 有害/偏见内容需要定向清除；(4) 商业知识版权争议。传统方案是全量重训，成本极高（GPT-4 级模型训练数亿美元），LUNE 提供了 1/10 成本的轻量替代。

**Q2：为什么 LUNE 使用 LoRA 而非全量微调来实现遗忘？**

A：三个核心原因：(1) **局部性**：LoRA 低秩约束限制权重更新范围，防止遗忘扩散影响无关知识；(2) **效率**：只需更新 r×(d_in+d_out) 参数，比全量微调少 100x+；(3) **可逆性**：LoRA adapter 可随时移除恢复原始模型，便于审计和回滚。全量微调一旦修改权重即不可逆，且容易产生灾难性遗忘。

**Q3：负样本（Negative Examples）在遗忘中的作用机制是什么？**

A：负样本遗忘通过**反向强化**压制目标知识：模型在训练时最大化在目标知识 query 上生成错误输出的概率（最大化 log p(y^- | x)）。这会干扰模型内部对目标知识的表征，导致对应的中间激活值改变，从而使原始知识在推理时无法被正确激活。关键是 LoRA 的低秩约束保证这种干扰是局部的，不会过度传播。

**Q4：遗忘和微调的目标函数有何不同？如何防止遗忘"过度"？**

A：标准微调目标：$\min \mathcal{L} = -\mathbb{E}[\log p(y|x)]$（最大化正确答案概率）。遗忘目标：$\max \mathcal{L} = -\mathbb{E}[\log p(y^-|x)]$（最大化错误输出概率，即最小化原正确答案概率）。防止过度遗忘：引入 Retain Set 约束 $\mathcal{L}_{\text{retain}}$，通过 $\lambda$ 参数平衡，在遗忘强度和知识保留之间找到最优点。实践中还需监控 PPL（困惑度）确保模型整体生成质量不退化。
