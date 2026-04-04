# HLLM: Enhancing Sequential Recommendations via Hierarchical Large Language Models

> 来源：arXiv 2409.12740 | 年份：2024 | 领域：ads/02_rank（LLM+推荐/序列推荐）

## 问题定义

将 LLM 用于推荐系统面临关键矛盾：

1. **语义理解需求**：推荐系统需要深入理解物品的文本描述（标题、属性、评论），LLM 在此有天然优势
2. **序列建模需求**：用户行为序列可达 1000+ 步，需要捕捉长期偏好和短期意图
3. **效率瓶颈**：单个 LLM 同时处理物品理解 + 序列建模，输入长度 = 序列长度 × 每物品 token 数，计算量爆炸
4. **现有方案不足**：
   - 纯 ID 模型（SASRec）：缺乏语义理解能力
   - 单一 LLM（P5, GPT4Rec）：效率低下，无法处理长序列
   - Embedding 注入（UniSRec）：语义信息被压缩损失

**HLLM 核心思路**：用层级分工解决效率和效果的矛盾——小 LLM 理解单个物品，大 LLM 建模用户序列。

## 模型结构图

```
┌───────────────────────────────────────────────────────┐
│                   HLLM Architecture                    │
│                                                       │
│   User Behavior Sequence: [item₁, item₂, ..., itemₙ]  │
│                                                       │
│   ┌─────┐  ┌─────┐  ┌─────┐       ┌─────┐           │
│   │Item₁│  │Item₂│  │Item₃│  ...  │Itemₙ│           │
│   │Text │  │Text │  │Text │       │Text │           │
│   └──┬──┘  └──┬──┘  └──┬──┘       └──┬──┘           │
│      ↓        ↓        ↓              ↓               │
│   ┌──┴──┐  ┌──┴──┐  ┌──┴──┐       ┌──┴──┐           │
│   │Small│  │Small│  │Small│  ...  │Small│ ← Item LLM │
│   │ LLM │  │ LLM │  │ LLM │       │ LLM │   (共享)   │
│   └──┬──┘  └──┴──┘  └──┬──┘       └──┬──┘           │
│      ↓        ↓        ↓              ↓               │
│   [CLS₁]  [CLS₂]  [CLS₃]   ...  [CLSₙ]             │
│      ↓        ↓        ↓              ↓    Projector  │
│   [emb₁]  [emb₂]  [emb₃]   ...  [embₙ]  ← 维度对齐  │
│      └────────┴────────┴──────────────┘               │
│                       ↓                               │
│              ┌────────┴────────┐                      │
│              │   Large LLM     │ ← User-level LLM    │
│              │  (Sequence      │    建模用户偏好        │
│              │   Modeling)     │                      │
│              └────────┬────────┘                      │
│                       ↓                               │
│              Next Item Prediction                     │
└───────────────────────────────────────────────────────┘
```

## 核心方法与完整公式

### 公式1：Item-level LLM 编码

$$
h_i = \text{ItemLLM}(\text{Tokenize}(t_i))[-1]
$$

$$
e_i = W_p \cdot h_i + b_p
$$

**解释：**
- $t_i$：第 $i$ 个物品的文本描述（标题+属性+品类）
- $\text{ItemLLM}$：小型语言模型（如 LLaMA-1B / Qwen-1.5B）
- $h_i$：最后一个 token 的隐藏状态（[CLS] 表示）
- $W_p, b_p$：线性 Projector，将 Item LLM 维度对齐到 User LLM 维度

### 公式2：User-level LLM 序列建模

$$
P(item_{n+1} | item_1, \ldots, item_n) = \text{UserLLM}(e_1, e_2, \ldots, e_n)
$$

**解释：**
- $e_i$：Item LLM 输出的物品 embedding（经 Projector 对齐后）
- $\text{UserLLM}$：大型语言模型（如 LLaMA-7B），处理 embedding 序列（非文本 token）
- 输出：下一个物品的概率分布

### 公式3：两级知识蒸馏

$$
\mathcal{L}_{distill} = \text{KL}(P_{user} \| P_{item}) + \lambda \cdot \text{MSE}(e_i^{user}, e_i^{item})
$$

**解释：**
- $P_{user}$：User LLM 学到的物品偏好分布
- $P_{item}$：Item LLM 的输出分布
- 第一项：分布级蒸馏（用户偏好传递给物品理解）
- 第二项：表示级蒸馏（embedding 空间对齐）
- $\lambda$：蒸馏损失权重

### 公式4：训练损失

$$
\mathcal{L} = \mathcal{L}_{rec} + \alpha \cdot \mathcal{L}_{distill}
$$

$$
\mathcal{L}_{rec} = -\sum_{n=1}^{N} \log P(item_{n+1} | item_1, \ldots, item_n)
$$

**解释：**
- $\mathcal{L}_{rec}$：下一个物品预测的交叉熵损失（序列推荐主任务）
- $\alpha$：蒸馏损失权重（通常 0.1-0.5）

## 与基线方法对比

| 方法 | 核心区别 | 优势 | 劣势 |
|------|---------|------|------|
| **SASRec** | 纯 ID + Self-Attention | 简单高效 | 无语义理解，冷启动差 |
| **BERT4Rec** | 双向 Attention + MLM | 双向建模 | 无语义，训练慢 |
| **P5/GPT4Rec** | 单一 LLM 全处理 | 统一框架 | 效率极低，长序列不可行 |
| **UniSRec** | 预训练 Embedding 注入 | 跨域迁移 | 语义被压缩损失 |
| **HLLM** | 层级 LLM（小+大） | 效率+效果 | 需要两个 LLM，架构复杂 |

## 实验结论

- **Amazon 商品推荐**：NDCG@10 提升约 10%（vs SASRec 等序列推荐基线）
- **vs 单一 LLM**：相同计算预算下效果提升约 5%
- **推理效率**：层级设计使推理速度提升约 3x（Item LLM 结果可缓存）
- **消融**：去掉 Item LLM 退化为纯 ID 模型（-8% NDCG），去掉 User LLM 退化为无序列建模（-12% NDCG）

## 工程落地要点

1. **Item LLM 缓存**：每个物品的 embedding 可预计算并缓存（物品更新频率远低于用户行为），大幅降低在线计算
2. **序列截断策略**：User LLM 上下文窗口有限（2048-4096 tokens），超长序列需截断——最近的 N 个行为 vs 基于重要性采样
3. **Projector 设计**：简单线性层 vs MLP，实验表明 2 层 MLP + ReLU 效果最好
4. **物品文本构建**：需要维护物品描述数据库，格式化为 "标题 | 品类 | 属性1 | 属性2"
5. **训练策略**：Item LLM 先冻结（预热 User LLM），再联合微调，避免梯度冲突

## 面试考点

**Q1：为什么用层级 LLM 而不是单一 LLM？**
> 单一 LLM 处理 N 个物品 × 每物品 M tokens = NM tokens，复杂度 $O((NM)^2)$ 不可接受。层级设计：Item LLM 处理 M tokens/物品（可缓存），User LLM 处理 N embeddings，总复杂度 $O(NM^2 + N^2)$，远优于 $O(N^2M^2)$。

**Q2：Item LLM 和 User LLM 如何选择模型大小？**
> Item LLM 用小模型（1B-3B），因为物品理解是相对简单的任务且需要高吞吐缓存。User LLM 用大模型（7B-13B），因为序列偏好建模更复杂。大小差 3-7 倍是常见配置。

**Q3：序列推荐模型（SASRec/BERT4Rec/GRU4Rec）的差异？**
> SASRec：单向 Self-Attention（causal mask），适合 next-item prediction。BERT4Rec：双向 Attention + MLM 训练，捕捉双向依赖但不能直接用于生成。GRU4Rec：RNN 顺序建模，效率低但参数少。

**Q4：LLM 推荐系统中如何处理物品 ID 与文本的对应？**
> 维护 ID-to-text 映射表，查找物品描述作为 LLM 输入。挑战：① 映射表需实时更新（新品上架）② 文本质量参差不齐需要清洗 ③ 多语言物品需要统一编码。

**Q5：HLLM 的冷启动优势体现在哪里？**
> 传统 ID-based 模型对新物品（无交互历史）无法生成 embedding。HLLM 通过 Item LLM 从物品文本生成 embedding，新物品只需有文本描述即可获得高质量表示。这是 LLM 推荐的核心优势之一。

**Q6：如何评估 LLM 推荐系统的效率-效果 tradeoff？**
> 关键指标：① NDCG/HR 提升幅度 vs 推理延迟增加 ② Item LLM 缓存命中率（通常 >95%）③ User LLM 的 tokens/second ④ GPU 显存占用 ⑤ 相比纯 ID 模型的成本增加倍数。通常需要效果提升 >5% 才值得 3x 的成本增加。
