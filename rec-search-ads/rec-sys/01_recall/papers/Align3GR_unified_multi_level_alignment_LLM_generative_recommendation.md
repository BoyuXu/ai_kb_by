# Align³GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation

> 来源：https://arxiv.org/abs/2511.11255 | 领域：rec-sys | 学习日期：20260329

## 问题定义

LLM 在语言建模任务中表现出色，但直接用于推荐系统时存在**语义与行为双重不对齐**问题：
1. **语义不对齐**：LLM 的 token 空间与推荐系统的 item ID 空间差异大，简单映射会造成语义漂移
2. **行为不对齐**：LLM 的 next-token 预测目标与推荐系统的隐式偏好建模目标不一致
3. **偏好不对齐**：静态训练无法持续适配用户的动态偏好变化

目标：在 token、行为建模、偏好三个层级统一对齐 LLM 与推荐系统。

## 核心方法与创新点

### 三层级对齐框架

**层级 1：Token-level 对齐 — 双重 SCID 编码（Dual SCID Tokenization）**
- 引入 Semantic Collaborative ID（SCID），同时融合语义信息和协同过滤信号
- **双重编码**：同时为 user 和 item 生成 SCID（以往方法只对 item 编码）
- 使用 RQ-VAE（3层，每层 codebook 大小 256）进行向量量化
- U2I 对齐损失函数：$\mathcal{L}_{U2I} = \alpha \mathcal{L}_{CF} + \gamma \mathcal{L}_{quant}$
- 训练策略：先设 α=1, γ=0 稳定行为对齐，再切换 α=0.1, γ=1 优化量化

**层级 2：行为建模对齐 — 多任务 SFT**
- 在 LC-Rec 基础上引入两个关键增强：
  1. 将 user SCID token 注入所有任务 prompt，提供更丰富的上下文
  2. 双向对齐任务（$B_2$）：text→SCID 和 SCID→text 的互转任务
- 多任务包括：Sequential Item Prediction、Asymmetric Item Prediction、Personalized Preference Inference 等

**层级 3：偏好对齐 — 渐进式 DPO（Progressive DPO）**
- **SP-DPO**（Self-Play DPO）：利用模型自身生成数据，按难度划分 Easy/Medium/Hard 三阶段
  - 基于 prefix-ngram 匹配指标衡量 chosen/rejected 的相似度
- **RF-DPO**（Real-world Feedback DPO）：利用真实用户反馈（点击/曝光/明确不喜欢）
- 使用 Softmax-DPO 变体，每个样本含 1 个 chosen + 20 个 rejected
- 渐进学习目标：$\mathcal{L}(\pi_\theta^i, \pi_{ref}^i) = -\mathbb{E} \log \sigma(-\log \sum_{y_l \in Y_l} \exp(\beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} - \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}))$

## 实验结论

**数据集**：Amazon Instruments、Beauty、Yelp（三个领域）

**公开数据集离线结果**（对比 SOTA EAGER-LLM）：
| 数据集 | Recall@10 提升 | NDCG@10 提升 |
|--------|---------------|-------------|
| Instruments | +17.8% | +20.2% |
| Beauty | +19.8% | +15.3% |
| Yelp | +19.3% | +27.9% |

**工业 A/B 测试**（40M+ 用户，10% 流量，多周）：
- Recall@100：Baseline 0.218 → TIGER 0.229 → **Align³GR 0.242**
- 广告收入提升：**+1.432%**（统计显著）
- 优于 industrial 双塔召回 baseline 和 TIGER

**消融实验关键发现**：
- Dual SCID（vs. 仅 Item 侧）：Recall@10 从 0.1322 → 0.1442（+9.1%）
- 渐进式 RF-DPO（vs. 静态 Softmax-DPO）：Recall@10 从 0.1295 → 0.1442（+11.3%）

## 工程落地要点

1. **分阶段训练**：三个对齐层级串行训练，每层收敛后再训下一层，避免相互干扰
2. **用户 SCID 双向部署**：user 和 item 模块独立部署，各自生成 SCID 供下游使用
3. **渐进式 DPO 数据构建**：无需大量标注，Self-Play 自动生成多难度负样本
4. **RQ-VAE 参数**：3层，codebook 256×32，扩充 LLM 词表以避免 OOV
5. **LoRA 微调**：在 Llama2-7B 上使用 LoRA 降低显存，4块 GPU 可训练
6. **工业集成**：Align³GR 在广告推荐场景中直接替换召回模块，对接现有精排

## 常见考点

**Q1：Align³GR 的三层对齐分别解决什么问题？**
> A：Token 层对齐 ID 空间（语义+协同信号融合）；行为建模层对齐训练目标（多任务 SFT 注入用户偏好）；偏好层解决静态偏好上限问题（渐进 DPO 动态适配）。三层递进，逐步缩小 LLM 与 RS 的 gap。

**Q2：为什么要同时对 user 和 item 做 SCID 编码？**
> A：传统方法仅对 item 编码，user 侧偏好用历史序列表示，两者在同一 token 空间中无法直接对齐。双重 SCID 使 user 和 item 处于同一协同语义空间，U2I 损失可以显式拉近匹配对的表示距离。

**Q3：渐进式 DPO 中 Self-Play 和 Real-world Feedback 各扮演什么角色？**
> A：SP-DPO 先用模型自身生成数据，通过 prefix-ngram 度量难度，解决初期偏好数据稀疏问题；RF-DPO 再引入真实用户行为（点击/曝光/厌恶）作为偏好信号，对齐真实业务目标。两者协同，SP-DPO 打底，RF-DPO 提精。

**Q4：Softmax-DPO 相比标准 DPO 有何改进？**
> A：标准 DPO 每条样本只有 1 个 rejected，Softmax-DPO 支持多个 rejected（本文 20 个），通过 softmax 聚合所有 rejected 的 log-ratio，提供更稳定的梯度估计，对生成式推荐的大候选空间更有效。

**Q5：在实际工业部署中，Align³GR 如何与传统双塔召回对接？**
> A：Align³GR 生成的候选集（通过 beam search，beam=20）与双塔召回并联，在精排前合并。在工业测试中 Recall@100 提升说明其补充了双塔未覆盖的个性化候选，最终广告收入提升 1.432%。

## 模型架构详解

### 候选编码
- **Item 表示**：Semantic ID（层次化离散编码）或稠密向量 Embedding
- **编码方式**：RQ-VAE（残差量化）/ K-Means 聚类 / 端到端学习的 Token 序列
- **多模态融合**：文本/图片/行为信号的统一表示空间

### 检索机制
- **生成式检索**：自回归解码器逐步生成 Item Token 序列
- **向量检索**：双塔编码 + ANN 索引（HNSW/IVF-PQ）
- **混合召回**：多路检索结果的统一评分与去重

### 训练策略
- **正样本**：用户交互（点击/购买/收藏）
- **负采样**：In-batch Negatives + 难负例挖掘
- **对比学习**：InfoNCE Loss 拉近正样本、推远负样本
- **课程学习**：从简单到困难逐步增加负例难度

## 与相关工作对比

| 维度 | 生成式召回 | 双塔向量召回 | 传统倒排 |
|------|-----------|------------|---------|
| 冷启动 | 好（内容特征） | 中（需行为） | 差 |
| 索引维护 | 无需显式索引 | 需 ANN 索引 | 需倒排表 |
| 推理延迟 | 中（自回归） | 低（一次编码） | 低 |
| 可扩展性 | 亿级 | 亿级 | 百万级 |
| 多模态 | 原生支持 | 需要适配 | 困难 |

## 面试深度追问

- **Q: Semantic ID 的设计思路和优势？**
  A: 将 Item 映射为离散 Token 序列（类似自然语言），使推荐问题转化为序列生成。优势：1) 天然支持自回归生成；2) 层次化结构（粗→细）提升检索效率；3) 避免连续向量的 ANN 近似误差。

- **Q: 生成式召回如何处理新物品？**
  A: 1) 内容特征驱动的 Semantic ID 分配（新物品基于属性分配 Token）；2) 增量学习更新 Codebook；3) 备用的 Content-based 召回通道兜底。

- **Q: 多路召回的融合策略？**
  A: 1) 统一打分：所有通道候选用同一模型重新打分；2) 配额分配：各通道按历史表现分配固定配额；3) 加权融合：考虑通道多样性的加权排序。

- **Q: 如何衡量召回质量？**
  A: 离线：Recall@K, HR@K, NDCG@K。在线：端到端 CTR/GMV 提升 + 召回覆盖率 + 新颖性。注意 K 值要与下游排序的候选集大小匹配。
