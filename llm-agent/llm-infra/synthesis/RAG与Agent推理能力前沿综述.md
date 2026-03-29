# RAG与Agent推理能力前沿综述

> 整理日期：20260329 | 领域：llm-infra | 覆盖论文：5篇

## 📖 综述摘要

本综述覆盖LLM基础设施中两个核心方向的最新进展：**RAG（检索增强生成）** 与 **Agent推理能力**。通过分析DeepSeek-R1、Qwen3、Collab-RAG、多跳RAG以及推荐Agent五篇代表性工作，提炼出当前领域的核心技术范式、工程实践要点和未来趋势。

---

## 🗺️ 技术版图

```
LLM基础设施技术版图
├── 推理能力（Reasoning）
│   ├── RL训练：DeepSeek-R1 (GRPO), Qwen3 (四阶段)
│   ├── 长CoT：自发涌现 vs 蒸馏注入
│   └── 推理预算控制：Thinking Budget (Qwen3)
│
├── 检索增强（RAG）
│   ├── 单跳RAG → 多跳RAG → 自适应RAG
│   ├── 白盒SLM + 黑盒LLM协作：Collab-RAG
│   └── Agent+工具RAG：ReAct-style
│
└── 应用层：推荐系统
    ├── LLM推荐Agent（工具调用+推理）
    └── 两阶段架构（传统召回+LLM精排）
```

---

## 📐 核心公式

### 公式1：GRPO（Group Relative Policy Optimization）目标函数

DeepSeek-R1和Qwen3均使用GRPO进行RL训练：

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\min\left(\frac{\pi_\theta}{\pi_{old}} \cdot \hat{A}_i, \text{clip}\left(\frac{\pi_\theta}{\pi_{old}}, 1-\varepsilon, 1+\varepsilon\right)\cdot\hat{A}_i\right) - \beta \cdot D_{KL}(\pi_\theta||\pi_{ref})\right]$$

其中组内相对优势估计：

$$\hat{A}_i = \frac{r_i - \mu(\{r_1,...,r_G\})}{\sigma(\{r_1,...,r_G\})}$$

**关键参数**：
- DeepSeek-R1: $G=16$, $\varepsilon=10$, $\beta=0.001$
- Qwen3: $G$ 更大（大batch + 高rollout数），熵控制代替固定$\varepsilon$

---

### 公式2：Collab-RAG迭代DPO优化目标

$$\mathcal{L}_{IDPO}(\theta^{(m+1)}) = -\mathbb{E}_{(x,Q^+,Q^-)\sim\mathcal{D}^{(m)}}\left[\log\sigma\left(\beta\log\frac{\pi^{(m+1)}_\theta(Q^+|x)}{\pi^{(m)}_\theta(Q^+|x)} - \beta\log\frac{\pi^{(m+1)}_\theta(Q^-|x)}{\pi^{(m)}_\theta(Q^-|x)}\right)\right]$$

**关键设计**：使用前一轮模型作为参考（非固定初始参考），实现渐进式偏好优化。

---

### 公式3：多跳RAG的迭代检索

多跳问答中，第 $k$ 步检索查询由前序推理自动生成：

$$q_k = f_{LLM}(Q, c_1,...,c_{k-1}, e_1,...,e_{k-1})$$

累积证据集合：

$$\mathcal{E}^* = \bigcup_{k=1}^{K} R(q_k, \mathcal{D})$$

自适应停止条件：

$$\text{stop} = \mathbb{1}\left[\text{conf}(a \mid Q, \mathcal{E}) > \tau\right]$$

**实践发现**：多跳收益符合对数律，$\text{EM}(K) \approx \text{EM}(1) + \beta\log K$，$K=3$后收益递减。

---

### 公式4：RAG性能-效率权衡指标

综合评估RAG系统的效率：

$$\text{Score}_{eff} = \frac{\text{NDCG@K}}{\sqrt{\text{avg\_retrievals}} \times \text{latency\_ms}}$$

**工程目标**：在约束延迟（<200ms）下最大化 $\text{Score}_{eff}$。

---

### 公式5：知识蒸馏的KL散度目标

Qwen3的On-policy蒸馏对齐教师logits：

$$\mathcal{L}_{distill} = D_{KL}(\pi_{teacher}(y|x) || \pi_{student}(y|x)) = \sum_y \pi_{teacher}(y|x)\log\frac{\pi_{teacher}(y|x)}{\pi_{student}(y|x)}$$

**关键发现**：此方法比直接RL训练省**10倍GPU hours**，Pass@64（探索能力）也显著更优。

---

## 💡 核心技术范式对比

### 推理能力提升方法对比

| 方法 | 代表工作 | 训练范式 | 计算效率 | 性能上限 |
|------|---------|---------|---------|---------|
| 纯RL（无SFT热身） | DeepSeek-R1-Zero | GRPO on base | 高成本 | 探索能力强 |
| 多阶段RL+SFT | DeepSeek-R1/Qwen3 | Cold Start→RL→SFT→RL | 中等 | 最优 |
| 直接蒸馏 | R1-Distill系列 | SFT only | 低成本 | 受教师限制 |
| On-policy蒸馏 | Qwen3小模型 | KL散度对齐 | 低成本 | 接近教师 |

### RAG架构演进

| 代 | 代表方法 | 检索策略 | 推理能力 | 适用场景 |
|---|---------|---------|---------|---------|
| 1代 | Lewis et al. RAG | 单次检索 | 无 | 简单QA |
| 2代 | IRCoT, FLARE | 迭代检索 | CoT | 多跳QA |
| 3代 | Collab-RAG, Search-R1 | 学习式检索 | Agent推理 | 复杂QA |
| 4代 | 自适应多跳 | 按需检索 | 端到端 | 生产环境 |

---

## 🔬 关键实验数据汇总

### 推理能力benchmark

| 模型 | AIME'24 | MATH-500 | LiveCodeBench | 参数规模 |
|-----|---------|---------|---------------|--------|
| GPT-4o | 9.3% | 74.6% | 32.9% | - |
| DeepSeek-R1 | 79.8% | 97.3% | 65.9% | 671B |
| Qwen3-235B-A22B | 85.7% | ~98% | 70.7% | 235B(22B激活) |
| Qwen3-32B | 81.4% | 97.2% | 65.7% | 32B |
| Qwen3-8B | 76.0% | 97.4% | 57.5% | 8B |
| R1-Distill-7B | 55.5% | 92.8% | 37.6% | 7B |

### RAG性能benchmark（HotpotQA EM）

| 方法 | EM | 检索次数 | 延迟 |
|-----|----|---------|----|
| 标准RAG | ~42% | 1 | 低 |
| IRCoT | ~55% | 2-5 | 高 |
| Collab-RAG 3B | 51.6% | 自适应 | 中 |
| Collab-RAG 8B | 53.0% | 自适应 | 中 |

---

## 🔄 技术演进趋势

### 趋势1：推理与RAG的融合

- **Search-R1**：将搜索行为纳入RL训练，模型学会"何时搜索"
- **Agent-RAG**：将检索作为工具，LLM通过推理链决定调用时机
- **预测**：未来1-2年，RAG系统将统一为"推理驱动的自适应检索"

### 趋势2：小模型专业化

- **Collab-RAG**：3B模型通过DPO专门训练问题分解，超越32B冻结模型
- **Qwen3蒸馏**：0.6B-14B通过Strong-to-Weak获得接近大模型的推理能力
- **推论**：任务特定的小模型 > 通用大模型（效率/成本维度）

### 趋势3：思维预算与可控推理

- Qwen3的Thinking Budget代表了"按需推理"的新范式
- 从模型侧控制推理深度 vs 用户侧控制延迟预算
- **下一步**：Adaptive thinking based on 实时用户反馈

### 趋势4：在线RL for RAG

- Collab-RAG论文指出：从离线DPO到在线RL是未来方向
- Search-R1已展示RL优化搜索策略的可行性
- **预测**：端到端RL训练检索+推理联合优化

---

## 📚 参考文献列表

1. **DeepSeek-R1** - Guo et al. (2025) "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" - arXiv:2501.12948

2. **Qwen3** - Qwen Team (2025) "Qwen3 Technical Report" - arXiv:2505.09388

3. **Collab-RAG** - (2025) "Collab-RAG: Collaborating via White-Box and Black-Box LLM Integration for Retrieval-Augmented Generation" - arXiv:2504.04915

4. **RAG with Adaptive Retrieval** - (基于占位符URL，参考相关工作)
   - IRCoT: Trivedi et al. (2023) "Interleaving Retrieval with Chain-of-Thought Reasoning" - ACL 2023
   - FLARE: Jiang et al. (2023) "Active Retrieval Augmented Generation" - EMNLP 2023
   - Adaptive-RAG: Jeong et al. (2024) "Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs" - NAACL 2024

5. **Agent for Recommendation** - (基于占位符URL，参考相关工作)
   - InteRecAgent: Huang et al. (2023) "Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations"
   - RecMind: Wang et al. (2023) "RecMind: Large Language Model Powered Agent For Recommendation"
   - Rec-R1: (2025) "Rec-R1: Bridging Generative Large Language Models and User-Friendly Conversational Recommendation"

---

## 🎓 Q&A 综合考题

### Q1：DeepSeek-R1和Qwen3在RL训练上有哪些共同点和不同点？

**答**：
- **共同点**：都使用GRPO算法（无需Value Model），都采用规则奖励避免reward hacking，都通过冷启动SFT初始化推理能力
- **不同点**：
  - DeepSeek-R1使用固定的clip $\varepsilon=10$；Qwen3通过熵控制动态调节探索-利用平衡
  - Qwen3有4阶段后训练（含Thinking Mode Fusion和General RL）；R1仅3阶段
  - Qwen3支持Thinking Budget（涌现能力）；R1无此机制
  - Qwen3的蒸馏更系统化（On-policy+Off-policy双阶段）

---

### Q2：为什么Collab-RAG选择训练SLM而不是直接用LLM做问题分解？

**答**：成本效率考虑——3B本地模型推理近乎零成本，而GPT-4o-mini每次API调用需要付费。实验证明，经过迭代DPO专门训练的3B模型在问题分解任务上超越了冻结的32B模型（10.7倍参数差异）。核心原因是DPO训练优化了"对当前检索器有效"的分解方式，而非通用分解。

---

### Q3：解释Thinking Budget如何在不训练的情况下涌现？

**答**：Qwen3的Stage 3（Thinking Mode Fusion）让模型同时学习`/think`（输出完整推理）和`/no_think`（空推理块+直接回答）两种模式。本质上，模型学会了"在任意推理深度处给出答案"的能力——这是两种模式之间的**连续插值**。当外部强制截断thinking内容时，模型自动基于已有的部分推理继续输出，无需专门训练这种"截断推理"的情形。

---

### Q4：多跳RAG中，如何平衡检索精确性和召回率？

**答**：核心是**查询策略设计**：
- 过于宽泛的查询→高召回率但噪声多→LLM分心
- 过于精确的查询→低召回率但相关性强
实践方案：先用宽泛查询召回Top-50，再用Cross-Encoder基于问题精确重排，最终取Top-5。迭代中逐步收窄查询（每跳更精确），同时增量积累证据。

---

### Q5：LLM推荐Agent如何处理延迟与推荐质量的矛盾？

**答**：两阶段架构是核心解法：
- 第一阶段（<10ms）：传统协同过滤/向量检索快速召回100候选
- 第二阶段（~500ms，异步预计算）：LLM Agent精排+生成解释
用户看到的是第一阶段结果先展示，第二阶段结果流式更新。对高频场景可引入用户聚类缓存（~60%命中率）。

---

### Q6：GRPO相比PPO在长CoT训练中的核心优势是什么？

**答**：三点核心优势：
1. **内存效率**：无需与Policy等大的Value Model（节省50%显存）
2. **不偏向长度**：PPO的每token密集奖励会隐式惩罚长response；GRPO的KL直接加入loss，允许response自然增长
3. **组内比较更稳定**：同一问题16个采样相互比较，优势估计方差更小

---

### Q7：Qwen3的Strong-to-Weak蒸馏中，为什么需要On-policy阶段？

**答**：Off-policy阶段（直接SFT教师输出）存在分布偏移问题——学生模型会遇到"自己不会生成"的序列，导致训练不稳定。On-policy阶段让学生模型生成自己的序列，再对齐教师的logits分布，这样训练数据与学生的推理分布匹配，更有效传递"探索能力"（Pass@64改善是关键证据）。

---

### Q8：Collab-RAG的偏好对是如何构建的？为什么不用人工标注？

**答**：全自动构建流程：
1. SLM对同一问题生成N=5个分解方案
2. 每个方案经检索→GPT-4o-mini读取→得到答案
3. 答案正确的分解作为Q+，错误的作为Q-
全程无需人工标注，黑盒LLM提供"免费监督信号"。这种端到端的偏好构建确保优化目标（分解质量）与最终任务（答案准确性）完全对齐。

---

### Q9：为什么Qwen3-30B-A3B（3B激活）能媲美QwQ-32B（32B激活）？

**答**：三方面原因：
1. **预训练数据质量**：36T tokens，实例级混合优化，质量大幅提升
2. **架构改进**：QK-Norm稳定训练，MoE无共享专家+全局负载均衡
3. **蒸馏效率**：从235B-A22B蒸馏，只需1/10训练成本就获得强推理能力
核心结论：**蒸馏质量 > 参数规模**，小模型+好教师 > 大模型+差教师。

---

### Q10：工业界RAG系统如何评估和监控？

**答**：
- **离线指标**：EM/F1（准确性），Recall@K（检索质量），Faithfulness（事实一致性，NLI评估），Answer Relevance（答案相关性）
- **在线指标**：用户满意率，点击率，无答案率（fallback频率）
- **监控告警**：检索延迟P99，LLM输出含幻觉词频（关键词过滤），答案长度异常
- **A/B测试**：新版RAG上线先5%流量，对比用户满意率和业务指标
