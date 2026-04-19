# 端侧LLM推荐 + 多场景兴趣演化 (2026-04-20)

> 覆盖论文：OD-LLM (WSDM 2026), Reinforcing User Interest Evolution (2506.17682)
> 主题：推荐系统的新部署范式（端侧）和新建模范式（跨场景RL）

---

## Paper 1: OD-LLM — 端侧大语言模型序列推荐 (WSDM 2026)

### 问题定义
LLM 用于序列推荐效果好，但推理成本高、隐私风险大。端侧部署是解决方案，但现有压缩方法（GPTQ, SparseGPT）不考虑推荐任务特性。

### 核心方法
首个面向序列推荐的任务自适应压缩框架 OD-LLM：
- 在压缩过程中融入推荐任务的特殊需求（序列依赖、item 表示质量）
- 兼顾压缩率和推荐效果

### 关键对比

| 方法 | 推荐准确度 | 推理延迟 | 压缩策略 |
|------|-----------|---------|---------|
| GPTQ | 略高于OD-LLM | ~3x 慢 | 通用量化 |
| SparseGPT | 最差 | 中等 | 通用剪枝 |
| **OD-LLM** | 竞争力强 | **最快** | 任务感知压缩 |

### 工业意义
- **隐私保护**：行为数据不出端，符合 GDPR 等法规
- **延迟降低**：消除网络往返，端侧推理更快
- **成本降低**：无需大规模 GPU 集群推理

### 面试考点

**Q：为什么端侧 LLM 推荐需要任务感知的压缩？**
A：通用压缩（GPTQ/SparseGPT）只考虑语言建模的 perplexity，不考虑推荐任务的特殊性。序列推荐对 item embedding 质量和序列位置编码精度要求高，通用压缩可能在这些关键位置引入较大误差。OD-LLM 在压缩目标中加入推荐 loss，确保压缩后推荐效果不大幅退化。

**Q：端侧 LLM 推荐的三大技术挑战？**
A：(1) 模型压缩与效果的 tradeoff — 推荐需要精确的 embedding 表示；(2) 端侧计算资源受限 — 手机 GPU/NPU 算力有限；(3) 模型更新 — 端侧模型如何持续学习新物品。OD-LLM 主要解决第一个问题。

---

## Paper 2: Reinforcing User Interest Evolution in Multi-Scenario Learning

### 问题定义
用户在不同场景（首页/搜索/相关推荐）展现不同兴趣面。现有多场景模型假设兴趣静态，忽略了兴趣的跨场景演化。

### 核心方法
- **Double Q-learning** 建模用户兴趣在场景间的动态演化
- **Q-value 优化对比学习损失**：用 RL 的 Q 值指导对比学习，使正样本选择更准确
- 将多场景学习建模为 MDP：状态=用户当前兴趣，动作=推荐物品，奖励=用户反馈

### 核心公式

**Q-value guided contrastive loss**:
$$\mathcal{L}_{\text{CL}} = -\log \frac{\exp(Q(s, a^+) \cdot \text{sim}(\mathbf{z}_i, \mathbf{z}_j^+) / \tau)}{\sum_k \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau)}$$

Q 值作为对比学习中正样本权重，高 Q 值的正样本对获得更大学习信号。

### 工业意义
- 电商平台用户在首页浏览（发现性兴趣）vs 搜索（目标性兴趣）vs 详情页推荐（关联性兴趣）是三种截然不同的模式
- RL 框架自然建模兴趣的动态转移
- Double Q-learning 缓解 Q 值过估计

### 面试考点

**Q：为什么多场景推荐需要 RL？传统多任务学习不够吗？**
A：传统多场景模型（STAR, SAR-Net）把场景当静态标签，学场景特有的参数。但用户兴趣在场景间是动态演化的：首页发现 → 搜索深挖 → 详情页关联，这是一个决策序列。RL 的 MDP 框架天然适合建模这种跨场景的兴趣转移。

**Q：Double Q-learning 在推荐中的具体好处？**
A：标准 Q-learning 在推荐中容易过估计（推荐空间大，max 操作放大噪声）。Double Q-learning 用两个独立 Q 网络：一个选动作，另一个评估，减少过估计偏差。在多场景中尤为重要，因为场景切换带来额外的状态空间不确定性。

**Tags:** #on-device #llm #sequential-rec #multi-scenario #reinforcement-learning #interest-evolution

---

## 相关概念

- [[sequence_modeling_evolution|序列建模演进]]
- [[generative_recsys|生成式推荐统一视角]]
- [[multi_objective_optimization|多目标优化]]
- [[20260407_GenRec_advances_synthesis|生成式推荐前沿进展]]
