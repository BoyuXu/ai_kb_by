# LLM 工程化知识库

> 模块: llm-infra/ | 维护者: MelonEggLearn

---

## 核心知识体系

### 1. LLM 在推荐/广告系统中的角色

```
离线阶段（Offline）：
├── 内容理解：商品/广告的多模态理解（图文 → 语义表示）
├── 数据增强：生成训练数据、标注数据
├── 特征工程：LLM 抽取高质量 item features
└── 知识蒸馏：LLM 知识迁移到轻量级模型

在线阶段（Online）：
├── 召回辅助：LLM embedding 用于语义检索（离线生成）
├── 排序辅助：LLM-enhanced feature（离线批量计算后缓存）
└── 直接排序（实验阶段，延迟受限）
```

### 2. LLM 推荐的主流范式

| 范式 | 代表工作 | 适用场景 |
|------|---------|---------|
| LLM as Ranker | RankGPT, PRP | 小样本 / 零样本排序 |
| LLM as Feature Extractor | IDProxy, TALLREC | 冷启动、跨域 |
| LLM as Generative Model | S-GRec, GR4Rec | 端到端生成式推荐 |
| LLM as Multi-Task Solver | AdNanny | 广告离线任务统一 |
| RAG + Rec | 学术探索 | 知识增强推荐 |

### 3. 推理优化关键技术

#### KV Cache
- 原理：缓存 attention 中间状态，避免重复计算
- 应用：序列推荐场景（同一 session 多次请求），广告 CTR 上下文建模（CADET）
- 注意：KV Cache 内存占用 = num_layers × num_heads × seq_len × head_dim × 2（K+V）× batch_size

#### 量化
- INT8 / INT4 量化：减少显存占用，加速推理，精度损失 < 1%
- 适用：边缘推断、大批量离线推理

#### 蒸馏
- 从 LLM 蒸馏到轻量级模型（BERT → 小型 Transformer）
- 在推荐场景：LLM 打分 → 精排模型学习

---

## 2026-03-12 更新

### 本次新增内容

**1. AdNanny - 推理 LLM 统一离线广告任务**
- 来源：arXiv 2026-02-01，微软团队
- 核心：单个推理 LLM（Chain-of-Thought）处理广告相关性标注、受众扩展、出价建议等离线任务
- 工程价值：减少为每个广告任务维护独立模型的成本，"一模型统治所有离线广告分析"
- 局限：目前仍是离线场景，在线延迟不满足 SLA

**2. IDProxy - LLM 赋能冷启动 CTR（详见 ads 模块）**
- 核心创新：多模态 LLM 生成 proxy embedding，对齐 CTR 模型 ID 空间
- 技术路径：VLM 推理（离线批量）→ proxy embedding 存储 → CTR 模型在线使用

**3. Decoder-Only LLM 架构在 CTR 预测中的应用（CADET）**
- GPT 式 causal attention 用于广告上下文序列建模
- KV Cache 复用使 Transformer CTR 模型在线可用（P99 延迟可控）

### LLM × 推广告系统的工程实践要点

1. **离线批量处理**：LLM 推理成本高，一定要离线生成，缓存结果
2. **embedding 对齐**：LLM embedding 空间 ≠ 协同过滤 embedding 空间，需要 adapter/alignment 训练
3. **版本管理**：LLM 更新后，embedding 可能漂移，需要重新对齐排序模型
4. **冷启动窗口**：LLM-based 方案在新 item 前 24-72 小时特别有价值，之后逐渐切换到 ID-based

---
