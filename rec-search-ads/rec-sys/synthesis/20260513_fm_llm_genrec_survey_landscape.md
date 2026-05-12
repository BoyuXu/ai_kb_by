# Foundation Model + LLM + 生成式推荐：Survey 全景与工业挑战 (2024-2026)

> **日期**：2026-05-13
> **覆盖论文**：FM4RecSys Survey (2504.16420) / GenRec Survey (2510.27157) / LLM4RecSys Review (2507.21117) / Scaling LRM (2412.00714) / Cold-Starts GenRec Reproducibility (2603.29845) / Task-Centric Perspective (2503.21188) / Real-World RecSys Challenges (2509.06002)
> **三大主题**：(1) FM/LLM 增强推荐的技术范式演进 (2) 生成式推荐的模型-数据-任务全景 (3) 工业落地挑战与学术-工业 Gap

**相关概念页**：[[生成式推荐]] | [[推荐中的注意力机制]] | [[序列建模演进]] | [[Embedding无处不在]] | [[多目标优化]]
**相关 synthesis**：[[20260504_scaling_coldstart_ctr_frontier]] | [[20260503_scaling_sequence_multitask_frontier]] | [[agentic-recsys-paradigm-shift]] | [[生成式推荐范式统一_20260403]] | [[生成式与LLM增强推荐系统前沿进展]]

---

## 总览表

| # | 论文 | 类型 | 核心贡献 | 覆盖范围 |
|---|------|------|----------|----------|
| 1 | **FM4RecSys Survey** (2504.16420) | Survey | Feature/Generative/Agentic 三范式全景 | TKDE, 2025 |
| 2 | **GenRec Survey** (2510.27157) | Survey | Data-Model-Task 三维框架 | 2025 |
| 3 | **LLM4RecSys Review** (2507.21117) | Review | LLM 解决 RecSys 6 大挑战 | 2025 |
| 4 | **Scaling LRM** (2412.00714) | Survey | Large Recommendation Model 定义与 Scaling | 2024 |
| 5 | **Cold-Start GenRec** (2603.29845) | Reprod. | 生成式推荐冷启动可复现性研究 | 2026 |
| 6 | **Task-Centric Perspective** (2503.21188) | Position | 推荐任务定义的反思 | 2025 |
| 7 | **Real-World Challenges** (2509.06002) | Survey | 学术 vs 工业 RecSys 鸿沟 | 2025 |

---

## Section 1: Foundation Model 增强推荐的三阶段演进

### 1.1 FM4RecSys: Feature-Based -> Generative -> Agentic (2504.16420)

**核心框架：基础模型赋能推荐的三个范式阶段**

| 阶段 | 范式 | FM 角色 | 典型方法 | 核心能力 |
|------|------|---------|----------|----------|
| **Phase 1** | Feature-Based | 特征增强器 | BERT4Rec, P5-ID | 文本/多模态表示 |
| **Phase 2** | Generative | 推荐生成器 | TIGER, HSTU, MTGR | 自回归生成 item |
| **Phase 3** | Agentic | 自主决策体 | AgenticRS, RecAgent | 规划/反思/工具调用 |

**三阶段的技术递进关系**：

```
Phase 1 (Feature-Based):
  FM 是"特征提取器"，输出 embedding → 传统 RecSys 消费
  局限：FM 的推理能力未被利用

Phase 2 (Generative):
  FM 是"推荐生成器"，直接输出推荐结果
  实现路径：
  - LLM-as-Ranker: prompt-based 排序
  - Generative Retrieval: SID 自回归生成
  - HSTU/MTGR: 行为序列生成式建模
  局限：单次推理，无法自我修正

Phase 3 (Agentic):
  FM 是"智能体"，具备规划-执行-反思-工具调用能力
  新能力：
  - 主动探索用户兴趣（多轮对话）
  - 调用外部工具（搜索/知识图谱/API）
  - 基于反馈自我修正推荐策略
  现状：早期研究阶段，工业落地有限
```

**Survey 覆盖范围**：
- 数据源：显式/隐式反馈 → 多模态内容 → 用户生成文本
- 任务：Top-N / Sequential / Zero-Shot / Conversational / Content Generation
- 挑战：跨域泛化 / 可解释性 / 公平性 / 多模态融合

### 1.2 Scaling New Frontiers: Large Recommendation Models (2412.00714)

**核心命题：推荐系统的参数规模扩展正在打破传统瓶颈**

**传统 RecSys 的 Scaling 困境**：
- Embedding 表可以扩展到 TB 级（工业常见几十 TB）
- 但网络参数（MLP/Attention）停滞在 ~10M 级别
- 更大的 embedding 不等于更强的模型能力

**Large Recommendation Model (LRM) 定义**：
> 一个可扩展的系统，设计用于处理多模态异构数据，支持广泛的推荐任务，通过增加模型参数和数据集来提升性能。

**关键里程碑**：

| 阶段 | 代表 | 参数量 | 创新 |
|------|------|--------|------|
| DLRM 时代 | DLRM/DCN-V2 | ~10M 网络 + TB Embedding | 特征交叉 |
| 序列时代 | SASRec/BERT4Rec | ~100M | Transformer 引入 |
| Scaling 时代 | HSTU | 1.5B → 1.5T | Scaling Law 验证 |
| LRM 时代 | ULTRA-HSTU/LUM | 7B+ | System Co-design |

**Scaling Law 在推荐中的特殊性**（与 LLM 的区别）：

| 维度 | LLM | RecSys |
|------|-----|--------|
| 数据类型 | 同质文本 | 异构（稀疏ID + 稠密特征 + 序列） |
| 延迟约束 | 秒级可接受 | P99 < 50ms |
| 数据质量 | 高质量语料 | 噪声大（隐式反馈） |
| 训练范式 | 预训练+微调 | 端到端/增量更新 |
| 推理模式 | 自回归生成 | 批量打分 |

---

## Section 2: 生成式推荐的 Data-Model-Task 全景

### 2.1 GenRec Survey: 三维解构 (2510.27157)

**统一框架：Data x Model x Task**

**数据层 (Data)**

| 策略 | 方法 | 效果 |
|------|------|------|
| 知识注入增强 | LLM 生成 item 描述/用户画像 | 补充稀疏信号 |
| Agent 模拟 | 用 LLM Agent 模拟用户交互 | 扩充训练数据 |
| 异构信号统一 | 文本/行为/图结构统一 tokenize | 消除模态壁垒 |

**模型层 (Model)**

| 类别 | 代表方法 | 核心能力 |
|------|----------|----------|
| LLM-Based | P5, GPT4Rec, InstructRec | NLU + 推理 |
| Large Rec Model | HSTU, LUM, ULTRA-HSTU | Scaling + 协同 |
| Diffusion-Based | DiffRec, DreamRec | 连续空间生成 |

**任务层 (Task)**

新兴任务能力（超越传统 Top-N）：
1. **对话式推荐**：多轮交互，主动探索用户需求
2. **可解释推理**：生成推荐理由，CoT 推理
3. **个性化内容生成**：不只推荐已有物品，生成新内容

**五大核心优势**：
1. 世界知识集成（World Knowledge）
2. 自然语言理解（NLU）
3. 推理能力（Reasoning）
4. Scaling Laws
5. 创意生成（Creative Generation）

### 2.2 Cold-Starts in GenRec: A Reproducibility Study (2603.29845)

**核心贡献：首次系统性、可复现地评估生成式推荐的冷启动能力**

**研究动机**：
- 生成式推荐（基于 PLM）理论上可用 item 语义信息缓解冷启动
- 但现有论文很少以冷启动为主要评估设置
- 设计选择（模型规模/ID 设计/训练策略）常同时改变，增益难归因

**评估协议**：统一的冷启动评估套件
- **User Cold-Start**：新注册用户，0-5 次交互
- **Item Cold-Start**：新引入物品，无/少量交互历史
- **评估标准**：控制变量（模型大小、ID 类型、训练策略分别单独变化）

**关键发现**：
1. **语义信息确实有帮助**：使用 item title/description 作为输入的生成式模型在冷启动上显著优于纯 ID 模型
2. **但提升幅度被高估**：控制模型规模后，语义 vs ID 的差距缩小
3. **ID 设计比模型规模更重要**：Semantic ID（RQ-VAE）vs Random ID vs Hash ID 的选择对冷启动影响最大
4. **训练策略关键**：预训练→微调 vs 端到端训练在冷启动场景下表现差异显著

**对实践的启示**：
- 冷启动场景优先考虑 Semantic ID + 语义特征输入
- 模型规模不是万能药，ID 设计和训练策略可能更重要
- 需要标准化的冷启动评估 benchmark

---

## Section 3: LLM 解决推荐核心挑战

### 3.1 LLM4RecSys Comprehensive Review (2507.21117)

**六大挑战与 LLM 解法**

| # | 推荐核心挑战 | LLM 解法 | 机制 |
|---|-------------|---------|------|
| 1 | 数据稀疏 | RAG 增强 | 检索外部知识补充稀疏信号 |
| 2 | 冷启动 | Zero/Few-Shot 推理 | 利用世界知识推断新用户/物品偏好 |
| 3 | 语义理解浅 | 语言原生表示 | LLM embedding 替代 ID embedding |
| 4 | 个性化不足 | Prompt-Driven 检索+排序 | 用户画像作为 prompt context |
| 5 | 可解释性差 | 推荐理由生成 | 自然语言解释推荐逻辑 |
| 6 | 跨域泛化弱 | 统一语言接口 | 不同域的 item 用文本统一表示 |

**LLM 在推荐链路中的角色映射**：

```
召回阶段：
  - Prompt-Driven Candidate Retrieval
  - LLM embedding 做 ANN 召回
  - Zero-shot 跨域召回

排序阶段：
  - Language-Native Ranking (直接用 LLM 排序)
  - LLM-as-Feature (LLM 输出作为排序特征)
  - Knowledge Distillation (LLM → 轻量排序模型)

重排阶段：
  - RAG 增强重排
  - CoT 推理重排
  - 对话式推荐

```

**核心权衡**：

| 维度 | LLM 原生推荐 | LLM 增强传统推荐 |
|------|-------------|-----------------|
| 准确率 | 中等（缺协同信号） | 高（保留 CF 优势） |
| 可扩展性 | 低（推理成本高） | 高（LLM 离线特征） |
| 实时性 | 差（秒级延迟） | 好（轻量模型在线） |
| 冷启动 | 强 | 中等 |
| 可解释性 | 强 | 中等 |
| 工业可行性 | 低 | 高 |

**结论**：当前工业最可行的路径是"LLM 增强传统推荐"而非"LLM 替代传统推荐"。

---

## Section 4: 推荐系统的元反思 — 任务定义与工业挑战

### 4.1 Task-Centric Perspective on RecSys (2503.21188)

**核心论点：推荐系统研究的问题定义太笼统，缺乏领域特异性**

**被忽视的任务维度**：

| 维度 | 学术通常假设 | 实际复杂性 |
|------|-------------|-----------|
| 输入-输出结构 | 用户-物品交互矩阵 | 多类型反馈 + 上下文 + 约束 |
| 时间动态 | 静态快照 | 实时流式 + 延迟反馈 |
| 候选集选择 | 全量物品 | 可用性/库存/合规过滤 |
| 决策成本 | 同质 | 高成本决策（房产）vs 低成本（短视频） |
| 交互深度 | 单步点击 | 多步骤（搜索→比较→购买→退货） |
| 不可观测交互 | 忽略 | 用户看了但没点（隐式负反馈） |

**对评估的影响**：
- 不同任务定义导致不同的评估指标合理性
- 某些 benchmark 的 "SOTA" 可能在实际部署中无意义
- 建议根据具体场景定制任务定义和评估方案

### 4.2 Real-World RecSys Challenges Survey (2509.06002)

**核心贡献：系统梳理学术 RecSys 与工业 RecSys 的鸿沟**

**六大鸿沟**：

| # | 维度 | 学术 | 工业 | 差距 |
|---|------|------|------|------|
| 1 | 数据规模 | 10K-1M 交互 | 10B+ 交互/天 | 4-5 个数量级 |
| 2 | 实时性 | 离线评估 | P99 < 50ms | 在线系统约束 |
| 3 | 评估方法 | 离线 AUC/NDCG | A/B 测试 + 长期留存 | 指标体系不同 |
| 4 | 特征工程 | 原始 ID/文本 | 1000+ 手工特征 | 工程投入 |
| 5 | 多目标 | 单目标优化 | 5-20 个 KPI 同时优化 | 复杂度指数增长 |
| 6 | 系统架构 | 单模型 | 召回→粗排→精排→重排 pipeline | 端到端 vs 分阶段 |

**工业特有挑战**（学术很少研究）：

1. **数据分布漂移**：用户兴趣变化 + 新物品涌入 → 模型每日/每小时更新
2. **公平性与合规**：内容审核 / 年龄限制 / 区域法规 / 广告合规
3. **多方利益平衡**：用户体验 vs 广告收入 vs 内容创作者 vs 平台长期价值
4. **系统可靠性**：降级策略 / 容灾 / 在线调试 / 灰度发布
5. **反作弊与对抗**：刷量 / 机器人 / 恶意点击

**推荐领域的学术-工业协作方向**：
- 开放工业级 benchmark（如 Meta 的 HSTU 开源）
- 系统论文（如 SparseCTR 的 system co-design）
- 因果推断方法在在线实验中的应用

---

## 综合洞察

### 2024-2026 推荐系统技术版图

```
                    Foundation Model 赋能
                           |
           +---------------+---------------+
           |               |               |
     Feature-Based    Generative      Agentic
     (成熟阶段)       (快速落地)     (探索阶段)
           |               |               |
    LLM Embedding     HSTU/MTGR      RecAgent
    知识蒸馏           SID 生成       多轮对话
    RAG 增强           扩散推荐       工具调用
           |               |               |
           +-------+-------+-------+-------+
                   |               |
            Scaling Law        工业挑战
            LRM (7B+)        实时/公平/多目标
                               学术-工业Gap
```

### 7 篇论文的互补关系

| 论文组合 | 关系 | 启示 |
|---------|------|------|
| FM4RecSys + GenRec Survey | 技术演进 + 细粒度分类 | 三范式→三维框架，互补理解 |
| LLM4RecSys + Cold-Start GenRec | 理论优势 + 实证检验 | LLM 理论上能解冷启动，但提升被高估 |
| Scaling LRM + Task-Centric | 技术可行性 + 问题合理性 | 先想清楚问题再 scale 模型 |
| Real-World Challenges + 以上所有 | 落地现实检验 | 学术 SOTA 不等于工业价值 |

### 面试串讲建议

**Survey 类问题回答框架**：
1. **技术演进线**：Feature-Based → Generative → Agentic（FM4RecSys Survey 的三阶段）
2. **能力增强线**：World Knowledge + NLU + Reasoning + Scaling + Generation（GenRec Survey 的五大优势）
3. **现实约束线**：延迟/规模/公平/多目标（Real-World Challenges）
4. **当前最优解**：LLM 增强传统推荐 > LLM 原生推荐（工业可行性）

**冷启动问题深入回答**：
"生成式推荐理论上能缓解冷启动，但 2603.29845 的可复现性研究发现，提升主要来自 Semantic ID 设计而非模型规模。工业实践中，IDProxy（小红书）和 SUIN（相似用户增强）是更务实的方案。"
