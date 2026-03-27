# Boyu 个人学习档案

> 📚 参考文献
> - [A Generative Re-Ranking Model For List-Level Multi](../../rec-sys/papers/A_Generative_Re_ranking_Model_for_List_level_Multi_object.md) — A Generative Re-ranking Model for List-level Multi-object...
> - [Din-Deep-Interest-Network](../../rec-sys/papers/DINDeep_Interest_Network.md) — DIN：深度兴趣网络（Deep Interest Network）
> - [Model Calibration Deep Dive](../../rec-sys/papers/model_calibration_deep_dive.md) — 模型校准（Model Calibration）完整学习笔记
> - [Gems-Breaking-The-Long-Sequence-Barrier-In-Gene...](../../rec-sys/papers/GEMs_Breaking_the_Long_Sequence_Barrier_in_Generative_Rec.md) — GEMs: Breaking the Long-Sequence Barrier in Generative Re...
> - [Mmoe-Multi-Task-Learning](../../rec-sys/papers/MMoEMulti_gate_Mixture_of_Experts.md) — MMoE：多门控混合专家（Multi-gate Mixture-of-Experts）
> - [10 Multimodal Rec Deep](../../rec-sys/papers/10_multimodal_rec_deep.md) — 多模态推荐系统深度解析（图文/视频融合）
> - [Reg4Rec Reasoning-Enhanced Generative Model For La](../../rec-sys/papers/REG4Rec_Reasoning_Enhanced_Generative_Model_for_Large_Sca.md) — REG4Rec: Reasoning-Enhanced Generative Model for Large-Sc...
> - [Multi-Behavior-Rec-Survey](../../rec-sys/papers/Multi_behavior_Recommender_Systems_A_Survey.md) — Multi-behavior Recommender Systems: A Survey


> 📅 生成日期: 2026-03-12 | 🎯 分析师: MelonEggLearn
> 
> 基于学习笔记深度分析 + AI知识库整合


## 📐 核心公式与原理

### 1. 矩阵分解
$$
\hat{r}_{ui} = p_u^T q_i
$$
- 用户和物品的隐向量内积

### 2. BPR 损失
$$
L_{BPR} = -\sum_{(u,i,j)} \ln \sigma(\hat{r}_{ui} - \hat{r}_{uj})
$$
- 正样本得分 > 负样本得分

### 3. 序列推荐
$$
P(i_{t+1} | i_1, ..., i_t) = \text{softmax}(h_t^T E)
$$
- 基于历史序列预测下一次交互

---

## 一、已掌握（强项）

### ⭐⭐⭐ 深度掌握领域

| 领域 | 掌握程度 | 具体表现 |
|------|---------|---------|
| **Attention机制** | 🔥 精通 | 完整理解Q/K/V计算流程、Soft/Hard Attention分类、多头注意力机制、dk缩放原理 |
| **Transformer架构** | 🔥 精通 | Encoder/Decoder结构、Position Encoding、残差连接、LayerNorm、Mask机制(Padding+Sequence) |
| **BERT vs GPT** | 🔥 精通 | 核心差异（Encoder-only vs Decoder-only）、Attention Mask差异（全向vs单向）、训练目标（MLM vs CLM） |
| **RNN/LSTM/Transformer对比** | ✅ 扎实 | 长期依赖问题、并行计算能力、信息损失程度的深入理解 |
| **大模型推理优化** | ✅ 扎实 | KV Cache原理、DeepSeek MLA、MoE优化（Shared Expert + Fine-Grained） |
| **推荐系统全链路** | ✅ 扎实 | 召回→粗排→精排→重排完整链路理解、各环节核心模型 |

### ⭐⭐ 良好掌握领域

| 领域 | 掌握程度 | 具体表现 |
|------|---------|---------|
| **经典排序模型** | ✅ 良好 | DIN/DIEN、MMoE/PLE、ESSM等模型的理解 |
| **序列推荐** | ✅ 良好 | SASRec、HSTU、自回归生成推荐的理解 |
| **生成式推荐** | ✅ 良好 | OneRec、TIGER/PLUM、RQ-VAE Tokenizer机制 |
| **Loss演进** | ✅ 良好 | BCE→BPR→Softmax/InfoNCE的演进路径理解 |
| **树模型基础** | ✅ 良好 | XGBoost分裂算法、特征重要度计算、Bagging vs Boosting |
| **Tokenization** | ✅ 良好 | BPE、WordPiece、SentencePiece的原理理解 |

---

## 二、待加强（弱项）

### ⚠️ 知识盲区识别

#### 1. 理论与实践Gap
| 盲区 | 具体问题 | 影响程度 |
|------|---------|---------|
| **代码实现** | 笔记多为理论，缺少实际代码练习 | 🔴 高 |
| **实验案例** | 缺少端到端项目实践经验 | 🔴 高 |
| **公式推导** | 部分公式（如反向传播、梯度更新）不够深入 | 🟡 中 |

#### 2. 评估与指标
| 盲区 | 具体问题 | 影响程度 |
|------|---------|---------|
| **GAUC/NDCG** | 笔记中提及较少，深入理解不足 | 🟡 中 |
| **离线vs在线指标** | 缺乏实际Gap分析和解决策略 | 🟡 中 |
| **A/B测试** | 实验设计、显著性检验等实践知识 | 🟡 中 |

#### 3. 工程与部署
| 盲区 | 具体问题 | 影响程度 |
|------|---------|---------|
| **模型Serving** | 推理优化、量化、剪枝实践 | 🔴 高 |
| **特征工程** | 实时特征、特征存储(Feature Store) | 🟡 中 |
| **分布式训练** | 大规模模型训练实践 | 🟡 中 |

#### 4. 前沿方向
| 盲区 | 具体问题 | 影响程度 |
|------|---------|---------|
| **多模态推荐** | 图文/视频融合理解较浅 | 🟡 中 |
| **Agent+推荐** | 对话式推荐实践较少 | 🟡 中 |
| **RLHF在推荐中应用** | 强化学习微调实践 | 🟢 低 |

---

## 三、个性化补充笔记

### 📚 针对盲区的补充内容

#### 1. 评估指标深度补充

```
GAUC (Group AUC) 深度理解
────────────────────────────────
定义: GAUC = Σ(wi * AUCi) / Σ(wi)

关键洞察:
• 为什么需要GAUC? 
  - 全局AUC可能被高频用户主导
  - 推荐系统关心每个用户的个性化排序质量
  
• 权重选择策略:
  - 按曝光数加权: wi = impressions_i
  - 按点击数加权: wi = clicks_i
  - 等权重: 每个用户同等重要（长尾友好）

• 面试高频考点:
  Q: GAUC比AUC更能反映推荐效果，为什么?
  A: 推荐是个性化排序，GAUC衡量每个用户的排序质量，
     避免被头部用户的主导行为掩盖长尾问题
```

#### 2. Loss函数演进完整图谱

```
推荐系统Loss演进路线图 (补充笔记细节)
────────────────────────────────────────

BCE (Point-wise)
├── 优点: 简单、并行计算友好
├── 缺点: 忽略物品间关系、对负采样敏感
└── 适用: 二分类CTR预估

BPR (Pair-wise)
├── 核心: 最大化正样本vs负样本的分数差距
├── 公式: L = -log(σ(score+ - score-))
├── 优点: 关注相对顺序
└── 缺点: 仅考虑一对，非真实排序

Softmax/InfoNCE (List-wise)
├── 核心: 把排序视为列表级分类问题
├── 温度系数τ: 控制分布尖锐程度
│   - τ→0: 接近one-hot（硬排序）
│   - τ→∞: 均匀分布（软排序）
└── 变体:
    • In-batch negatives: 使用同batch其他样本做负样本
    • Time-aware: 考虑时间衰减的权重
    • Importance weighted: 高价值样本加权
```

#### 3. 推荐系统部署优化实践

```
模型Serving优化 checklist
──────────────────────────

□ 模型压缩
  ├── 量化 (FP32 → FP16/INT8)
  │   └── TensorRT、ONNX Runtime
  ├── 剪枝 (移除不重要权重)
  │   └── 结构化剪枝 vs 非结构化剪枝
  └── 蒸馏 (大模型→小模型)
      └── 软标签训练、特征蒸馏

□ 推理加速
  ├── 批处理 (Batching)
  ├── 缓存策略 (Embedding Cache)
  ├── 异步预测 (Async Inference)
  └── 模型分片 (Model Sharding)

□ 系统架构
  ├── 特征服务 (Feature Service)
  ├── 模型版本管理 (A/B测试支持)
  └── 监控告警 (延迟/准确率/资源)
```

#### 4. 经典模型细节补充

```
DIN (Deep Interest Network) 关键细节
─────────────────────────────────────

1. Attention计算:
   e_ij = DNN([v_i, v_j, v_i-v_j, v_i*v_j])
   α_ij = softmax(e_ij)
   v_u = Σ(α_ij * v_j)

2. 激活函数选择:
   - 使用Dice激活函数（自适应PReLU）
   - 根据数据分布自适应调整激活点

3. 工业实践技巧:
   - 温度缩放：防止attention分数过于尖锐
   - 注意力 dropout：防止过拟合
   - 历史序列长度：通常50-100，过长引入噪声

DIEN (Deep Interest Evolution Network)
──────────────────────────────────────

关键创新: 兴趣进化建模
• GRU + Attention = AUGRU
• 引入时间间隔信息 (Time Interval)
• 兴趣抽取层 + 兴趣进化层
```

---

## 四、推荐复习顺序

### 🗓️ 4周复习计划

#### Week 1: 基础巩固 + 代码实践
```
Day 1-2: Attention + Transformer 代码实现
├── 手/手撕Self-Attention (PyTorch)
├── Mini Transformer 实现
└── 练习: 实现Multi-Head Attention

Day 3-4: 经典排序模型复盘
├── Wide&Deep / DeepFM 结构手绘
├── DIN Attention机制代码实现
└── 对比DIN vs DIEN的差异

Day 5-7: 序列推荐实践
├── SASRec 论文精读 + 代码
├── 理解Causal Mask实现
└── 实验: 在MovieLens上跑通SASRec
```

#### Week 2: 评估指标 + 工程实践
```
Day 1-2: 评估指标深入
├── AUC/GAUC/NDCG 代码实现
├── 理解指标间的权衡关系
└── 实现一个简单的Evaluator类

Day 3-4: Loss函数实践
├── 实现BCE/BPR/Softmax Loss
├── 对比不同Loss在相同数据上的表现
└── 理解温度系数的作用

Day 5-7: 特征工程专题
├── 特征处理Pipeline设计
├── 序列特征编码实践
└── 实时特征 vs 离线特征
```

#### Week 3: LLM+推荐深化
```
Day 1-2: BERT/GPT架构对比复习
├── 手撕BERT Encoder
├── 手撕GPT Decoder
└── 对比实验: 相同数据下的表现

Day 3-4: 生成式推荐深入
├── RQ-VAE原理 + 代码
├── OneRec论文精读
└── 理解Tokenization在推荐中的作用

Day 5-7: LLM推理优化
├── KV Cache实现
├── MQA/GQA对比
└── DeepSeek MLA论文精读
```

#### Week 4: 前沿探索 + 面试准备
```
Day 1-2: 多模态推荐入门
├── CLIP原理理解
├── 图文融合策略
└── 阅读一篇多模态推荐论文

Day 3-4: Agent+推荐探索
├── ReAct模式理解
├── LangChain/LlamaIndex实践
└── 构建一个简单的对话推荐Demo

Day 5-7: 面试题库复习
├── 推荐系统高频面试题
├── 手撕代码题练习
└── 项目经历梳理
```

---

## 五、知识库关联资源

### 📖 推荐学习材料

| 主题 | 知识库文件 | 建议阅读顺序 |
|------|-----------|-------------|
| 推荐系统全链路 | `rec-sys/推荐系统全链路架构概览.md` | ⭐ 必读 |
| 排序模型详解 | `rec-sys/CTR模型深度解析.md` | ⭐ 必读 |
| 机器学习基础 | `rec-sys/04_ml_fundamentals.md` | ⭐ 必读 |
| LLM+推荐前沿 | `llm-infra/01_llm_rec.md` | ⭐ 必读 |
| 特征工程 | `rec-sys/推荐系统特征工程深度笔记.md` | 🟡 选读 |
| 工业论文 | `rec-sys/精排工业界精华_AlgoNotes精读.md` | 🟡 选读 |

### 🎯 重点关注章节

```
rec-sys/04_ml_fundamentals.md
├── 损失函数对比 (BCE/Focal Loss/MSE/Huber)
├── 优化器对比 (Adam/AdamW/LAMB)
├── 评估指标详解 (AUC/GAUC/NDCG)
└── 高频面试题 1-8

llm-infra/01_llm_rec.md
├── P5/TALLRec/LLMRec架构对比
├── LLM特征增强实践
├── Agent+推荐系统架构
└── RAG在推荐中的应用
```

---

## 六、学习建议

### 💡 针对Boyu的个性化建议

1. **强化代码能力**
   - 目前理论知识扎实，但缺少代码实践
   - 建议：每周至少手撕一个模型组件（如Attention、GRU等）

2. **建立评估直觉**
   - 对GAUC/NDCG等指标理解较浅
   - 建议：在实际数据集上计算这些指标，理解其物理意义

3. **关注工程实践**
   - 部署优化知识相对薄弱
   - 建议：学习TensorRT/ONNX模型部署，了解工业界Serving架构

4. **深化前沿方向**
   - 对LLM+推荐理解较好，可多模态/Agent方向可加强
   - 建议：跟进2024-2025的ACL/SIGIR推荐相关论文

---

> 📝 **Last Updated**: 2026-03-12
> 
> 📧 **反馈**: 此档案基于笔记深度分析生成，可根据后续学习进度更新调整

### Q1: 推荐系统的实时性如何保证？
**30秒答案**：①用户特征实时更新（Flink 流处理）；②模型增量更新（FTRL/天级重训）；③索引实时更新（新物品上架）；④特征缓存+预计算降低延迟。

### Q2: 推荐系统的 position bias 怎么处理？
**30秒答案**：训练时：①加 position feature 推理时固定；②IPW 加权；③PAL 分解 P(click)=P(examine)×P(relevant)。推理时：设置固定 position 或用 PAL 只取 P(relevant)。

### Q3: 工业推荐系统和学术研究的差距？
**30秒答案**：①规模（亿级 vs 百万级）；②指标（商业指标 vs AUC/NDCG）；③延迟（<100ms vs 不关心）；④迭代（A/B 测试 vs 离线评估）；⑤工程（特征系统/模型服务 vs 单机实验）。

### Q4: 推荐系统面试中设计题怎么答？
**30秒答案**：按层回答：①明确场景和指标→②召回策略（多路）→③排序模型（DIN/多目标）→④重排（多样性）→⑤在线实验（A/B）→⑥工程架构（特征/模型/日志）。

### Q5: 2024-2025 推荐技术趋势？
**30秒答案**：①生成式推荐（Semantic ID+自回归）；②LLM 增强（特征/数据增广/蒸馏）；③Scaling Law（Wukong）；④端到端（OneRec 统一召排）；⑤多模态（视频/图文理解）。

### Q6: 推荐系统的 EE（Explore-Exploit）怎么做？
**30秒答案**：①ε-greedy：ε 概率随机推荐；②Thompson Sampling：从后验分布采样；③UCB：置信上界探索；④Boltzmann Exploration：按 softmax 温度控制探索度。工业实践：对新用户多探索，老用户少探索。

### Q7: 推荐系统的负反馈如何利用？
**30秒答案**：①隐式负反馈：曝光未点击（弱负样本）、快速划过（中等负样本）；②显式负反馈：「不喜欢」按钮（强负样本）。处理：加大显式负反馈的权重，用 skip 行为做弱负样本。

### Q8: 多场景推荐（Multi-Scenario）怎么做？
**30秒答案**：同一用户在首页/搜索/详情页/直播间有不同推荐需求。方法：①STAR：场景自适应 Tower；②共享底座+场景特定头；③Scenario-aware Attention。核心：共享知识避免数据孤岛，同时保留场景差异。

### Q9: 推荐系统的内容理解怎么做？
**30秒答案**：①文本理解（NLP/LLM 提取标题、标签语义）；②图片理解（CNN/ViT 提取视觉特征）；③视频理解（时序模型提取关键帧+音频）；④多模态融合（CLIP-style 对齐文本和视觉）。

### Q10: 推荐系统的公平性问题？
**30秒答案**：①供给侧公平（小创作者也有曝光机会）；②需求侧公平（不同用户群体获得同等服务质量）；③内容公平（避免信息茧房）。方法：公平约束重排、多样性保障、定期公平性审计。
