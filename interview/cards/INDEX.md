# 面试知识卡片库 索引

> 生成时间：2026-03-16 | MelonEggLearn

---

## 使用指南

- **Anki 导入**：File → Import → 选择 .txt 文件 → Fields separated by: Tab → Tags in field 3
- **学习顺序**：L1→L2→L3（理解原理）→ L4（练输出）→ L5（防踩坑）
- **标签系统**：`{模块}::L{层级}::{类型}`，可在 Anki 按标签筛选复习
- **复习建议**：面试前一天过 L1+L5（概念+陷阱），面试当天过快速总览

---

## 模块总览

| 模块 | 总卡片数 | Markdown | Anki TSV |
|------|---------|---------|---------|
| 推荐系统 | 35张 | rec_sys_cards.md | rec_sys_anki.txt |
| CTR预估 | 39张 | ctr_prediction_cards.md | ctr_prediction_anki.txt |
| 搜索算法 | 36张 | search_cards.md | search_anki.txt |
| A/B实验 | 36张 | ab_test_cards.md | ab_test_anki.txt |
| LLM工程 | 37张 | llm_engineering_cards.md | llm_engineering_anki.txt |
| 广告系统 | 36张 | ads_system_cards.md | ads_system_anki.txt |
| Auto Bidding | 30张 | auto_bidding_cards.md | auto_bidding_anki.txt |
| **合计** | **249张** | | |

---

## 各模块核心考点速览

### 📊 推荐系统（35张）
1. **召回→粗排→精排→重排全链路**：双塔召回、HNSW/FAISS 检索、各层模型选型
2. **用户兴趣建模**：DIN 注意力机制、SIM 超长序列、DIEN 兴趣演化
3. **多目标排序**：MMOE/PLE 多任务框架、帕累托前沿、目标权重调节

### 🎯 CTR预估（39张）
1. **模型演进脉络**：LR → FM → Wide&Deep → DIN → Transformer → DSSM，每代解决什么问题
2. **特征工程**：离散特征 Embedding、特征交叉（DCN-V2/FiBiNet）、序列特征建模
3. **训练技巧**：样本构造（负采样策略）、在线学习（FTRL）、校准（Platt Scaling/Isotonic）

### 🔍 搜索算法（36张）
1. **相关性 vs 排序**：BM25 精确匹配 vs 语义向量召回（ANN），混合检索策略
2. **Query 理解**：意图识别、词权重（IDF/BM25）、Query 改写与扩展
3. **Learning to Rank**：PointWise/PairWise/ListWise 区别，LambdaMART/LambdaRank 原理

### 🧪 A/B 实验（36张）
1. **实验设计**：分流粒度（用户/请求/设备）、最小样本量计算（功效分析）
2. **统计检验**：t检验/卡方检验适用场景、p值 vs 置信区间、多重检验校正（Bonferroni/BH）
3. **实验陷阱**：网络效应（SUTVA违背）、新奇效应、辛普森悖论、AA测试的作用

### 🤖 LLM工程（37张）
1. **微调策略**：SFT vs RLHF vs DPO 适用场景，LoRA/QLoRA 参数高效微调原理
2. **推理优化**：KV Cache、Flash Attention、连续批处理（Continuous Batching）、量化（GPTQ/AWQ）
3. **RAG系统**：检索增强生成架构、Chunk策略、Reranker、幻觉检测与缓解

### 📢 广告系统（36张）
1. **RTB全链路**：DSP/SSP/ADX架构，100ms内竞价时序，eCPM统一排序公式
2. **CVR预估难题**：ESMM多任务框架（pCVR=pCTCVR/pCTR），样本偏差+延迟转化+位置偏差三类问题
3. **冷启动与Pacing**：UCB/Thompson Sampling探索策略，PID预算控制，Lookalike扩量

### 💰 Auto Bidding（30张）
1. **出价策略演进**：规则出价→oCPC/oCPA→LP对偶优化→RL强化学习，各阶段核心方法
2. **RL出价建模**：状态空间（预算余量/时段/ROI）、动作空间（出价系数）、奖励设计
3. **多约束优化**：ROI+预算双约束的拉格朗日对偶，对偶变量λ的在线更新

---

## 标签检索速查

```
按层级复习：
  Anki 搜索 "tag:*::L1::*"  → 所有模块概念卡（What）
  Anki 搜索 "tag:*::L2::*"  → 所有模块原理卡（How）
  Anki 搜索 "tag:*::L5::*"  → 所有模块陷阱卡（Trap）

按模块复习：
  Anki 搜索 "tag:ads_system::*"    → 广告系统全部卡片
  Anki 搜索 "tag:auto_bidding::*"  → Auto Bidding 全部卡片
  Anki 搜索 "tag:*::formula"       → 所有含公式的卡片
```

---

*📦 文件位置：`/Users/boyu/Documents/ai-kb/interview/cards/`*  
*🔄 更新方式：重新运行 MelonEggLearn 对应模块任务，覆盖对应 .md 和 .txt 文件*
