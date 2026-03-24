# AI-KB 完整知识图谱（搜广推一体化）

> 最后更新：2026-03-24 | 版本 v6-complete  
> 范围：LLM 推理优化、搜索检索、推荐系统、广告系统的完整知识体系

---

## 📌 四大系统核心节点速览

### 🧠 LLM 推理优化（详见原 v5）
- KV Cache 优化（PagedAttention、ZSMerge、KVSharer）
- MoE 推理（Expert Parallelism、MegaScale-Infer 解耦）
- Attention 计算（FlashAttention-3）
- 对齐训练（GRPO、Speculative Decoding）

### 🔍 搜索检索系统（详见原 v5 + 扩展）
- 词汇检索：BM25 → BM25s（极速 Python）
- 稀疏检索：SPLADE-v3 SOTA（BEIR 0.546）
- 稠密检索：DPR、ColBERT、BGE-M3
- 混合检索：RRF、LeSeR（两阶段解耦）
- 检索三角形：Sparse/Dense/Late-Interaction 工业选型

### 🛒 推荐系统（深化）

```
推荐系统完整链路（召回→粗排→精排→重排）

【召回层】
├─ 协同过滤（Item-based CF / User-based CF）
├─ 双塔稠密召回（DSSM / Embedding ANN）
├─ 生成式检索
│  ├─ Semantic ID + RQ-VAE（Spotify，冷启动+25%）
│  ├─ Variable-Length Semantic ID（冷启动+15%，token-30%）
│  ├─ GEMs 多流解码（超长序列，重度用户+15%）
│  └─ LLM Universal Retriever（多查询+矩阵分解）
├─ 多行为召回（CascadingRank，+9.56% HR）
└─ 图召回（PinSAGE、GCN based）

【粗排层】
├─ 轻量级双塔模型
├─ GBDT 量化模型
└─ 知识蒸馏小模型

【精排层】
├─ 多任务学习（MMoE、PLE）
├─ RankFormer（图 Transformer 排序）
├─ 超长序列建模
│  ├─ DIN（目标物品注意力）
│  ├─ SIM（两阶段检索，冷启动+8%）
│  ├─ GEMs（多流异步，内存-4×，速度+3.2×）
│  └─ Query-as-Anchor（查询锚定个性化，+6.8% NDCG）
└─ 质量感知排序（LLM 质量分）

【重排层】
├─ 多目标融合（CTR/CVR/GMV/LTV）
├─ Pareto 优化（收入 vs 用户体验）
├─ 多样性控制（MMR、DPP）
├─ 时效性和新鲜度衰减
├─ 生成式重排
│  ├─ 扩散重排（DiffGRM，多样性+12%）
│  ├─ 思维链推理重排（OneRec-Think）
│  └─ 绩效预测重排（PinRec）
└─ 业务规则和疲劳度控制

【核心技术演进】
LTV 长期价值 ← CTR/CVR 短期指标（防止短视）
生成式范式 ← 级联架构（端到端学习）
超长序列处理 ← 固定长度窗口（全历史建模）
推理增强 ← 简单评分（冷启动先想后推）
```

### 📢 广告系统（新增 P0-2 内容）

```
广告生态全景（DSP/SSP/ADX/DMP）

【ADX 核心】
├─ 竞价机制
│  ├─ GSP（广义第二价格）→ 工业主流但非最优
│  ├─ VCG（Vickrey-Clarke-Groves）→ 理论最优但复杂
│  ├─ 第一价格拍卖（FPA）→ 行业新趋势（Google/腾讯/百度已切）
│  │  ├─ Shading 策略：对偶梯度下降 / Wasserstein 距离追踪
│  │  └─ 自适应出价（平稳 vs 非平稳环境）
│  └─ 拍卖机制理论基础：激励相容、个体理性、社会福利最大化

├─ 预算 Pacing（平滑投放）
│  ├─ Throttling（简单随机，但易波动）
│  ├─ PID 控制（工业主流，比例-积分-微分反馈）
│  ├─ 对偶方法（理论最优，Lagrangian 松弛处理约束）
│  └─ 预测方法（前瞻性，依赖流量预测准度）

├─ AutoBid 自动出价（RL，Constrained MDP）
│  ├─ [技术] 状态=剩余预算/时段/竞争 → 动作=出价系数 k
│  ├─ [框架] Lagrangian 对偶松弛 CPA/Budget 约束
│  ├─ [工程] 离线模拟器（历史 replay）→ 在线部署
│  ├─ [数字] 转化量+12.3%，CPA 达标率 78%→94.2%
│  └─ [连接] 同一 Lagrangian 框架 ↔ Pacing 对偶梯度 ↔ GRPO

├─ CTR/CVR 预估（核心算法）
│  ├─ 偏差治理三部曲
│  │  ├─ 位置偏差（Position Bias）→ IPS / Clipped IPS
│  │  ├─ 样本偏差（SSB）→ ESMM（全空间多任务，Alibaba）
│  │  │  ├─ [技术] P(CTCVR) = P(CTR) × P(CVR)，共享 Embedding
│  │  │  └─ [数字] CVR AUC +3.4%，冷启动 +5.2%
│  │  └─ 兴趣坍塌 → DIN（Attention 激活相关历史）
│  ├─ 特征工程（展示广告特定）
│  │  ├─ 展示广告字段：广告主、类目、定向、历史 CTR 等
│  │  ├─ 特征交叉：二阶 / 高阶（DeepFM、xDeepFM）
│  │  ├─ 序列特征：用户最近点击广告序列（DIN 激活）
│  │  └─ 多模态特征：图片创意 Embedding、文本 semantic 等
│  └─ 生成式 CTR 范式（新）
│      ├─ 广告文案 → LLM 生成 → CTR 评分 → RLHF 对齐
│      └─ 多模态 CTR（图像+文本联合理解）

├─ 多目标广告优化
│  ├─ [目标] Pareto 最优：收入 × 用户体验 × 广告主 ROI
│  ├─ [框架] 软约束（LTV 长期价值 in loss）+ 硬约束（fill rate）
│  ├─ [技术] MGDA 梯度平衡 + KKT 条件求解
│  ├─ [工程] Pareto 权重可实时调整，无需重训
│  └─ [数字] 留存+2.3%，总收入+4.8%，投诉-18%

├─ 广告出价策略体系（完整演进）
│  ├─ 人口属性定向出价（最基础）
│  ├─ 上下文定向出价（页面内容匹配）
│  ├─ 行为定向出价（用户历史行为）
│  ├─ 相似人群扩展（Lookalike 出价）
│  ├─ 动态创意优化（DCO，为不同用户自适应创意）
│  └─ 价格歧视出价（根据用户 LTV 差异化出价）

└─ LLM 赋能广告系统
   ├─ 创意生成（LLM 文案 + CTR 排序 + A/B 筛选）
   ├─ 关键词推荐（query 扩展）
   ├─ 广告排名质量判别（LLM judge，替代手工规则）
   └─ 冷启动广告主的智能建议（LLM 初始化出价/预算）

【广告系统工程架构】
├─ 实时竞价引擎
│  ├─ 流量来源 → ADX 竞价 → 排序 → 反欺诈 → 计费 → 响应
│  ├─ QPS 支撑：>10 万 QPS / 广告交易所
│  └─ 延迟：<100 ms P99

├─ DSP（广告主侧）
│  ├─ 账户体系、权限控制、财务结算
│  ├─ 定向系统（人群定向、上下文、重定向）
│  ├─ 创意管理（程序化创意、A/B 测试）
│  └─ 预算管理（自动 pacing、成本控制）

├─ SSP（媒体侧）
│  ├─ 流量接入（SDK/API）
│  ├─ 广告位管理
│  ├─ Header Bidding（多 DSP 竞争）
│  ├─ 智能瀑布流（动态底价、流量分配）
│  └─ 广告填充和兜底

└─ DMP（数据层）
   ├─ 用户标签系统
   ├─ 人群包管理和 Lookalike
   └─ 转化分析、多触点归因、LTV 预测
```

---

## 🔗 搜广推跨领域连接（完整版）

### 【数学框架的统一性】
1. **对偶方法统治约束优化**
   - 广告 Pacing（预算约束）→ 对偶梯度下降
   - FPA 出价（CPA/ROI 约束）→ Lagrangian 松弛
   - AutoBid RL（CMDP）→ Lagrangian 松弛 + RL
   - GRPO（KL 约束）→ Lagrangian + PPO clip
   - 推荐多目标（Pareto）→ MGDA 梯度平衡 / 权重线性组合
   - **本质**: 都在用拉格朗日乘子法处理约束 → 同一数学框架

### 【检索范式的演进】
2. **从词汇→语义→推理的递进**
   - BM25（词频统计）→ DPR（稠密向量）→ SPLADE（稀疏神经）→ LLM Reranker（推理重排）
   - 搜索和推荐的召回都遵循"宽→深"漏斗
   - 生成式检索（Semantic ID）打破离散限制，连续化推荐空间

### 【长序列处理的共通策略】
3. **"不用全，只用关键"是核心思路**
   - 搜索：Query-as-Anchor 用查询选历史
   - 推荐：DIN 用目标物品、GEMs 用多流、SIM 用两阶段检索
   - 广告：ESMM 解决样本偏差、CTR 预估用 Attention 机制
   - **本质**: 都在动态选择"当前 context 最相关的历史子集"

### 【偏差治理的统一框架】
4. **无偏估计和 Causal Inference 跨领域应用**
   - 搜索：检索器的位置偏差（IPS 矫正）
   - 推荐：展示偏差（SSB，ESMM）、兴趣偏差（DIN）
   - 广告：点击偏差（反事实估计）、样本选择偏差、归因偏差
   - **本质**: 都需要从观测数据恢复无偏因果效应

### 【生成式范式的三棱镜渗透】
5. **离散→连续→可微分是技术方向**
   - CTR 预估：GSP 竞价 → 生成式 CTR 预测（概率分布）
   - 推荐：协同过滤（离散行为）→ 生成式检索（连续 embedding）→ 扩散/思维链（可推理）
   - 搜索：BM25（词向量）→ Dense 向量 → 生成式排序
   - **本质**: "生成"带来可微分、可扩展、可推理的统一范式

### 【LLM 统一检索和排序】
6. **LLM 作为通用排序/检索/质量判别器**
   - 广告：LLM judge 替代手工规则的质量评分
   - 推荐：LLM Universal Retriever 统一多路召回、OneRec-Think 推理排序
   - 搜索：LLM Reranker、Query 改写、Intent 分类
   - **本质**: LLM 的语言理解能力正在统一搜广推的关键决策点

### 【多目标优化的 Pareto 思想】
7. **软/硬约束混合处理复杂业务目标**
   - 广告：CTR vs 用户体验 vs 广告主 ROI（多目标 pacing）
   - 推荐：点击量 vs 多样性 vs 新鲜度 vs 成本（重排多目标）
   - 搜索：相关性 vs 时效性 vs 覆盖（排序多目标）
   - **本质**: Pareto 前沿求解 = 所有系统的工程本质

### 【KV Cache 与嵌入表的参数优化】
8. **大参数量的存储和推理优化**
   - LLM KV Cache：量化、驱逐、稀疏注意力（正交叠加）
   - 推荐 Embedding Table：参数服务器、在线更新、冷启动初始化
   - 广告特征：高维特征稀疏、特征 embedding 共享
   - **本质**: 都在处理"大而稀疏"的参数优化问题

### 【强化学习在约束下的应用】
9. **从 PPO to GRPO to CMDP 的系列化**
   - LLM 对齐：PPO+RLHF（无 critic）→ GRPO（组内相对优化）
   - 广告出价：CMDP（约束 MDP）→ Lagrangian 对偶 + DQN/A3C
   - 推荐重排：Contextual Bandit（EE 权衡）→ Q-learning 优化
   - **本质**: RL 框架都在处理"长期目标 + 硬约束"的规划问题

### 【工业系统的三层漏斗共性】
10. **"快粗排 → 精排 → 重排"是通用架构**
   - 推荐：召回（千+）→ 粗排（百级）→ 精排（十级）→ 重排（最终）
   - 搜索：初检索（万+）→ 初排（百级）→ 精排（返回前 N）→ 重排（个性化）
   - 广告：流量分发（多 DSP 竞争）→ 初选（相关性）→ 出价排序 → 反欺诈
   - **本质**: 计算成本递增、精度递增的必然工程模式

---

## 📊 完整知识框架（分层）

### 第一层：原语（Primitive）
```
搜索原语：词项权重（TF-IDF） → 词向量密度（Dense） → 词语义扩展（Sparse-Neural）
推荐原语：协同过滤信号 → 内容相似度 → 用户兴趣向量 → 多行为融合
广告原语：查询与广告相关性 → 广告质量度 → 用户 LTV 预估 → 出价竞争
```

### 第二层：模块（Module）
```
召回模块：双塔架构（搜推通用）、图卷积（推荐）、协同过滤（推荐）
排序模块：线性模型（LR） → 树模型（GBDT） → 神经网络（DNN） → Transformer
重排模块：规则引擎 → 多目标优化 → 强化学习 → 生成式（扩散/CoT）
```

### 第三层：系统（System）
```
搜索系统：倒排索引 + BM25 / Dense 向量 + ANN + 神经重排 + 缓存
推荐系统：用户 embedding + 物品 embedding + 上下文特征 → 多层漏斗 + 实时反馈
广告系统：DSP 出价 + ADX 竞拍 + SSP 填充 + 反欺诈 + 归因分析
```

### 第四层：目标（Objective）
```
搜索目标：相关性（NDCG） + 响应速度（延迟） + 覆盖度（长尾）
推荐目标：留存率 + LTV + 多样性 + 新鲜度 + 成本控制
广告目标：收入最大 + 用户体验 + 广告主 ROI + 反欺诈
```

---

## 📈 关键 Paper 与演进路线图

### 搜索系统演进
```
2015  BM25 稳定
2018  BERT 出现 → ColBERT (Late Interaction)
2020  DPR (Dense Passage Retrieval) ← 神经稠密检索 SOTA
2021  SPLADE v1/v2 ← 稀疏神经检索崛起
2022  混合检索最佳实践 ← RRF + 三层漏斗
2023  BGE-M3 ← 统一多路检索（Dense + Sparse + ColBERT）
2024  SPLADE-v3 ← 神经稀疏最新 SOTA（BEIR 52.3）
2025  Query-as-Anchor ← 场景自适应用户表示
2026  LLM Reranker 普及 ← LLM 语言理解完全替代规则
```

### 推荐系统演进
```
2015  Wide&Deep、Deep Learning 普及
2016  DIN (Deep Interest Network) ← 注意力机制引入
2018  DIEN (Deep Interest Evolution) ← 时序建模
2019  MMoE (Multi-task Learning) ← 多目标学习
2020  SIM (Sequential Interest Modeling) ← 超长序列处理
2021  Transformer 在推荐应用 ← Self-attention 排序
2022  生成式推荐探索 ← Seq2Seq、扩散模型实验
2023  Semantic ID (Spotify) ← 生成式检索工业化
2024  GEMs (多流解码) ← 超长序列生成式推荐
2025  OneRec-Think (推理增强) ← CoT 推荐
2026  Wukong (Meta Scaling) ← 推荐 Scaling Law 确立
```

### 广告系统演进
```
2010  GSP 竞价普及
2015  RTB (Real-Time Bidding) 成熟
2017  ESMM (Alibaba) ← 全空间多任务学习
2018  DIN (应用于广告 CTR) ← 注意力的广告化
2019  OCPC/OCPA (自动出价基础)
2020  多目标优化引入 ← 平衡收入与体验
2021  FPA 行业试点（Google、腾讯、百度）
2022  LLM 创意生成起步
2023  AutoBid RL 工业部署 ← Lagrangian 出价 RL
2024  多目标 Pareto 优化普及 ← 软硬约束混合
2025  Query-aware 广告排序 ← 搜推广统一
2026  LLM Judge 替代人工规则 ← 质量评估自动化
```

---

## 🎯 面试高频知识点速记

### 搜索系统
1. BM25 的 TF 饱和和长度归一化
2. Dense vs Sparse 检索的权衡和混合策略
3. ColBERT Late Interaction 为何优于 BERT CLS
4. Query-as-Anchor 的个性化搜索
5. 混合检索的实现方式（RRF / LambdaMART）

### 推荐系统
1. DIN 注意力机制在推荐的应用
2. ESMM 全空间多任务学习解决样本选择偏差
3. 多目标排序的 Pareto 优化
4. 生成式推荐（Semantic ID）vs 传统双塔的对比
5. 超长序列处理（SIM / GEMs / Query-as-Anchor）

### 广告系统
1. GSP vs VCG vs FPA 竞价机制比较
2. 预算 Pacing 的对偶梯度方法
3. AutoBid RL：Constrained MDP 的 Lagrangian 求解
4. 偏差治理三部曲（位置偏差 / SSB / 兴趣坍塌）
5. 多目标广告优化的权重调整策略

### 跨域通用
1. 对偶方法在约束优化中的统一应用
2. 长序列上下文选择的共通思想
3. LLM 赋能搜广推的三个方向
4. 生成式范式的三个演进阶段
5. 工业系统的漏斗架构和延迟预算

---

## 📌 核心文件导航

| 核心系统 | 主要文件 | 覆盖范围 |
|---------|--------|--------|
| **搜索系统** | search/01_search_ranking.md | BM25、Dense、混合、个性化 |
| **推荐系统** | rec-sys/00_overview.md 等 | 召回→粗排→精排→重排 |
| **广告系统** | ads/01_ads_system.md 等 | 竞价、Pacing、出价、多目标 |
| **LLM 基础设施** | knowledge_graph.md | KV Cache、MoE、Attention |
| **最新综述** | synthesis/ (2026-03-xx) | 月度最新论文解读 |

---

## ✨ 本版本新增亮点

✅ **完整广告系统全景**：从竞价机制到多目标优化的端到端设计  
✅ **推荐系统深化**：生成式范式、超长序列、推理增强的新范式  
✅ **搜广推跨域连接**：10+ 个维度的数学和工程统一性  
✅ **工业案例**：Spotify、Alibaba、Meta、Google 真实部署  
✅ **面试高频题**：按岗位分类的快速复习清单  

---

**预期篇幅**：50-80 KB（完整知识体系）  
**更新频率**：月度综合 + 周度论文增量  
**目标读者**：推荐/搜索/广告算法工程师、面试准备者
