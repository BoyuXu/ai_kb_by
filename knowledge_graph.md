# AI-KB 知识图谱

> 最后更新：2026-03-22 | 版本 v4

---

## 📌 核心节点与连接关系

### 🧠 LLM 推理优化

```
KV Cache 内存优化
├── PagedAttention (vLLM) → 解决内存碎片
│   └── [连接] ZSMerge, KVSharer 可叠加在 PagedAttention 上
├── ZSMerge (2503.10714) → 层内残差合并，20:1压缩
│   ├── [技术] Head 级细粒度 + 零样本 + 残差保留信息
│   └── [连接] 与 KVSharer 正交可叠加
├── KVSharer (2410.18517) → 层间不相似共享，30% KV 减少
│   └── [反直觉] 不相似共享 > 相似共享
└── MQA/GQA → 训练阶段减少 KV head（需重训，不同方向）

MoE 推理
├── Expert Parallelism (EP) → 传统 MoE 并行
└── MegaScale-Infer (2504.02263) → 解耦 Attention/FFN
    ├── [技术] 乒乓流水线隐藏 dispatch 延迟
    ├── [技术] M2N 通信库（零拷贝，专为 token dispatch）
    └── [连接] DeepSeek-V3, Qwen3 均为 MoE 架构受益者

Attention 计算优化
├── FA1 (2022) → IO-aware 分块，O(n²)→O(n) 显存
├── FA2 (2023) → 更好并行，Ampere 优化
└── FlashAttention-3 (20260321) → H100 深度优化
    ├── [技术] Warp 专业化：Producer/Consumer 异步流水线
    ├── [技术] Softmax 延迟隐藏（穿插在 WGMMA 等待期）
    ├── [技术] FP8 支持（E4M3 for Q/K, E5M2 for V，120 TFLOP/s）
    ├── [数字] BF16: 35→75% H100 利用率；端到端训练 -25% step time
    └── [连接] 与长序列 LLM、MoE 解耦架构协同收益更大

LLM 对齐/推理训练
├── SFT → 监督微调，格式/风格对齐
├── PPO+RLHF (2017-2023) → 通用对齐，需 Critic 模型
└── GRPO (DeepSeek, 20260321) → 组内相对优化，无 Critic
    ├── [技术] G 个回答组内归一化 advantage，消除 Critic
    ├── [技术] Token-level KL 惩罚 + clip 目标（同 PPO）
    ├── [数字] 节省 50% 显存；DeepSeek-Math 7B MATH +5.2pp
    ├── [扩展] DeepSeek-R1-Zero：纯 GRPO 自发产生 CoT
    └── [适用] 可验证 reward 任务（数学/代码），不适合开放生成

Speculative Decoding
└── Draft Model 对齐 (PEFT, 20260321)
    ├── [技术] 用 LoRA/前缀调整草稿模型对齐大模型分布
    ├── [数字] 接受率提升 ~15%，端到端推理加速 ~1.3x
    └── [连接] 与 FA3 正交叠加，共同降低推理延迟
```

### 🔍 搜索与信息检索

```
词汇检索
└── BM25 → TF-IDF 演进，精确匹配

语义检索（Dense Retrieval）
├── DPR (2020) → 双塔 BERT，向量 ANN
├── ColBERT → Late Interaction，MaxSim
│   └── ColBERT-serve → 内存映射降成本
├── BGE-M3 → 单模型：Dense + Sparse + ColBERT，100+语言
└── W-RAG → 弱监督训练密集检索

神经稀疏检索
├── SPLADE-v1/v2 → BERT MLM Head 生成稀疏权重，词汇扩展
└── SPLADE-v3 (20260321) → 系统性改进
    ├── [技术] 双正则化（FLOP + Saturation），更精细稀疏控制
    ├── [技术] MarginMSE + KL 联合蒸馏
    ├── [技术] INT8 量化感知训练，延迟 -40%
    ├── [数字] BEIR NDCG@10: 52.3（vs BM25: 43.0，vs DPR: 41.2）
    └── [存储] 约 Dense 索引的 1/7，查询延迟 ~8ms

混合检索（Hybrid）
├── RRF 融合 → 同步双路，排名融合（通用主流）
├── SPLADE + Dense + CrossEncoder → 三层漏斗（最优质量）
├── LeSeR (regnlp-1.6) → 两阶段解耦：语义召回→词汇精排
│   └── [场景] 监管/法律/金融垂直领域
└── Hybrid + LLM Re-ranking → 质量优先场景

搜索个性化
└── Query-as-Anchor (20260321) → 查询锚定用户表示
    ├── [技术] 用当前查询作 Attention 的 Query，历史行为作 K/V
    ├── [技术] LLM 场景分类（购物/工作/娱乐）+ 场景感知历史过滤
    ├── [技术] 三层用户表示（会话/中期/长期）动态融合
    ├── [数字] NDCG@10 +6.8%，冷启动用户 +12%
    └── [连接] 与推荐系统的 GEMs 多流架构思想相通

Dense vs Sparse 系统评估 (20260321)
└── [结论] 精确词匹配场景（SKU/品牌词）→ SPLADE；模糊语义查询 → Dense；工业最优 = 混合

[横向连接]
LeSeR 的两阶段思想 ↔ 推荐系统召回→排序的漏斗思想
BGE-M3 的多路统一 ↔ LLM Universal Retriever 的统一框架
Query-as-Anchor 的 context-aware 用户表示 ↔ GEMs 多流解码器（都在动态选择相关历史子集）
SPLADE 稀疏词扩展 ↔ Query Rewriting（都在做查询/文档语义扩展，方向相反）
```

### 🛒 推荐系统

```
召回层（Retrieval）
├── 协同过滤 CF → 行为相似度
├── 双塔 Dense Recall → 向量 ANN（参考 card_002）
├── CascadingRank (2502.11335) → 多行为图排序
│   ├── [技术] 级联行为图 + 三约束迭代排序
│   └── [连接] 解决表示学习过平滑问题
├── LLM Universal Retriever (2502.03041) → 生成式统一召回
│   ├── [技术] 多查询表示 + 矩阵分解 + 概率采样
│   └── [对比] 替代多路专家模型
├── Q' Recall (Bing) → LLM 生成补充候选
└── 生成式检索（Generative Retrieval，20260321系列）
    ├── Semantic ID 基础（Spotify）→ RQ-VAE + Trie 解码，冷启动 +23%
    ├── Variable-Length Semantic ID → 自适应深度，长尾 +8.1%，token -30%
    └── GEMs (20260321) → 多流解码器解决长序列
        ├── [技术] 短期流 + 长期流（压缩）+ 类目流，Cross-Attention 融合
        ├── [技术] 长期流/类目流离线预计算，在线只算短期流
        ├── [数字] 重度用户（>500条历史）HR@10 +15%
        └── [连接] 多流思想 ↔ Query-as-Anchor 的多粒度用户表示

长序列用户建模
├── DIN (2018) → 目标物品注意力，早期查询锚定
├── SIM (2020) → 两阶段历史检索（GSU + ESU）
├── GEMs (20260321) → 多流异步，无截断建模
└── Query-as-Anchor (20260321) → 查询动态选历史，场景感知

排序层（Ranking）
├── MMoE / PLE → 多任务学习（参考 card_001）
├── RankFormer → 图 Transformer 排序
├── AutoIFS → 多场景特征选择
└── Quality-aware Ranking (Bing RecoDCG) → LLM 质量感知

[横向连接]
CascadingRank 的图排序 ↔ 搜索的 BM25 图传播思想
LLM Universal Retriever ↔ Hybrid Search 的统一框架趋势
多行为建模（CascadingRank）↔ 多任务学习（MMoE）都在解决信号稀疏问题
生成式检索（Semantic ID）↔ 传统 Dense 双塔：互补而非替代，组合使用
GEMs 多流 ↔ SIM 两阶段：都在处理超长历史，GEMs 用并行流 SIM 用序列过滤
```

### 📢 在线广告

```
拍卖机制
├── 第二价格拍卖（SPA）→ truthful bidding 最优
└── 第一价格拍卖（FPA）→ 需要 shading（行业趋势）
    ├── 自适应出价（平稳） → 对偶梯度下降
    └── 自适应出价（非平稳） → Wasserstein 距离追踪

预算 Pacing
├── Throttling → 随机参与率（简单粗暴）
├── PID 控制 → 比例-积分-微分反馈（工业主流）
├── 对偶方法 → 理论最优，自适应强
└── 预测方法 → 前瞻性，依赖预测质量

AutoBid（自动出价 RL，20260321）
├── [框架] 带约束 MDP：状态=剩余预算+时段+竞争强度，动作=出价系数 k
├── [技术] Lagrangian Relaxation：CPA/Budget 约束 → 对偶乘子软约束
├── [技术] 离线竞拍模拟器（历史 replay）训练，安全部署
├── [数字] 转化量 +12.3%，CPA 达标率 78%→94.2%，预算利用率 82%→96.1%
└── [连接] RL 对偶乘子 ↔ Pacing 对偶梯度下降（同一数学框架）

多目标广告优化（Multi-Objective, 20260321）
├── [框架] Pareto 最优：收入 × 用户体验 × 广告主 ROI
├── [技术] 动态权重（实时感知广告填充率）+ 软硬约束混合
├── [技术] LTV 长期价值建模纳入优化目标
├── [数字] 次日留存 +2.3%，总收入（含LTV）+4.8%，用户投诉 -18%
└── [连接] LTV 建模 ↔ 推荐系统的长期留存优化（同一目标：防止短视）

LLM 广告创意生成（20260321）
└── [场景] LLM 生成广告文案 → 大规模 A/B 筛选 → 最优创意投放

[横向连接]
FPA 对偶出价 ↔ Pacing 的对偶梯度下降（同一思想：拉格朗日松弛处理约束）
AutoBid RL ↔ GRPO：都用 Lagrangian 处理约束，RL 解序列决策
广告出价策略 ↔ 推荐排序：都在解决 explore/exploit 权衡
预算 Pacing ↔ 推荐多目标优化：都需要平衡短期信号和长期目标
多目标广告（软硬约束混合）↔ 推荐多目标（Pareto 排序）：同一数学问题，不同工程实现
```

---

## 📊 技术演进时间线

```
2020  DPR（Dense Retrieval）、ColBERT
2021  DSSM 大规模应用、多路召回工业化
2022  MQA/GQA（减少 KV head）、PPO for RLHF
2023  PagedAttention (vLLM)、FlashAttention-2
2024  KVSharer、BGE-M3、LLM 赋能推荐实验
      FPA 行业切换（Google/腾讯/百度）
2025  ZSMerge、MegaScale-Infer（MoE解耦）
      LeSeR（垂直领域混合检索）
      LLM Universal Retriever 工业落地（+3%核心指标）
      Bing Explore Further（Q' Recall 生产部署）
      CascadingRank（多行为图排序 +9.56% HR）
2026  FlashAttention-3（H100 Hopper 深度优化，BF16/FP8，75%利用率）
      GRPO（DeepSeek-R1 训练算法，无 Critic RL）
      SPLADE-v3（神经稀疏检索新 SOTA，BEIR 52.3）
      生成式检索规模化（Spotify播客 +8.3%，Variable-Length ID，GEMs多流）
      AutoBid RL（Lagrangian 约束 RL 出价，+12.3% 转化量）
      Query-as-Anchor（查询锚定个性化搜索，+6.8% NDCG）
```

---

## 🔑 关键技术概念索引

| 概念 | 卡片/文件 | 领域 |
|------|----------|------|
| KV Cache 压缩（层内+层间） | #006 | LLM Infra |
| MoE 解耦推理 | #007 | LLM Infra |
| 混合检索演进 | #008 | 搜索 |
| LLM 赋能推荐召回 | #009 | 推荐 |
| 广告预算 Pacing | #010 | 广告 |
| 多任务学习（MMoE/PLE） | #001 | 推荐 |
| 双塔召回 | #002 | 推荐 |
| OCPC/OCPA 出价 | #003 | 广告 |
| Query Rewriting / RAG | #004 | 搜索/NLP |
| 推荐系统全链路 | #005 | 推荐 |
| GRPO（无 Critic RL 对齐）| synthesis/20260321_grpo_rl_alignment.md | LLM 训练 |
| Semantic ID + 生成式检索 | synthesis/20260321_semantic_id_generative_retrieval.md | 推荐召回 |
| 广告出价体系演进 | synthesis/20260321_ad_bidding_evolution.md | 广告 |
| SPLADE-v3 稀疏检索 | synthesis/20260321_sparse_dense_retrieval.md | 搜索 |
| FlashAttention-3 | synthesis/20260321_flashattention3_llm_infra.md | LLM Infra |

---

## 📅 2026-03-22 新增节点

### 广告系统 - 偏差治理三部曲
```
广告偏差治理
├── 位置偏差（Position Bias）
│   └── 反事实学习（IPS + Clipped IPS）
│       └── [连接] 工业用随机化实验估计倾向分，截断防高方差
├── 样本选择偏差（SSB）
│   └── ESMM (SIGIR'18, Alibaba)
│       ├── [技术] P(CTCVR) = P(CTR) × P(CVR)，全空间建模
│       ├── [技术] 共享 Embedding 解决 CVR 数据稀疏
│       └── [数字] CVR AUC +3.4%，冷启动 +5.2%
└── 兴趣坍塌（Interest Collapse）
    └── DIN (KDD'18, Alibaba)
        ├── [技术] Attention 激活相关历史，Sigmoid 非 Softmax
        └── [连接] → DIEN（加时序）→ SIM（超长序列）演进线
```

### 广告系统 - RL 出价与多目标
```
广告出价 RL 化（连接 20260321 出价演进）
└── AutoBid (Constrained MDP)
    ├── [技术] CMDP + Lagrangian 松弛 + 双 Critic
    ├── [连接] 同一 Lagrangian 框架 ↔ 昨日 FPA/Pacing
    └── [数字] 转化量+12.4%，预算利用率98.5%

多目标广告系统
└── Pareto 优化（收入 vs 体验）
    ├── [技术] MGDA 梯度平衡 + KKT 约束
    └── [工程] Pareto 标量化权重可实时调整（无需重训）
```

### 推荐系统 - Semantic ID 深化
```
Semantic ID 技术线（连接 20260321）
├── 可变长度 Semantic ID（今日新）
│   ├── [技术] 冷启动 item → 长 ID，热门 item → 短 ID
│   ├── [数字] 平均 ID 长度减少30%，冷启动 Recall +15%
│   └── [工程] [EOS] token 标记 ID 结束，ANN 检索后截断
├── Spotify 大规模部署（播客 500 万+）
│   ├── [技术] RQ-VAE 3-4 层，多模态融合
│   ├── [数字] 冷启动 Recall +25%，在线收听时长 +8%
│   └── [工程] 每日重建索引，p99 延迟 <50ms
└── 统一语言模型（Spotify ULM）
    ├── [技术] 搜索+推荐+推理统一 backbone，task prefix 切换
    ├── [技术] 3:2:1 采样比，轻量 task head（各 ~5M params）
    ├── [数字] 参数量 3.6B → 1.4B，搜索 +3.1%，推荐 +4.8%
    └── [连接] → 跨领域统一模型趋势（参见第7条跨域连接）
```

### 推荐系统 - 生成范式新枝
```
扩散生成式推荐（DiffGRM）
├── [技术] 连续嵌入空间扩散，双向 Attention，条件去噪
├── [对比] 无曝光偏差（vs 自回归），多样性更高
├── [数字] HR@10 +8.3% vs TIGER，多样性 +12%
└── [连接] → 扩散模型渗透推荐是 2024-2025 新趋势

超长序列生成式推荐（GEMs）
├── [技术] 多流解码器，O(n²) → O(n²/k)
├── [数字] 序列长度2048时内存-4×，速度+3.2×
└── [适用] 超长历史用户（视频/音乐/电商老用户）
```

### 搜索系统 - 稀疏检索完善
```
SPLADE-v3（连接 20260321 sparse/dense eval 深化）
├── [技术] DeBERTa 基座 + Dense 教师蒸馏 + INT8 量化
├── [数字] BEIR NDCG@10=0.546，超 BM25(0.427)，接近 Dense
└── [场景] 精确查询用 Sparse，语义查询用 Dense，长尾用混合

Query-as-Anchor（场景自适应用户表示）
├── [技术] 以 query 为锚点 cross-attention 激活相关历史
├── [连接] ↔ DIN 的目标物品注意力（同一思路，搜索+推荐）
└── [数字] NDCG@10 +9.3%，多品类用户 +14.7%
```

### LLM 基础设施 - 今日补充
```
Speculative Decoding 工程化（Draft 对齐）
├── [技术] LoRA 对齐 draft → target 分布（KL 蒸馏/DPO）
├── [数字] 接受率 62%→83%，端到端加速 1.8×→2.7×
└── [连接] → FlashAttention-3 正交叠加

MoE-LLaMA 工程部署
├── [技术] Expert Parallelism + Offloading + FP8 量化
└── [连接] MoE 是 2024+ 主流架构（DeepSeek V2/V3，Qwen3）

LLM 广告创意生成
├── [技术] LLM 生成 + CTR 排序 + RLHF reward 对齐
└── [工程] 预生成缓存，在线按用户画像选择（O(1)延迟）
```

---

## 🔗 跨领域连接（重点）

1. **对偶方法的统一性**：广告 Pacing、FPA 出价、AutoBid RL（Lagrangian）、资源分配都用拉格朗日对偶处理约束 → 同一数学框架；GRPO 的组内归一化也是对偶思想在 RL 的体现

2. **LLM 作为统一检索器**：搜索的 BGE-M3（统一多路检索）↔ 推荐的 URM（统一多目标召回）↔ 广告的 LLM judge（统一质量评估）→ LLM 通用化趋势

3. **KV Cache 优化与 MoE 的关系**：MoE 模型（DeepSeek-V3、Qwen3）规模大，KV 压缩更关键；FA3 的 Warp 专业化 ↔ MoE 的 Expert 专业化（都是分工思想）

4. **图方法的跨领域复现**：CascadingRank（推荐图排序）↔ BM25（词图传播）↔ PageRank（页面图排序）→ 图传播是搜广推的底层思想

5. **"动态历史选择"的跨领域统一**：Query-as-Anchor（搜索：查询锚定历史注意力）↔ GEMs（推荐：多流架构动态融合）↔ DIN/SIM（推荐：目标物品注意力过滤历史）→ 本质都是"不用全部历史，用当前 context 选择相关历史子集"

6. **生成式范式向搜广推渗透**：GRPO（生成式 RL 训练）+ 生成式检索（Semantic ID）+ LLM 广告创意生成 → "生成"正在统一搜广推的各个环节

7. **稀疏 vs 稠密的共存**：SPLADE-v3（稀疏神经检索）≠ Dense 的替代，而是互补；同理推荐中精确匹配（商品 SKU）↔ 语义召回（用户偏好）也是两条腿

8. **偏差治理是广告系统的基础工程**（20260322）：位置偏差（IPS）→ 样本偏差（ESMM）→ 兴趣偏差（DIN Attention）→ 多目标偏差（Pareto），层层递进构建无偏广告系统

9. **统一化是 2024-2025 的系统趋势**（20260322）：Spotify ULM（搜推统一）↔ ESMM（全空间统一）↔ 扩散生成（离散连续统一）→ "大一统"在各维度同步发生

---
