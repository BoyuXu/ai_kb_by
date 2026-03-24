# 知识卡片 #005：推荐系统全链路串联

> 📚 参考文献
> - [Action Is All You Need Dual-Flow Generative Ran...](../../ads/papers/20260323_action_is_all_you_need_dual-flow_generative_ranking.md) — Action is All You Need: Dual-Flow Generative Ranking Netw...
> - [Counterfactual-Learning-For-Unbiased-Ad-Ranking...](../../ads/papers/20260321_counterfactual-learning-for-unbiased-ad-ranking-in-industrial-search-systems.md) — Counterfactual Learning for Unbiased Ad Ranking in Indust...
> - [Linear-Item-Item-Session-Rec](../../rec-sys/papers/20260319_linear-item-item-session-rec.md) — Linear Item-Item Model with Neural Knowledge for Session-...
> - [Jensen-Gap-Fairness-Rec](../../rec-sys/papers/20260319_jensen-gap-fairness-rec.md) — Bridging Jensen Gap for Max-Min Group Fairness Optimizati...
> - [Query As Anchor Llm User Repr](../../search/papers/20260322_query_as_anchor_llm_user_repr.md) — Query as Anchor: Scenario-Adaptive User Representation vi...
> - [Query-As-Anchor-Scenario-Adaptive-User-Represen...](../../search/papers/20260321_query-as-anchor-scenario-adaptive-user-representation-via-large-language-model-for-search.md) — Query as Anchor: Scenario-Adaptive User Representation vi...
> - [Multi-Objective-Optimization-For-Online-Adverti...](../../ads/papers/20260321_multi-objective-optimization-for-online-advertising-balancing-revenue-and-user-experience.md) — Multi-Objective Optimization for Online Advertising: Bala...
> - [Esmm-Cvr](../../ads/papers/20260317_esmm-cvr.md) — ESMM：全空间多任务 CVR 预估


> 创建：2026-03-20 | 领域：推荐系统·全局视野 | 难度：⭐⭐⭐⭐

---

## 🌟 一句话解释

推荐系统是一个从「亿级物料库」到「几十条精准结果」的**多阶段漏斗**，每个阶段权衡效率与精度，最终通过多目标优化实现用户体验和商业目标双赢。

---

## 🏗️ 全链路架构

```
亿级物料库
    │
    ▼
【召回层】多路并行，高召回率
├── 双塔向量召回（行为相关）
├── 协同过滤（I2I / U2I）
├── 图召回（GRIT / U2I2I）
├── 内容召回（文本/图像相似度）
└── 规则召回（新品保量 / 运营策略）
    │ ~1万候选
    ▼
【粗排层】轻量交互模型，快速过滤
├── 少量交叉特征（User-Item Interaction）
├── 低延迟（<5ms）
└── 通常是 Embedding + 轻量 MLP
    │ ~1000候选
    ▼
【精排层】完整交叉特征，多目标预估
├── CTR 预估（DeepFM / DCN / DIN）
├── CVR 预估（ESMM / AITM）
├── 多任务：MMoE / PLE 同时优化多个目标
└── 行为序列建模（DIN / SIM / Transformer）
    │ ~200候选
    ▼
【重排层】多样性与业务约束
├── 多样性（MMR / DPP 行列式点过程）
├── 打散（同类目/同作者物品去重）
├── 强插（广告 / 运营位 / 保量策略）
└── 公平性约束（Jensen Gap / 弹性方法）
    │ ~50展示结果
    ▼
用户看到的推荐列表
```

---

## 🔄 各阶段模型演进

### 召回演进
| 时代 | 代表模型 | 核心改进 |
|------|---------|---------|
| 2013 | ALS矩阵分解 | 隐式反馈协同过滤 |
| 2016 | YouTube DNN | Deep Learning + 行为序列 |
| 2019 | MIND | 多兴趣胶囊网络 |
| 2020 | Facebook EBR | Hard Negative Mining |
| 2022+ | 图召回 | 多跳关系建模 |

### 排序演进
| 时代 | 代表模型 | 核心改进 |
|------|---------|---------|
| 2016 | Wide&Deep | 记忆+泛化联合 |
| 2017 | DeepFM | FM+DNN端到端 |
| 2018 | DIN | 注意力机制建模用户兴趣 |
| 2018 | MMoE | 多任务负迁移解决 |
| 2019 | ESMM | CVR偏差修正 |
| 2020 | SIM | 超长行为序列 |
| 2021 | PLE | 进阶多任务 |
| 2022+ | LLM增强 | 语义特征+指令微调 |

---

## 🎯 核心技术串联图

```
特征工程
├── 用户特征：ID嵌入 + 画像 + 统计
├── 物品特征：ID嵌入 + 内容 + 统计
└── 上下文：时间/位置/设备

       ↓ 喂给

召回模型（双塔）─────── 负采样策略
   用户向量 ←→ 物品向量     in-batch + hard negative
                            sampling bias correction

       ↓ 候选集

排序模型（MMoE/DIN）── 行为序列建模
   CTR Tower               DIN: 注意力机制
   CVR Tower ←─ ESMM      SIM: 检索式长序列
   多目标融合              Transformer: 全局依赖

       ↓ 得分

重排（DPP/MMR）─────── 业务约束
   多样性                  强插规则
   公平性                  A/B实验框架
```

---

## 🏭 工业实践要点

1. **离线指标 vs 在线效果**：AUC提升0.001不一定带来线上CTR正收益，必须A/B验证
2. **特征重要性**：行为序列特征 > 统计特征 > ID特征，内容特征在冷启动最关键
3. **实时推荐**：用户实时行为（过去5分钟点击）比历史行为更重要，需流式特征平台
4. **探索与利用（E&E）**：纯利用导致信息茧房，需ε-greedy或UCB策略保持探索
5. **位置偏差**：精排得分高不等于展示在好位置，需要position bias建模

---

## 🎯 面试考点（综合型）

**Q1：如果离线AUC高但线上CTR没有提升，可能是什么原因？**
> ①训练/线上特征分布不一致（Feature Drift）；②样本构造问题（离线负样本不代表线上）；③位置偏差：AUC没有考虑曝光位置；④模型训练数据与真实推荐场景的分布偏差（Exposure Bias）；⑤重排阶段规则强插覆盖了精排结果。

**Q2：推荐系统中如何处理冷启动问题？**
> **物品冷启动**：内容特征（图文Embedding）做内容召回，不依赖行为数据；增加曝光保量；迁移学习（相似物品的ID Embedding）。**用户冷启动**：注册时收集偏好（Onboarding）；人口统计学特征替代行为特征；协作过滤（根据少量点击找相似用户）；UCB保证探索。

**Q3：多目标推荐中如何平衡短期CTR和长期用户价值？**
> 短期：CTR高但可能是标题党（点击->快速滑走）；长期：完播率、收藏、用户留存更有价值。解决：①多目标联合优化（权重需调）；②加入负向信号（举报/不感兴趣）；③时长/完播率作为额外目标；④双Q值框架（即时奖励 + 长期价值）。

---

## 🔗 知识关联图

```
特征工程 → [双塔召回] → [粗排] → [MMoE精排] → [重排多样性]
              ↑              ↑           ↑
         负采样策略      行为序列      多任务联合
         (in-batch      (DIN/SIM)    (ESMM/PLE)
         +hard neg)
              
广告系统 = 推荐 + [CTR×CVR出价] + [竞价机制GSP/VCG] + [oCPC控制]
搜索系统 = 推荐 + [Query理解] + [BM25+向量混合] + [Reranking]
```

---

> 💡 **面试黄金法则**：推荐系统面试必考"你对整个链路的理解"，要从全局视角说清楚每个阶段的目标、难点和代表方法，再深挖某个你擅长的点。
