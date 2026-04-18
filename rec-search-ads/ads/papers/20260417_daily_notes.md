# 广告系统论文笔记 — 2026-04-17

## 1. CTCVR Optimization with Heterogeneous Hierarchical Decoder (GPR 框架核心)

**来源：** https://arxiv.org/abs/2511.10138 (Tencent)
**领域：** CTCVR 优化 × 生成式广告推荐
**核心定位：** GPR 框架中的核心组件 — 异构层级解码器（HHD）

**核心贡献：**
- 双解码器架构：解耦用户意图建模（粗粒度）与广告生成（细粒度）
- 统一 token schema 编码广告上下文多个维度
- 三阶段训练管线：MTP 预训练 → VAFT 价值对齐 → HEPO 策略优化

**工业验证：** 部署于微信视频号广告，CTCVR 和 GMV 指标显著提升

**面试考点：** HHD 双解码器训练效率 vs 推理灵活性 trade-off、token 类型设计动机

---

## 2. Value-Aware Fine-Tuning (VAFT) for Advertising CTR Prediction

**来源：** GPR 框架 (Tencent)
**领域：** CTR 预估 × 价值感知训练
**核心创新：** 训练 loss 中融入商业价值信号（eCPM）进行加权

**关键公式：**
- VAFT loss: L = Σ(weight_i × loss_i)
- weight 融合行为类型（点击/转化/支付）和归一化 eCPM
- 实现预测目标与商业价值的对齐

**面试考点：** VAFT vs 传统 CTR-only 训练的区别、eCPM 加权的数值稳定性、多目标对齐

---

## 3. IDProxy: Cold-Start CTR Prediction at Xiaohongshu with Multimodal LLMs

**来源：** https://arxiv.org/abs/2603.01590 (Xiaohongshu)
**领域：** 冷启动 × 多模态 LLM × CTR 预估
**核心问题：** 新物品无历史行为数据，传统 ID embedding 失效

**核心方法：**
- 多模态 LLM 从内容信号（文本+图片）生成代理嵌入
- 轻量级粗到精对齐机制
- 代理嵌入与已有 ID embedding 空间显式对齐
- 端到端 CTR 目标优化

**关键公式：**
- Proxy 对齐: minimize ||proxy_embed - id_embed||² (热门物品)
- CTR loss: BCE(pred_ctr, label) 融合 proxy embeddings

**工业验证：** 部署于小红书内容 Feed 和展示广告

**面试考点：** 如何不微调 MLLM 实现 proxy 对齐、哪些内容信号最有信息量、持续演化中的粗到精机制

---

## 4. OneTrans: Scaling Up One-Stage Transformers for CTR Prediction

**来源：** https://arxiv.org/abs/2510.26104 (ByteDance)
**领域：** CTR 预估 × 统一 Transformer
**核心贡献：** 统一特征交互和序列建模到单一 Transformer，验证统一扩展优于分别扩展

**关键结果：**
- OneTrans-L (330M params): CTR AUC +1.53%, CVR AUC +1.14% (vs DCNv2+DIN)
- 在线 GMV +5.68%

**面试考点：** 统一扩展 vs 分别扩展的实验证据、KV caching 在广告推理中的优化

---

## 5. Real-Time Bidding by Reinforcement Learning in Display Advertising

**来源：** https://arxiv.org/abs/1701.02490
**领域：** RTB × 强化学习
**核心贡献：** 将 RTB 出价优化建模为 MDP，无需完整市场信息学习最优出价策略

**关键技术：**
- 状态空间：竞价信息 + 广告活动参数
- 动作：出价价格
- 神经网络近似状态价值函数
- Constrained MDP (CMDP) 处理预算约束

**关键公式：** V(s) = E[r + γV(s')]，神经网络近似 V_θ(s)

**面试考点：** MDP 建模竞价动态、CMDP vs 无约束 MDP 的预算处理、神经网络近似 vs 表格 Q-learning

---

## 6. HLLM-Creator: Hierarchical LLM-based Personalized Creative Generation

**来源：** https://arxiv.org/abs/2508.18118 (Douyin/ByteDance)
**领域：** LLM × 个性化广告创意生成
**核心贡献：** 三层 LLM 分层框架实现高效个性化广告标题生成

**三层架构：**
1. **Item LLM：** 编码广告标题
2. **User LLM：** 聚合用户历史行为
3. **Creative LLM：** 生成个性化标题

**效率优化：** 用户聚类 + 匹配预测剪枝
**数据构建：** Chain-of-thought 数据构建应对稀缺性

**工业验证：** 部署于抖音搜索广告，广告指标 +0.476%

**面试考点：** 三层分解 vs 端到端的优劣、CoT 数据构建方法、剪枝策略效率贡献

---

## 7. PRO-Bid: Constraint-Aware Generative Auto-bidding via Pareto-Prioritized Regret Optimization

**来源：** https://arxiv.org/abs/2602.08261
**领域：** 智能出价 × 生成式框架
**核心贡献：** 将全局比率约束分解为递归状态流

**关键技术：**
- **CDPR (Constraint-Decoupled Pareto Representation)：** 
  - RRV (Remaining Required Value) = target_value - cumsum_value[:t]
  - RAC (Remaining Allowable Cost) = budget - cumsum_cost[:t]
- Return-to-Go (RTG) 条件化增强
- Pareto 优先化遗憾优化

**面试考点：** RRV/RAC 分解 vs 直接约束表示的优势、Pareto 优先化的遗憾上界、prompt-based vs CDPR 的扩展性

---

## 8. RTBAgent: A LLM-based Agent System for Real-Time Bidding

**来源：** https://arxiv.org/abs/2502.00792 (ACM WWW 2025)
**领域：** LLM Agent × 实时竞价
**核心创新：** 首个 LLM 驱动的 RTB Agent 系统

**系统设计：**
- 4 工具：CTR 预估器、策略库、出价计算器、历史检索器
- 3 类记忆：历史决策、交易记录、专家知识
- 两步决策：规划 → 执行
- 每日反思循环

**工业验证：** 在真实广告数据集上显著优于 RL 基线（RLB, DRLB, USCB）

**面试考点：** LLM 推理 vs 纯 RL 在市场动态中的优势、专家知识库价值、每日反思机制转化为出价调整

---

## 9. GPR (Ads Version): Generative Pre-training Approach for Ads Ranking

**来源：** https://arxiv.org/abs/2506.07634
**领域：** 生成式预训练 × 广告排序
**核心贡献：** 生成式预训练在广告排序任务的具体应用

**与 rec-sys GPR 的关系：** 共享统一模型哲学，但聚焦广告排序的特定优化

---

## 10. LLM-Generated Ads: From Personalization Parity to Persuasion Superiority

**来源：** 工业研究
**领域：** LLM × 广告文案生成
**核心发现：**
- LLM 生成的个性化广告与人类写作统计平价
- 基于人格特质的说服力定向显示 LLM 优势
- 特定人格维度下 LLM 广告表现超越人类

**面试考点：** LLM vs 人类广告的统计对比方法论、人格特质感知的 prompt 工程、大规模 LLM 广告生成的可行性
