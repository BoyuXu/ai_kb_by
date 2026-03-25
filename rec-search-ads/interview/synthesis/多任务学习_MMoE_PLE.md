# 知识卡片 #001：多任务学习（MMoE / PLE）

> 📚 参考文献
> - [Moe-Llama Mixture-Of-Experts For Efficient Larg...](../../../llm-agent/llm-infra/20260323_moe-llama_mixture-of-experts_for_efficient_large_la.md) — MoE-LLaMA: Mixture-of-Experts for Efficient Large Languag...
> - [Moe-Llama-Mixture-Of-Experts-Efficient-Llm-Serving](../../../llm-agent/llm-infra/20260321_moe-llama-mixture-of-experts-efficient-llm-serving.md) — MoE-LLaMA: Mixture-of-Experts for Efficient Large Languag...
> - [Moe-Llama-Mixture-Of-Experts-For-Efficient-Larg...](../../../llm-agent/llm-infra/20260321_moe-llama-mixture-of-experts-for-efficient-large-language-model-serving.md) — MoE-LLaMA: Mixture-of-Experts for Efficient Large Languag...
> - [Esmm-Cvr](../../ads/papers/esmm_cvr.md) — ESMM：全空间多任务 CVR 预估
> - [Action Is All You Need Dual-Flow Generative Ran...](../../ads/papers/Action_is_All_You_Need_Dual_Flow_Generative_Ranking_Netwo.md) — Action is All You Need: Dual-Flow Generative Ranking Netw...
> - [Llm-Enhanced-Ad-Creative-Generation-And-Optimiz...](../../ads/papers/LLM_Enhanced_Ad_Creative_Generation_and_Optimization_for.md) — LLM-Enhanced Ad Creative Generation and Optimization for ...
> - [Mmoe-Multi-Task-Learning](../../rec-sys/papers/MMoEMulti_gate_Mixture_of_Experts.md) — MMoE：多门控混合专家（Multi-gate Mixture-of-Experts）
> - [Llm-For-Ir-Survey](../../search/papers/Large_Language_Models_for_Information_Retrieval_A_Survey.md) — Large Language Models for Information Retrieval: A Survey


> 创建：2026-03-20 | 领域：推荐系统·排序 | 难度：⭐⭐⭐


## 📐 核心公式与原理

### 1. 推荐系统漏斗
$$\text{全量} \xrightarrow{召回} 10^3 \xrightarrow{粗排} 10^2 \xrightarrow{精排} 10^1 \xrightarrow{重排} \text{展示}$$
- 逐层过滤，平衡效果和效率

### 2. CTR 预估
$$pCTR = \sigma(f_{DNN}(x_{user}, x_{item}, x_{context}))$$
- 排序核心：预估用户点击概率

### 3. 在线评估
$$\Delta metric = \bar{X}_{treatment} - \bar{X}_{control}$$
- A/B 测试量化策略效果

---

## 🌟 一句话解释

多任务学习让模型同时优化多个目标（点击率、转化率、时长），**MMoE 通过"多个专家 + 各任务独立门控"解决任务冲突导致的负迁移**。

---

## 🎭 生活类比

想象一家餐厅需要同时做中餐、日料、西餐：
- **Shared-Bottom（硬共享）**：三种厨师共用同一个厨房配方，做出来的东西四不像
- **MMoE**：有8个专长不同的厨师（Experts），中餐主厨（任务门控）决定今天用哪3个厨师协作，日料主厨用另一组组合——各取所需，互不干扰
- **PLE**：除了共用厨师，中餐还有专属的2个厨师，绝对不参与日料，防止技术污染

---

## ⚙️ 核心机制

```
输入 x
  │
  ├──▶ Expert_1(x)  ─┐
  ├──▶ Expert_2(x)   │
  ├──▶ Expert_3(x)   │  ← K个专家，每个是独立MLP
  │  ...              │
  └──▶ Expert_K(x)  ─┘
          │
    ┌─────┴──────┐
    │            │
  Gate_A(x)   Gate_B(x)  ← 每个任务的独立软注意力（Softmax(W·x)）
    │            │
  f_A(x)      f_B(x)    ← 加权融合各专家输出
    │            │
  Tower_A    Tower_B    ← 各任务独立预测头
    │            │
  y_CTR      y_CVR
```

**PLE 改进**：区分 Shared Experts（所有任务共用）和 Task-specific Experts（任务专属），层次化提取。

---

## 🔄 横向对比

| 方法 | 参数共享方式 | 任务冲突处理 | 适用场景 |
|------|------------|------------|---------|
| Shared-Bottom | 完全共享底层 | ❌ 无 | 任务高度相关 |
| One-gate MoE | 软共享（单门控） | 弱 | 一般场景 |
| MMoE | 软共享（任务独立门控） | ✅ 强 | 任务相关性低 |
| PLE | 硬+软混合，分层提取 | ✅✅ 最强 | 工业级多任务 |

---

## 🏭 工业落地

- **阿里 ESMM**：CTR × CVR 联合建模，在全空间训练 CVR（解决样本偏差）
- **腾讯 PLE**：微信视频号多目标排序，同时优化播放完成率、点赞、分享
- **快手 AITM**：转化路径建模（展示→点击→关注→购买），用 Adaptive Information Transfer
- **YouTube**：同时优化 CTR + 观看时长，时长用回归而非分类避免标签偏差

**工程注意事项：**
1. 专家数 K 通常 4~8，过多导致专家坍塌（collapse）
2. 各任务 loss weight 需调优，梯度量级差距大时用 GradNorm
3. 门控加温度参数防止 winner-take-all

---

## 🎯 面试考点

**Q1（基础）：什么是负迁移？为什么 Shared-Bottom 会产生？**
> 任务目标不一致时（如 CTR 与 CVR 梯度方向可能相反），共享参数被矛盾梯度拉扯，每个任务都无法学好。

**Q2（中等）：MMoE 中如果所有任务门控权重都集中在同一个 Expert，会怎样？**
> 退化成 Shared-Bottom，其他 Expert 梯度稀疏、参数得不到训练。解决：Expert 均衡损失（类 Switch Transformer）、Expert Dropout。

**Q3（高难）：PLE 和 MMoE 最本质的区别是什么？**
> PLE 将 Shared 和 Task-specific Experts 物理隔离，task-specific experts 的梯度不会流向其他任务，从根本上杜绝梯度污染；MMoE 虽然多门控，但所有 Expert 对所有任务都可见，梯度依然会交叉影响。

---

## 🔗 知识关联

- 上游：特征工程 → 用户行为序列建模（DIN/SIM）
- 同层：ESMM（CVR 偏差修正）、MMOE 变体（MoE-LLM）
- 下游：排序后的重排（MMR、DPP 多样性）

### Q1: 面试项目介绍的 STAR 框架？
**30秒答案**：Situation（背景）→Task（任务）→Action（方案）→Result（结果）。关键：量化结果（AUC +0.5%, 线上 CTR +2%），突出个人贡献，准备 follow-up 追问。

### Q2: 算法面试如何展现系统性思维？
**30秒答案**：①先说全局架构再说细节；②主动分析 trade-off；③提及工程约束（延迟/资源）；④讨论 A/B 测试验证；⑤对比多种方案优劣。

### Q3: 面试中遇到不会的问题怎么办？
**30秒答案**：①诚实说不了解具体细节；②从已知相关知识推导思路；③说明学习路径（"我会从 XX 论文入手了解"）。比胡编强 100 倍。

### Q4: 简历中项目经历怎么写？
**30秒答案**：①每个项目 3-5 行；②突出方法创新点和业务效果；③用数字量化（AUC/CTR/时长提升 X%）；④技术关键词匹配 JD；⑤按相关度排序而非时间顺序。

### Q5: 如何准备系统设计面试？
**30秒答案**：①准备推荐/搜索/广告各一个完整系统设计；②每个系统能说清召回→排序→重排全链路；③准备 scalability 方案（如何从百万到亿级）；④准备 failure mode 和降级方案。

### Q6: 八股文和实际项目经验如何结合？
**30秒答案**：八股文提供理论框架，项目经验证明落地能力。面试时：先用八股文回答「是什么/为什么」，再用项目经验回答「怎么做/效果如何」。纯八股文没有竞争力。

### Q7: 面试中如何展示 leadership？
**30秒答案**：①描述自己在项目中的角色和贡献；②说明如何推动跨团队协作；③展示主动发现问题并推动解决的案例；④分享技术方案选型的决策过程。

### Q8: 被问到不会的论文怎么办？
**30秒答案**：①说清楚自己了解的相关工作；②从论文标题推断可能的方法（如 xxx for recommendation 可能是把 xxx 技术迁移到推荐）；③承认不了解但表达学习意愿。

### Q9: 算法岗面试的常见流程？
**30秒答案**：①简历筛选→②一面（算法基础+项目）→③二面（系统设计+深度追问）→④三面（部门 leader，考察思维+潜力）→⑤HR 面→Offer。每轮约 45-60 分钟。

### Q10: 如何准备不同公司的面试？
**30秒答案**：①字节：重工程实现+大规模系统+实际效果；②阿里：重业务理解+电商场景+系统设计；③腾讯：重算法深度+创新性+论文理解；④快手/小红书：重内容推荐+短视频场景+多模态。
