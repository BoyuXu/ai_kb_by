# 知识卡片 #003：oCPC/oCPA 智能出价系统

> 📚 参考文献
> - [Ocpc Ocpa Deep](../../ads/04_bidding/papers/ocpc_ocpa_deep.md) — 深度转化出价：oCPC/oCPA原理与工程落地
> - [Esmm-Cvr](../../ads/02_rank/papers/esmm_cvr.md) — ESMM：全空间多任务 CVR 预估
> - [Gsp-Vcg-Auction](../../ads/04_bidding/papers/gsp_vcg_auction.md) — 广告竞价机制：GSP 与 VCG 详解
> - [Budget-Pacing-Strategies-For-Large-Scale-Ad-Cam...](../../ads/04_bidding/papers/Budget_Pacing_Strategies_for_Large_Scale_Ad_Campaigns.md) — Budget Pacing Strategies for Large-Scale Ad Campaigns
> - [Practical-Guide-Budget-Pacing](../../ads/04_bidding/papers/A_Practical_Guide_to_Budget_Pacing_Algorithms_in_Digital.md) — A Practical Guide to Budget Pacing Algorithms in Digital ...
> - [Real-Time-Bidding-Optimization-With-Multi-Agent...](../../ads/04_bidding/papers/Real_Time_Bidding_Optimization_with_Multi_Agent_Deep_Rein.md) — Real-Time Bidding Optimization with Multi-Agent Deep Rein...
> - [Ocpc-Ocpa-Optimization](../../ads/04_bidding/papers/ocpc_ocpa_optimization.md) — oCPC/oCPA 优化：原理、工程实现与调优
> - [Gsp-Vcg-Auction](../../ads/04_bidding/papers/gsp_vcg_auction_v2.md) — GSP/VCG 拍卖机制（广告竞价理论）

> 创建：2026-03-20 | 领域：广告系统·出价 | 难度：⭐⭐⭐

## 📐 核心公式与原理

### 1. 推荐系统漏斗

$$
\text{全量} \xrightarrow{召回} 10^3 \xrightarrow{粗排} 10^2 \xrightarrow{精排} 10^1 \xrightarrow{重排} \text{展示}
$$

- 逐层过滤，平衡效果和效率

### 2. CTR 预估

$$
pCTR = \sigma(f_{DNN}(x_{user}, x_{item}, x_{context}))
$$

- 排序核心：预估用户点击概率

### 3. 在线评估

$$
\Delta metric = \bar{X}_{treatment} - \bar{X}_{control}
$$

- A/B 测试量化策略效果

---

## 🌟 一句话解释

oCPC/oCPA 让广告主只需设定转化目标出价，平台**用 pCTR × pCVR 自动换算成竞价金额**，同时通过 PID 控制器实时反馈调节，使实际 CPA 收敛到广告主目标。

---

## 🎭 生活类比

开车使用定速巡航（Cruise Control）：
- **传统 CPC**：你手动踩油门（手动出价），盯着车速（点击率）开车，路况变了还没反应
- **oCPC**：你设定目标车速100km/h（目标 CPA），定速巡航系统（PID控制器）自动踩油门/刹车，上坡时加力，下坡时减力，让实际车速稳定在100km/h（实际 CPA≈目标）

---

## ⚙️ 核心机制

```
广告主设定：target_CPA = 50元

【竞价公式】
bid = target_CPA × pCTR × pCVR × 调节系数

其中：
- pCTR：点击率预估（深度学习模型）
- pCVR：点击后转化率预估（ESMM系列）
- 调节系数：由 PID 控制器实时输出

【PID 反馈控制】
error = actual_CPA - target_CPA
bid_multiplier = Kp·error + Ki·∫error·dt + Kd·Δerror/Δt

┌─────────────────────────────────────┐
│ 实际CPA > 目标CPA → 降低出价系数     │
│ 实际CPA < 目标CPA → 提高出价系数     │
└─────────────────────────────────────┘

【oCPC生命周期】
探索期(0~转化积累) → 学习期(模型训练) → 稳定期(PID控制)
```

**CVR 预估的核心难点（ESMM 解决方案）：**
```
问题：CVR 只有"曝光→点击→转化"样本，存在样本选择偏差

ESMM：联合建模
  pCTCVR = pCTR × pCVR  (在全曝光空间训练)
  
  Loss = L_CTR(y_ctr, p_ctr) + L_CTCVR(y_ctcvr, p_ctcvr)
```

---

## 🔄 横向对比

| 出价方式 | 广告主操作 | 平台保障 | 风险承担方 |
|---------|----------|---------|----------|
| CPC | 手动设 CPC 出价 | 无 | 广告主 |
| oCPC | 设目标 CPA | 平台优化出价 | 平台（学习期保量） |
| oCPA | 设目标 CPA，按转化计费 | 平台最大化转化 | 平台全承担 |
| tROAS | 设目标 ROI | 平台优化收益 | 平台 |

---

## 🏭 工业落地

- **百度凤巢**：国内最早推广 oCPC，搜索广告场景
- **腾讯广告**：微信/朋友圈 oCPA，深度转化（如付费、注册）
- **字节跳动穿山甲**：程序化广告 oCPX，同时支持多种转化目标
- **阿里妈妈**：eROAS 优化，品效合一场景

**工程关键点：**
1. **探索期保护**：新广告转化数据不足，PID 可能震荡，需设置出价上下限
2. **CVR 延迟问题**：转化可能发生在点击后数天，需要 delayed conversion 建模
3. **多转化目标**：电商有"加购→下单→支付"漏斗，用 AITM 建模转化路径
4. **归因问题**：最后点击归因 vs 多触点归因，影响 CVR 估计准确性

---

## 🎯 常见考点

**Q1（基础）：oCPC 和 CPC 最本质的区别是什么？**
> CPC 广告主手动出价、自行承担点击是否转化的风险；oCPC 广告主设定转化目标，平台通过 pCTR×pCVR 自动换算出竞价 bid，平台承诺优化 CPA 趋近目标，广告主按点击付费但CPA被托管。

**Q2（中等）：CVR 预估为什么存在样本选择偏差？ESMM 如何解决？**
> CVR 训练样本只有"点击"样本，而真实CVR应该在全曝光空间计算。点击用户本身是有偏的（更可能转化）。ESMM 通过联合建模 pCTCVR = pCTR × pCVR，在全曝光空间建模整体转化概率，CVR 模型参数通过 CTCVR loss 间接在全空间更新。

**Q3（高难）：为什么使用 PID 控制器而不是直接用 CVR 模型输出调整出价？**
> CVR 模型预估存在系统性偏差（训练分布 ≠ 线上分布，特别是分布漂移时），单纯依赖 pCVR 会导致出价持续偏高或偏低。PID 是反馈控制，用实际 CPA 与目标的差值做闭环校正，能自动补偿模型偏差，对模型精度要求更低，系统更鲁棒。

---

## 🔗 知识关联

- 上游：CTR 预估（DeepFM/DCN）、CVR 预估（ESMM/AITM）
- 同层：竞价机制（GSP/VCG）、预算分配（Budget Pacing）
- 下游：广告主 ROI 分析、LTV 建模

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


## 📐 核心公式直观理解

### oCPC 出价公式

$$
\text{bid} = \text{CPA}_{\text{target}} \times \frac{p\text{CVR}}{p\text{CTR}} \times p\text{CTR} = \text{CPA}_{\text{target}} \times p\text{CVR}
$$

**直观理解**：广告主设定"每个转化最多愿意付 X 元"（CPA target），系统根据每次请求的转化概率自动出价——高转化概率的请求出高价（值得抢），低转化概率的请求出低价或不出（不值得）。自动化替代了人工调价。

### PID 控制器调节出价

$$
b_t = b_{t-1} + K_p \cdot e_t + K_i \cdot \sum_{s=1}^{t} e_s + K_d \cdot (e_t - e_{t-1})
$$

- $e_t = \text{CPA}_{\text{target}} - \text{CPA}_{\text{actual}}$：实际 CPA 与目标的偏差

**直观理解**：实际 CPA 高于目标（花多了）→ 降低出价；低于目标（花少了）→ 提高出价。$K_p$ 管当前偏差（立即纠偏），$K_i$ 管累积偏差（消除长期偏移），$K_d$ 管变化趋势（预防过冲）。就像开车——看到偏离车道就打方向盘，PID 三个参数控制"打多狠"。

### 预算约束下的 Lagrangian 出价

$$
\text{bid}^*(x) = \text{CPA}_{\text{target}} \times p\text{CVR}(x) \times \frac{1}{1 + \lambda}
$$

**直观理解**：$\lambda$ 是预算约束的拉格朗日乘子。预算快花完时 $\lambda$ 增大，出价整体压低（省钱）；预算充裕时 $\lambda \approx 0$，出价 ≈ 不受约束的最优出价。自动化的"花钱节奏控制"。

