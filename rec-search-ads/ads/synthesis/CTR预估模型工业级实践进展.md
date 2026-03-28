# CTR 预估模型工业级实践进展

> 综合总结 | 领域：ads | 整理日期：20260328 | 涵盖论文：12篇

---

## 一、全景概述

近年来工业级 CTR 预估和广告系统的技术演进呈现三条主线：

1. **特征交叉深化**：从浅层 FM 到深层矩阵交叉（DCN V2、DHEN、Wukong），Transformer 结构引入
2. **生成式范式兴起**：LLM 和生成式模型进入广告排序、内容生成（GRR、HiGR、BannerAgency）
3. **强化学习落地**：从即时 CTR 优化转向长期用户价值优化（HEPO、DAPO）

---

## 二、核心技术体系

### 2.1 特征交叉方法演进

```
FM (2010) → NFM → DeepFM → DCN V1 → xDeepFM
         ↓
DCN V2 (矩阵交叉) → DHEN (层次Ensemble) → Wukong (深层+并行训练)
         ↓
MaskNet (实例级乘法门控) → SENet (全局注意力) → AutoInt (Transformer)
```

**关键公式对比：**

DCN V1 cross layer（标量权重）：
$$x_{l+1} = x_0 \cdot x_l^T \cdot w_l + b_l + x_l$$

DCN V2 cross layer（矩阵权重）：
$$x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l, \quad W_l \in \mathbb{R}^{d \times d}$$

MaskNet 实例级调制：
$$V_{mask} = LayerNorm(f_{mask}(e_{instance}) \odot V_{emb})$$

DHEN 层次 Ensemble：
$$h^{(l+1)} = Aggregate\left(\{Cross(h^{(l)}), Bilinear(h^{(l)}), Attn(h^{(l)}), MLP(h^{(l)})\}\right)$$

### 2.2 规模化训练技术

Wukong 提出的大规模并行 CTR 训练框架：

**Pre-LN 深层稳定性**（关键公式）：
$$h^{(l)} = h^{(l-1)} + FFN(LN(Cross(LN(h^{(l-1)}))))$$

Pre-LN 使得 $\nabla_{h^{(0)}} h^{(L)}$ 不受 LayerNorm 干扰，梯度传播更稳定，支持 10+ 层。

**Large-Batch 训练学习率缩放**：
$$lr_{new} = lr_{base} \times \frac{B_{new}}{B_{base}}$$

配合线性 warmup 避免早期训练不稳定。

### 2.3 召回与个性化统一

Etsy Unified Embedding 的核心建模：

$$\text{Query Tower: } e_q = Encoder_Q(q_{text}, u_{history}, u_{context})$$
$$\text{Item Tower: } e_i = Encoder_I(title, desc, category, ...)$$
$$\text{Loss: } \mathcal{L} = -\log \frac{\exp(e_q \cdot e_{i^+}/\tau)}{\sum_{j} \exp(e_q \cdot e_{i_j}/\tau)}$$

其中 $\tau$ 为温度系数，in-batch negatives 作为负样本。

---

## 三、生成式广告技术

### 3.1 生成式重排（GRR）

将重排分解为推理 + 排序两步：
$$P(rank | q, u, \mathcal{C}) = P(reasoning | q, u) \cdot P(rank | reasoning, \mathcal{C})$$

显式推理链（CoT）迫使模型先理解用户意图再排序，比直接端到端排序更准确。

### 3.2 列表级生成推荐（HiGR）

层次规划的效率公式：
$$\text{搜索空间: } |\mathcal{C}|^K \text{（类别规划）} + K \cdot |\mathcal{I}_{per\_cat}| \text{（item选择）}$$
$$\ll |\mathcal{I}|^K \text{（暴力搜索）}$$

类别优先生成将候选空间从 $|\mathcal{I}|^K$ 降至可控范围。

### 3.3 DAPO：LLM 强化学习

DAPO 的非对称 Clip 策略：
$$\mathcal{L}_{DAPO} = -\mathbb{E}_{(s,a,A)} \left[ \min\left(r_t A_t, \text{clip}(r_t, 1-\epsilon_l, 1+\epsilon_h) A_t\right) \right]$$

其中 $\epsilon_h > \epsilon_l$，对正向 advantage 给更大更新空间，加速学习。

Token-level 梯度（避免长序列信号稀释）：
$$\mathcal{L}_{token} = -\frac{1}{\sum_t |tokens_t|} \sum_t \sum_k A_t \log \pi_\theta(a_{t,k} | s_{t,k})$$

---

## 四、多场景与成本效率

### 4.1 Meta Lattice：参数共享框架

场景适配的低秩分解：
$$\theta_{scene_k} = \theta_{shared} + A_k B_k^T, \quad A_k, B_k \in \mathbb{R}^{d \times r}, r \ll d$$

参数效率：$N$ 个场景的参数从 $N \times |\theta|$ 降为 $|\theta_{shared}| + N \times 2dr$

当 $r=16, d=512, N=50$ 时，参数节省约 $50 \times (512^2 - 2 \times 512 \times 16) = 50 \times 245760 \approx 12M$ 参数。

### 4.2 多智能体设计（BannerAgency）

广告创意生成的 Agent 协作流：
```
Brief → [Creative Director] → Strategy
      → [Copywriter] → Text
      → [Visual Designer] → Layout + Image
      → [QA Reviewer] → Feedback Loop (≤5轮)
      → Final Banner
```

---

## 五、面试 Q&A（综合篇）

**Q1：工业级 CTR 模型的关键设计原则是什么？**
A：1) 特征工程优先：高质量特征 > 复杂模型架构；2) 计算效率：embedding 稀疏查询 + dense 网络高效计算；3) 训练稳定性：Pre-LN、梯度裁剪、学习率 warmup；4) 在线更新：增量训练保持时效性；5) A/B 测试驱动：所有改动必须线上验证。

**Q2：DCN V2 的 Low-rank 近似为什么有效？**
A：特征交叉矩阵 $W_l$ 的有效秩往往远低于全秩（特征空间存在低维流形结构），低秩近似 $W_l = UV^T$（$U,V \in \mathbb{R}^{d \times r}$）捕捉了主要变化方向，丢弃的是噪声维度，因此在大幅减少参数（从 $d^2$ 到 $2dr$）的同时效果几乎无损。

**Q3：MaskNet 和 DCN V2 解决的是什么不同问题？**
A：DCN V2 解决的是**特征阶次**问题（如何学习高阶显式特征交叉）；MaskNet 解决的是**实例差异**问题（同一特征对不同用户/场景的重要性不同）。两者可以叠加使用：先用 DCN V2 做特征交叉，再用 MaskNet 做实例级调制。

**Q4：为什么大规模 CTR 模型需要 Pre-LN 而不是 Post-LN？**
A：Post-LN 在残差路径上放置 LayerNorm，深层时 $\frac{\partial Loss}{\partial h^{(0)}}$ 的梯度需要穿过多个 LayerNorm 的 Jacobian，可能趋近于零（梯度消失）。Pre-LN 将 LayerNorm 放在子层输入前，残差分支保持干净，梯度可以直接流过残差跳连，支持 10+ 层稳定训练。

**Q5：Slate 推荐（列表级推荐）为什么比单 item 推荐更难？**
A：1) 组合空间爆炸：$C_N^K$ 的搜索空间无法暴力枚举；2) 列表内依赖：item 间的多样性、互补性需要联合建模；3) 位置偏差：列表不同位置的曝光效果不同；4) 联合优化困难：需要同时优化列表整体的点击率、转化率和用户满意度。

**Q6：双塔模型中 user side 和 item side 为什么要用不同的更新频率？**
A：item embedding 相对静态（商品属性变化慢），可以离线批量计算建索引，每日更新即可；user embedding 需要捕捉用户实时兴趣（最近 1 小时的点击行为），必须在线计算，延迟要求毫秒级。分离更新策略在实时性和计算效率间取得平衡。

**Q7：广告强化学习中探索（exploration）的最大风险是什么？如何缓解？**
A：最大风险是探索成本高——展示非最优广告直接损失收入，且用户体验损伤可能有长期负面效应。缓解：1) 仿真预训练，线上探索比例<5%；2) Constrained RL 设置收入下界；3) 时间段选择（低流量时段探索）；4) 快速回滚机制。

**Q8：DAPO 去掉 KL 散度约束有什么风险？如何替代？**
A：KL 约束防止策略偏离 reference model 太远（避免"奖励黑客"和分布崩溃）。DAPO 用动态 clip 范围替代：当 ratio 过大时自动截断更新，实现软约束效果，同时避免 KL 项引入的超参数调优负担。风险是无 KL 约束时需要更仔细的 clip 超参数调整。

**Q9：Meta Lattice 的场景 Adapter 和 LoRA 有什么关系？**
A：本质相同——都是用低秩矩阵 $AB^T$ 来表示对 base model 的适配增量，避免全量 fine-tuning 的高成本。区别在于：LoRA 用于 LLM 的语言任务 fine-tuning；Meta Lattice 用于推荐系统的多场景适配，场景间差异通常更结构化（不同广告位的行为分布差异）。

**Q10：生成式重排（GRR）的推理链（CoT）训练数据如何获取？**
A：1) 人工标注：让标注员写出排序理由，但成本高；2) 强模型蒸馏：用 GPT-4 等大模型对训练集生成推理链，作为小模型的训练目标；3) 自动生成+过滤：用规则或弱监督生成推理链，结合 RL reward 过滤低质量推理；4) 程序生成：对于有结构化解释（如价格匹配）的排序依据，用程序生成推理链。

---

## 六、技术选型速查表

| 问题类型 | 推荐方案 | 核心论文 |
|---------|---------|---------|
| 特征交叉深化 | DCN V2 (矩阵交叉) + Parallel 结构 | DCN V2 |
| 实例级特征调制 | MaskNet (Serial 3-4层) | MaskNet |
| 多模块集成 | DHEN (Cross+Bilinear+Attn+MLP) | DHEN |
| 大规模深层训练 | Wukong Pre-LN 架构 + 分布式 | Wukong |
| 个性化召回统一 | 双塔 + 多任务 + Hard Negative | Etsy Unified |
| 多场景参数共享 | Meta Lattice (rank=16~32) | Meta Lattice |
| 可解释重排 | LLM Re-ranker + 蒸馏 | LLM Re-Ranker |
| 列表级生成推荐 | HiGR 层次规划 | HiGR |
| LLM RL 训练 | DAPO (Clip-Higher + Token-level) | DAPO |
| 广告创意生成 | BannerAgency 多 Agent 框架 | BannerAgency |
| 长期价值优化 | HEPO 层次 RL | HEPO |

---

## 📚 参考文献

1. Wang et al., *DCN V2: Improved Deep & Cross Network for Feature Cross Learning in Web-Scale Learning to Rank Systems*, 2020. [arxiv: 2008.13535]
2. Wang et al., *MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask*, 2021. [arxiv: 2102.07619]
3. Chen et al., *DHEN: A Deep and Hierarchical Ensemble Network for Large-Scale CTR Prediction*, 2022. [arxiv: 2203.11014]
4. *Wukong CTR: Scalable Deep CTR Prediction via Massive Parallel Training*, 2023. [arxiv: 2312.01399]
5. *Unified Embedding Based Personalized Retrieval in Etsy Search*, 2023. [arxiv: 2306.04833]
6. *Meta Lattice: Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations*, 2024. [arxiv: 2512.09200]
7. *LLM as Explainable Re-Ranker for Recommendation System*, 2024. [arxiv: 2512.03439]
8. *Generative Reasoning Re-ranker*, 2025. [arxiv: 2602.07774]
9. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, 2025. [arxiv: 2503.14476]
10. *HiGR: Efficient Generative Slate Recommendation via Hierarchical Planning*, 2024. [arxiv: 2512.24787]
11. *BannerAgency: Advertising Banner Design with Multimodal LLM Agents*, 2025. [arxiv: 2503.11060]
12. *Hierarchy Enhanced Policy Optimization for Ad Ranking*, 2026. [arxiv: 2603.xxxx]
