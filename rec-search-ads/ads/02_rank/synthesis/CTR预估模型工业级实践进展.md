# CTR 预估模型工业级实践进展

> 综合总结 | 领域：ads | 整理日期：20260329 | 涵盖论文：14篇（更新 +2）

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

$$
x_{l+1} = x_0 \cdot x_l^T \cdot w_l + b_l + x_l
$$

DCN V2 cross layer（矩阵权重）：

$$
x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l, \quad W_l \in \mathbb{R}^{d \times d}
$$

MaskNet 实例级调制：

$$
V_{mask} = LayerNorm(f_{mask}(e_{instance}) \odot V_{emb})
$$

DHEN 层次 Ensemble：

$$
h^{(l+1)} = Aggregate\left(\{Cross(h^{(l)}), Bilinear(h^{(l)}), Attn(h^{(l)}), MLP(h^{(l)})\}\right)
$$

### 2.2 规模化训练技术

Wukong 提出的大规模并行 CTR 训练框架：

**Pre-LN 深层稳定性**（关键公式）：

$$
h^{(l)} = h^{(l-1)} + FFN(LN(Cross(LN(h^{(l-1)}))))
$$

Pre-LN 使得 $\nabla_{h^{(0)}} h^{(L)}$ 不受 LayerNorm 干扰，梯度传播更稳定，支持 10+ 层。

**Large-Batch 训练学习率缩放**：

$$
lr_{new} = lr_{base} \times \frac{B_{new}}{B_{base}}
$$

配合线性 warmup 避免早期训练不稳定。

### 2.3 召回与个性化统一

Etsy Unified Embedding 的核心建模：

$$
\text{Query Tower: } e_q = Encoder_Q(q_{text}, u_{history}, u_{context})
$$

$$
\text{Item Tower: } e_i = Encoder_I(title, desc, category, ...)
$$

$$
\text{Loss: } \mathcal{L} = -\log \frac{\exp(e_q \cdot e_{i^+}/\tau)}{\sum_{j} \exp(e_q \cdot e_{i_j}/\tau)}
$$

其中 $\tau$ 为温度系数，in-batch negatives 作为负样本。

---

## 三、生成式广告技术

### 3.1 生成式重排（GRR）

将重排分解为推理 + 排序两步：

$$
P(rank | q, u, \mathcal{C}) = P(reasoning | q, u) \cdot P(rank | reasoning, \mathcal{C})
$$

显式推理链（CoT）迫使模型先理解用户意图再排序，比直接端到端排序更准确。

### 3.2 列表级生成推荐（HiGR）

层次规划的效率公式：

$$
\text{搜索空间: } |\mathcal{C}|^K \text{（类别规划）} + K \cdot |\mathcal{I}_{per\_cat}| \text{（item选择）}
$$

$$
\ll |\mathcal{I}|^K \text{（暴力搜索）}
$$

类别优先生成将候选空间从 $|\mathcal{I}|^K$ 降至可控范围。

### 3.3 DAPO：LLM 强化学习

DAPO 的非对称 Clip 策略：

$$
\mathcal{L}_{DAPO} = -\mathbb{E}_{(s,a,A)} \left[ \min\left(r_t A_t, \text{clip}(r_t, 1-\epsilon_l, 1+\epsilon_h) A_t\right) \right]
$$

其中 $\epsilon_h > \epsilon_l$，对正向 advantage 给更大更新空间，加速学习。

Token-level 梯度（避免长序列信号稀释）：

$$
\mathcal{L}_{token} = -\frac{1}{\sum_t |tokens_t|} \sum_t \sum_k A_t \log \pi_\theta(a_{t,k} | s_{t,k})
$$

---

## 四、多场景与成本效率

### 4.1 Meta Lattice：参数共享框架

场景适配的低秩分解：

$$
\theta_{scene_k} = \theta_{shared} + A_k B_k^T, \quad A_k, B_k \in \mathbb{R}^{d \times r}, r \ll d
$$

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
13. *RankMixer: Scaling Up Ranking Models in Industrial Recommenders*, 2025. [arxiv: 2507.15551]
14. *Scaling Laws for Recommendation Models*, 2026. [研究综合]

---

## 🆕 2026-03-29 新增：精排 Scaling 与 GPU 效率革命

### RankMixer：从 CPU 时代架构迈向 GPU-Native 排序

**核心问题**：传统精排模型（FM/DIN/CIN）继承自 CPU 时代，MFU 仅约 4.5%，GPU 大量空置，无法有效 Scaling。

**三大技术创新**：
1. **Token Mixing 替代 Self-Attention**：保留 Transformer 并行性，用 MLP-Mixer 风格 token 混合替代 O(n²) 注意力，内存访问连续，MFU 从 4.5% → **45%**（10×）
2. **Per-token FFN**：每个特征 token 有独立 FFN 参数，同时建模 intra-feature 和 inter-feature 交互
3. **Sparse-MoE 扩展**：动态路由策略解决专家负载不均衡，参数扩展 **100×** 而推理延迟基本不变

**工业验证**（万亿样本数据集）：
- 1B Dense RankMixer 全量上线：用户活跃天数 +0.3%，应用内总时长 +1.08%
- 在推荐 + 广告两大业务场景同时验证正收益，通用性强

### 推荐系统 Scaling Law（2026 综合视角）

**核心规律**：

$$
L(N, D) \approx \frac{A}{N^{0.07}} \cdot \frac{B}{D^{0.04}}
$$

与 LLM Scaling 对比：
| 维度 | LLM（Chinchilla）| 推荐（工业综合）|
|------|-----------------|----------------|
| 主要参数 | Attention+FFN（稠密）| Embedding（稀疏）|
| 指数 α | 0.07-0.15 | 0.05-0.10（低） |
| Scaling 瓶颈 | 计算量（FLOPs）| 内存带宽（Embedding 查找）|
| 最优路径 | Dense Transformer | MoE 稀疏激活 |

**工程结论**：
- 先优化 MFU（从 4.5% → 45%），再谈 Scaling；否则 Scaling 毫无意义
- 推荐 Scaling 优先级：Embedding 扩展 >> MLP 加深 >> MoE 专家数量
- 数据 10× ≈ 参数 3× 的等效收益（数据飞轮效应更强）


## 📐 核心公式直观理解

### DeepFM 的特征交叉

$$
\hat{y} = \sigma\left(w_0 + \sum_i w_i x_i + \underbrace{\sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j}_{\text{FM 二阶交叉}} + \underbrace{\text{DNN}(\mathbf{x})}_{\text{高阶交叉}}\right)
$$

- $\mathbf{v}_i \in \mathbb{R}^k$：特征 $i$ 的隐向量
- $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$：FM 用内积近似任意两个特征的交叉权重

**直观理解**：FM 部分自动学"用户年龄×商品品类"这类二阶交叉（不需要手动构造），DNN 部分学更复杂的高阶交叉。两者互补——FM 对稀疏特征交叉高效，DNN 对稠密信号捕获非线性关系。

### Calibration（校准）

$$
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} |\bar{y}_m - \bar{p}_m|
$$

- $B_m$：第 $m$ 个分桶内的样本集合
- $\bar{y}_m$：桶内真实正样本比例
- $\bar{p}_m$：桶内预测概率均值

**直观理解**：如果模型预测"点击率 10%"的那群用户真实点击率也是 10%，说明校准良好。ECE 衡量"模型说的概率和实际频率差多远"。广告出价直接乘以 pCTR，所以校准比排序更重要——排序错了少挣钱，校准错了可能亏钱。

### 在线学习的增量更新

$$
\theta_{t+1} = \theta_t - \eta_t \cdot \frac{\partial \ell(y_t, \hat{y}_t)}{\partial \theta_t}, \quad \eta_t = \frac{\alpha}{\beta + \sqrt{\sum_{s=1}^{t} g_s^2}}
$$

**直观理解**：广告 CTR 模型必须实时更新——昨天的热门商品今天可能无人问津。FTRL/AdaGrad 的自适应学习率让频繁更新的特征步长小（稳定），罕见特征步长大（抓住稀疏信号）。


---

## 面试高频考点（10题 Q&A）

### Q1: 工业级 CTR 系统的典型四级漏斗？
> ① 召回（多路并行，千万→万）② 粗排（轻量模型，万→千）③ 精排（复杂模型如 [DeepFM](../papers/DeepFMDeep_Factorization_Machine.md)+[DIN](../papers/DIN_Deep_Interest_Network_for_Click_Through_Rate_Predicti.md)，千→百）④ 重排（多样性、频控，百→十）。精排是 CTR 模型核心战场。

### Q2: CTR 模型的校准（Calibration）为什么重要？
> 广告出价依赖精确的 CTR 值（不仅是排序）。pCTR=0.1 意味着每 10 次展示有 1 次点击。校准不准 → eCPM 计算偏差 → 广告主 ROI 异常。常用 Platt Scaling 或 Isotonic Regression 后校准。

### Q3: 特征哈希（Feature Hashing）的原理和风险？
> 将超大 ID 空间（10⁹）通过哈希函数映射到较小空间（10⁷），接受少量碰撞。风险：不同 ID 碰撞到同一 bucket 导致信息混淆。缓解：① 多哈希函数 ② 保留高频 ID 的独立 embedding ③ signed hash 降低偏差。

### Q4: 增量训练 vs 全量训练的 tradeoff？
> 增量训练：每小时/天用新数据更新模型，追踪分布漂移，但可能逐渐遗忘早期模式。全量训练：定期用全量数据重训，全局最优但成本高。工业实践：增量为主（追踪时效性），定期全量重训（防止漂移累积）。

### Q5: 在线学习（Online Learning）在 CTR 系统中的优势？
> 实时更新部分参数（如 FTRL 更新 embedding），能快速适应用户兴趣变化和新物品上架。挑战：① 数据顺序偏差 ② 灾难性遗忘 ③ 需要可靠的流式特征管道。

### Q6: 模型蒸馏在 CTR 系统中的应用？
> 大模型（Teacher，如 [DHEN](../papers/DHEN_deep_hierarchical_ensemble_network_CTR_prediction.md) 集成）离线训练达到最佳效果。小模型（Student）通过蒸馏学习 Teacher 的 soft label，在线推理用 Student 满足延迟要求（<10ms）。

### Q7: Embedding 维度如何选择？
> 经验公式：$d = 6 \times (\text{category\_size})^{1/4}$（Google）。高频特征用较高维度（16-64），低频用较低维度（4-8）。也可用 NAS 自动搜索每个 field 的最优维度。

### Q8: CTR 模型中如何处理正负样本不均衡？
> ① 负采样（1:n），需校准 ② Focal Loss 对难分样本加权 ③ 正样本加权 ④ 分层采样（按用户活跃度分层）⑤ 校准步骤恢复真实 CTR 分布。

### Q9: [MaskNet](../papers/MaskNet_feature_wise_multiplication_CTR_instance_guided_mask.md) 的核心创新是什么？
> 引入实例引导的乘性特征交互（element-wise product），替代传统 MLP 的纯加性操作。通过动态掩码让同一特征对不同请求有不同权重，同时 LayerNorm 保证数值稳定性。

### Q10: 特征交叉方法的演进路线？
> LR（手工交叉）→ FM（自动二阶）→ [DeepFM](../papers/DeepFMDeep_Factorization_Machine.md)（FM+DNN）→ DCN/DCNv2（Cross Network 显式多阶）→ AutoInt（Attention 交叉）→ [MaskNet](../papers/MaskNet_feature_wise_multiplication_CTR_instance_guided_mask.md)（乘性交互）→ [DHEN](../papers/DHEN_deep_hierarchical_ensemble_network_CTR_prediction.md)（异构集成）。

---

## 补充内容（合并自 广告系统CTR预估与排序前沿综述.md）

### 一、技术脉络概览

广告推荐系统的核心链路：**召回 → 粗排 → 精排（CTR预估）→ 重排 → 广告创意生成**

本批论文覆盖了这条链路的各个关键阶段：

| 阶段 | 代表论文 | 核心技术 |
|------|---------|---------|
| CTR 精排 | DHEN, MaskNet, Wukong CTR | 异构集成/乘性交互/大规模并行 |
| 重排 | GR2, LLM Reranker | 生成式推理/LLM+传统模型混合 |
| 列表推荐 | HiGR | 层次化生成式 Slate 推荐 |
| 广告排序策略 | Hierarchy Policy Opt | 层次化强化学习 |
| 系统架构 | Meta Lattice | 多域多目标模型空间重设计 |
| LLM RL 基础 | DAPO | 大规模强化学习算法 |
| 创意生成 | BannerAgency | 多模态 LLM 广告素材自动生成 |

---


### 二、核心技术主题深度解析

### 主题1：特征交互建模的演进

**从加性到乘性的演进路径：**

```
FM (2010): w₁x₁ + w₂x₂ + v₁·v₂·x₁x₂   ← 引入乘性二阶交互
    ↓
DeepFM (2017): FM + DNN                    ← 加性高阶 + 乘性二阶
    ↓
xDeepFM (2018): Compressed Interaction Network ← 显式高阶乘性
    ↓
MaskNet (2021): Instance-Guided Mask ⊙ FFN ← 实例动态乘性
    ↓
DHEN (2022): 异构模块层次集成              ← 多种交互方式集成
    ↓
Wukong (2023): 低秩压缩交互 + 大规模并行   ← 可扩展高效乘性交互
```

**关键公式：MaskBlock**

$$
\text{MaskBlock}(X) = \text{FFN}\left(\text{LayerNorm}\left(\text{MLP}}_{\text{{mask}}(X) \odot X\right)\right)
$$

**关键公式：DHEN 层次集成**

$$
h_l = \text{Aggregate}\left(\text{FM}(h_{l-1}), \text{DCN}(h_{l-1}), \text{DIN}(h_{l-1}), ...\right)
$$

---

### 主题2：生成式推荐与重排的兴起

**LLM 在广告推荐系统中的定位演变：**

```
早期尝试：LLM 作为独立推荐器 → 失败（ID 空间爆炸，协同过滤信号缺失）
...(已整合)

### 三、关键指标对照表

| 论文 | 数据集/平台 | 关键指标提升 |
|------|-----------|------------|
| HiGR | 腾讯（数亿用户） | 观看时长+1.22%，视频播放+1.73%，推理5× |
| DHEN | Meta 广告系统 | NE +0.27%，训练吞吐 1.2× |
| MaskNet | 3个公开数据集 | AUC +0.1%~0.5% vs DeepFM |
| GR2 | 2个真实数据集 | Recall@5 +2.4%，NDCG@5 +1.3% vs SOTA |
| Meta Lattice | Meta 全产品线 | 收入 +10%，满意度 +11.5%，容量节省 20% |
| DAPO | AIME 2024 | 50分（Qwen2.5-32B base） |

---


### 四、📐 核心公式汇总（≥3个）

### 公式1: MaskBlock 特征乘性交互

$$
\text{MaskBlock}(X) = \text{FFN}\left(\text{LayerNorm}\left(\text{MLP}}_{\text{{mask}}(X) \odot X\right)\right)
$$

- $\odot$：逐元素乘法（Hadamard product）
- $\text{MLP}}_{\text{{mask}}(X)$：基于输入实例动态生成的掩码
- **意义**：引入实例引导的乘性特征交互，突破加性 FFN 的局限

### 公式2: DHEN 层次集成

$$
h_l = \text{Aggregate}\left(\{\text{Module}}_{\text{k(h}}_{\text{{l-1}})\}_{k=1}^K\right)
$$

$$
\mathcal{L}_{NE} = -\frac{1}{N}\sum_{i=1}^N \left[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right] / H(p)
$$

- $H(p) = -p\log p - (1-p)\log(1-p)$：背景 CTR $p$ 对应的熵
- **意义**：Normalized Entropy 归一化评估，消除不同数据集 CTR 差异的影响

### 公式3: HiGR CRQ-VAE 对比量化损失

$$
\mathcal{L}_{CRQ-VAE} = \mathcal{L}_{recon} + \lambda_1 \mathcal{L}_{global\_quan} + \lambda_2 \mathcal{L}_{cont}
$$

$$
\mathcal{L}_{cont} = -\frac{1}{D-1}\sum_{d=1}^{D-1} w_d \log \frac{\exp(\cos(e_a^d, e_p^d)/\tau)}{\exp(\cos(e_a^d, e_p^d)/\tau) + \sum_{n\neq a,p}\exp(\cos(e_a^d, e_n^d)/\tau)}
$$

- **意义**：Prefix-level InfoNCE 使相似物品共享 ID 前缀，提升语义 ID 的结构化

### 公式4: DAPO 解耦裁剪目标

$$
...(已整合)

### 五、📚 参考文献列表

1. **HiGR** - Pang et al. "HiGR: Efficient Generative Slate Recommendation via Hierarchical Planning." arXiv:2512.24787 (2025).

2. **BannerAgency** - Wang et al. "BannerAgency: Advertising Banner Design with Multimodal LLM Agents." EMNLP 2025. arXiv:2503.11060.

3. **DHEN** - Zhang et al. "DHEN: A Deep and Hierarchical Ensemble Network for Large-Scale CTR Prediction." arXiv:2203.11014 (2022).

4. **GR2** - Liang et al. "Generative Reasoning Re-ranker." arXiv:2602.07774 (2026).

5. **Meta Lattice** - Luo et al. "Meta Lattice: Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations." KDD 2026. arXiv:2512.09200.

6. **LLM Reranker** - Sun et al. "LLM as Explainable Re-Ranker for Recommendation System." arXiv:2512.03439 (2025).

7. **DAPO** - Yu et al. "DAPO: An Open-Source LLM Reinforcement Learning System at Scale." arXiv:2503.14476 (2025).

8. **Wukong CTR** - "Wukong CTR: Scalable Deep CTR Prediction via Massive Parallel Training." (2023).

9. **MaskNet** - Zhang et al. "MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask." DLP-KDD 2021. arXiv:2102.07619.

10. **Hierarchy Policy Optimization** - "Hierarchy Enhanced Policy Optimization for Ad Ranking." arXiv:2603.xxxx (2026).

### 相关基础工作
- **DeepFM** - Guo et al. (2017). Combined FM + DNN for CTR.
- **xDeepFM** - Lian et al. (2018). Compressed Interaction Network.
- **DCN** - Wang et al. (2017). Deep & Cross Network.
- **BERT4Rec** - Sun et al. (2019). Self-attentive sequential recommendation.
- **OneRec** - 字节跳动. 统一检索与精排的生成式推荐框架.
- **TIGER** - Rajput et al. (2023). Semantic ID-based generative recommendation.

---


### 六、🎓 面试 Q&A（≥10道）

### Q1: CTR 预估中，加性特征交互和乘性特征交互有什么本质区别？各自的代表方法是什么？

**A**: 
- **加性交互**：特征通过加权求和组合，如 $o = \sigma(\sum_i w_i \cdot h_i)$。代表方法：MLP/DNN（全连接层本质是加性操作）
- **乘性交互**：特征通过逐元素或矩阵乘法组合，如 FM 的 $\langle v_i, v_j \rangle x_i x_j$。代表方法：FM、MaskNet（element-wise mask）、DCN（cross product）
- **本质区别**：加性操作从数学角度无法高效表达两个特征的乘性关系（需要无限多个隐层单元），乘性操作直接建模特征间的交叉影响

---

### Q2: 工业 CTR 系统为什么用 NE（Normalized Entropy）而不是 AUC 来评估模型？

**A**: 
- **AUC** 衡量排序能力，与绝对概率值无关（量级无关）
- **NE** 衡量概率校准精度，归一化后消除不同数据集背景 CTR 不同的影响
- 工业广告系统的出价（bidding）依赖精确的 CTR 预估值（$\text{bid} = \text{eCPM} / \text{CTR}$），AUC 好但 NE 差会导致出价严重失真，影响收入
- 因此：**NE** 更能反映工业系统的真实价值，0.1% NE 改进通常对应可观收入提升

---

### Q3: Multi-Domain Multi-Objective（MDMO）推荐系统的主要挑战是什么？Meta Lattice 如何解决？

**A**: 
主要挑战：①数据碎片化（跨产品线信号无法共享）②负迁移（强域压制弱域）③参数爆炸（N×M个独立模型）④系统运维复杂度
Meta Lattice 解决方案：①跨域知识共享（统一 User Tower）②Domain-adaptive 损失权重（防负迁移）③统一模型+路由（从N×M缩至1）④蒸馏+系统优化（保效果降成本）
结果：一个模型同时服务所有产品线，+10%收入，-20%算力。

---

### Q4: 生成式推荐（Generative Recommendation）相比判别式推荐（Discriminative Recommendation）有什么优势和劣势？

**A**: 
| | 生成式 | 判别式 |
|--|--------|--------|
| **优势** | ①端到端优化整个列表 ②天然支持多样性控制 ③可利用 LLM 世界知识 ④扩展能力强（Scaling Law） | ①延迟低 ②对超大候选集高效 ③协同过滤信号更充分 |
| **劣势** | ①推理延迟高 ②ID 空间大时难处理 ③冷启动依赖语义 | ①缺乏全局列表规划 ②多样性需要额外 post-processing ③可解释性差 |

**工业实践**：两者结合，判别式做粗排，生成式做精排/重排（如 GR2、HiGR）。

...(已整合)

### 七、技术趋势与展望

1. **生成式 Ranking 全面兴起**：从判别式 CTR 到生成式 Slate，端到端优化列表质量
2. **LLM 与传统模型深度融合**：LLM 不是替代者，而是传统模型的上层"理解与对齐"模块
3. **大规模 RL 普及**：DAPO 等算法使工业 RL 从实验室走向大规模生产部署
4. **多模态广告创意**：从文字优化到图文联合优化，素材自动生成成为核心竞争力
5. **系统级协同设计**：模型架构与训练/服务系统协同设计（DHEN、Wukong）成为主流范式
6. **统一模型架构**：从N×M个专门模型到统一架构（Meta Lattice），降低运维成本同时提升效果


