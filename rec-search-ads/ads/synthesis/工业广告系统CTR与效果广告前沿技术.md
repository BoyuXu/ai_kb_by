# 工业广告系统 CTR 与效果广告前沿技术（2025）

> 综合总结 | 领域：ads | 学习日期：20260404

## 主题概述

2025 年工业广告系统的核心趋势：**Scaling 效率优化**、**多任务统一建模**、**冷启动突破**、**意图感知 CTR** 以及 **推荐-广告一体化**。

---

## 一、CTR Scaling Laws

**EST 核心发现**：CTR 模型也遵循 Power-Law Scaling，但配比与 LLM 不同：

$$\mathcal{L}(N, D) = A \cdot N^{-\alpha} + B \cdot D^{-\beta} + C$$

最优扩展配比（类 Chinchilla）：

$$N_{\text{opt}} \propto C^{0.55}, \quad D_{\text{opt}} \propto C^{0.45}$$

关键洞察：
- Embedding 参数收益 > Dense 参数：$\Delta \text{AUC} \propto d_{\text{emb}}^{0.3}$ vs $\Delta \text{AUC} \propto L_{\text{MLP}}^{0.1}$
- 数据质量 3x 高于数量：高质量 D ≡ 3× 同量低质数据
- 大多数 CTR 模型是"数据欠拟合"而非"模型过小"

---

## 二、动态特征交叉

**DLF 动态阶数感知**：

$$\hat{y} = \sum_{k=1}^{K} w_k(x) \cdot f_k(x), \quad w_k \text{ 由门控网络预测}$$

FM（二阶）与 DNN（高阶）融合：

$$h = \text{DNN}(x) + \alpha \cdot \text{FM}(x)$$

多样性约束防止阶数崩塌：

$$\mathcal{L}_{\text{div}} = \sum_k w_k \log w_k \quad \text{(熵最大化)}$$

---

## 三、统一 Transformer 架构

**OneTrans** 用单 Transformer 统一特征交叉与序列建模：

| 组件 | Attention 类型 | 目的 |
|------|--------------|------|
| 行为序列 | Causal Mask | 时序建模 |
| 上下文特征 | Full Attention | 特征交叉 |
| 目标广告 | Cross Attention | 意图对齐 |

统一 loss：

$$\mathcal{L} = \mathcal{L}_{\text{CTR}} + \lambda \cdot \mathcal{L}_{\text{seq\_pred}}$$

参数量 -35%，训练速度 +1.4x，AUC +0.9‰。

---

## 四、冷启动广告建模

**IDProxy 渐进迁移**（小红书）：

初始化：

$$\text{ProxyIDs} = \text{TopK}_{j \in \mathcal{H}}[\text{sim}(e_i^{\text{content}}, e_j^{\text{content}})]$$

迁移：

$$e_{\text{final}(t) = (1 - e^{-\mu N_{\text{clicks}}}) \cdot e_{\text{proxy}} + e^{-\mu N_{\text{clicks}}} \cdot e_{\text{id}}$$

---

## 五、多任务 CVR 建模

**No One Left Behind** 利用标签层次性：

$$P(\text{pay}) = P(\text{pay} | \text{add\_cart}) \cdot P(\text{add\_cart} | \text{click}) \cdot P(\text{click})$$

不确定性软标签：

$$\hat{y}_{\text{soft}} = \begin{cases} y & \text{labeled} \\ P_{\text{impute}(y=1) & \text{missing} \end{cases}$$

偏斜校正 Focal Loss：

$$\mathcal{L}_{\text{focal}} = -\alpha_t (1-p_t)^{\gamma} \log p_t, \quad \gamma_{\text{pay}}=2.5, \gamma_{\text{click}}=0.5$$

---

## 六、Offline RL 出价优化

**MTORL** 保守离线 RL：

$$\mathcal{L}_{\text{CQL}} = \alpha \cdot \mathbb{E}_{s}[\log \sum_a \exp Q(s,a)] - \mathbb{E}_{(s,a) \sim D}[Q(s,a)] + \mathcal{L}_{\text{Bellman}}$$

行为约束（BCO）防止策略漂移：

$$\mathcal{L}_{\text{BCO}} = \mathbb{E}[D_{KL}(\pi_\theta || \pi_\beta)]$$

---

## 七、终身用户建模

**R2LED** 检索精炼蒸馏：

$$h_u = \text{Transformer}\left(\text{TopK}_{i}[\text{sim}(e_a, e_i)], a\right)$$

蒸馏保证精炼质量：

$$\mathcal{L}_{\text{distill}} = ||h_u^{\text{teacher}} - h_u^{\text{student}}||_2^2$$

在线延迟：检索 8ms + 精炼 12ms = 20ms（工业可用）。

---

## 🎓 面试高频 Q&A（10题）

**Q1**: CTR 模型该如何扩展？先加参数还是先加数据？  
**A**: 先扩 Embedding 维度（收益最高）→ 加数据（清洗质量优先于数量）→ 最后加深 MLP。按 CTR Scaling Law 配比计算最优分配。

**Q2**: DLF 动态阶数与固定阶数的 tradeoff？  
**A**: 动态阶数适应不同样本复杂度，门控网络轻量（+3% 参数），AUC +0.6‰。适合特征交叉模式多变的大规模系统。

**Q3**: 触发式广告（DAIAN）与普通展示广告的建模差异？  
**A**: 触发广告有即时意图信号（用户正在搜索），需实时推断意图向量并与广告对齐；普通展示广告依赖历史兴趣建模。

**Q4**: Offline RL（MTORL）与在线 RL 相比的优劣？  
**A**: 优：安全（无在线探索风险）、利用海量历史数据；劣：外推误差（OOD 动作 Q 值高估）、需要特殊处理（CQL/BCO）。

**Q5**: ESMM 的核心思想？解决了什么问题？  
**A**: 在全曝光空间联合建模 CTCVR=CTR×CVR，解决 CVR 训练的样本选择偏差（只在点击样本上训练 CVR 导致偏差）。

**Q6**: 广告冷启动为什么用代理 ID 而非直接用内容 Embedding？  
**A**: 保持模型架构统一（无需修改 Embedding Table 结构）；代理 ID 携带协同过滤信息（热门广告的用户行为统计），纯内容特征没有。

**Q7**: 推荐-广告统一（One Model Two Markets）的主要挑战？  
**A**: 目标冲突（用户满意度 vs 商业收入）；出价引入分布偏移（高出价广告偏多）；需要用户体验保护机制（兴趣阈值+混排比例）。

**Q8**: 多路相关性（PRECTR-V2 跨用户挖掘）如何提升冷启动广告效果？  
**A**: 冷启动用户自身信号少，相似用户群的行为统计提供代理信号；cross-user attention 学习"相似用户对同一广告的反应"。

**Q9**: CQL（保守 Q-learning）如何防止 Offline RL 的外推误差？  
**A**: 惩罚 OOD 动作的 Q 值（最小化 OOD 动作 Q 函数的 log-sum-exp）+ 最大化已见动作 Q 值，使策略保持在历史数据支撑的范围内。

**Q10**: 广告系统多任务学习（CTR/CVR/留存）的参数共享策略？  
**A**: 底层特征提取共享（用户/物品/上下文 Embedding）+ 任务特定头（每个目标独立预测层）+ 门控自适应权重（根据样本动态调整任务权重）。

---

## 📚 参考文献

1. EST: Efficient Scaling Laws in CTR Prediction (2025)
2. DLF: Dynamic Low-Order-Aware Fusion for CTR (2025)
3. OneTrans: Unified Feature Interaction via One Transformer (2025)
4. IDProxy: Cold-Start CTR at Xiaohongshu (2025)
5. MTORL: Multi-task Offline RL for Advertising (2025)
6. No One Left Behind: Multi-Label CVR Prediction (2025)
7. PRECTR-V2: Relevance-CTR with Cross-User Mining (2025)
8. DAIAN: Intent-Aware CTR for Trigger Advertising (2025)
9. R2LED: Retrieval+Refinement Lifelong User Modeling (2025)
10. One Model Two Markets: Bid-Aware Generative Recommendation (2025)

---

## 相关概念

- [[embedding_everywhere|Embedding 技术全景]]
- [[multi_objective_optimization|多目标优化]]
- [[sequence_modeling_evolution|序列建模演进]]
