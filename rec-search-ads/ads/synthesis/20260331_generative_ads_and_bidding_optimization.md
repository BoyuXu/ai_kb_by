# 计算广告前沿综合：生成式广告与智能竞价

> 综合日期：20260331 | 领域：计算广告 | 覆盖论文：10篇

## 主题概述

本批次广告方向论文呈现三大技术趋势：**生成式CTR预测**（DGenCTR、MTGR）、**智能竞价优化**（Bid2X、GBS）、以及**LLM轻量化应用**。传统判别式CTR模型正向生成式范式演进。

## 核心技术脉络

### 1. 从判别式到生成式CTR

DGenCTR标志着CTR预测从判别式向生成式的范式转变：

$$
p_\theta(x_0 | x_t, c) = \prod_{i} p_\theta(x_0^{(i)} | x_t, c)
$$

核心思想是将特征统一token化后，用离散扩散模型生成预测。美团MTGR将这一思路应用到多场景统一推荐，通过decoder-only架构和场景Prompt服务12+个场景。

### 2. DCN-V2与特征交叉

DCN-V2通过矩阵式Cross层突破了V1的表达力瓶颈：

$$
x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l
$$

低秩分解 $W = UV^T$ 在 $O(d \times r)$ 参数量下达到接近 $O(d^2)$ 的表达能力。

### 3. 智能竞价新范式

FPA（First-Price Auction）时代，出价策略的重要性凸显：
- **Bid2X**：预测竞价环境动态，前瞻性调整出价
- **GBS**：生成式出价分布建模，量化不确定性
- **GenCI**：建模用户兴趣漂移，提升CTR预估精度

$$
p(w | x) = \sum_{k=1}^{K} \pi_k(x) \cdot \mathcal{N}(w | \mu_k(x), \sigma_k^2(x))
$$

### 4. LLM轻量化融入CTR

直接使用LLM做CTR推理延迟200ms+不可接受。轻量化方案：离线生成LLM Embedding → 对齐 → 查表融合，仅增加2ms延迟。

## 关键公式汇总

**DCN-V2 Cross层**：

$$
x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l
$$

**GBS混合高斯出价分布**：

$$
p(w | x) = \sum_{k=1}^{K} \pi_k(x) \cdot \mathcal{N}(w | \mu_k(x), \sigma_k^2(x))
$$

**跨模态对齐损失**：

$$
L_{align} = -\log \frac{\exp(\text{sim}(e_{LLM}, e_{ID}) / \tau)}{\sum_j \exp(\text{sim}(e_{LLM}, e_{ID_j}) / \tau)}
$$

## Q&A 面试精选

**Q1: 生成式CTR与判别式CTR的本质区别？**
A: 判别式建模 $P(y|x)$，生成式建模 $P(x, y)$（联合分布）。生成式通过建模数据生成过程获得更好的泛化性。

**Q2: DCN-V2为什么需要低秩分解？**
A: 全矩阵参数量 $O(d^2)$ 在feature维度 $d=10000+$ 时不可承受。低秩分解以 $O(d \times r)$ （$r \ll d$）近似。

**Q3: 什么是Bid Shading？为什么在FPA中重要？**
A: FPA中广告主出多少付多少，不像GSP只付第二高价。Bid Shading通过降低出价避免过度支付，同时保持竞争力。

**Q4: 美团MTGR用一个模型替代12个的风险？**
A: 主要风险是大流量场景被小场景拖累。对策：场景权重调整、知识蒸馏、分阶段上线验证。

**Q5: 显式反馈和隐式反馈如何混合训练？**
A: 显式反馈作为高质量锚点校准隐式反馈中的噪声，两类loss加权融合，显式反馈loss给更高权重。

**Q6: LLM Embedding如何高效服务CTR线上推理？**
A: 离线批量计算并存入Feature Store/Redis，在线推理只做查表（<1ms）+ MLP融合（~1ms）。

**Q7: 竞价环境预测中如何处理竞争者不可观测？**
A: 通过历史市场价格分布反推竞争强度，结合时序模型预测趋势。本质是间接推断。

**Q8: DGenCTR的自适应步数如何决定？**
A: 基于模型在每步去噪后的置信度（输出分布的熵）。简单样本1-2步即可，复杂样本需要4-5步。

**Q9: 用户兴趣漂移建模中，群组vs个体的trade-off？**
A: 个体粒度更精准但数据稀疏噪声大。群组聚合后信号清晰但损失个体差异。实践中先群组后个性化。

**Q10: 统一Embedding搜索中如何防止个性化覆盖搜索相关性？**
A: 注意力权重 + query-product相关性score作为基准分，个性化只做微调。对强意图query，个性化权重自动降低。

## 参考文献

1. Unified Embedding Based Personalized Retrieval in Etsy Search (arXiv:2306.04833)
2. DCN V2: Improved Deep & Cross Network (arXiv:2008.13535)
3. Revisiting Explicit vs. Implicit Feedback (arXiv:2302.03437)
4. Bid2X: Revealing Dynamics of Bidding Environment (arXiv:2510.23410)
5. GenCI: Generative Modeling of User Interest Shift (arXiv:2601.18251)
6. A Lightweight LLM-enhanced Method for CTR Prediction (arXiv:2505.14057)
7. DGenCTR: Universal Generative Paradigm for CTR (arXiv:2508.14500)
8. DGenCTR: Discrete Diffusion for CTR (arXiv:2508.14500)
9. MTGR: Industrial-Scale Generative Recommendation in Meituan (arXiv:2505.18654)
10. GBS: Generative Bid Shading in RTB (arXiv:2508.06550)
