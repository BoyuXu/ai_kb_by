# UFO: Unfair-to-Fair Evolving Mitigates Unfairness in LLM-based Recommender Systems via Self-Play Fine-tuning

> 来源：arXiv 2025 | 领域：rec-sys | 学习日期：20260404

## 问题定义

LLM 推荐系统中存在严重的 **公平性问题**：
- **流行度偏差**：高曝光物品被过度推荐
- **人口统计偏差**：不同性别/年龄用户的推荐质量差异显著
- **位置偏差**：模型倾向推荐 prompt 中靠前位置的物品

$$\text{Unfairness} = \mathbb{E}_{g \in G}[|R@K(g) - \bar{R@K}|]$$

## 核心方法与创新点

**UFO** 用 Self-Play 微调迭代消除偏差：

1. **偏差识别器（Bias Detector）**：
   - 自动识别当前模型的推荐中哪些是有偏推荐
   - 分析推荐理由中的偏差词（"热门"、"大众喜爱"等 vs 真正匹配用户兴趣）

2. **反事实对抗生成（Counterfactual Adversarial Generation）**：
   - 生成 "公平版" 推荐对照样本
   - 对比公平推荐 vs 有偏推荐，构造偏好对 $(y_{\text{fair}}, y_{\text{biased}})$

3. **自对弈 DPO 优化**：
   
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_f|x)}{\pi_{\text{ref}}(y_f|x)} - \beta \log \frac{\pi_\theta(y_b|x)}{\pi_{\text{ref}}(y_b|x)}\right)\right]$$

4. **迭代进化（Evolving）**：
   - 每轮用新模型重新生成对抗样本，更精准定位残余偏差
   - 通常 3 轮迭代即收敛

## 实验结论

- 人口统计组间 Recall 差距缩小 **67%**
- 流行度偏差（popular item ratio）从 43% 降至 **28%**
- 整体 NDCG@10 几乎无损（-0.8%）
- 自对弈迭代比静态 DPO 额外提升 **+15%** 公平性

## 工程落地要点

- 偏差识别无需人工标注，全自动（Self-Play）
- DPO 微调成本低：对比 SFT 减少 80% 计算
- 需定期（月度）重新运行 UFO，因新数据可能引入新偏差
- 监控指标：推荐多样性 + 组间公平性指标（定期 A/B）

## 面试考点

1. **Q**: 推荐系统公平性的主要维度有哪些？  
   **A**: 流行度公平（避免马太效应）、人口统计公平（性别/年龄/地域）、物品提供者公平（新卖家被曝光）。

2. **Q**: DPO 在公平性场景的应用？  
   **A**: 构造 (公平推荐, 有偏推荐) 偏好对，用 DPO 训练模型倾向公平推荐，无需人工标注，成本低。

3. **Q**: Self-Play 相比单次 DPO 的优势？  
   **A**: 每轮用新模型生成更精准的对抗样本，迭代消除残余偏差，而单次 DPO 只能纠正静态样本中识别的偏差。
