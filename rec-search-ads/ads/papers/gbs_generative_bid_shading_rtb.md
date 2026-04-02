# GBS: Generative Bid Shading in Real-Time Bidding Advertising

> 来源：https://arxiv.org/abs/2508.06550 | 领域：计算广告 | 学习日期：20260331

## 问题定义

RTB从GSP向FPA转型后，广告主需要主动竞价遮蔽（bid shading）避免过度出价，但市场价格分布未知。

## 核心方法与创新点

1. **生成式出价模型**：条件生成模型预测最优出价分布
2. **市场价格分布建模**：

$$
p(w | x) = \sum_{k=1}^{K} \pi_k(x) \cdot \mathcal{N}(w | \mu_k(x), \sigma_k^2(x))
$$

3. **Shading率自适应**：根据预测的市场价格分布动态计算最优shading比例
4. **探索-利用平衡**：Thompson Sampling实现出价空间高效探索

## 实验结论

在DSP真实数据上，GBS节省15%广告支出，ROI提升12%，Win Rate不变。

## 工程落地要点

- 生成模型推理延迟需<10ms（RTB要求）
- 可部署为轻量级在线服务
- 需持续学习应对市场变化
- 与预算控制系统联合优化

## 常见考点

1. **什么是Bid Shading？** FPA中降低出价避免超付
2. **为什么生成式？** 分布而非点估计，更好量化不确定性
3. **FPA vs GSP对广告主影响？** FPA中付第一价，需更精准出价
