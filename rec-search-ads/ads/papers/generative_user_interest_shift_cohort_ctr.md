# Generative Modeling of User Interest Shift via Cohort-based Intent Learning for CTR

> 来源：https://arxiv.org/abs/2601.18251 | 领域：计算广告 | 学习日期：20260331

## 问题定义

用户兴趣会随时间漂移（interest shift），传统CTR模型假设兴趣静态，无法捕获突发兴趣变化。

## 核心方法与创新点

1. **群组意图学习**：将用户分组（cohort），学习群组级别的兴趣漂移模式
2. **VAE生成式建模**：

$$
z_t \sim q_\phi(z_t | z_{t-1}, x_t), \quad \hat{x}_{t+1} = p_\theta(x_{t+1} | z_t)
$$

3. **时间感知Cohort划分**：基于行为模式的动态用户分组
4. **兴趣转移矩阵**：建模不同兴趣类别间的转移概率

## 实验结论

在电商和新闻推荐数据上，interest shift明显场景（大促期间）CTR预测AUC提升0.8-1.2%。

## 工程落地要点

- Cohort划分可周期性更新（日级）
- VAE推理增加一次前向传播，延迟增加<5ms
- 群组级建模减少用户级噪声
- 适合有明显季节性/事件性业务

## 面试考点

1. **为什么群组而非个体？** 个体数据稀疏，群组聚合模式更清晰
2. **VAE在兴趣建模的优势？** 生成连续的兴趣演化轨迹
3. **检测interest shift？** 监控点击分布的KL散度变化
