# Kunlun: Establishing Scaling Laws for Massive-Scale Recommendation Systems
> 来源：arXiv:2602.10016 | 领域：rec-sys | 学习日期：20260330

## 问题定义
LLM 领域的 Scaling Laws（损失随参数量、数据量、算力的幂律关系）在推荐系统中是否成立？推荐系统的参数主要分布在 embedding table（稀疏）和 DNN（稠密），两者 scaling 规律不同。Kunlun 通过系统实验建立推荐系统的 Scaling Laws，指导工业级模型设计。

## 核心方法与创新点
1. **双轨 Scaling**：区分 Embedding Scaling（embedding 维度/数量）和 DNN Scaling（层数/宽度），分别建立幂律关系：
   $$\mathcal{L}(N_e, N_d) = A \cdot N_e^{-\alpha} + B \cdot N_d^{-\beta} + C$$
2. **Compute-Optimal 点**：固定算力 budget，找到 embedding 和 DNN 的最优分配比例（类似 Chinchilla 对 LLM 的贡献）。
3. **数据 Scaling**：用户行为数据量 $D$ 满足 $\mathcal{L} \propto D^{-\gamma}$，验证数据飞轮效应有理论支撑。
4. **跨平台验证**：在视频推荐、电商推荐、信息流三类场景验证 scaling law 的一致性（指数略有差异）。

## 实验结论
- Embedding 维度从 32 → 512，离线 loss 持续下降（幂律 $\alpha \approx 0.3$）
- DNN 层数从 2 → 8，收益递减明显（$\beta \approx 0.2$，小于 embedding 收益）
- 数据量翻倍带来稳定 ~3% loss 下降（$\gamma \approx 0.25$）
- Compute-optimal 推荐：70% 算力用于 embedding，30% 用于 DNN

## 工程落地要点
- Embedding table 是推荐系统扩展的核心，优先扩 embedding 维度而非 DNN 层数
- 稀疏特征 embedding 的存储 scaling 需配合 Distributed Parameter Server
- 数据新鲜度（data freshness）影响 scaling 效果，需控制变量
- 建议在小规模实验（1/100 参数量）验证 scaling 趋势，再外推大模型配置

## 面试考点
- Q: 推荐系统 scaling law 和 LLM scaling law 的核心差异？
  - A: LLM 主要是 dense 参数（transformer 权重），推荐系统 embedding 是稀疏的高维查表，两者的 compute/memory 权衡完全不同
- Q: 为什么推荐系统中 embedding 比 DNN 更 scale-efficient？
  - A: embedding 直接学习 ID 级别的语义，数据量充足时容量是主要瓶颈；DNN 容量饱和后增大参数量收益边际递减
- Q: 如何实验验证 compute-optimal 点？
  - A: 固定 FLOPs，grid search embedding 维度 × DNN 层数组合，绘制等 FLOP 曲线取最低 loss 点

## 数学公式
$$\mathcal{L}(N) = A \cdot N^{-\alpha} + C, \quad \alpha > 0$$

$$N^*_{\text{embed}} = \arg\min_{N_e + c \cdot N_d = \text{Budget}} \mathcal{L}(N_e, N_d)$$
