# Large Memory Network for Recommendation (LMN) - ByteDance Douyin
> 来源：arXiv:2502.05558 | 领域：rec-sys | 学习日期：20260419

## 问题定义
用户序列建模需要同时捕捉短期意图和长期兴趣。传统方法（DIN/SIM/SDIM）在长序列场景面临：①序列截断丢失长期信息；②全序列建模计算开销大；③不同用户序列分布差异大，泛化困难。

## 核心方法与创新点
1. **大规模记忆块（Large-Scale Memory Block）**：用外部记忆网络压缩和存储用户兴趣
2. **空间感知（Spatial Perception）**：记忆可在不同用户间共享，提升对稀疏用户的泛化能力
3. **时间记忆（Temporal Memory）**：User-aware memory block 记忆用户长期兴趣演化
4. **Product Quantization Memory Decomposition**：PQ分解降低记忆参数量和计算开销
5. **工业级在线 Memory Parameter Server**：专为在线服务设计的记忆参数服务框架

## 部署情况
- **已全量部署于字节跳动抖音电商搜索（Douyin ECS）**
- 服务百万级日活用户
- 训练数据：36亿样本（2024.10.1-10.28，3周训练/1周评估）

## 工程落地要点
- Memory Parameter Server 需支持高并发读写（分布式缓存 + 异步更新）
- PQ分解后记忆检索延迟 < 1ms
- 记忆更新频率：近线（分钟级）+ 在线增量

## 面试考点
- Q: 外部记忆网络 vs Transformer长序列建模的优劣？
  - A: 记忆网络：O(1)检索复杂度，易扩展，但需维护额外存储；Transformer：端到端，但O(n²)注意力开销
- Q: PQ在推荐系统中的应用？
  - A: 向量检索（ANN）、Embedding压缩、Memory分解
