# 实时推荐与在线学习

## 文件索引

| 文件 | 内容 | 行数 |
|------|------|------|
| [online-learning.md](online-learning.md) | FTRL 算法详解、模型漂移检测与应对、增量训练、在线多任务学习、E&E 探索利用 | ~290 |
| [realtime-arch.md](realtime-arch.md) | 实时特征工程（Flink/Kafka）、Lambda/Kappa 架构、Feature Store、全链路架构与性能优化 | ~290 |

## 知识点覆盖

### online-learning.md
- 在线学习 vs 批量学习对比（核心区别与混合策略）
- FTRL 算法详解（原理、参数、优劣势、适用场景）
- FTRL vs OGD/FOBOS/RDA 对比
- 模型漂移（数据漂移/概念漂移/标签漂移）
- 漂移检测方法（PSI / KS 检验 / AUC 趋势）
- 增量训练 vs 全量重训（灾难性遗忘与缓解方案）
- 模型热加载与灰度发布
- 在线多任务学习（MMoE/PLE + 动态权重）
- 参数服务器架构与样本拼接
- 探索与利用（Epsilon-Greedy / UCB / Thompson Sampling / LinUCB）
- 在线学习安全与对抗

### realtime-arch.md
- 实时推荐 vs 离线推荐（三层实时性级别）
- Lambda 架构 vs Kappa 架构（原理与选型）
- 实时特征类型（用户短期兴趣/物品实时统计/交叉/上下文）
- Flink 实时特征计算全流程（keyBy + 窗口 + 状态 + Redis）
- Flink 关键技术（事件时间/水印/状态管理/精确一次语义）
- Feature Store 架构与关键能力（时间旅行/血缘/双写）
- 实时推荐系统整体架构（召回→排序→重排协作）
- 延迟预算分配与超时降级
- 高可用容灾设计（微服务/降级/熔断/多活）
- 性能优化全链路（特征/推理/系统三层优化）
- 数据一致性保障（流处理一致性/特征一致性）

## 对应 PDF 章节
- 第 13 章全部 4 节（13.1-13.4），约 35 题
- PDF 页码：940-989
