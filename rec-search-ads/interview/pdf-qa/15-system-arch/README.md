# 推荐系统架构与高性能服务

## 本章文件索引

| 文件 | 内容 | 行数 |
|------|------|------|
| [fundamentals.md](fundamentals.md) | 四层架构/召回策略/排序模型/多任务学习/重排/评估体系 | ~230 |
| [large-scale.md](large-scale.md) | 多阶段漏斗/ANN索引/特征工程/排序模型演进/重排优化/AB测试 | ~440 |
| [serving-optimization.md](serving-optimization.md) | 模型部署/推理加速/实时特征/全链路监控/云原生/延迟优化 | ~210 |
| [feature-store.md](feature-store.md) | Feature Store架构/实时特征Pipeline/离线特征/特征一致性/缓存策略 | ~250 |
| [model-serving.md](model-serving.md) | Serving框架对比/量化推理/动态Batching/微服务部署/K8s弹性伸缩 | ~280 |

## 阅读建议

- 入门：先读 fundamentals.md 掌握推荐系统四层架构全貌
- 进阶：large-scale.md 覆盖各层的工业级实现细节
- 工程深度：serving-optimization.md + feature-store.md + model-serving.md 覆盖高性能服务的核心技术栈
- 面试准备：每个文件末尾的「面试高频问题」可直接用于模拟问答
