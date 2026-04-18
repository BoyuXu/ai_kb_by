# 多目标优化与多任务学习

## 子主题索引

| 文件 | 内容 | 行数 |
|------|------|------|
| [mmoe-ple.md](mmoe-ple.md) | MMoE/PLE/CGC 架构详解、Expert 设计、Gate 机制、训练技巧 | ~220 |
| [loss-design.md](loss-design.md) | 多任务 Loss 设计、权重策略（Uncertainty/GradNorm/PCGrad）、Pareto 优化 | ~210 |
| [esmm-cvr.md](esmm-cvr.md) | ESMM/全空间建模、CVR 样本偏差、AITM、多目标融合公式、在线部署监控 | ~220 |

## 核心概念

多任务学习(MTL)的本质：通过共享表示在一个模型中同时优化多个相关目标（如CTR、CVR、完播率），利用任务间的信息互补提升泛化能力。

关键动机：
- 单目标模型无法同时优化多个业务指标
- 多个独立模型维护成本高、特征不一致
- 相关任务的辅助信号可缓解数据稀疏（如CVR样本少，可借助CTR信号）

## 经典模型架构

### Shared-Bottom
所有任务共享底层网络，顶部各任务独立 tower。
- 优点：参数高效，结构简单
- 缺点：任务差异大时，共享层被迫妥协，引发负迁移

### MMoE (Multi-gate Mixture-of-Experts)
- 多个 Expert 网络并行提取特征
- 每个任务有独立的 Gate 网络，输出各 Expert 的软权重
- Gate(x) = Softmax(W_g * x)，加权求和 Expert 输出
- 解决了 Shared-Bottom 的负迁移问题：任务可以选择性使用不同 Expert
- 局限：Expert 之间仍可能趋同，Gate 可能坍塌
- 详见 [mmoe-ple.md](mmoe-ple.md)

### PLE (Progressive Layered Extraction)
- CGC 结构：每层既有共享 Expert 又有任务专属 Expert
- 任务 Gate 同时接收共享 Expert 和本任务专属 Expert 的输出
- 多层堆叠，逐步提炼任务特定信息
- 比 MMoE 更显式地分离共享与私有知识，缓解 seesaw 效应
- 工业界主流选择（腾讯提出，广泛应用于推荐排序）
- 详见 [mmoe-ple.md](mmoe-ple.md)

### ESMM (Entire Space Multi-Task Model)
- 解决 CVR 预估的样本选择偏差（传统 CVR 只在点击样本上训练）
- 核心思路：pCVR = pCTCVR / pCTR，在全样本空间建模
- 两个 tower：CTR tower + CVR tower，共享 embedding
- pCTCVR = pCTR * pCVR，用全量曝光样本训练
- 详见 [esmm-cvr.md](esmm-cvr.md)

### AITM (Adaptive Information Transfer Multi-task)
- 不假设严格的概率乘积关系
- 用可学习的 Gate 控制上游任务向下游任务的信息传递
- 比 ESMM 更灵活，适合复杂任务依赖
- 详见 [esmm-cvr.md](esmm-cvr.md)

## 负迁移与 Seesaw 现象

负迁移：一个任务的优化损害另一个任务的性能，常见于任务差异大时。

Seesaw 效应：A 任务指标提升时 B 任务下降，像跷跷板。

产生原因：
- 任务梯度方向冲突（共享参数被拉向不同方向）
- 任务学习速度不一致（快任务主导共享层更新）
- 数据分布差异（不同任务对应不同样本子集）

检测方法：
- 监控各任务 loss 曲线是否此消彼长
- 计算任务间梯度余弦相似度（负值说明冲突）

缓解策略：
- 增加任务专属参数（PLE 方案）
- 梯度调节（GradNorm, PCGrad）→ 详见 [loss-design.md](loss-design.md)
- 任务权重动态调整

## 损失函数权重策略

详见 [loss-design.md](loss-design.md)，包括：
- 固定权重、不确定性加权、GradNorm、PCGrad、GradVac
- Pareto 优化框架
- 各方法的适用场景和工业实践

## 多目标在线融合与部署

详见 [esmm-cvr.md](esmm-cvr.md)，包括：
- 加法/乘法/混合融合公式设计
- Bandit 自适应权重调优
- 在线监控体系（Flink、告警分级）
- AB 实验分析方法

## 面试高频考点

1. MMoE vs PLE 区别：PLE 增加任务专属 Expert + 多层渐进提取
2. ESMM 解决什么问题：CVR 的样本选择偏差，在全空间建模
3. 如何检测和缓解负迁移：梯度余弦、GradNorm、增加专属参数
4. 损失权重怎么设：不确定性加权 > 固定权重，GradNorm 动态调整
5. 线上融合公式设计：乘法融合 + 指数调权，通过 AB 实验确定
6. AITM vs ESMM：严格概率链式分解 vs 可学习门控信息传递
