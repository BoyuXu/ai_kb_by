# MMoE / PLE / CGC 架构详解

## 1. 多任务学习架构演进路线

Shared-Bottom → MMoE → CGC → PLE

核心矛盾：共享带来正迁移 vs 任务差异导致负迁移。架构演进本质上是在「共享程度」和「任务独立性」之间找更好的平衡点。

## 2. Shared-Bottom 基线

结构：所有任务共享一个底层 DNN，顶部各 task 独立 tower。

```
输入特征 → [共享底层 DNN] → shared_repr
                              ├→ Tower_A → pred_A (CTR)
                              └→ Tower_B → pred_B (CVR)
```

优点：参数高效，结构最简单，适合高度相关的任务组合。

致命缺陷：
- 任务差异大时，共享层被迫在冲突梯度间妥协
- 一个任务的噪声会通过共享层污染其他任务
- 无法让不同任务选择性地使用不同特征模式

## 3. MMoE (Multi-gate Mixture-of-Experts)

### 3.1 核心思想

用多个 Expert 替代单一共享底层，每个任务通过独立的 Gate 网络选择性地组合 Expert 输出。

```
输入特征 x
├→ Expert_1(x) → e_1
├→ Expert_2(x) → e_2
├→ ...
└→ Expert_n(x) → e_n

Gate_A(x) = Softmax(W_gA * x) → [g_1, g_2, ..., g_n]  (任务A的门控权重)
Gate_B(x) = Softmax(W_gB * x) → [g_1, g_2, ..., g_n]  (任务B的门控权重)

任务A输入 = sum(g_i * e_i)  → Tower_A → pred_A
任务B输入 = sum(g_i * e_i)  → Tower_B → pred_B
```

### 3.2 关键设计细节

Expert 网络：
- 通常 4-8 个 Expert，每个是 2-3 层 MLP
- Expert 之间结构相同但参数独立，允许学习不同的特征模式
- Expert 数量过少 → 表达力不够；过多 → 训练不稳定、门控坍塌

Gate 网络：
- 每个任务一个独立 Gate，输入是原始特征（或底层 embedding）
- Gate 输出维度 = Expert 数量，经 Softmax 归一化
- Gate 的作用：根据输入特征动态决定各 Expert 的贡献比例
- 不同样本对同一任务，Gate 权重也不同（输入依赖）

### 3.3 MMoE 的局限

Expert 趋同问题：
- 多个 Expert 可能学到几乎相同的特征模式
- 原因：没有显式约束 Expert 的多样性
- 结果：退化为 Shared-Bottom

Gate 坍塌问题：
- 某些 Expert 的 Gate 权重趋近 0，形同虚设
- 原因：训练初期某些 Expert 被偶然忽略，后续无法恢复
- 缓解：Expert dropout、正则化、均匀初始化

仍存在负迁移：
- 所有 Expert 对所有任务共享，任务间仍在同一参数空间竞争
- 任务差异极大时（如 CTR vs 停留时长），效果提升有限

## 4. CGC (Customized Gate Control)

PLE 的单层基本单元，核心改进：引入任务专属 Expert。

```
输入特征 x
├→ Shared Expert_1(x), Shared Expert_2(x)    # 共享 Expert
├→ Task_A Expert_1(x), Task_A Expert_2(x)    # A 专属 Expert
└→ Task_B Expert_1(x), Task_B Expert_2(x)    # B 专属 Expert

Gate_A(x) → 加权组合 [Shared Experts + Task_A Experts]  (不看 B 的专属 Expert)
Gate_B(x) → 加权组合 [Shared Experts + Task_B Experts]  (不看 A 的专属 Expert)
```

关键区别：
- Gate_A 只能看到共享 Expert 和 A 的专属 Expert，看不到 B 的专属 Expert
- 专属 Expert 保证每个任务有「私有」的特征提取能力
- 共享 Expert 负责任务间的正迁移
- 比 MMoE 更显式地分离共享知识与任务特定知识

## 5. PLE (Progressive Layered Extraction)

### 5.1 核心思想

多层堆叠的 CGC 结构，逐层渐进式地分离和提炼任务特定信息。

```
Layer 1:
  共享 Expert × k + 任务A Expert × k + 任务B Expert × k
  → Gate_A_L1, Gate_B_L1 → 各自加权输出

Layer 2:
  共享 Expert × k + 任务A Expert × k + 任务B Expert × k
  (输入来自 Layer 1 的 Gate 输出)
  → Gate_A_L2, Gate_B_L2 → 各自加权输出

...
→ Tower_A → pred_A
→ Tower_B → pred_B
```

### 5.2 渐进提取机制

第一层：粗粒度特征分离，共享 Expert 提取通用模式
第二层：在第一层基础上进一步精炼，任务专属 Expert 提取更具区分性的特征
更深层：逐步增强任务特异性，共享信息越来越抽象

每层的输入不是原始特征，而是上一层各 Gate 的输出——这使得 Expert 可以在前一层的特征基础上做更高阶的提取。

### 5.3 PLE 的核心优势

对比 MMoE：
- 显式的共享/私有分离 → 更有效缓解负迁移和 seesaw 效应
- 多层渐进提取 → 更好的特征层次化
- Gate 的选择范围被限制 → 降低 Gate 坍塌风险

工业落地：
- 腾讯提出，在微信看一看、QQ 浏览器等多个场景验证
- 短视频推荐中同时优化 CTR + 完播率 + 点赞率，PLE 是业界主流选择
- 典型配置：2-3 层 CGC，每层 2-4 个 Expert/任务，Expert 为 2 层 MLP

## 6. Expert 设计实践

### 6.1 Expert 数量选择

经验法则：
- 共享 Expert：2-4 个
- 每个任务专属 Expert：1-3 个
- 总 Expert 数不宜超过 12 个（训练稳定性）

验证方法：
- 观察 Gate 权重分布——如果某 Expert 的权重持续极低，说明它冗余
- 逐步增加 Expert 并监控各任务指标，找到边际收益递减的拐点

### 6.2 Expert 多样性保证

正则化：
- Expert dropout：训练时随机丢弃部分 Expert，迫使其他 Expert 学习互补特征
- 互信息最小化：添加 Expert 输出间的互信息惩罚项
- 梯度疫苗 (GradVac)：将冲突梯度投影，保持 Expert 的特征空间正交性

结构差异化：
- 不同 Expert 使用不同的隐层宽度或深度
- 部分 Expert 专门处理稀疏特征，部分处理稠密特征

### 6.3 Gate 网络设计

基本 Gate：单层线性 + Softmax（足够大多数场景）
增强 Gate：两层 MLP + Softmax（任务关系复杂时）
温度参数：Softmax(logits / tau)，tau < 1 更尖锐（专家更特化），tau > 1 更平滑（更多共享）

## 7. 训练技巧

### 7.1 Expert 初始化
- Xavier/He 初始化，各 Expert 使用不同随机种子
- Gate 权重初始化为近均匀，避免初期 Expert 被忽略

### 7.2 学习率
- Expert 和 Gate 可用不同学习率
- 专属 Expert 学习率可略高于共享 Expert（加速任务特化）
- Gate 学习率通常与 Tower 一致

### 7.3 训练顺序
- 可先预训练共享 Expert（用全部任务 loss），再加入专属 Expert 微调
- 或端到端训练（更常用，但需要更仔细的 loss 权重调节）

## 8. 面试高频问题

Q: MMoE 相比 Shared-Bottom 的本质改进是什么？
A: 从「强制共享」变为「选择性共享」。Gate 网络让每个任务可以根据输入动态选择使用哪些 Expert 的知识，而不是被迫接受同一个共享表示。

Q: PLE 相比 MMoE 解决了什么问题？
A: MMoE 的所有 Expert 仍然是共享的，任务间在同一参数空间竞争。PLE 引入任务专属 Expert，显式分离共享知识和私有知识，更有效缓解负迁移和 seesaw 效应。

Q: 怎么判断多任务模型是否发生了负迁移？
A: 对比单任务模型和多任务模型在各任务上的指标。如果某任务的多任务版本不如单任务版本，说明发生了负迁移。还可以监控训练过程中各任务 loss 曲线是否出现此消彼长。

Q: Expert 数量怎么定？
A: 没有理论最优解。经验上从少到多实验：先 4 个共享 Expert + 2 个专属 Expert/任务，观察 Gate 权重和各任务指标。如果 Gate 权重过于均匀，说明 Expert 不够多样，需要增加并配合正则化。

Q: CGC 和 PLE 的关系？
A: CGC 是 PLE 的单层基本单元。PLE = 多层堆叠的 CGC。单层 CGC 已经比 MMoE 更好，多层 PLE 通过渐进提取进一步提升效果。

Q: Gate 网络为什么用 Softmax 而不是 Sigmoid？
A: Softmax 保证权重和为 1，是 Expert 输出的凸组合，物理意义更清晰（各 Expert 的贡献比例）。Sigmoid 各维度独立，权重和不为 1，容易导致表示尺度不稳定。

Q: 线上服务时 MMoE/PLE 的推理开销如何？
A: 所有 Expert 需要对每个请求都推理一次，开销约为单 Expert 的 N 倍。但 Gate + 加权求和开销极小。实际优化：知识蒸馏到轻量模型、Expert 剪枝、量化。共享底层只推理一次，各 tower 并行出分。
