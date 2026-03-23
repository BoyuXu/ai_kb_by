# 推荐系统的 Scaling Law：Wukong（Meta）
> 知识卡片 | 创建：2026-03-23 | 领域：ads / rec-sys

---

**一句话**：Meta 的 Wukong 论文回答了「推荐系统加大模型有没有用」这个问题——有用，但规律和 LLM 不一样：Embedding Table 是主角，MLP 是配角。

**类比**：LLM 的 Scaling 像「大脑皮层越大越聪明」（稠密参数），推荐的 Scaling 像「记忆容量越大越了解用户」（稀疏 Embedding）——两者增长的东西不同。

---

## 核心发现

```
发现 1：推荐系统 Scaling Law 成立，但形式不同于 LLM
  ├── 效果 ∝ 参数量^α（幂律增长），α ≈ 0.05-0.1
  ├── LLM 中 α 更大（0.07-0.15），推荐的 scaling 效率略低
  └── 但推荐参数量从 10M → 1T 时效果持续提升（无明显 plateau）

发现 2：Embedding Table > MLP，优先扩大 Embedding
  ├── 固定 FLOPs 预算：扩大 Embedding >> 加深 MLP
  ├── 原因：Embedding 直接存储 user/item 协同过滤信号
  └── MLP 做特征交叉，但信息上界受 Embedding 制约

发现 3：数据质量 > 数据数量
  ├── 用等量高质量数据（去噪）vs 大量原始数据
  └── 高质量数据胜出约 15-20%

发现 4：稀疏-稠密最优比例
  ├── Embedding(稀疏) : MLP(稠密) ≈ 8:1 ~ 16:1（参数量比）
  └── 过度扩大 MLP 边际收益快速递减
```

---

## 与 LLM Scaling Law 的对比

| 维度 | LLM（Chinchilla）| 推荐（Wukong）|
|------|-----------------|--------------|
| 主要参数 | Attention + FFN（稠密）| Embedding Table（稀疏）|
| Scaling 瓶颈 | 计算量（FLOPs）| 内存带宽（Embedding 查找）|
| 数据类型 | 连续文本序列 | 用户行为事件（稀疏点击）|
| Compute-optimal | token 数 ≈ 20× 参数数 | 训练步数 ≈ 样本数 × epochs |
| 推理模式 | 自回归，顺序 | 实时查表 + 前向，并行 |

---

## 工程落地的特殊挑战

```
万亿参数 Embedding Table 的分布式存储：
├── 无法单机存储：需要参数服务器（PS）或分布式 Embedding
├── 通信是瓶颈：worker 计算 → 推送梯度到 PS → PS 更新 → worker 拉取
├── 异步 SGD：接受参数陈旧（staleness），用 staleness penalty 修正
└── 内存带宽：Embedding 查表是内存密集型，GPU SM 利用率低（~20%）

解决方案：
├── 混合并行：Embedding Table 用 PS，MLP 用数据并行
├── SSD 卸载：不活跃 Embedding 卸载到 NVMe（DLRM/ZionEX）
└── Embedding 压缩：Product Quantization（PQ）/ Hashing（Hash Embedding）
```

---

## 推荐 Scaling 的三个实用结论

1. **给定固定预算，先扩 Embedding，再深化 MLP**
   - 1TB Embedding + 简单 MLP >> 100GB Embedding + 深层 MLP

2. **特征工程仍然关键，Scaling 不能替代特征质量**
   - 垃圾特征 × 万亿参数 = 更精准的垃圾
   - 高质量行为特征（精确时间戳、序列顺序、上下文）比粗粒度特征值 10x

3. **在线增量更新比全量重训更重要**
   - 用户行为分布随时变化，模型每小时更新 >> 每天更新
   - 增量更新 Embedding + 全量更新 MLP 是常见工业方案

---

## 面试考点

1. **Q: 推荐系统的 Scaling Law 和 LLM 最大的不同是什么？**
   A: 推荐的参数主要在 Embedding Table（稀疏，直接存储 user/item 协同信号），LLM 在注意力/FFN（稠密，学习通用语义）；推荐的计算瓶颈是内存带宽，LLM 是算力

2. **Q: 为什么 Embedding 的 Scaling 比 MLP 边际收益更高？**
   A: Embedding 直接增加模型「记住」用户和物品的容量；MLP 做的是特征交叉，但信息上界受 Embedding 质量制约，加深 MLP 不能突破 Embedding 的信息瓶颈

3. **Q: 参数服务器（PS）的基本工作流程？**
   A: ①worker 前向→计算局部梯度 → ②push 梯度到对应 PS shard → ③PS 更新参数 → ④worker pull 最新参数；PS 负责对应 Embedding 行的存储和更新

4. **Q: Embedding 哈希（Hash Embedding）的优缺点？**
   A: 优：无需存储 item-to-index 映射，内存可控；缺：哈希碰撞（不同 item 共享同一 Embedding），精度下降，热门 item 受影响

5. **Q: 给你 100 亿预算（参数数），如何分配推荐模型的架构？**
   A: 参考 Wukong：Embedding Table 80-90 亿（稀疏）+ 3-4 层 MLP（约 10 亿）；特征：user/item ID + 交叉特征 + 上下文；优先做好 Embedding 的增量更新和数据质量
