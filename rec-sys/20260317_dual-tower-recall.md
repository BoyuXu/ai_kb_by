# 双塔召回模型（Two-Tower / Dual Encoder）

> 来源：工程实践 / Google DSSM 系列 | 日期：20260317

## 问题定义

召回阶段需要从亿级物料库中快速找到用户可能感兴趣的 Top-K 候选，要求：
- **效率**：每次请求 <10ms，不能做用户-物品交叉特征在线计算
- **质量**：召回结果要覆盖精排最终推荐的物品

传统协同过滤（ALS/矩阵分解）无法融入丰富的 side information（图文特征、上下文）。双塔模型通过**离线建立物品索引 + 在线用户向量检索**解决这一问题。

## 核心方法与创新点

1. **双塔结构**
   - **User Tower**：输入用户特征（ID、画像、历史行为序列聚合）→ DNN → 用户向量 $u \in \mathbb{R}^d$
   - **Item Tower**：输入物品特征（ID、内容特征、统计特征）→ DNN → 物品向量 $v \in \mathbb{R}^d$
   - 相似度：$\text{score}(u, v) = u \cdot v$（内积）或 $\cos(u, v)$

2. **训练目标**
   - **In-batch Negative Sampling**：batch 内其他正样本的物品作为当前用户的负样本，高效且实现简单
   - **Sampled Softmax Loss**：$\mathcal{L} = -\log \frac{e^{u\cdot v^+/\tau}}{e^{u\cdot v^+/\tau} + \sum_j e^{u\cdot v_j^-/\tau}}$
   - **Hard Negative Mining**：混合随机负样本 + 难负样本（被召回但未点击），提升模型区分能力

3. **工程优化**
   - **物品向量离线预算**：Item Tower 输出预先存入向量数据库（Faiss/Milvus）
   - **用户向量实时计算**：User Tower 在线推理，latency ~2ms
   - **ANN 检索**：HNSW 或 IVF+PQ 量化检索 Top-K

4. **变体**
   - **YoutubeDNN**（2016）：基础双塔，使用均值池化聚合历史行为
   - **MIND**：用户侧多兴趣，Capsule Network 生成多个用户向量
   - **Facebook EBR**：混合 in-batch + hard negatives
   - **SimCSE 风格 CL**：对比学习增强向量质量

## 实验结论

- 相比矩阵分解，在有 side information 的场景 Recall@100 提升 20~30%
- Hard negative mining 相比纯随机负样本 Recall@100 提升约 5~10%
- 温度参数 τ 对训练稳定性影响显著，通常 0.05~0.1

## 工程落地要点

1. **特征实时性**：用户行为序列需 near-real-time 更新（流式特征平台），冷热分离
2. **向量更新频率**：Item 向量通常每小时/每天更新，需与模型版本对齐
3. **负样本偏差修正**：In-batch negative 存在曝光偏差（热门物品被负采到的概率高），需做 sampling bias correction：$s = u \cdot v - \log p(v)$
4. **召回路数**：通常部署 3~5 路召回（双塔 + 协同过滤 + 内容），取并集
5. **评估指标**：Recall@K, Hit Rate@K, 离线召回率（精排结果在召回中的覆盖率）

## 面试考点

- **Q: 双塔模型为什么不能做用户-物品交叉特征？**
  A: 因为需要离线预算所有物品向量并建立索引，如果引入交叉特征，物品向量就依赖用户，无法离线预算，在线需要对所有候选做用户-物品联合推理，计算量爆炸。

- **Q: In-batch negative 的问题是什么？**
  A: 1) 热门 item 作为负样本的概率更高，引入 exposure bias；2) 同一批次内可能存在假负样本（实际是用户喜欢的但没有正样本标签）。解决：sampling bias correction + hard negative 混合。

- **Q: 双塔和 DSSM 的关系？**
  A: DSSM（2013 Microsoft）是双塔模型的先驱，用于搜索 Query-Document 匹配，后被推荐系统采用并扩展了行为序列建模、多兴趣等能力。
