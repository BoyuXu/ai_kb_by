# 互联网大厂推荐算法实战：精排/粗排/重排/召回全链路解析

> **来源**：《互联网大厂推荐算法实战》赵传霖（快手算法专家，清华博士）
>
> **关联概念页**：[[attention_in_recsys]] | [[embedding_everywhere]] | [[multi_objective_optimization]]
>
> **关联 synthesis**：[[20260411_sequential_and_generative_rec]] | [[recsys-infrastructure-landscape-2025-2026]] | [[推荐广告生成式范式统一全景]]

---

## 1. 全链路架构：级联漏斗

```
全量物料池 (~百万/千万)
    ↓ 召回 (Recall)
候选集 (~数千)     ← 多路召回: 双塔/图/序列/热门
    ↓ 粗排 (Pre-Ranking)
候选集 (~数百)     ← 轻量模型蒸馏: 双塔/轻量交叉
    ↓ 精排 (Ranking)
候选集 (~数十)     ← 重模型: DIN/DIEN/DCN/Transformer + 多目标
    ↓ 重排 (Re-Ranking)
展示结果 (~十几)   ← 多样性/去重/业务规则/上下文
```

**核心原则**：每一级用更复杂的模型处理更少的候选。上层负责"不漏"，下层负责"精准"。

## 2. 召回 (Recall)

### 双塔模型（工业主力）

**架构**：User Tower + Item Tower → 内积/余弦相似度 → ANN 检索

**四条改进双塔的道路**（书中总结）：
1. **特征交互增强**：在塔内加深特征交叉（如 SENet、DSSM 变体）
2. **负样本优化**：困难负例采样、混合负采样、batch 内负采样
3. **多兴趣建模**：MIND（多头）/ ComiRec（动态路由）
4. **对齐与蒸馏**：用精排模型蒸馏召回模型，减少级间 gap

### 多路召回融合
- 协同过滤（ItemCF/UserCF）：简单有效的基线
- 向量召回（双塔）：泛化能力强
- 图召回（GraphSAGE/PinSage）：利用关系网络
- 序列召回（SASRec/GRU4Rec）：捕获实时兴趣
- 热门/规则召回：保底策略

## 3. 粗排 (Pre-Ranking)

### 定位
召回和精排之间的"过滤器"，面对数千候选，需在毫秒级完成。

### 典型方案
| 方案 | 思路 | 优劣 |
|------|------|------|
| 双塔打分 | 和召回类似但特征更多 | 快但交叉不够 |
| **精排蒸馏** | 用精排 teacher 训练轻量 student | 效果好，工业主流 |
| 轻量交叉 | 简单特征交叉（如浅层 DCN） | 平衡速度和效果 |

### 粗排蒸馏要点
- Teacher: 精排模型（重特征交叉）
- Student: 轻量双塔或浅层网络
- Loss: KD loss (soft label) + 硬标签 loss 联合
- **关键**：蒸馏的 gain 来自把精排的"排序知识"压缩进粗排

## 4. 精排 (Ranking) — 最核心环节

### 经典模型演进
```
LR → FM → DeepFM → DCN → DIN → DIEN → HSTU → SORT
```

### 核心技术
- **特征交叉**：显式（FM/DCN）+ 隐式（DNN）
- **用户行为建模**：DIN (target attention) → DIEN (兴趣演化) → SIM (长序列检索)
- **多目标学习**：MMOE/PLE/AITM，同时优化 CTR/CVR/时长/多样性等
- **Transformer 精排**：HSTU、SORT 代表最新方向

### 精排的工业约束
| 约束 | 影响 |
|------|------|
| 延迟 <100ms | 限制模型复杂度 |
| 特征稀疏 | 需要 embedding + 交叉 |
| 样本选择偏差 | 需要纠偏（IPW/DR） |
| 多目标冲突 | 需要 Pareto 优化 |

## 5. 重排 (Re-Ranking)

### 核心目标
不是"找最好的"，而是"排出最好的列表"——考虑 item 间关系和整体体验。

### 关键技术
| 技术 | 目的 | 方法 |
|------|------|------|
| **多样性** | 避免信息茧房 | MMR, DPP, 滑窗打散 |
| **去重** | 避免重复 | 内容指纹 + 类目打散 |
| **上下文感知** | 考虑位置效应 | PRM, SetRank |
| **业务规则** | 广告插入/置顶/过滤 | 规则引擎 |

### 多样性核心方法
- **MMR (Maximal Marginal Relevance)**：$\text{score} = \lambda \cdot \text{relevance} - (1-\lambda) \cdot \max_{j \in S} \text{sim}(i,j)$
- **DPP (Determinantal Point Process)**：行列式点过程，数学上优雅但计算贵
- **滑窗打散**：工业最常用，简单规则保证窗口内类目多样

## 6. 实践难题

### 冷启动
| 场景 | 解法 |
|------|------|
| 新用户 | 人口统计特征 + 热门推荐 + Exploration (ε-greedy/UCB) |
| 新物料 | 内容特征 (文本/图像 embedding) + 流量扶持 + 生成式方法 |

### 多任务推荐
- 共享底层 + 任务特定塔 (MMOE/PLE)
- 梯度冲突处理：GradNorm, PCGrad
- 工业经验：不同任务的 loss 权重需要动态调整

### 多场景推荐
- 场景间知识迁移（STAR, SAMD）
- 共享 embedding + 场景特定参数

## 7. 面试高频考点

**Q1: 推荐系统为什么要分召回/粗排/精排/重排四级？**
> 计算量约束。全量物料百万级，不可能对每个都跑重模型。级联漏斗逐级缩减候选集，每级用更复杂模型处理更少候选，在效果和效率间取平衡。

**Q2: 双塔模型的核心局限？**
> User 和 Item 在最后才做内积交互，无法建模细粒度特征交叉。解法：① 塔内加深交叉 ② 多向量表示 ③ 蒸馏精排知识。

**Q3: DIN 的 target attention 和标准 attention 有什么区别？**
> DIN 的 attention 是"target-aware"：用候选 item 作为 query，对用户历史行为加权。标准 attention 是自注意力。DIN 的核心公式：$\text{weight}_i = \text{softmax}(f(e_{item}, e_{history_i}))$

**Q4: 精排多目标怎么平衡？CTR 和时长冲突怎么办？**
> MMOE: 多个 expert 共享 + 门控选择。PLE: 任务特定 expert + 共享 expert + 渐进萃取。实际工业中，最终融合分常用加权公式：$\text{score} = \text{CTR}^{w_1} \times \text{CVR}^{w_2} \times \text{duration}^{w_3}$，权重人工或自动调。

**Q5: 重排的多样性为什么重要？不多样化会怎样？**
> 短期看推荐相似内容 CTR 更高，长期看导致信息茧房、用户疲劳、留存下降。多样性是长期价值的投资。DPP 数学最优但 O(k³)，工业中常用滑窗打散 O(kn)。

**Q6: 级联架构 vs 端到端生成式（OneRec-V2），工业中选哪个？**
> 目前绝大多数公司仍用级联架构（成熟、可解释、各级可独立优化）。端到端生成式是前沿方向（快手 OneRec-V2 已落地），优势是避免级间误差传播，但对算力要求高、调试复杂。短期面试答级联为主+了解生成式趋势。

---

*Last updated: 2026-04-11*

---

## 相关概念

- [[sequence_modeling_evolution|序列建模演进]]
- [[generative_recsys|生成式推荐统一视角]]
