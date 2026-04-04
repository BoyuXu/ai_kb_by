# GeoGR: A Generative Retrieval Framework for Spatio-Temporal Aware POI Recommendation

> 来源：arXiv 2025 | 领域：rec-sys | 学习日期：20260404

## 问题定义

兴趣点（POI）推荐需要同时建模：
- **空间约束**：用户当前位置 → 距离衰减
- **时序规律**：工作日 vs 周末、早中晚的不同偏好
- **个人兴趣**：历史访问模式

传统生成式推荐忽略地理时序约束，直接生成 POI ID 不考虑空间可达性。

$$P(\text{POI}_t | u, \text{loc}_t, \text{time}_t, \text{hist}_{<t})$$

## 核心方法与创新点

1. **时空感知 ID 编码（STID）**：
   - 将地理坐标量化为分层地理 token（城市→区→格网→POI）
   - 时间编码：小时 + 星期 + 节假日 → 时间 token
   
$$\text{STID}(p) = [\text{geo\_token}_{L1}, \text{geo\_token}_{L2}, \text{geo\_token}_{L3}, \text{poi\_token}]$$

2. **时空约束解码**：
   - 在 beam search 中加入空间可行性约束
   - 距离超过阈值 $d_{\max}$ 的 token 概率置零（Hard Constraint）
   
$$P'(t | \text{context}) = P(t | \text{context}) \cdot \mathbb{1}[\text{dist}(\text{loc}_t, \text{POI}(t)) \leq d_{\max}]$$

3. **时序模式融合**：
   - 按时段划分用户偏好子空间（早餐/午餐/晚餐/夜间）
   - 时段感知 Adapter 动态调整模型参数

## 实验结论

- Recall@10 在 Foursquare-NYC: **+12.3%** vs 最强基线
- 空间约束将「距离超标」推荐率从 31% 降至 **<2%**
- 时序建模在时段切换场景（工作日→周末）提升 **+18%**

## 工程落地要点

- 分层地理 token 编码：建议 4 层（国→市→区→格网500m）
- 格网粒度影响精度-覆盖率权衡：细格网覆盖率低（POI 偏少），粗格网区分度差
- 约束解码在 beam search 中实现，无额外延迟
- 时段 Adapter 可用 LoRA 替代，减少存储

## 面试考点

1. **Q**: POI 推荐与普通 item 推荐的核心区别？  
   **A**: POI 有空间约束（距离可达性）+ 时序规律（时段偏好），需在模型和解码阶段同时建模。

2. **Q**: 分层地理 ID 的优势？  
   **A**: 自然编码空间层级关系（区域相似性），减少 Vocabulary 大小，同时保持空间连续性。

3. **Q**: 约束解码如何影响多样性？  
   **A**: Hard Constraint 牺牲一定多样性（过滤远距离 POI），可改为 Soft Constraint（距离衰减权重）保持多样性。
