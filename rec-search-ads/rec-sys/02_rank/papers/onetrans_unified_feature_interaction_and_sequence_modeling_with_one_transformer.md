# OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer
> 来源：arxiv/2306.xxxxx | 领域：rec-sys | 学习日期：20260326

## 问题定义
传统推荐模型将特征交叉（Feature Interaction）和序列建模（Sequential Modeling）分为两个独立模块：
- FM/DCN 做特征交叉，LSTM/Transformer 做序列建模
- 两模块串联/并联，参数分散，特征融合不充分
- 序列中的 item 特征无法与上下文特征进行深度交互
- 工程上维护两套系统复杂度高

## 核心方法与创新点
**OneTrans**：用单一 Transformer 统一特征交叉和序列建模。

**核心洞察：** 特征交叉 = 特征之间的 attention；序列建模 = 时间步之间的 attention。Two in one！

**统一 Attention 机制：**
```
# 将所有特征（用户画像、上下文、历史序列item）拼接为 token 序列
tokens = [u_feat_1, ..., u_feat_m, ctx_feat_1, ..., item_1, item_2, ..., item_t, target_item]

# 统一自注意力（不同 token 类型有不同位置编码）
H = TransformerEncoder(tokens)  # 所有 token 相互 attend

# 输出：target_item token 的表示用于打分
score = MLP(H[target_item])
```

**位置编码设计：**
- 特征维度用 Feature Type Embedding 区分
- 序列维度用时序位置编码
- 支持 token 级别的掩码策略（特征 dropout 正则）

**计算优化：**
- 特征 token（非序列）数量固定，序列 token 可变长
- 用 Flash Attention 优化超长序列

## 实验结论
- 腾讯广告平台 A/B：CTR +2.1%，RPM +1.7%
- MovieLens-1M AUC：0.832（vs DIN 0.815，vs DCN-V2 0.821）
- 参数效率：比 DIN+DCN 并联少 30% 参数，性能更好

## 工程落地要点
1. **特征 tokenization**：每个 field 的 embedding 作为一个 token，注意 embedding 维度对齐
2. **序列长度控制**：超长序列用 SIM（Search-based Interest Model）预筛选 Top-K 相关 item
3. **计算预算**：特征数 × 序列长度 = token 总数，需平衡模型效果和延迟
4. **增量序列更新**：实时追加新行为 token，旧 token 缓存（不重新计算 KV）

## 面试考点
**Q1: OneTrans 为什么能统一特征交叉和序列建模？**
A: 本质上两者都是注意力机制：特征交叉关注不同 field 之间的相关性，序列建模关注不同时间步之间的依赖。统一为 token 序列后，自注意力自然捕获两种模式。

**Q2: 与 DIN 相比，OneTrans 的优势？**
A: DIN 只做 target item 与历史序列的 attention，忽略特征之间的交叉。OneTrans 所有 token 互相 attend，特征交叉和序列依赖同时建模，表达更丰富。

**Q3: OneTrans 的计算复杂度问题如何解决？**
A: ①Flash Attention 减少内存带宽②特征 token 数量固定（通常 20-50 个），序列 token 限制最大长度③可对历史序列做分块注意力（局部+全局）。

**Q4: 如何做特征 dropout 正则化？**
A: 训练时随机 mask 一定比例的特征 token（类似 BERT 的 [MASK]），强迫模型学习不依赖单一特征的鲁棒表示，减少过拟合。

**Q5: 线上部署时序列缓存如何设计？**
A: KV Cache 机制：历史行为的 K/V 矩阵预计算并缓存，新请求只需计算新 token 的 Q，与缓存 KV 做 attention，大幅降低在线推理延迟。
