# OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender

> 来源：arXiv 2025 | 领域：ads | 学习日期：20260404

## 问题定义

工业推荐/广告系统通常是 **堆叠架构**：
- 独立特征交叉模块（FM/DCN）
- 独立序列建模模块（Transformer/GRU）
- 独立多任务输出头

各模块分离训练，特征交叉与序列建模无法相互增强，信息传递低效。

## 核心方法与创新点

**OneTrans** 用单一 Transformer 架构统一两个任务：

1. **统一序列化输入**：
   - 将所有特征（用户行为序列 + 上下文特征 + 目标广告）格式化为 token 序列
   
```
[user_feat_tokens] [behavior_seq_tokens] [ctx_tokens] [target_ad_token]
```

2. **Attention 掩码设计**：
   - 行为序列 token：因果 Attention（防止未来泄露）
   - 上下文 + 目标 token：全局 Attention（允许看全局特征）
   - 跨域 Attention：特征交叉 token 可 attend 到序列 token

3. **位置编码适配**：
   - 行为序列：时序位置编码（RoPE）
   - 静态特征：无位置编码（顺序无关）

4. **共享 Transformer 的双任务 Loss**：

$$\mathcal{L} = \mathcal{L}_{\text{CTR}} + \lambda \cdot \mathcal{L}_{\text{sequence\_pred}}$$

- 序列预测辅助任务：预测下一个行为 item，增强序列建模质量
- 主任务：CTR 预测

## 实验结论

- CTR AUC vs 堆叠架构（DCN + BST）: **+0.9‰**
- 参数量减少 **35%**（统一 Transformer vs 独立模块之和）
- 训练速度提升 **1.4x**（统一反向传播 vs 多模块独立训练）
- 行为序列长度 50→200：OneTrans 收益 +0.4‰，独立模块 +0.1‰（统一架构序列扩展更高效）

## 工程落地要点

- Attention Mask 矩阵需定制（稀疏掩码），避免全量 $O(n^2)$ 注意力
- token 序列总长度建议 ≤ 256（Flash Attention 优化区间）
- 多任务 loss 权重 λ：建议 0.1-0.3（辅助任务不能压制主任务）
- 推理时去掉序列预测头，只保留 CTR 头（零额外开销）

## 面试考点

1. **Q**: 为什么用 Transformer 统一特征交叉和序列建模？  
   **A**: Self-Attention 天然是特征交叉（任意两个 token 的交互），同时 Causal Attention 建模序列时序依赖。统一后特征交叉与序列信息可以互相增强。

2. **Q**: OneTrans 的 Attention Mask 如何设计？  
   **A**: 行为序列用因果掩码（看历史不看未来）；上下文特征用全局掩码（无时序约束）；特征 token 允许 attend 到序列，但序列 token 不能 attend 到"未来"的上下文（避免泄露）。

3. **Q**: 辅助任务（序列预测）为什么能提升 CTR？  
   **A**: 序列预测迫使模型更好地建模用户行为规律，学到更好的序列表示，作为 CTR 的上游特征。多任务学习的正则化效果也减少过拟合。
