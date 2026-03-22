# CADET: Context-Conditioned Ads CTR Prediction With a Decoder-Only Transformer

> arXiv: 2502.XXXXX | 发布: 2026-02-11 | 重要程度: ⭐⭐⭐⭐⭐

---

## 1. 问题定义

**广告 CTR 预测的上下文建模问题：**
- 传统 CTR 模型（DCN、DIN、DIEN）对 user/ad feature 建模，但对**上下文信息**（用户当前浏览会话、页面上下文、相邻广告）建模不足
- 序列推荐领域已有 Transformer 大显身手，但广告 CTR 场景有额外挑战：高稀疏性、实时性要求、上下文依赖

**核心问题：** 如何用 Decoder-Only Transformer 架构捕捉广告曝光上下文，提升 CTR 预估精度？

---

## 2. 核心方法（关键创新）

### CADET 架构

```
上下文序列 [ad_1, ad_2, ..., ad_{t-1}] → Decoder-Only Transformer
                                                    ↓
                                         条件化 CTR 预测 P(click | context, ad_t)
```

**三大创新：**

1. **Decoder-Only 架构**：借鉴 GPT 范式，用 causal attention 建模广告展示序列，每个位置的 CTR 预测都依赖前序广告上下文（而非双向 attention）

2. **Context Conditioning**：将页面上下文（用户会话 query、页面类型、时段）注入 cross-attention，实现"同一 ad 在不同上下文下预测不同 CTR"

3. **工业化训练技巧**：
   - 稀疏 ID feature 用 embedding hash，dense feature 直接拼接
   - 位置编码改为 Rotary Position Embedding (RoPE)，更好处理变长序列
   - 混合精度训练 + 序列截断，控制训练/推理成本

---

## 3. 实验结论

- 在大规模广告 CTR 预测 benchmark 上显著超越 DCN-V2、DIN、BST 等 baseline
- AUC 提升 **+0.5%~1.2%**（工业场景 AUC 0.1% 即有显著商业价值）
- 推理延迟：通过 KV Cache 复用，推理时延增加 **<20%**（相比 DIN 等 baseline）
- 特别在"短 session 内高频曝光"场景（如信息流广告）效果最显著

---

## 4. 工程价值（如何落地）

**适用场景：**
- 信息流广告（上下文依赖强）
- 搜索广告（query 上下文明确）
- 大规模广告平台（已有 Transformer infra）

**工程要点：**
1. **KV Cache 复用**：同一 session 内多次预测时，前序上下文的 KV 缓存可以复用，大幅降低推理成本
2. **序列长度控制**：建议 ctx_len=32~64，平衡效果与延迟
3. **Causal Mask**：避免未来信息泄露，训练和推理行为一致

**与 DIN/DIEN 对比：**
| 模型 | 上下文建模 | 计算复杂度 | 延迟 |
|------|-----------|-----------|------|
| DIN | 用户历史行为 attention | O(n) | 低 |
| DIEN | GRU + attention | O(n) | 中 |
| CADET | Decoder-Only Transformer + 上下文 | O(n²) | 中高（KV Cache 优化后可接受）|

---

## 5. 面试考点

**Q1: Decoder-Only vs Encoder-Only Transformer 在 CTR 预测中的区别？**
> Decoder-Only 用 causal attention，天然适合序列自回归场景，推理时可复用 KV Cache；Encoder-Only（如 BERT4Rec）双向 attention，适合离线建模但在线推理成本高

**Q2: 广告 CTR 模型中如何引入上下文信息？**
> 方法：① 直接拼接上下文 feature ② Cross-Attention（CADET 的方式）③ 把上下文序列 prepend 到用户行为序列

**Q3: 工业广告系统 AUC 提升 0.1% 有多大价值？**
> 头部广告平台（BAT/字节）AUC 每提升 0.1%，广告 RPM 可以提升 1-3%，对应数亿级别的年收入增量

---

*笔记生成时间: 2026-03-12 | MelonEggLearn*
