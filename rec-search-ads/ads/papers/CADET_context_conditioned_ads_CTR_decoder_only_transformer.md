# CADET: Context-Conditioned Ads CTR Prediction with Decoder-Only Transformer
> 来源：arXiv:2602.11410 | 领域：ads | 学习日期：20260330

## 问题定义
广告 CTR 预估传统上用 Encoder-based 模型（DIN/DIEN），对用户历史序列和广告特征做 attention。CADET 将 Decoder-Only Transformer（GPT 风格）引入 CTR 预估，利用 causal attention 建模用户行为序列的因果依赖关系，并以广告特征作为 condition 生成 CTR 预估值。

## 核心方法与创新点
1. **Decoder-Only for CTR**：将用户行为序列 $[a_1, a_2, ..., a_T, ad]$ 作为因果序列，每个 position 的 hidden state 只 attend 到前面的 token，天然建模时序依赖。
2. **广告 Conditioning**：广告特征（AD token）作为 context appended 到序列末尾，通过 causal attention 聚合所有历史行为信息生成 CTR 打分。
3. **Pre-training on Behavior**：在用户行为序列上做 next-item prediction 预训练，学习通用用户兴趣表征，CTR 任务 fine-tune。
4. **Flash Attention 优化**：超长行为序列（>1000 步）用 Flash Attention 降低显存，使工业级序列长度可行。
5. **多尺度位置编码**：区分时间位置（行为发生时间）和序列位置（行为顺序），双维度位置编码捕捉时序规律。

## 实验结论
- 某大型电商广告平台：CTR AUC +0.85%，相比 DIN baseline
- 长序列（>500 步历史）效果提升最显著（+1.3% AUC）
- Pre-training 带来冷启动广告 CTR AUC +2.1%（行为泛化优势）

## 工程落地要点
- Decoder-Only 推理时需维护 KV Cache，避免历史 token 重复计算
- 行为序列动态更新（实时点击需即时拼接）需流式推理支持
- Causal attention 使历史行为不能双向 attend，可能损失部分信息（可用 Prefix attention 缓解）
- 建议 batch size ≥ 128（Transformer 小 batch 效率低），配合混合精度（BF16）训练

## 常见考点
- Q: Decoder-Only 和 Encoder-Only 在 CTR 场景的区别？
  - A: Encoder（BERT）双向 attention，全局理解历史；Decoder 因果 attention，建模序列生成过程，更符合用户行为时序因果。CTR 场景两者均可，Decoder 更自然地支持 next-item 预训练
- Q: KV Cache 在 CTR 预估中的作用？
  - A: 用户历史序列较固定（只有末尾新增），KV Cache 避免重新计算历史 K/V，推理加速 ~5-10x
- Q: Pre-training 在广告 CTR 中如何避免数据泄露？
  - A: Pre-training 用有机行为数据（自然点击/浏览），不包含广告曝光数据；Fine-tuning 才引入广告 label

## 数学公式

$$
\text{CTR} = \sigma(W \cdot h_T^{\text{ad}}), \quad h_T^{\text{ad}} = \text{Decoder}([a_1,...,a_T, ad])
$$

$$
\mathcal{L}_\text{pretrain} = -\sum_t \log P(a_{t+1} | a_1,...,a_t)
$$
