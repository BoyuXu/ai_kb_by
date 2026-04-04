# EST: Towards Efficient Scaling Laws in Click-Through Rate Prediction

> 来源：arXiv 2025 | 领域：ads | 学习日期：20260404

## 问题定义

LLM 的 Scaling Law 在 NLP 取得巨大成功，但 CTR 预估有其特殊性：
- CTR 数据是稀疏 ID 特征，不是连续文本
- 过度扩大模型规模在 CTR 可能产生负收益（过拟合）
- 工业界对延迟有严格约束，不能无限扩容

**核心问题**：CTR 模型的最优规模如何确定？存在类似 Chinchilla 定律的 CTR Scaling Law 吗？

$$\mathcal{L}(N, D) = A \cdot N^{-\alpha} + B \cdot D^{-\beta} + C$$

## 核心方法与创新点

**EST（Efficient Scaling for CTR）** 的三大发现与方法：

1. **CTR Scaling Law 验证**：
   - 实验验证：CTR 模型也遵循 Power-Law Scaling
   - 最优计算配比（如 Chinchilla）：$N_{\text{opt}} \propto C^{0.55}$（模型参数），$D_{\text{opt}} \propto C^{0.45}$（训练数据量）
   - 结论：CTR 模型普遍欠训练（数据不足），而非欠参数

2. **Embedding 参数 vs Dense 参数的 Scaling 差异**：
   - Embedding（稀疏）参数缩放收益 > Dense（MLP）参数
   - 建议：优先扩大 Embedding 维度，而非加深 MLP
   
$$\Delta \text{AUC} \propto d_{\text{emb}}^{0.3} \quad \text{vs} \quad \Delta \text{AUC} \propto L_{\text{MLP}}^{0.1}$$

3. **数据质量 > 数据量**：
   - 去噪数据（去除随机点击、机器人流量）的 Scaling 效率高 **3x**
   - 高质量 D 的等效价值 ≈ 3x 同量低质量数据

4. **高效 Scaling 策略**：
   - 先 Embedding 扩维 → 然后增加训练数据 → 最后加深 MLP
   - 避免均匀 Scaling（浪费计算）

## 实验结论

- 按 EST 策略 vs 均匀 Scaling：相同计算预算下 AUC 高 **+1.2‰**
- 验证数据集：百亿规模工业广告数据
- Scaling 效率：数据驱动 > 模型驱动（对 CTR 成立）

## 工程落地要点

- Embedding 维度扩大：从 64→128，注意 Serving 内存（亿级 Item → 内存翻倍）
- 数据清洗优先：随机点击过滤、曝光时长过滤（<0.5s 点击视为误点）
- 按 CTR Scaling Law 预测最优训练数据量，避免过早停止训练

## 面试考点

1. **Q**: CTR 模型存在 Scaling Law 吗？与 LLM 的异同？  
   **A**: 存在，都遵循 Power Law。区别：CTR 以 Embedding 为主（稀疏参数），LLM 以 Dense 参数为主；CTR 数据量比参数量更重要（与 LLM Chinchilla 比例不同）。

2. **Q**: Embedding 参数和 Dense 参数扩展的效果为何不同？  
   **A**: Embedding 扩维直接提升特征表达能力（每个稀疏特征的表示更丰富），收益直接；Dense 层加深面临梯度消失和过拟合，收益递减更快。

3. **Q**: 数据质量如何影响 CTR 模型效果？  
   **A**: 噪声标签（误点）引入训练信号噪声，降低 Scaling 效率。高质量数据的信息密度更高，等效于 3x 数据量。
