# A Comprehensive Survey on Advertising Click-Through Rate Prediction Algorithm

> 来源：综合调研 | 年份：2024 | 领域：ads/02_rank（CTR预估综述）

## 问题定义

点击率（CTR）预测是计算广告系统的核心任务：预测用户在看到某广告时点击的概率。

**业务重要性**：
- eCPM = bid × pCTR（CPC出价）或 bid × pCTR × pCVR（CPA出价），CTR 预估精度直接影响广告收入
- 用户体验：低相关性广告浪费用户注意力，精准 CTR 预估 = 更好的广告相关性
- 广告主 ROI：CTR 预估偏差导致出价不合理，广告主流失

**技术挑战**：
- 高维稀疏特征（用户ID、广告ID、上下文，特征空间 10⁸-10¹⁰）
- 低延迟要求（<10ms serving，QPS 数十万）
- 数据分布漂移（用户兴趣随时间变化）
- 正负样本极度不均衡（CTR 通常 1-5%）

## 模型结构图：CTR 模型演进全景

```
时间线：2010 ─────────────────────────────────────── 2024

手工特征时代:
  LR (2010) ──→ GBDT+LR (2014) ──→ FM (2010, 工业化2014)
                                        ↓
浅层交叉时代:                          FFM (2016)
  FM ──→ DeepFM (2017) ──→ DCN (2017) ──→ DCNv2 (2020)
         Wide&Deep (2016)    xDeepFM (2018)  AutoInt (2019)
                                              FiBiNET (2019)
                                        ↓
序列建模时代:
  DIN (2018) ──→ DIEN (2019) ──→ BST (2019) ──→ SIM (2020)
                                                 HSTU (2024)
                                        ↓
多任务时代:
  ESMM (2018) ──→ MMoE (2018) ──→ PLE (2020) ──→ AITM (2021)
                                        ↓
集成/高级时代:
  MaskNet (2021) ──→ DHEN (2022) ──→ HLLM (2024)
```

## 核心方法与完整公式

### 公式1：LR 基线

$$
P(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}
$$

**解释：** 线性模型，特征交叉完全依赖手工构造。

### 公式2：FM 二阶交叉

$$
\hat{y}_{FM} = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j
$$

**解释：** 自动学习二阶特征交叉，参数量 $O(nk)$，是 CTR 模型的里程碑。

### 公式3：DCN Cross Network

$$
x_{l+1} = x_0 \cdot x_l^T \cdot w_l + b_l + x_l
$$

**解释：**
- $x_0$：原始输入
- $x_l$：第 $l$ 层输出
- 每层显式增加一阶交叉，$L$ 层 = $L+1$ 阶交叉
- DCNv2 改进：$w_l$ 从向量升级为矩阵，表达能力更强

### 公式4：DIN Attention

$$
\text{Attention}(e_i, e_a) = \sigma(W[e_i; e_a; e_i - e_a; e_i \odot e_a])
$$

**解释：** 目标广告驱动的注意力机制，动态激活用户兴趣。

### 公式5：多任务 eCPM 排序

$$
\text{eCPM} = \text{bid} \times \hat{p}_{CTR} \times \hat{p}_{CVR} \times f(\text{quality}_{	ext{score}})
$$

**解释：** 综合广告出价、点击率、转化率和质量分进行广告排序。

## 与基线方法对比

### 特征交叉方法对比

| 方法 | 交叉方式 | 交叉阶数 | 额外参数 | 核心创新 |
|------|---------|---------|---------|---------|
| FM | 内积 | 2阶 | $O(nk)$ | 自动二阶交叉 |
| DeepFM | FM+DNN | 2阶+高阶 | FM+DNN | 共享Embedding |
| DCN | Cross Network | 显式L+1阶 | $O(nd)$ | bit-wise交叉 |
| DCNv2 | 矩阵Cross | 显式高阶 | $O(d^2)$ | 更强表达力 |
| AutoInt | Multi-head Attn | 自适应 | $O(d^2)$ | Attention交叉 |
| xDeepFM | CIN | 显式高阶 | $O(mHd)$ | vector-wise交叉 |

### 序列建模方法对比

| 方法 | 序列建模 | 序列长度 | 创新点 |
|------|---------|---------|-------|
| DIN | Target Attention | ~50 | 局部激活 |
| DIEN | GRU + Attention | ~50 | 兴趣演化 |
| BST | Transformer | ~50 | Self-Attention |
| SIM | 检索 + Attention | 10000+ | 超长序列 |
| HSTU | 生成式Transformer | 10000+ | 统一序列特征 |

## 实验结论

- DeepFM 在 Criteo 数据集上比 LR 提升约 1-2% AUC
- DIN 对有历史行为的用户 AUC 提升 0.5-1.5%
- 多任务学习在 CVR 任务上提升 2-5% AUC（解决样本选择偏差）
- 工业界共识：0.1% AUC 提升 ≈ 1% RPM 提升 ≈ 数百万美元年收入增长

## 工程落地要点

1. **特征工程**：用户历史序列截断长度 50-200，ID 类特征 embedding 维度 8-64
2. **分布式训练**：PS（Parameter Server）架构处理万亿级稀疏参数，Dense 部分用数据并行
3. **特征哈希**：超大 ID 空间（10⁹）用 Hash Trick 降维到 10⁷，接受少量碰撞
4. **实时特征**：近实时更新用户行为特征（<1分钟延迟），使用 Flink/Kafka 流处理
5. **模型蒸馏**：大模型 Teacher → 小模型 Student，在线推理用 Student 满足延迟要求
6. **样本构建**：展示未点击为负样本，注意负采样比例和校准（calibration）

## 面试考点

**Q1：FM 和 DNN 结合的原理？（以 DeepFM 为例）**
> FM 层学习精确的二阶交叉（$\langle v_i, v_j \rangle$），DNN 层学习隐式高阶非线性交叉。两者共享 Embedding，最终输出相加。优势：无需手工特征工程，同时捕获低阶精确交叉和高阶复杂交叉。

**Q2：DIN 的核心思想？**
> 用目标广告作为 query，对历史行为（keys）做注意力加权，相关行为权重更高。同一用户面对不同广告激活不同历史行为，实现"局部兴趣激活"。用 sigmoid（非 softmax）允许多兴趣共存。

**Q3：CTR 中如何处理样本不均衡（CTR 通常 1-5%）？**
> ① 负采样（1:n），但需校准：$CTR_{real} = CTR_{sampled} \times \frac{n_{neg}}{n_{neg} + n_{pos}} \times \frac{1}{sampling\_rate}$ ② Focal Loss：对难分样本加权 ③ 正样本加权：$w_{pos} = n_{neg}/n_{pos}$。

**Q4：DCN 和 DeepFM 的区别？**
> DeepFM 用 FM 做显式二阶交叉 + DNN 做隐式高阶。DCN 用 Cross Network 做显式多阶交叉（每层增加一阶）+ DNN。DCN 的显式交叉阶数可控（L层=L+1阶），但 Cross 向量的表达能力有限（DCNv2 用矩阵改进）。

**Q5：工业级 CTR 系统的典型架构？**
> 四级漏斗：① 召回（多路并行，千万→万）② 粗排（轻量模型，万→千）③ 精排（复杂模型如 DeepFM+DIN，千→百）④ 重排（多样性、频控，百→十）。精排是 CTR 模型核心战场。

**Q6：如何处理特征的分布漂移？**
> ① 增量训练（每小时/每天用新数据更新模型）② 在线学习（实时更新部分参数）③ 特征时效性控制（用最近 N 天数据）④ Batch Normalization 缓解分布偏移 ⑤ 定期全量重训。

**Q7：Embedding 维度如何选择？**
> 经验公式：$d = 6 \times (\text{category}_{	ext{size}})^{1/4}$（Google 经验）。高频特征可用较高维度（16-64），低频特征用较低维度（4-8）。也可以用 NAS 自动搜索每个 field 的最优维度。

**Q8：CTR 模型的校准（Calibration）为什么重要？**
> 广告出价依赖精确的 CTR 值（不仅是排序），pCTR=0.1 意味着广告系统认为每 10 次展示有 1 次点击。校准不准 → 出价偏差 → 广告主 ROI 异常。常用 Platt Scaling 或 Isotonic Regression 做后校准。
