# CTR/CVR 预估与校准全景

> 创建：2026-04-13 | 领域：广告系统 | 类型：综合分析
> 来源文件：
> - `ads/02_rank/synthesis/CTR预估模型工业级实践进展.md`
> - `ads/02_rank/synthesis/广告CTR_CVR预估与校准.md`
> - `ads/synthesis/20260407_CTR_scaling_advances_synthesis.md`
> - `ads/synthesis/工业广告系统CTR与效果广告前沿技术.md`
> - `ads/02_rank/synthesis/ESMM系列CVR估计演进_从整体空间到因果推断.md`

---

## 一、特征交叉方法演进

```
FM (2010) → NFM → DeepFM → DCN V1 → xDeepFM
  → DCN V2 (矩阵交叉) → DHEN (层次Ensemble) → Wukong (深层+并行训练)
  → MaskNet (实例级乘法门控) → SENet → AutoInt (Transformer)
```

### 1.1 核心公式对比

**DeepFM**：

$$
\hat{y} = \sigma\left(w_0 + \sum_i w_i x_i + \underbrace{\sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j}_{\text{FM 二阶交叉}} + \underbrace{\text{DNN}(\mathbf{x})}_{\text{高阶交叉}}\right)
$$

**DCN V2**（矩阵交叉，替代 V1 的标量权重）：

$$
x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l, \quad W_l \in \mathbb{R}^{d \times d}
$$

Low-rank 近似 $W_l = UV^T$（$U,V \in \mathbb{R}^{d \times r}$），参数从 $d^2$ 降至 $2dr$。

**MaskNet**（实例级乘法门控）：

$$
\text{MaskBlock}(X) = \text{FFN}(\text{LayerNorm}(\text{MLP}_{\text{mask}}(X) \odot X))
$$

**DHEN**（异构集成）：

$$
h^{(l+1)} = \text{Aggregate}(\{Cross(h^{(l)}), Bilinear(h^{(l)}), Attn(h^{(l)}), MLP(h^{(l)})\})
$$

### 1.2 加性 vs 乘性交互

| 类型 | 代表方法 | 本质 |
|------|---------|------|
| 加性 | MLP/DNN | $o = \sigma(\sum w_i h_i)$，无法高效表达乘性关系 |
| 乘性 | FM/MaskNet/DCN | $\langle v_i, v_j \rangle x_i x_j$，直接建模特征交叉 |

两者可叠加：DCN V2 做特征交叉，MaskNet 做实例级调制。

### 1.3 动态特征交叉（DLF）

$$
\hat{y} = \sum_{k=1}^{K} w_k(x) \cdot f_k(x), \quad w_k \text{ 由门控网络预测}
$$

动态感知不同样本所需的特征交叉阶数，多样性约束防止阶数崩塌：

$$
\mathcal{L}_{\text{div}} = \sum_k w_k \log w_k \quad \text{(熵最大化)}
$$

---

## 二、CTR Scaling Laws

### 2.1 EST 核心发现

CTR 模型也遵循 Power-Law Scaling：

$$
\mathcal{L}(N, D) = A \cdot N^{-\alpha} + B \cdot D^{-\beta} + C
$$

最优扩展配比（类 Chinchilla）：

$$
N_{\text{opt}} \propto C^{0.55}, \quad D_{\text{opt}} \propto C^{0.45}
$$

**关键洞察**：
- Embedding 参数收益 > Dense 参数：$\Delta\text{AUC} \propto d_{\text{emb}}^{0.3}$ vs $\Delta\text{AUC} \propto L_{\text{MLP}}^{0.1}$
- 数据质量 3x 高于数量
- 大多数 CTR 模型是"数据欠拟合"而非"模型过小"

### 2.2 推荐 Scaling Law（综合）

$$
L(N, D) \approx \frac{A}{N^{0.07}} \cdot \frac{B}{D^{0.04}}
$$

与 LLM 对比：

| 维度 | LLM（Chinchilla） | 推荐（工业综合） |
|------|------------------|----------------|
| 主要参数 | Attention+FFN（稠密） | Embedding（稀疏） |
| Scaling 瓶颈 | 计算量（FLOPs） | 内存带宽（Embedding 查找） |
| 最优路径 | Dense Transformer | MoE 稀疏激活 |

### 2.3 RankMixer: GPU-Native 排序

传统精排 MFU 仅 4.5%，RankMixer 用 Token Mixing + Per-token FFN + Sparse-MoE 将 MFU 提升至 45%（10x）。工程结论：先优化 MFU，再谈 Scaling。

---

## 三、统一 Transformer 架构（OneTrans）

用单 Transformer 统一特征交叉与序列建模：

| 组件 | Attention 类型 | 目的 |
|------|--------------|------|
| 行为序列 | Causal Mask | 时序建模 |
| 上下文特征 | Full Attention | 特征交叉 |
| 目标广告 | Cross Attention | 意图对齐 |

$$
\mathcal{L} = \mathcal{L}_{\text{CTR}} + \lambda \cdot \mathcal{L}_{\text{seq\_pred}}
$$

参数量 -35%，训练速度 +1.4x，AUC +0.9‰。

---

## 四、ESMM 系列 CVR 估计演进

### 4.1 ESMM（KDD 2018）

**核心问题**：CVR 只在点击样本训练（SSB），但推理在全曝光空间。

$$
P(\text{CTCVR}) = P(\text{CTR}) \times P(\text{CVR})
$$

$$
\mathcal{L}_{\text{ESMM}} = \mathcal{L}_{\text{CTR}}(\hat{p}_{\text{CTR}}, y_{\text{click}}) + \mathcal{L}_{\text{CTCVR}}(\hat{p}_{\text{CTR}} \cdot \hat{p}_{\text{CVR}}, y_{\text{click}} \cdot y_{\text{conv}})
$$

CVR 塔通过 CTCVR loss 间接获得全空间训练信号。共享 Embedding 将 CTR 丰富样本迁移给 CVR。

### 4.2 ESMM2 / 多阶段扩展

引入中间行为（加购/收藏）：

$$
P(\text{购买}) = P(\text{点击}) \times P(\text{加购}|\text{点击}) \times P(\text{购买}|\text{加购})
$$

多阶段链路缓解数据稀疏，但因果问题仍未解决。

### 4.3 ESCM²（SIGIR 2022）

**IPW 去偏**：

$$
\hat{\mathcal{L}}_{\text{IPW}} = \sum_{i:O_i=1} \frac{1}{p_i^{\text{CTR}}} \ell(y_i^{\text{CVR}}, f_{\text{CVR}}(x_i))
$$

**DR 双重鲁棒估计器**：

$$
\hat{\mathcal{L}}_{\text{DR}} = \sum_{i=1}^{N} \ell(y_i^{\text{imputed}}, f) + \sum_{i:O_i=1} \frac{1}{p_i^{\text{CTR}}}(\ell(y_i^{\text{actual}}, f) - \ell(y_i^{\text{imputed}}, f))
$$

只要 outcome 模型或 propensity 模型任一正确，估计就无偏。

### 4.4 ECAD：跨域归因去偏

多域广告系统中同一商品在搜索/Feed/首页多场景曝光：

$$
p_{\text{cross}}^{\text{CTR}}(x) = \sum_{d \in \mathcal{D}} w_d \cdot p_d^{\text{CTR}}(x)
$$

### 4.5 四代模型对比

| 代际 | 方法 | 偏差处理 | 方差 | 落地难度 |
|------|------|---------|------|---------|
| ESMM | 乘法分解 | 有偏（假设过强） | 低 | 低（工业首选） |
| ESMM2 | 多阶段链路 | 有偏 | 低 | 低 |
| ESCM²-IPW | 逆倾向加权 | 无偏 | 高（极小倾向分时爆炸） | 中 |
| ESCM²-DR | 双重鲁棒 | 无偏 | 低 | 高 |
| ECAD | 跨域因果 | 无偏 | 中 | 高 |

---

## 五、校准（Calibration）

### 5.1 为什么需要校准

广告出价 $\text{bid} = \text{CPA}_{target} \times pCVR$，pCVR 不准直接导致出价偏差。

典型问题：模型输出 pCTR=0.05 但实际 CTR=0.02（系统性高估 2.5x），原因通常是训练集正负比例与线上不一致。

### 5.2 Platt Scaling

$$
p_{\text{calib}}(x) = \sigma(a \cdot f(x) + b)
$$

- $a < 1$：模型过度自信，需压缩
- $b \neq 0$：系统性偏差
- 等价于在 logit 上训练 1D Logistic Regression

### 5.3 Isotonic Regression

非参数保序回归：将预测值分桶，每桶内用实际正样本比例作为校准值。更灵活但需更多数据。

### 5.4 ECE（Expected Calibration Error）

$$
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} |\bar{y}_m - \bar{p}_m|
$$

理想情况 ECE=0：预测概率 = 实际发生率。

### 5.5 校准实战效果

| 分桶 | 校准前实际CTR | 校准后实际CTR | 误差改善 |
|------|------------|------------|---------|
| 0-0.01 | 0.004 | 0.004 | - |
| 0.01-0.05 | 0.015 | 0.016 | -6% |
| 0.05-0.15 | 0.03 | 0.05 | +40% |
| 0.15+ | 0.06 | 0.14 | +57% |

高分段误差最大，校准后 ECE 从 0.08 降至 0.02。

---

## 六、在线学习（FTRL）

$$
w_{t+1,i} = \begin{cases} 0 & |z_{t,i}| \leq \lambda_1 \\ -\frac{z_{t,i} - \text{sgn}(z_{t,i})\lambda_1}{\frac{1}{\alpha}\sum_{\tau=1}^t \sigma_{\tau,i} + \lambda_2} & \text{otherwise} \end{cases}
$$

- L1 产生稀疏性（95%+ 特征维度为零），自动特征选择
- 自适应学习率（类 AdaGrad）：高频特征步长小，低频特征步长大
- 支持亿级特征维度的实时更新

---

## 七、前沿进展（2025-2026）

| 方法 | 核心创新 | 效果 |
|------|---------|------|
| GRAB | CamA + 自回归序列范式 | Revenue +3.05%, CTR +3.49% |
| EST | LCA + CSA + 统一序列 | Power-law scaling 验证 |
| DAES | 水库采样 + 分布调制 | 数值特征 SOTA |
| RQ-GMM | GMM 软赋值量化 | Adv Value +1.502% vs RQ-VAE |
| HeMix | 动态+固定查询 + HeteroMixer | 数亿用户部署 |
| RankMixer | Token Mixing + Sparse-MoE | MFU 4.5%→45%, 参数扩展 100x |

---

## 八、面试高频 Q&A（20 题）

**Q1: 广告 CTR 为什么需要校准？**
出价 bid = CPA × pCVR，pCVR 偏高→出价过高→广告主亏损；偏低→竞价失败。广告对概率绝对值的准确性要求远高于推荐。

**Q2: Platt Scaling vs Isotonic Regression？**
Platt 用 sigmoid 全局线性校准（参数少），Isotonic 用分段常数非参数校准（更灵活需更多数据）。

**Q3: ESMM 如何解决 CVR 的 SSB？**
通过 pCTCVR = pCTR × pCVR 在全曝光空间建模，CVR 塔通过 CTCVR loss 间接获得全空间训练信号。共享 Embedding 将 CTR 丰富样本迁移给 CVR。

**Q4: ESCM² 比 ESMM 好在哪？**
ESMM 乘法分解假设过强（隐含独立性）。ESCM² 用 IPW 去偏 + DR 双重鲁棒估计器，理论上无偏。DR 在倾向分极小时比纯 IPW 更稳定。

**Q5: DCN V2 的 Low-rank 为什么有效？**
特征交叉矩阵的有效秩远低于全秩，低秩近似捕捉主要变化方向、丢弃噪声维度，参数从 $d^2$ 降至 $2dr$ 但效果几乎无损。

**Q6: MaskNet 和 DCN V2 解决什么不同问题？**
DCN V2 解决特征阶次问题（高阶显式交叉），MaskNet 解决实例差异问题（同一特征对不同用户重要性不同）。可叠加使用。

**Q7: CTR 模型该如何扩展？**
先扩 Embedding 维度（收益最高）→ 加数据（清洗质量优先）→ 最后加深 MLP。按 EST Scaling Law 配比计算最优分配。

**Q8: NE 和 AUC 的区别？**
AUC 衡量排序能力，NE 衡量概率校准精度。广告出价依赖精确概率值，NE 更能反映工业真实价值。

**Q9: 延迟转化怎么处理？**
等待窗口法（简单但浪费数据）、Elapsed-Time Model（时间作为特征）、Fake Negative Calibration（事后回补）、多窗口建模（1d/7d/30d CVR）。

**Q10: FTRL 在线学习的核心优势？**
L1 正则产生稀疏解（亿级特征中 95%+ 为零），自适应学习率处理高维稀疏特征，支持流式更新追踪分布漂移。

**Q11: 位置偏差怎么处理？**
IPS（逆倾向加权）、PAL（分解 P(click) = P(examine|pos) × P(click|examine,ad)）、Shallow Tower（浅层塔单独建模位置偏差，推理时丢弃）。

**Q12: 工业级 CTR 系统的四级漏斗？**
召回（千万→万）→ 粗排（万→千）→ 精排（千→百，CTR 模型核心战场）→ 重排（百→十，多样性+频控）。

**Q13: Embedding 维度如何选择？**
经验公式 $d = 6 \times (\text{category\_size})^{1/4}$（Google）。高频特征 16-64 维，低频 4-8 维。也可用 NAS 搜索。

**Q14: 特征哈希的原理和风险？**
将超大 ID 空间（10^9）哈希映射到较小空间（10^7），接受少量碰撞。缓解：多哈希函数、高频 ID 保留独立 embedding、signed hash。

**Q15: 增量训练 vs 全量训练？**
增量追踪时效性但可能遗忘。全量全局最优但成本高。工业实践：增量为主（追踪漂移），定期全量重训（防累积漂移）。

**Q16: 大规模 CTR 模型为什么用 Pre-LN？**
Post-LN 深层梯度需穿过多个 LayerNorm Jacobian（梯度消失），Pre-LN 残差分支干净，梯度直接流过，支持 10+ 层稳定训练。

**Q17: DLF 动态阶数的 tradeoff？**
适应不同样本复杂度，门控网络轻量（+3% 参数），AUC +0.6‰。适合特征交叉模式多变的大规模系统。

**Q18: OneTrans 统一架构的优势？**
用单 Transformer 统一特征交叉（Full Attention）和序列建模（Causal Mask），参数 -35%、速度 +1.4x、AUC +0.9‰。

**Q19: 多模态 Embedding 如何与 CTR 对齐？**
离散化（SID）是主要方案。RQ-VAE → RQ-GMM（GMM 软赋值量化），关键是 codebook 利用率和语义判别性。

**Q20: 工业落地选 ESMM 还是 ESCM²？**
ESMM 简单可控，是大多数系统首选。ESCM² 需精确倾向分估计（倾向分不准反而更差）。通常先上 ESMM，验证后增量实验 ESCM²-DR。

---

## 参考文献

1. ESMM (KDD 2018), ESCM² (SIGIR 2022)
2. DCN V2 (2020), MaskNet (2021), DHEN (2022), Wukong (2023)
3. EST: Efficient Scaling Laws in CTR (2025)
4. DLF: Dynamic Low-Order-Aware Fusion (2025)
5. OneTrans: Unified Feature Interaction (2025)
6. RankMixer: GPU-Native Ranking (2025)
7. GRAB, DAES, RQ-GMM, HeMix (2025)

---

## 相关概念

- [[concepts/embedding_everywhere|Embedding 技术全景]]
- [[concepts/attention_in_recsys|Attention 在搜广推中的演进]]
- [[concepts/sequence_modeling_evolution|序列建模演进]]
- [[concepts/multi_objective_optimization|多目标优化]]
- [[synthesis/ads/02_广告排序系统演进|广告排序系统演进]]
- [[synthesis/ads/06_冷启动与偏差治理|冷启动与偏差治理]]
