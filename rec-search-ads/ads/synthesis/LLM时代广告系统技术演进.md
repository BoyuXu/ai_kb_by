# LLM时代广告系统技术演进（持续更新）

> 领域：ads | 类型：综合综述 | 覆盖论文：26篇 | 最近更新：2026-04-02

## 一、技术演进脉络

```
传统广告 (2018-2022)                 生成式广告 1.0 (2023-2024)          生成式广告 2.0 (2025-2026)
──────────────────────────────       ──────────────────────────────       ──────────────────────────────
DIN/DIEN 序列建模 CTR               CADET Decoder-Only CTR               DGenCTR 离散扩散 CTR
FM/DeepFM 特征交叉                   DCN-V2 矩阵 Cross 特征交叉           统一 token 化生成式预测（MTGR多任务）
GSP 拍卖 + 规则出价                  Lagrangian 对偶自动出价              FPA Bid Shading + 竞价动态（Bid2X）
人工设计广告素材                     MLLM 驱动创意生成                    多场景统一生成式排序（MTGR美团）
独立 CTR/CVR/LTV 模型               UniROM 统一排序模型                  ELEC 离线 LLM 特征工厂（<2ms延迟）
纯 ID 特征冷启动困难                 Multimodal Proxy 解决冷启动          广告基础模型预训练 + Adapter 微调
静态 ID Embedding                    隐式→显式反馈联合训练               GRAD 生成式预训练通用出价模型
单广告主 RL 出价                     Offline RL 出价（BCQ/CQL）          短视频多模态个性化（NextAds）
                                                                         毫秒级生成式广告推荐（GR4AD）
```

## 二、核心技术维度

### 2.1 CTR 预估：从判别式到生成式

**技术演进路径**：
```
FM (2010) → DeepFM (2017) → DCN (2017) → DCN-V2 (2021)
     ↓ 序列建模维度
DIN (2018) → DIEN (2019) → SIM (2020)
     ↓ LLM 融合维度（2023-2026）
ELEC (离线特征) → CADET (Decoder-Only) → DGenCTR (离散扩散)
```

**代表工作**：CADET、DGenCTR、DCN-V2、Lightweight-LLM、MTGR

#### DCN-V2：特征交叉的里程碑

通过矩阵式 Cross 层突破 DCN-V1 的表达力瓶颈：

$$
x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l
$$

低秩分解 $W = UV^T$（$U,V \in \mathbb{R}^{d \times r}, r \ll d$）：
- 参数量从 $O(d^2)$ 降至 $O(d \times r)$
- 保留了显式高阶特征交叉能力
- Parallel 和 Stacked 两种结构适配不同业务

#### DGenCTR：离散扩散 CTR（2025年最前沿）

CTR 预测从判别式 $P(y|x)$ 演进到生成式 $P(x,y)$：

**离散扩散前向过程**（从干净特征到噪声）：

$$
q(x_t | x_{t-1}) = \text{Cat}(x_t; \alpha_t x_{t-1} + (1-\alpha_t)/V)
$$

**反向去噪预测**：

$$
p_\theta(x_0 | x_t, c) = \prod_{i} p_\theta(x_0^{(i)} | x_t, c)
$$

关键创新：
1. 将用户特征、商品特征统一 token 化（每个离散值为一个 token）
2. 离散扩散在 token 空间而非连续空间操作，更适合 ID/类别特征
3. 自适应去噪步数（简单样本 1-2 步，复杂样本 4-5 步）
4. 联合建模特征间依赖（传统 CTR 视特征独立），获得更好泛化

**与 CADET 对比**：
| 维度 | CADET (Decoder-Only) | DGenCTR (离散扩散) |
|------|---------------------|-------------------|
| 建模方式 | 自回归序列预测 | 联合特征分布建模 |
| 对序列的建模 | 显式时序因果 | 隐式特征协同 |
| 特征类型适配 | 文本/ID 均可 | 离散 ID 特征更优 |
| 推理延迟 | 自回归慢 | 少步去噪快 |
| 泛化能力 | 预训练迁移强 | 特征补全能力强 |

#### 美团 MTGR：工业多场景统一

decoder-only 架构统一 12+ 业务场景：

$$
P(i_1,...,i_K | s, \text{Prompt_{s) = }\prod_{k=1}}^K P(i_k | i_{<k}, s, \text{Prompt}_{\text{s)
$$

- 场景 Prompt 区分不同业务语境
- 服务美食/酒旅/闪购等异质场景
- 在线延迟通过 Speculative Decoding 控制在 50ms 内

#### LLM 轻量化 CTR

直接 LLM 在线推理延迟 >200ms 不可接受。工业解法：

```
离线                      在线
LLM → Embedding 生成  →  Feature Store → 查表(1ms) → 融合 MLP(1ms)
```

**对齐损失**（将 LLM Embedding 对齐到 ID Embedding 空间）：

$$
L}_{{\text{align}}} = -\log \frac{\exp(\text{sim}(e_{LLM}, e_{ID}) / \tau)}{\sum_j \exp(\text{sim}(e_{LLM}, e_{ID_j}) / \tau)}
$$

#### ELEC：LLM 离线特征工厂（2025 工业最实用方案）

**三层特征增强架构**：

| 粒度 | LLM 输出 | 在线使用 |
|------|----------|---------|
| Item-level | 品类语义标签 + 跨模态描述 | 物品内容 Embedding |
| User-level | 用户画像 summary | 用户兴趣 Embedding |
| User×Item | 交互强度评分 | 个性化交叉特征 |

**关键工程**：4096 → 64 维压缩（Knowledge Distillation），在线零 LLM 开销，延迟 +0.3ms。

**ELEC vs 在线 LLM 推理对比**：

| 维度 | ELEC（离线） | 在线 LLM |
|------|------------|---------|
| 延迟增量 | +0.3ms | +100-200ms |
| 特征时效 | 小时级 | 实时 |
| 冷启动效果 | AUC +2.8% | AUC +3.5% |
| 工程复杂度 | 中（Feature Store） | 高（在线 LLM 集群） |

**结论**：ELEC 是 LLM × CTR 的工业最优解；在线 LLM 仅在超低流量/高价值广告场景可考虑。

#### 广告基础模型（Foundation Model for Ads）

借鉴 NLP Scaling Law，广告系统也存在**数据规模 → 模型性能**的幂律关系：

$$
\text{AUC} \propto D^{\alpha}, \quad \alpha \approx 0.07 \text{（广告域经验值）}
$$

**场景自适应设计**（Adapter 架构）：
```
基础模型（跨平台预训练）
    ↓
Adapter Layer（场景专有，参数量仅 2-5%）
    ↓
搜索广告 / 信息流广告 / 开屏广告
```

**价值**：新场景冷启动减少 70% 标注数据，是 LLM fine-tuning 范式向广告系统的迁移。

#### 美团 MTGR 多任务生成式排序

在生成式架构基础上引入多任务：

$$
\mathcal{L}_{MTGR} = \sum_{k} w_k \mathcal{L}_k + \underbrace{\mathcal{L}_{cross-task}_{任务间注意力}}
$$

**关键发现**：生成式架构 ≠ 放弃 DLRM 特征交叉，两者互补：
- 生成式 HSTU 负责序列建模（时序依赖）
- DLRM Cross 层负责特征交叉（协同关系）
- Pareto 搜索替代人工调权，效率 ×10

**GMV +2.1%，CTR +0.8%，CVR +1.5%（美团广告在线 A/B）**

### 2.2 自动出价：从规则到生成式分布

**技术演进路径**：
```
手动出价 → 规则出价（ROI = 目标/成本）→ RL 出价（策略梯度）
    ↓ FPA 时代（2020+）
Bid Shading（避免过度支付）→ 生成式出价分布（GRAD/GBS）→ 竞价动态预测（Bid2X）
```

**统一数学框架**（约束优化）：

$$
\max_\pi \mathbb{E}_\pi\left[\sum_t v_t\right] \quad \text{s.t.} \quad \mathbb{E}\left[\sum_t b_t\right] \leq B
$$

**Lagrangian Relaxation**（所有生成式出价的共同底层）：

$$
\mathcal{L}(b, \lambda) = \mathbb{E}\left[\sum_t(v_t - \lambda b_t)\right] + \lambda B
$$

对偶变量 $\lambda$ = 预算的影子价格，自适应调整预算消耗速率。

#### GBS：生成式 Bid Shading（RTB 专用）

FPA 拍卖中，广告主出多少付多少，因此需要 Bid Shading 降低出价：

$$
b^* = \arg\max_b \int_0^b (v - w) dF(w|x)
$$

GBS 用**混合高斯分布**建模 winning price：

$$
p(w | x) = \sum_{k=1}^{K} \pi_k(x) \cdot \mathcal{N}(w | \mu_k(x), \sigma_k^2(x))
$$

优势：分布建模而非点估计，能量化竞价不确定性。

#### Bid2X：竞价动态预测

不建模分布，而是预测**未来竞价环境动态**：
- 竞争对手数量变化
- 平均出价水平趋势
- 时序模型（LSTM/Transformer）预测市场信号

**与 GBS 对比**：
| 维度 | GBS | Bid2X |
|------|-----|-------|
| 建模对象 | 当前 winning price 分布 | 未来市场动态趋势 |
| 时序性 | 弱（单次竞价） | 强（前瞻性调整） |
| 适用拍卖 | FPA RTB | FPA + 程序化广告 |
| 核心优化目标 | 避免过度支付 | 预判市场趁机出价 |

#### GRAD：生成式预训练通用出价模型

**核心范式**：跨广告主大规模预训练，单一模型服务所有广告主：

$$
P(\text{bid_{t | }\text{bid}_{t-k:t-1}, \text{budget}, \text{KPI}) = p_\theta(\cdot)
$$

Diffusion 在连续出价空间建模（多峰分布）：

$$
b_0 \sim p_\theta(b_0 | b_T, c) = \text{DDIM}(b_T, c, T_{\text{steps}})
$$

其中 $c$ = 广告主 prompt（KPI + 历史表现），$T_{\text{steps}}$ = 10-20（DDIM 加速）。

**工业价值**：1000 个广告主 → 1 个 GRAD 模型，维护成本 -99%；新广告主（<7天历史）ROI +9.8%（预训练市场知识迁移）。

**四种出价方案对比**：
| 方法 | 核心思想 | 稳定性 | 最优性 | 工业可行性 |
|------|----------|--------|--------|-----------|
| 规则出价 | 固定系数 × ROI | ★★★ | ★ | ★★★★ |
| RL 出价 | 策略梯度长期 ROI | ★★ | ★★★ | ★★★ |
| GRAD/GBS 生成式 | Diffusion 出价分布 | ★★★ | ★★★ | ★★★ |
| KBD 双过程 | 知识 + RL 自适应 | ★★★ | ★★★ | ★★★★ |
| Bid2X 动态 | 预测市场趋势 | ★★★ | ★★★★ | ★★★ |

### 2.3 显式与隐式反馈联合训练

**核心问题**：显式反馈（评分/收藏）高质量低数量；隐式反馈（点击/浏览）高数量高噪声。

**联合训练策略**：

$$
L = \alpha L_{explicit} + (1-\alpha) L_{implicit} + \beta L_{denoising}
$$

- 显式反馈作为"锚点"校准隐式反馈中的噪声
- 去噪损失 $L_{denoising}$ 识别并降低噪声隐式样本的权重
- $\alpha > 0.5$：显式反馈给更高权重

**实践启示**：收藏/加购等强信号比点击有更高的 CVR 相关性，应差异化处理而非混同。

### 2.4 Etsy 统一 Embedding：个性化检索的工业实践

**核心创新**：搜索检索和推荐用统一 Embedding 空间，一个模型解决两类需求：

$$
e_{query} = f_q(q, u), \quad e_{item} = f_i(i)
$$

$$
\text{Score}(q, u, i) = \text{sim}(e_{query}, e_{item})
$$

个性化通过 $f_q$ 融合用户历史行为，而 $f_i$ 与查询无关（可离线预计算）。

### 2.5 广告创意的 AI 生成

**CTR-as-Reward 训练循环**：
```
广告需求 → MLLM prompt 生成 → Diffusion 图像生成
                                      ↓
                              CTR Predictor 打分
                                      ↓
                              RL 微调 Diffusion（PPO/DDPO）
```

**防止 Reward Hacking**（Goodhart's Law）：

$$
\mathcal{L} = r_\text{CTR} - \beta \cdot \text{KL}(\pi_\theta || \pi_\text{ref}) + \gamma \cdot r_\text{quality}
$$

### 2.6 短视频广告多模态个性化（NextAds）

**挑战**：短视频广告 = 多模态理解（视频帧 + 音频 + 字幕）× 强用户体验约束（用户容忍度低）。

**技术架构**：
```
视频内容 → ViT + AudioEncoder + ASR/OCR → 融合 Embedding
用户历史视频行为 → 完播率 + 互动率 → 视频消费偏好
                                           ↓
                              广告-内容语义对齐（场景融合）
                                           ↓
                              动态出价（高沉浸溢价 +12% CVR）
```

**核心 insight**："场景融合广告"（广告风格贴近有机内容）VTR 比强制插入高 23%。这是广告系统从"曝光最大化"到"体验-收入双优"的典型范式转变。

**关键指标**：VTR +8.2%，CVR +4.5%，高沉浸场景 CVR +12%，eCPM +6%。

### 2.7 毫秒级生成式广告推荐（GR4AD）

**问题**：生成式推荐的自回归解码天然慢（O(K) 步），在线广告 SLA = 50ms P99。

**解法矩阵**：

| 优化技术 | 延迟收益 | 质量损失 |
|---------|---------|---------|
| 动态束搜索（Beam Width 自适应） | 中 | 低 |
| 预计算 KV Cache 复用 | 高（缓存命中率 >85%）| 极低 |
| 投机解码（Speculative Decoding） | 高 | 低 |
| INT8 量化 | 中 | 低 |

**组合优化结果**：P99 延迟 5ms 内，相比贪心解码 CTR +1.3%。

**工程决策**：GR4AD 的延迟优化路径 = 缓存命中率优先（ROI 最高），其次量化，最后投机解码。

### 2.8 广告拍卖机制创新

LLM 时代对话式广告拍卖：

$$
\text{Score}_{\text{i = b_{i }}\times q}_{i(query), \quad q_{i = }\text{LLM-Relevance}(ad_i, query)
$$

Quality-adjusted 定价（Vickrey 变体）：

$$
p_i = \frac{b_{2\text{nd}} \times q_{2\text{nd}}}{q_i}
$$

## 三、📐 关键数学公式全集

| 公式 | 名称 | 用途 |
|------|------|------|
| $x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l$ | DCN-V2 Cross 层 | 显式特征交叉 |
| $p_\theta(x_0 \| x_t, c) = \prod_i p_\theta(x_0^{(i)} \| x_t, c)$ | DGenCTR 去噪 | 生成式 CTR |
| $p(w \| x) = \sum_k \pi_k \mathcal{N}(w \| \mu_k, \sigma_k^2)$ | GBS 出价分布 | 混合高斯 Bid Shading |
| $\mathcal{L}(b,\lambda) = \mathbb{E}[\sum_t(v_t - \lambda b_t)] + \lambda B$ | Lagrangian 出价 | 预算约束优化 |
| $\text{eCPM} = \text{CTR} \times \text{CVR} \times \text{Bid} \times 1000$ | eCPM | 广告排序核心指标 |
| $L_{\text{align}} = -\log \frac{\exp(\text{sim}(e_{LLM}, e_{ID})/\tau)}{\sum_j \exp(\cdot)}$ | 跨模态对齐 | LLM 特征融入 CTR |
| $\mathcal{L} = r_\text{CTR} - \beta \cdot \text{KL}(\pi_\theta \|\| \pi_\text{ref}) + \gamma r_q$ | 创意生成 | 防 Reward Hacking |
| $\mathcal{L}_{MTGR} = \sum_k w_k \mathcal{L}_k + \mathcal{L}_{cross-task}$ | MTGR 多任务 | 生成式多目标排序 |
| $\text{AUC} \propto D^{0.07}$ | 广告 Scaling Law | 基础模型预训练 |
| $b_0 \sim \text{DDIM}(b_T, c, T_{\text{steps}})$ | GRAD Diffusion 出价 | 跨广告主通用出价 |

## 四、🎯 核心洞察

1. **生成式范式正在统一广告漏斗的每个环节**：CTR（DGenCTR 离散扩散）、出价（GBS/GRAD 分布建模）、素材（Diffusion 创意生成）、拍卖（LLM-AUCTION）——不是局部优化，是整个广告系统的重构。

2. **DGenCTR 的核心 insight：特征间依赖建模**：传统 CTR 隐含假设特征独立（给定 y），DGenCTR 通过联合分布 $P(x,y)$ 建模特征间协变关系，在稀疏场景（冷启动/长尾）泛化能力显著更强。

3. **FPA 时代的出价策略颠覆**：GSP（second-price）时代，出价=真实价值是弱均衡；FPA 时代，出价=预测 winning price 才能最大化 ROI。Bid Shading 从可选项变成必选项，但分布建模（GBS）比规则调整更鲁棒。

4. **Bid2X 的前瞻性 vs GBS 的分布估计**：这是两种互补的出价哲学——GBS 问"现在出多少合适"，Bid2X 问"未来市场会怎么变、提前调整"。工业上两者可叠加：先用 Bid2X 调整基础出价，再用 GBS 做分布级精调。

5. **LLM 进入广告的正确姿势：离线重于在线**：在线 LLM 推理的延迟成本在广告场景（50ms SLA）不可接受，轻量化方案（离线 Embedding → 对齐 → 查表）是工业落地的唯一现实路径，延迟增量 <2ms。

6. **DCN-V2 的工程价值：低秩分解解决了特征维度灾难**：广告系统特征维度动辄 10 万+，全矩阵参数量爆炸；低秩分解让高阶特征交叉在工业可承受的参数量下实现，这是 DCN-V2 成为广告系统基础架构的根本原因。

7. **显隐式反馈统一是信号质量问题而非数据融合问题**：不是"多少数据"而是"多干净的信号"，去噪和质量感知加权才是核心，而非简单的多任务联合训练。

8. **广告基础模型的 Scaling Law 是降低工程成本的杠杆**：广告 AUC ∝ D^0.07 的幂律意味着数据量翻倍只带来约 5% AUC 提升，但多场景 Adapter 微调使新场景上线成本降低 70% 标注。"预训练一次 + 微调多次"的 NLP 范式终于在广告域找到落地路径。

9. **短视频广告的核心差异化：场景融合而非强制插入**：NextAds 的实验数据（场景融合 VTR 比强插高 23%）表明，广告与有机内容的风格一致性是短视频广告效果的关键变量，优先于内容本身的质量。动态出价（高沉浸溢价）是与之配套的竞价策略。

10. **GR4AD 的工程经验：延迟优化 = 缓存命中率 > 量化 > 投机解码**：生成式广告推荐的延迟瓶颈在自回归步数，缓存热门用户的部分解码状态（命中率 >85%）是 ROI 最高的优化手段，其次才是模型量化和投机解码。

## 五、🎓 面试 Q&A（15题）

**Q1**: 广告 eCPM 的计算公式和优化目标？
> $\text{eCPM} = \text{CTR} \times \text{CVR} \times \text{Bid} \times 1000$（CPA 广告）。平台优化：最大化广告主价值 + 平台收益；广告主优化：在预算约束下最大化转化/GMV。

**Q2**: 生成式 CTR（DGenCTR）与判别式 CTR 的本质区别？
> 判别式建模 $P(y|x)$，假设特征独立；生成式建模 $P(x,y)$（联合分布），能捕捉特征间协变关系。冷启动场景中，联合分布建模能利用特征间依赖补全稀疏信号。

**Q3**: DCN-V2 为什么需要低秩分解？
> 全矩阵参数量 $O(d^2)$，当 $d=10000+$ 时内存和计算不可承受。低秩分解以 $O(d \times r)$（$r \ll d$）近似，在表达力和效率间取得工业可行的平衡点。

**Q4**: GSP 拍卖为什么不是 DSIC？
> GSP 中虚报出价可能比真实出价更优（赢得更多流量且成本更低）；只有 VCG 是真正 DSIC，但 VCG 计算复杂度高。FPA 中无 DSIC，Bid Shading 是正确策略。

**Q5**: Lagrangian 对偶出价中 λ 的经济学含义？
> λ 是预算的"影子价格"：λ 大 → 预算稀缺 → 出价保守；λ 小 → 预算充裕 → 出价激进。在线梯度更新 λ 实现预算动态管控，是所有自动出价方案的共同数学基础。

**Q6**: GBS 和 Bid2X 分别适用什么场景？
> GBS：适合 RTB 场景，通过分布建模"当前最优出价"，量化竞价不确定性。Bid2X：适合预算规划场景，前瞻性预测市场动态趋势，提前调整策略。工业上可叠加使用。

**Q7**: LLM Embedding 如何高效服务 CTR 线上推理？
> 离线批量计算并存入 Feature Store/Redis，在线推理只做查表（<1ms）+ MLP 融合（~1ms）。关键是训练时做好 LLM/ID Embedding 的对齐，保证离线特征在线可用。

**Q8**: 美团 MTGR 用一个模型替代 12 个的核心风险？
> 主要风险：大流量场景被小场景拖累；异质场景目标函数冲突（美食 vs 酒旅用户行为差异大）。对策：场景权重均衡采样、场景 Prompt 区分语境、分阶段上线验证。

**Q9**: 广告冷启动的核心挑战和解法？
> 挑战：新广告 ID 稀疏，CTR 模型打分不准，导致竞价劣势。解法：① 内容/语义代理（IDProxy，MLLM 生成 Embedding）；② 相似广告主迁移；③ 探索策略（UCB 保量）。

**Q10**: DGenCTR 的自适应去噪步数如何决定？
> 基于模型在每步去噪后输出分布的熵（置信度）。简单样本（主流行为，特征密集）1-2 步即可收敛，复杂/稀疏样本需要 4-5 步充分去噪。

**Q11**: 显式和隐式反馈如何混合训练？
> 显式反馈（高质量低噪声）作为锚点：① 赋更高 Loss 权重；② 校准隐式样本的质量评分；③ 去噪 Loss 降低噪声隐式样本影响。关键不是融合数据量，而是感知信号质量。

**Q12**: 广告 Bid Shading 的数学最优解？
> $b^* = \arg\max_b \int_0^b (v - w) dF(w|x)$，即最大化期望利润。一阶条件：$(v - b^*) f(b^*|x) = F(b^*|x)$。GBS 通过混合高斯近似 $F(w|x)$，避免直接计算积分。

**Q13**: Etsy 统一 Embedding 如何平衡搜索相关性和个性化？
> Query Embedding $f_q(q, u)$ 融合搜索意图（强）+ 用户偏好（弱），通过注意力权重动态平衡。强意图 Query（品牌词/型号）个性化权重自动降低，探索型 Query 个性化权重提升。

**Q14**: CADET 相比 DIN 的核心改进？
> DIN 用 attention 加权历史行为（判别式），CADET 用 causal LM 预训练（生成式），能做 next-item prediction 预训练，天然建模行为时序因果，冷启动和长尾泛化显著更好。

**Q15**: 统一广告排序模型（UniROM）的工程挑战？
> ① 各任务数据量级差异大（CTR >> CVR >> LTV），需均衡采样；② 任务更新频率不同（实时 CTR vs 天级 LTV）；③ 梯度冲突时的 PCGrad/MGDA 优化；④ 多维指标内卷需系统化 A/B。

**Q16**: ELEC 与广告基础模型（Foundation Model）的区别与分工？
> ELEC（离线 LLM 特征工厂）：用已有 LLM（如 BERT/LLaMA）生成增强特征，**不改变 CTR 模型结构**，工程改动小，是近期最可落地方案。Foundation Model for Ads：从零预训练专属广告大模型，参数量更大，需要跨平台数据，是 **3-5 年的工程路线**。两者并不对立——ELEC 用于今日业务，Foundation Model 是未来方向。

**Q17**: GRAD（Diffusion 出价）相比 RL 出价的核心优势？
> RL 出价：每广告主独立训练，维护 1000 个模型，成本高；冷启动广告主无历史数据，策略质量差。GRAD：**一个模型服务所有广告主**，通过 Prompt condition 个性化；预训练学到了市场通用规律，新广告主（<7天历史）直接受益（ROI +9.8%）。不足：Diffusion 推理需控制步数，DDIM 10-20 步满足实时约束，但比 RL 单步前向仍慢。

**Q18**: 短视频广告"场景融合"的技术实现路径？
> ① 提取有机视频和广告视频的共同语义 Embedding（ViT + 文本）；② 计算风格相似度（内容主题 + 画面节奏 + 情感色调）；③ 候选广告按相似度排序，优先展示风格一致的广告；④ 动态出价：相似度高的场景出价上限提升 20-30%（高 VTR 潜力），相似度低的降价保量。工程挑战：视频特征提取成本高，必须离线预计算 + 定期批量更新。

**Q19**: 生成式广告推荐（GR4AD）的延迟优化为什么以缓存为主？
> 广告场景用户-广告交互具有高度重复性（热门广告主 × 主流用户群），KV Cache 缓存高频解码状态命中率可达 85%+。而量化（INT8/INT4）需要重新评估模型精度损失，投机解码需要额外草稿模型，工程复杂度高。缓存的工程 ROI 最高：**简单、高效、无精度损失**。缓存失效策略（TTL + LRU）需精心设计以维持命中率。

**Q20**: 广告 Scaling Law（$\text{AUC} \propto D^{0.07}$）意味着什么？
> 幂律指数 0.07 意味着数据量翻倍只带来 ~5% AUC 提升（$2^{0.07} \approx 1.05$），但与 NLP（~0.1-0.15）相比更平缓，说明广告系统的提升瓶颈不仅在数据量，也在**特征质量和模型架构**。实践启示：① 追求数据量翻倍的边际效益在下降，特征工程和架构创新更有价值；② 多场景 Adapter 微调（参数量仅 2-5%）可以在不增加基础模型成本的前提下快速获取新场景收益。

## 六、📐 横向对比：工程落地视角

```
                 精度    延迟    冷启动    大规模可扩展性
DGenCTR 离散扩散  ★★★★   ★★★    ★★★★      ★★★
CADET 自回归      ★★★★   ★★     ★★★       ★★★
DCN-V2 交叉网络   ★★★    ★★★★★  ★★        ★★★★★
LightweightLLM   ★★★    ★★★★★  ★★★★     ★★★★★
```

**工程选型建议**：
- **主排序模型**：DCN-V2 / DNN 为主干，ELEC 离线 LLM 特征为补充（+0.3ms 零感知增强）
- **生成式探索**：DGenCTR 可作为离线评估 baseline，上线需专门优化延迟（GR4AD 路线）
- **新场景快速上线**：MTGR 统一模型 + 场景 Prompt，比训练独立模型快 3-5 倍；或 Foundation Model + Adapter 微调
- **自动出价选型**：单广告主高价值 → KBD（稳定+知识注入）；多广告主平台 → GRAD（跨广告主通用，维护成本低）
- **短视频广告**：NextAds 的场景融合 + 动态出价组合，是 2026 年短视频广告系统的参考架构

## 📚 参考文献

> - [[unified_embedding_personalized_retrieval_etsy|unified_embedding_personalized_retrieval_etsy]] — 统一 Embedding 个性化检索（Etsy 工业实践）
> - [bid2x_bidding_dynamics_forecasting](../papers/bid2x_bidding_dynamics_forecasting.md) — 竞价动态预测前瞻性出价优化
> - dcn_v2_deep_cross_network — DCN-V2 改进的深度交叉网络
> - [mtgr_meituan_generative_recommendation](../papers/mtgr_meituan_generative_recommendation.md) — 美团工业级多场景生成式推荐
> - [mtgr_multi_task_generative_ranking](../papers/mtgr_multi_task_generative_ranking.md) — 美团多任务生成式排序（HSTU + DLRM + Pareto）
> - [dgenctr_discrete_diffusion_ctr](../papers/dgenctr_discrete_diffusion_ctr.md) — 离散扩散生成式 CTR 预测
> - [generative_user_interest_shift_cohort_ctr](../papers/generative_user_interest_shift_cohort_ctr.md) — 生成式用户兴趣漂移群组建模
> - [revisiting_explicit_implicit_feedback](../papers/revisiting_explicit_implicit_feedback.md) — 重新审视显式与隐式反馈
> - [ELEC_efficient_llm_empowered_click_through_rate_prediction](../papers/ELEC_efficient_llm_empowered_click_through_rate_prediction.md) — LLM 离线特征工厂高效 CTR 增强
> - [foundation_model_ads_ctr](../papers/foundation_model_ads_ctr.md) — 广告基础模型预训练 + Scaling Law
> - [NextAds_next_generation_personalized_video_advertising](../papers/NextAds_next_generation_personalized_video_advertising.md) — 短视频多模态个性化广告
> - [gr4ad_generative_recommendation_advertising](../papers/gr4ad_generative_recommendation_advertising.md) — 毫秒级生成式广告推荐（GR4AD）
> - [gbs_generative_bid_shading_rtb](../papers/gbs_generative_bid_shading_rtb.md) — 生成式 Bid Shading RTB
> - [dgenctr_universal_generative_ctr](../papers/dgenctr_universal_generative_ctr.md) — 通用生成式 CTR 范式
> - CADET_context_conditioned_ads_CTR_decoder_only_transformer — Decoder-Only 因果建模广告 CTR
> - [GRAD_generative_pretrained_models_automated_ad_bidding](../papers/GRAD_generative_pretrained_models_automated_ad_bidding.md) — 生成式预训练模型自动出价（跨广告主通用）
> - [GAVE_generative_auto_bidding_value_guided_explorations](../papers/GAVE_generative_auto_bidding_value_guided_explorations.md) — 价值引导扩散模型出价探索
> - [KBD_knowledge_informed_bidding_dual_process_control](../papers/KBD_knowledge_informed_bidding_dual_process_control.md) — 知识引导双过程控制自动出价
> - CTR_driven_advertising_image_generation_MLLM — CTR 驱动广告图像生成
> - [UniROM_unifying_online_advertising_ranking_one_model](../papers/UniROM_unifying_online_advertising_ranking_one_model.md) — 统一在线广告排序单模型
> - [LLM_AUCTION_generative_auction_llm_native_advertising](../papers/LLM_AUCTION_generative_auction_llm_native_advertising.md) — LLM 原生对话广告生成式拍卖

---

## 相关概念

- [[generative_recsys|生成式推荐统一视角]]
- [[attention_in_recsys|Attention 在搜广推中的演进]]
- [[embedding_everywhere|Embedding 技术全景]]
- [[multi_objective_optimization|多目标优化]]
- [[sequence_modeling_evolution|序列建模演进]]
