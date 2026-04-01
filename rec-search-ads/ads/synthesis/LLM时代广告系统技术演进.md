# LLM时代广告系统技术演进（持续更新）

> 领域：ads | 类型：综合综述 | 覆盖论文：20篇 | 最近更新：2026-03-31

## 一、技术演进脉络

```
传统广告 (2018-2022)                 生成式广告 1.0 (2023-2024)          生成式广告 2.0 (2025-2026)
──────────────────────────────       ──────────────────────────────       ──────────────────────────────
DIN/DIEN 序列建模 CTR               CADET Decoder-Only CTR               DGenCTR 离散扩散 CTR
FM/DeepFM 特征交叉                   DCN-V2 矩阵 Cross 特征交叉           统一 token 化生成式预测
GSP 拍卖 + 规则出价                  Lagrangian 对偶自动出价              FPA 时代 Bid Shading + 竞价动态
人工设计广告素材                     MLLM 驱动创意生成                    多场景统一生成式排序（MTGR）
独立 CTR/CVR/LTV 模型               UniROM 统一排序模型                  LLM 轻量离线特征增强
纯 ID 特征冷启动困难                 Multimodal Proxy 解决冷启动          显隐式反馈联合训练
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
P(i_1,...,i_K | s, \text{Prompt}}_{\text{s) = \prod}}_{\text{{k=1}}^K P(i_k | i_{<k}, s, \text{Prompt}}_{\text{s)
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
L}}_{\text{{align}} = -\log \frac{\exp(\text{sim}(e_{LLM}, e_{ID}) / \tau)}{\sum_j \exp(\text{sim}(e_{LLM}, e_{ID_j}) / \tau)}
$$

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

### 2.6 广告拍卖机制创新

LLM 时代对话式广告拍卖：

$$
\text{Score}}_{\text{i = b}}_{\text{i \times q}}_{\text{i(query), \quad q}}_{\text{i = \text{LLM-Relevance}}(ad_i, query)
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
| $L_{align} = -\log \frac{\exp(\text{sim}(e_{LLM}, e_{ID})/\tau)}{\sum_j \exp(\cdot)}$ | 跨模态对齐 | LLM 特征融入 CTR |
| $\mathcal{L} = r_\text{CTR} - \beta \cdot \text{KL}(\pi_\theta \|\| \pi_\text{ref}) + \gamma r_q$ | 创意生成 | 防 Reward Hacking |

## 四、🎯 核心洞察

1. **生成式范式正在统一广告漏斗的每个环节**：CTR（DGenCTR 离散扩散）、出价（GBS/GRAD 分布建模）、素材（Diffusion 创意生成）、拍卖（LLM-AUCTION）——不是局部优化，是整个广告系统的重构。

2. **DGenCTR 的核心 insight：特征间依赖建模**：传统 CTR 隐含假设特征独立（给定 y），DGenCTR 通过联合分布 $P(x,y)$ 建模特征间协变关系，在稀疏场景（冷启动/长尾）泛化能力显著更强。

3. **FPA 时代的出价策略颠覆**：GSP（second-price）时代，出价=真实价值是弱均衡；FPA 时代，出价=预测 winning price 才能最大化 ROI。Bid Shading 从可选项变成必选项，但分布建模（GBS）比规则调整更鲁棒。

4. **Bid2X 的前瞻性 vs GBS 的分布估计**：这是两种互补的出价哲学——GBS 问"现在出多少合适"，Bid2X 问"未来市场会怎么变、提前调整"。工业上两者可叠加：先用 Bid2X 调整基础出价，再用 GBS 做分布级精调。

5. **LLM 进入广告的正确姿势：离线重于在线**：在线 LLM 推理的延迟成本在广告场景（50ms SLA）不可接受，轻量化方案（离线 Embedding → 对齐 → 查表）是工业落地的唯一现实路径，延迟增量 <2ms。

6. **DCN-V2 的工程价值：低秩分解解决了特征维度灾难**：广告系统特征维度动辄 10 万+，全矩阵参数量爆炸；低秩分解让高阶特征交叉在工业可承受的参数量下实现，这是 DCN-V2 成为广告系统基础架构的根本原因。

7. **显隐式反馈统一是信号质量问题而非数据融合问题**：不是"多少数据"而是"多干净的信号"，去噪和质量感知加权才是核心，而非简单的多任务联合训练。

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

## 六、📐 横向对比：工程落地视角

```
                 精度    延迟    冷启动    大规模可扩展性
DGenCTR 离散扩散  ★★★★   ★★★    ★★★★      ★★★
CADET 自回归      ★★★★   ★★     ★★★       ★★★
DCN-V2 交叉网络   ★★★    ★★★★★  ★★        ★★★★★
LightweightLLM   ★★★    ★★★★★  ★★★★     ★★★★★
```

**工程选型建议**：
- **主排序模型**：DCN-V2 / DNN 为主干，LightweightLLM 为补充特征
- **生成式探索**：DGenCTR 可作为离线评估 baseline，上线需专门优化延迟
- **新场景快速上线**：MTGR 统一模型 + 场景 Prompt，比训练独立模型快 3-5 倍

## 📚 参考文献

> - [unified_embedding_personalized_retrieval_etsy](../papers/unified_embedding_personalized_retrieval_etsy.md) — 统一 Embedding 个性化检索（Etsy 工业实践）
> - [bid2x_bidding_dynamics_forecasting](../papers/bid2x_bidding_dynamics_forecasting.md) — 竞价动态预测前瞻性出价优化
> - [dcn_v2_deep_cross_network](../papers/dcn_v2_deep_cross_network.md) — DCN-V2 改进的深度交叉网络
> - [mtgr_meituan_generative_recommendation](../papers/mtgr_meituan_generative_recommendation.md) — 美团工业级多场景生成式推荐
> - [dgenctr_discrete_diffusion_ctr](../papers/dgenctr_discrete_diffusion_ctr.md) — 离散扩散生成式 CTR 预测
> - [generative_user_interest_shift_cohort_ctr](../papers/generative_user_interest_shift_cohort_ctr.md) — 生成式用户兴趣漂移群组建模
> - [revisiting_explicit_implicit_feedback](../papers/revisiting_explicit_implicit_feedback.md) — 重新审视显式与隐式反馈
> - [lightweight_llm_enhanced_ctr](../papers/lightweight_llm_enhanced_ctr.md) — 轻量化 LLM 增强 CTR 预测
> - [gbs_generative_bid_shading_rtb](../papers/gbs_generative_bid_shading_rtb.md) — 生成式 Bid Shading RTB
> - [dgenctr_universal_generative_ctr](../papers/dgenctr_universal_generative_ctr.md) — 通用生成式 CTR 范式
> - [CADET_context_conditioned_ads_CTR_decoder_only_transformer](../papers/CADET_context_conditioned_ads_CTR_decoder_only_transformer.md) — Decoder-Only 因果建模广告 CTR
> - [GRAD_generative_pretrained_models_automated_ad_bidding](../papers/GRAD_generative_pretrained_models_automated_ad_bidding.md) — 生成式预训练模型自动出价
> - [GAVE_generative_auto_bidding_value_guided_explorations](../papers/GAVE_generative_auto_bidding_value_guided_explorations.md) — 价值引导扩散模型出价探索
> - [KBD_knowledge_informed_bidding_dual_process_control](../papers/KBD_knowledge_informed_bidding_dual_process_control.md) — 知识引导双过程控制自动出价
> - [CTR_driven_advertising_image_generation_MLLM](../papers/CTR_driven_advertising_image_generation_MLLM.md) — CTR 驱动广告图像生成
> - [UniROM_unifying_online_advertising_ranking_one_model](../papers/UniROM_unifying_online_advertising_ranking_one_model.md) — 统一在线广告排序单模型
> - [LLM_AUCTION_generative_auction_llm_native_advertising](../papers/LLM_AUCTION_generative_auction_llm_native_advertising.md) — LLM 原生对话广告生成式拍卖
