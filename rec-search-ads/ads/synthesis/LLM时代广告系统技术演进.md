# LLM时代广告系统技术演进（2026-03-30）

> 领域：ads | 类型：综合综述 | 覆盖论文：10篇

## 一、技术演进脉络

```
传统广告技术 (2018-2022)          LLM时代广告 (2023-2026)
──────────────────────────        ────────────────────────────────
DIN/DIEN 序列建模 CTR            CADET Decoder-Only CTR
规则/RL 出价策略                  生成式出价 (GRAD, GAVE, KBD)
人工设计广告素材                  MLLM 驱动创意生成
独立 CTR/CVR/LTV 模型           统一排序模型 (UniROM)
纯 ID 特征冷启动困难             Multimodal Proxy 解决冷启动 (IDProxy)
```

## 二、核心技术维度

### 2.1 CTR 预估的 LLM 化

**代表工作**：CADET、ELEC、IDProxy

三种 LLM 融合路径：

| 路径 | 代表方法 | 优点 | 缺点 |
|------|----------|------|------|
| LLM 在线推理 | 直接 LLM CTR | 特征最新 | 延迟高（>100ms） |
| LLM 离线特征 | ELEC | 零在线延迟 | 特征可能过时 |
| Multimodal Proxy | IDProxy | 冷启动强 | 模型维护复杂 |

**CADET Decoder-Only CTR 核心**：
$$\text{CTR} = \sigma(W \cdot h_T^{ad}), \quad h_T^{ad} = \text{CausalDecoder}([a_1,...,a_T, ad])$$

因果 attention 建模行为序列的时序因果依赖，比双向 encoder 更符合用户行为生成过程。

### 2.2 自动出价的生成式范式

**代表工作**：GRAD、GAVE、KBD

**核心数学框架**：
$$\max_\pi \mathbb{E}_\pi[\sum_t v_t] \quad \text{s.t.} \quad \mathbb{E}[\sum_t b_t] \leq B$$

**Lagrangian Relaxation**：
$$\mathcal{L}(b, \lambda) = \mathbb{E}[\sum_t(v_t - \lambda b_t)] + \lambda B$$

对偶变量 $\lambda$ 作为"预算价格"，自适应调整预算消耗速率。

**三种出价方案对比**：

| 方法 | 核心思想 | 稳定性 | 最优性 |
|------|----------|--------|--------|
| 规则出价 | 固定出价 × 调节系数 | ★★★ | ★ |
| RL 出价 | 策略梯度优化长期 ROI | ★★ | ★★★ |
| KBD 双过程 | 市场知识 + RL 自适应 | ★★★ | ★★★ |
| GRAD 生成式 | Diffusion 建模出价分布 | ★★★ | ★★★ |

### 2.3 广告创意的 AI 生成

**代表工作**：CTR-Driven Advertising Image Generation、NextAds

**CTR-as-Reward 训练循环**：
```
广告需求 → MLLM prompt 生成 → Diffusion 图像生成
                                      ↓
                              CTR Predictor 打分
                                      ↓
                              RL 微调 Diffusion（PPO/DDPO）
```

**防止 Reward Hacking（Goodhart's Law）**：
$$\mathcal{L} = r_\text{CTR} - \beta \cdot \text{KL}(\pi_\theta || \pi_\text{ref}) + \gamma \cdot r_\text{quality}$$

KL 约束防止过度优化 CTR predictor，quality reward 保证视觉合理性。

### 2.4 广告拍卖机制创新

**代表工作**：LLM-AUCTION

LLM 时代的对话式广告拍卖：
$$\text{Score}_i = b_i \times q_i(query), \quad q_i = \text{LLM-Relevance}(ad_i, query)$$

Quality-adjusted 定价（Vickrey 变体）：
$$p_i = \frac{b_{2\text{nd}} \times q_{2\text{nd}}}{q_i}$$

保证 DSIC（诚实出价是最优策略）+ 广告质量正向激励。

### 2.5 统一排序

**代表工作**：UniROM

$$\text{eCPM} = \text{CTR} \times \text{CVR} \times \text{Bid} \times 1000$$

统一模型一次 forward 同时输出 CTR/CVR/LTV，端到端优化 eCPM，延迟降低 35%。

## 三、📐 关键数学公式

**1. 自动出价对偶优化**：
$$\lambda^* = \arg\min_{\lambda \geq 0} \max_\pi \left[\mathbb{E}[\sum_t(v_t - \lambda b_t)] + \lambda B\right]$$

**2. CADET Pre-training Loss**：
$$\mathcal{L}_\text{pretrain} = -\sum_t \log P(a_{t+1} | a_1,...,a_t;\theta)$$

**3. Diffusion 预算约束出价采样**：
$$x_{t-1} = \alpha x_t + \beta \epsilon_\theta(x_t, t) + \gamma \nabla_{x_t} \text{Budget-Constraint}(x_t)$$

## 四、🎓 面试 Q&A（10题）

**Q1**: 广告 eCPM 的计算公式和优化目标？
> $\text{eCPM} = \text{CTR} \times \text{CVR} \times \text{Bid} \times 1000$（CPA 广告）。平台优化：最大化广告主价值 + 平台收益；广告主优化：在预算约束下最大化转化/GMV

**Q2**: GSP 拍卖为什么不是 DSIC？
> GSP 中，当其他人出价固定时，虚报出价可能比真实出价更优（赢得更多流量且成本更低）；只有 VCG 是真正 DSIC，但 VCG 计算复杂度高

**Q3**: 广告冷启动的核心挑战和解法？
> 挑战：新广告 ID 稀疏，CTR 模型打分不准，导致竞价劣势。解法：① 内容/语义代理（IDProxy）；② 探索策略（UCB）；③ 相似广告主迁移；④ 业务加权保护

**Q4**: Lagrangian 对偶出价中 λ 的经济学含义？
> λ 是"预算的影子价格"（边际价值）：λ 大 → 预算稀缺 → 出价保守；λ 小 → 预算充裕 → 出价激进。在线调整 λ 实现预算动态管控

**Q5**: LLM 广告素材生成如何避免 Reward Hacking？
> ① KL 散度约束（不偏离原始分布太远）；② 多维 Reward（CTR + 视觉质量 + 品牌合规）；③ 定期更新 CTR predictor（防止生成模型学会"欺骗"）

**Q6**: CTR 模型中如何处理位置偏差（Position Bias）？
> ① Inverse Propensity Scoring（IPS）：用展示位置的逆倾向分加权样本；② PAL（Position-Aware LTR）：分离 position 和 relevance 预测；③ 随机实验去除偏差（但成本高）

**Q7**: CADET 相比 DIN 的核心改进是什么？
> DIN 用 attention 对历史行为加权，是判别式 encoder；CADET 用 causal LM，能做 next-item 预训练，天然建模行为时序因果，冷启动更好

**Q8**: 广告出价中如何建模竞争对手出价（Bid Landscape）？
> 用历史竞价日志拟合 winning price 分布 $P(w|x)$（常用 Log-normal / 混合高斯），结合 bid landscape 做出价优化：$b^* = \arg\max_b \int_0^b (v-w)dF(w|x)$

**Q9**: 视频广告和图文广告 CTR 预估的核心差异？
> 视频：优化 VTR（3s/完播率）而非纯 CTR；特征提取需多模态（帧+音频+文字）且离线完成；用户跳过行为建模（skip pattern）是额外难点

**Q10**: 统一广告排序模型（UniROM）的工程挑战？
> ① 各任务数据量级差异大（CTR > CVR > LTV），需均衡采样；② 任务更新频率不同（实时 CTR vs 天级 LTV）；③ 指标内卷风险（A/B 需多维评估）

## 🎯 核心洞察

1. **生成式范式正在统一广告系统的每个环节**：CTR（CADET 生成式序列建模）、出价（GRAD/GAVE 扩散模型出价）、素材（Diffusion 创意生成）、拍卖（LLM-AUCTION 对话式机制）——生成式不是一个点，是整个广告漏斗的重构。

2. **Decoder-Only CTR 的本质是把"行为预测"变成"语言模型预训练目标"**：CADET 用 next-item prediction 预训练 CTR 模型，获得了 LLM 预训练的语义泛化能力，同时保留了 CTR 模型的精排特性——这是把两个生态打通的关键。

3. **出价策略的 Lagrangian 框架是统一语言**：GRAD/GAVE/KBD 出发点不同（扩散/价值/知识），但底层都是同一个约束优化问题：$\max \mathbb{E}[V] \text{ s.t. } \text{Budget} \leq B$。Lagrangian 对偶变量 $\lambda$ 是预算价格，这一经济学直觉比模型架构更重要。

4. **广告创意的 Goodhart's Law 是最大工程陷阱**：用 CTR 作为 reward 训练 Diffusion 会产生"对抗性"图像（CTR predictor 打高分但用户不喜欢）——KL 约束 + 多维 reward 是工业界已经踩过坑后的标配解法。

5. **LLM-Native 广告机制颠覆了广告 ≠ 搜索结果的假设**：传统广告是"展示位"竞争；LLM 对话广告是"推荐权重"竞争，边界模糊（自然回答 vs 广告植入）。机制设计的 DSIC 证明意味着广告主诚实出价是均衡，比 GSP 更优雅。

6. **冷启动的根本解法是语义代理**：IDProxy 用 MLLM 为新广告生成 embedding，绕过了 ID 稀疏问题——这不仅是广告技巧，也是所有 ID-based 系统冷启动的通用思路。

7. **统一排序（UniROM）降低的不只是延迟，而是系统复杂性熵**：维护 CTR/CVR/LTV 三个独立模型带来三套训练/部署/监控，统一后减少的认知负担和运维成本才是最大收益。

## 📚 参考文献

> - [CADET_context_conditioned_ads_CTR_decoder_only_transformer](../papers/CADET_context_conditioned_ads_CTR_decoder_only_transformer.md) — Decoder-Only 因果建模广告 CTR 预估
> - [ELEC_efficient_llm_empowered_click_through_rate_prediction](../papers/ELEC_efficient_llm_empowered_click_through_rate_prediction.md) — 离线 LLM 特征高效增强 CTR
> - [IDProxy_cold_start_CTR_multimodal_LLM_Xiaohongshu](../papers/IDProxy_cold_start_CTR_multimodal_LLM_Xiaohongshu.md) — 小红书多模态 LLM 代理解决冷启动 CTR
> - [KBD_knowledge_informed_bidding_dual_process_control](../papers/KBD_knowledge_informed_bidding_dual_process_control.md) — 知识引导双过程控制自动出价
> - [GRAD_generative_pretrained_models_automated_ad_bidding](../papers/GRAD_generative_pretrained_models_automated_ad_bidding.md) — 生成式预训练模型自动出价
> - [GAVE_generative_auto_bidding_value_guided_explorations](../papers/GAVE_generative_auto_bidding_value_guided_explorations.md) — 价值引导扩散模型出价探索
> - [CTR_driven_advertising_image_generation_MLLM](../papers/CTR_driven_advertising_image_generation_MLLM.md) — CTR 作为 reward 的广告图像生成
> - [NextAds_next_generation_personalized_video_advertising](../papers/NextAds_next_generation_personalized_video_advertising.md) — 下一代个性化视频广告 MLLM
> - [LLM_AUCTION_generative_auction_llm_native_advertising](../papers/LLM_AUCTION_generative_auction_llm_native_advertising.md) — LLM 原生对话广告生成式拍卖机制
> - [UniROM_unifying_online_advertising_ranking_one_model](../papers/UniROM_unifying_online_advertising_ranking_one_model.md) — 统一在线广告排序单模型
