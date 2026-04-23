# LLM 驱动广告冷启动 CTR 技术演进

> 领域：ads | 类型：synthesis | 覆盖论文：5篇 | 创建日期：2026-04-20

---

## 一、问题定义与技术脉络

### 冷启动的本质

广告冷启动的核心矛盾：**新广告没有交互数据 → CTR 模型打分不准 → 拍卖中竞争力差 → 曝光少 → 数据更少（恶性循环）**。

传统解法（content-based、meta-learning、相似 item 迁移）的共同瓶颈：**信息源单一**，只能利用广告自身的结构化特征（类目、关键词），无法深度理解广告内容语义。

### LLM 带来的范式转变

LLM（尤其是多模态 LLM）天然具备对文本+图像内容的深度理解能力，为冷启动提供了全新的信息源：

```
传统冷启动                          LLM 驱动冷启动
────────────                       ────────────────
结构化特征 → 规则映射              多模态内容 → LLM 语义理解
    ↓                                  ↓
相似 ID Embedding 迁移              生成 Proxy Embedding / 模型参数
    ↓                                  ↓
粗粒度冷启动                        细粒度、零样本冷启动
```

### 技术演进路径

```
Phase 1: 探索驱动（2015-2023）     Phase 2: LLM 特征增强（2024-2025）    Phase 3: LLM 生成式（2026+）
─────────────────────             ─────────────────────────────        ──────────────────────────
ε-greedy 随机探索                 ELEC 离线 LLM 特征工厂               LLM-HYPER 生成模型参数
UCB 置信上界探索                   IDProxy MLLM Proxy Embedding         LLM Agent 实时推理出价
Thompson Sampling                 MAB + UCB 系统性探索                  端到端 LLM 冷启动系统
```

---

## 二、核心方法对比

### 2.1 IDProxy（小红书，2026）

**核心思想**：MLLM 从多模态内容生成 Proxy ID Embedding，对齐到已有 ID Embedding 空间。

$$
e_{\text{proxy}} = \text{Align}(\text{MLLM}(\text{image}, \text{title}, \text{tags}), \mathcal{E}_{\text{ID}})
$$

**对齐损失**：

$$
\mathcal{L}_{\text{align}} = \mathcal{L}_{\text{CTR}}(e_{\text{proxy}}) + \lambda \cdot \text{MSE}(e_{\text{proxy}}, e_{\text{warm}}^{\text{nn}})
$$

其中 $e_{\text{warm}}^{\text{nn}}$ 是最近邻暖启动 item 的 ID Embedding。

**关键设计**：
- Proxy Embedding 与 CTR 模型端到端联合训练
- 暖启动后平滑过渡：$e_{\text{final}} = (1-\alpha_t) \cdot e_{\text{proxy}} + \alpha_t \cdot e_{\text{ID}}$，$\alpha_t$ 随曝光量递增
- 部署于小红书 Explore Feed，覆盖内容推荐 + 展示广告，**服务亿级用户**

### 2.2 LLM-HYPER（美国头部电商，2026）

**核心思想**：LLM 作为 Hypernetwork，直接生成 CTR 预测器的特征权重。

$$
w = \text{LLM}(\text{Prompt}(q, \{(x_i, w_i^*)\}_{i=1}^k)), \quad \hat{y} = \sigma(w^T x)
$$

**关键设计**：
- Few-shot CoT：检索 $k$ 个语义相似的历史广告作为 demonstration
- CLIP Embedding 做相似度检索
- **Training-free**：纯 in-context learning，不需要微调 LLM
- 适用于**极短生命周期**的促销广告（数天即下线）

### 2.3 MAB-ColdStart（UCB 探索，2025）

**核心思想**：用 UCB 多臂老虎机在拍卖框架下系统性探索新广告。

$$
\text{pCTR}_{\text{UCB}}(a) = \hat{\mu}_a + \sqrt{\frac{2\log t}{n_a}}
$$

$$
\text{eCPM}_{\text{UCB}}(a) = \text{bid}(a) \times \text{pCTR}_{\text{UCB}}(a)
$$

**关键设计**：
- 在 PBM（Position-Based Model）下推导理论 Regret 上界
- Budget Regret 而非 Click Regret（更符合广告场景）
- 真实广告平台数据验证

### 2.4 Online Learning for Auctions（UCB + 博弈论，2024）

**核心思想**：平台同时学习 CTR 和做拍卖分配，分析不同广告主行为模型下的 Regret。

$$
\text{Score}_a(t) = b_a \cdot \text{UCB}_a(t)
$$

**关键发现**：
- Myopic 广告主场景：$O(\sqrt{T})$ worst-case Regret（tight）
- 有间隔时可实现**负 Regret**（UCB 探索反而增收）
- Non-myopic 广告主会策略性操纵平台学习过程

---

## 三、横向对比

| 维度 | IDProxy | LLM-HYPER | MAB-ColdStart | Online Learning |
|------|---------|-----------|---------------|-----------------|
| **LLM 角色** | 生成 Proxy Embedding | 生成模型参数 | 无 LLM | 无 LLM |
| **是否需要训练** | 是（端到端） | 否（in-context） | 否（在线学习） | 否（在线学习） |
| **冷启动零样本** | 支持 | 支持 | 需几次曝光 | 需几次曝光 |
| **理论保证** | 无 | 无 | 有 Regret 上界 | 有 Regret 上界 |
| **工业验证** | 小红书亿级 | 美国头部电商 | 合成+真实数据 | 理论为主 |
| **冷启动阶段** | 取代 ID Embedding | 取代 CTR 模型 | 补充 CTR 估计 | 补充 CTR 估计 |
| **暖启动后** | 平滑过渡到真实 ID | 切换到正常模型 | UCB 项自然衰减 | UCB 项自然衰减 |

---

## 四、核心公式全集

| 公式 | 名称 | 用途 |
|------|------|------|
| $e_{\text{proxy}} = \text{Align}(\text{MLLM}(x), \mathcal{E}_{\text{ID}})$ | IDProxy 生成 | MLLM → Proxy Embedding |
| $w = \text{LLM}(\text{Prompt}(q, \text{demos}))$ | LLM-HYPER 生成 | LLM → CTR 模型参数 |
| $\text{UCB}_a = \hat{\mu}_a + \sqrt{2\log t / n_a}$ | UCB 置信上界 | 探索-利用平衡 |
| $\text{Regret}(T) = O(\sum_{a:\Delta_a>0} \log T / \Delta_a)$ | UCB Regret 上界 | 理论保证 |
| $\text{sim}(q, x_i) = \cos(\text{CLIP}(q), \text{CLIP}(x_i))$ | CLIP 相似检索 | 历史广告检索 |

---

## 五、工业实践指南

### 冷启动方案选型决策树

```
新广告上线
    ├── 有多模态内容（图文/视频）？
    │   ├── Yes → 生命周期 > 7 天？
    │   │   ├── Yes → IDProxy（端到端训练 Proxy Embedding，暖启动后平滑过渡）
    │   │   └── No  → LLM-HYPER（Training-free，即时生效）
    │   └── No  → 纯结构化特征 → 相似广告主迁移 + UCB 探索
    │
    └── 需要理论保证/合规要求？
        ├── Yes → MAB-ColdStart（有 Regret 上界）
        └── No  → IDProxy / LLM-HYPER（更高精度但黑盒）
```

### 工程实现要点

1. **IDProxy 部署**：MLLM 推理离线批处理（新 item 入库时触发），Proxy Embedding 存入 Feature Store，在线零额外延迟
2. **LLM-HYPER 部署**：LLM 离线生成权重 → 存储 → 在线查表。CLIP 检索需要建索引（ANN），延迟控制在 10ms 内
3. **UCB 部署**：UCB bonus 计算极轻量（$O(1)$），直接嵌入 eCPM 排序公式，无额外系统组件
4. **混合方案**：UCB 做系统性探索保底 + IDProxy/LLM-HYPER 提供更准的初始估计，两者互补

---

## 六、面试考点

**Q1**: LLM 驱动冷启动 vs 传统冷启动的本质区别？
> 传统方法用**结构化特征的统计关联**做冷启动（如同类目 item 的平均 CTR），LLM 方法用**内容语义理解**做冷启动（理解"这个广告在说什么，用户可能怎么反应"）。后者信息维度更高，但依赖 LLM 的语义理解质量。

**Q2**: IDProxy 的 Proxy Embedding 和直接用 BERT/CLIP Embedding 做冷启动的区别？
> 直接用 BERT/CLIP Embedding 是"通用语义空间"，与 CTR 模型的 ID Embedding 空间不对齐，效果有限。IDProxy 的核心在于**端到端对齐**：Proxy Embedding 在 CTR 目标下联合优化，确保语义信息被映射到 CTR 有用的方向。

**Q3**: LLM-HYPER 的 Training-free 是真正的零成本吗？
> 不是。代价在于：(1) LLM 推理成本（虽然离线，但每个新广告需要一次 LLM 调用）；(2) 检索质量依赖历史广告库的覆盖度；(3) 生成的权重是线性模型参数，表达力有限。适合**极短生命周期**场景，长期广告还是 IDProxy 更优。

**Q4**: UCB 冷启动在拍卖中的特殊考虑？
> 拍卖中 UCB 的 bonus 直接影响 eCPM 排序，过大的 bonus 会让新广告"挤掉"高价值老广告，导致平台收入短期下降。需要 budget-aware UCB：设定探索预算上限，或用 budget regret 替代 click regret 做优化。

**Q5**: 如何衡量冷启动方案的好坏？
> (1) **冷启动期 CTR 预估 AUC**：新广告前 N 次曝光的预估准确性；(2) **收敛速度**：达到暖启动水平所需的曝光次数；(3) **平台收入影响**：冷启动探索对整体 eCPM 的影响；(4) **广告主体验**：新广告上线后多快能获得稳定曝光。

---

## 相关链接

- [[IDProxy_cold_start_CTR_ads_recommendation_xiaohongshu]] — 小红书多模态冷启动
- [[LLM_HYPER_generative_ctr_cold_start_hypernetworks]] — LLM Hypernetwork 冷启动
- [[mab_cold_start_auction_dynamics]] — MAB 拍卖冷启动
- [[improved_online_learning_ctr_ad_auctions]] — 在线学习 CTR 拍卖
- [[ELEC_efficient_llm_empowered_click_through_rate_prediction]] — LLM 离线特征工厂

## 相关概念

- [[concepts/embedding_everywhere]] — Embedding 技术全景
- [[concepts/generative_recsys]] — 生成式推荐统一视角
- [[concepts/attention_in_recsys]] — Attention 在搜广推演进
