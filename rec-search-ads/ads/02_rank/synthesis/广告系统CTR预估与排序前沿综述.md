# 广告系统 CTR 预估与排序前沿综述
> 综合学习笔记 | 领域：ads | 日期：20260329

---

## 一、技术脉络概览

广告推荐系统的核心链路：**召回 → 粗排 → 精排（CTR预估）→ 重排 → 广告创意生成**

本批论文覆盖了这条链路的各个关键阶段：

| 阶段 | 代表论文 | 核心技术 |
|------|---------|---------|
| CTR 精排 | DHEN, MaskNet, Wukong CTR | 异构集成/乘性交互/大规模并行 |
| 重排 | GR2, LLM Reranker | 生成式推理/LLM+传统模型混合 |
| 列表推荐 | HiGR | 层次化生成式 Slate 推荐 |
| 广告排序策略 | Hierarchy Policy Opt | 层次化强化学习 |
| 系统架构 | Meta Lattice | 多域多目标模型空间重设计 |
| LLM RL 基础 | DAPO | 大规模强化学习算法 |
| 创意生成 | BannerAgency | 多模态 LLM 广告素材自动生成 |

---

## 二、核心技术主题深度解析

### 主题1：特征交互建模的演进

**从加性到乘性的演进路径：**

```
FM (2010): w₁x₁ + w₂x₂ + v₁·v₂·x₁x₂   ← 引入乘性二阶交互
    ↓
DeepFM (2017): FM + DNN                    ← 加性高阶 + 乘性二阶
    ↓
xDeepFM (2018): Compressed Interaction Network ← 显式高阶乘性
    ↓
MaskNet (2021): Instance-Guided Mask ⊙ FFN ← 实例动态乘性
    ↓
DHEN (2022): 异构模块层次集成              ← 多种交互方式集成
    ↓
Wukong (2023): 低秩压缩交互 + 大规模并行   ← 可扩展高效乘性交互
```

**关键公式：MaskBlock**

$$
\text{MaskBlock}(X) = \text{FFN}\left(\text{LayerNorm}\left(\text{MLP}}_{\text{{mask}}(X) \odot X\right)\right)
$$

**关键公式：DHEN 层次集成**

$$
h_l = \text{Aggregate}\left(\text{FM}(h_{l-1}), \text{DCN}(h_{l-1}), \text{DIN}(h_{l-1}), ...\right)
$$

---

### 主题2：生成式推荐与重排的兴起

**LLM 在广告推荐系统中的定位演变：**

```
早期尝试：LLM 作为独立推荐器 → 失败（ID 空间爆炸，协同过滤信号缺失）
现在共识：LLM 作为重排器 → 成功（候选集已由传统模型筛选）
前沿探索：生成式 Slate 推荐（HiGR）→ 端到端生成整个推荐列表
```

**HiGR 的三阶段目标优化（ORPO）：**

$$
\mathcal{L}_{post} = -\log \pi_\theta(y^+|x) - \alpha \log \sigma\left(\sum_{t=1}^{MD} l_\theta(x, y^+_t) - \sum_{t=1}^{MD} l_\theta(x, y^-_t)\right)
$$

其中 $l_\theta(x, y_t) = \log \frac{\pi_\theta(y_t|x, y_{<t})}{1 - \pi_\theta(y_t|x, y_{<t})}$ 是每步的 log-odds。

---

### 主题3：大规模 RL 训练在广告系统的应用

**DAPO 解耦裁剪：**

$$
\mathcal{L}_{DAPO} = \mathbb{E}\left[\min\left(r_t \hat{A}_t, \text{clip}(r_t, 1-\varepsilon_{low}, 1+\varepsilon_{high})\hat{A}_t\right)\right]
$$

其中正优势样本用大 $\varepsilon_{high}$，负优势样本用小 $\varepsilon_{low}$，解耦提升训练稳定性。

**RL 在广告系统中的主要应用场景：**

| 应用场景 | 奖励设计 | 代表工作 |
|---------|---------|---------|
| 广告排序策略 | 多目标业务指标（RPM+满意度） | Hierarchy Policy Opt |
| 重排对齐 | 可验证重排奖励（NDCG提升） | GR2 |
| 列表对齐 | 三目标偏好对（排名/兴趣/多样性） | HiGR ORPO |

---

### 主题4：工业级系统架构的演进

**Meta Lattice 的 MDMO 架构转变：**

```
传统：N产品线 × M目标 = N×M 个独立模型
↓
Lattice: 1个统一模型 + 轻量级 domain/task 路由
↓
结果：+10%收入，+11.5%用户满意度，-20%算力
```

**Wukong CTR 大规模并行训练：**

$$
\mathbf{W}_{interaction} = \mathbf{U} \cdot \mathbf{V}^T, \quad \mathbf{U} \in \mathbb{R}^{d \times r}, \mathbf{V} \in \mathbb{R}^{d \times r}, r \ll d
$$

低秩分解使参数从 $O(d^2)$ 降至 $O(dr)$，实现百亿参数级别的高效训练。

---

## 三、关键指标对照表

| 论文 | 数据集/平台 | 关键指标提升 |
|------|-----------|------------|
| HiGR | 腾讯（数亿用户） | 观看时长+1.22%，视频播放+1.73%，推理5× |
| DHEN | Meta 广告系统 | NE +0.27%，训练吞吐 1.2× |
| MaskNet | 3个公开数据集 | AUC +0.1%~0.5% vs DeepFM |
| GR2 | 2个真实数据集 | Recall@5 +2.4%，NDCG@5 +1.3% vs SOTA |
| Meta Lattice | Meta 全产品线 | 收入 +10%，满意度 +11.5%，容量节省 20% |
| DAPO | AIME 2024 | 50分（Qwen2.5-32B base） |

---

## 四、📐 核心公式汇总（≥3个）

### 公式1: MaskBlock 特征乘性交互

$$
\text{MaskBlock}(X) = \text{FFN}\left(\text{LayerNorm}\left(\text{MLP}}_{\text{{mask}}(X) \odot X\right)\right)
$$

- $\odot$：逐元素乘法（Hadamard product）
- $\text{MLP}}_{\text{{mask}}(X)$：基于输入实例动态生成的掩码
- **意义**：引入实例引导的乘性特征交互，突破加性 FFN 的局限

### 公式2: DHEN 层次集成

$$
h_l = \text{Aggregate}\left(\{\text{Module}}_{\text{k(h}}_{\text{{l-1}})\}_{k=1}^K\right)
$$

$$
\mathcal{L}_{NE} = -\frac{1}{N}\sum_{i=1}^N \left[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right] / H(p)
$$

- $H(p) = -p\log p - (1-p)\log(1-p)$：背景 CTR $p$ 对应的熵
- **意义**：Normalized Entropy 归一化评估，消除不同数据集 CTR 差异的影响

### 公式3: HiGR CRQ-VAE 对比量化损失

$$
\mathcal{L}_{CRQ-VAE} = \mathcal{L}_{recon} + \lambda_1 \mathcal{L}_{global\_quan} + \lambda_2 \mathcal{L}_{cont}
$$

$$
\mathcal{L}_{cont} = -\frac{1}{D-1}\sum_{d=1}^{D-1} w_d \log \frac{\exp(\cos(e_a^d, e_p^d)/\tau)}{\exp(\cos(e_a^d, e_p^d)/\tau) + \sum_{n\neq a,p}\exp(\cos(e_a^d, e_n^d)/\tau)}
$$

- **意义**：Prefix-level InfoNCE 使相似物品共享 ID 前缀，提升语义 ID 的结构化

### 公式4: DAPO 解耦裁剪目标

$$
\mathcal{L}_{DAPO} = \mathbb{E}\left[\min\left(r_t \hat{A}_t, \text{clip}(r_t, 1-\varepsilon_{low}, 1+\varepsilon_{high})\hat{A}_t\right)\right]
$$

$$
r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}, \quad \varepsilon_{low} < \varepsilon_{high}
$$

- **意义**：正负样本解耦裁剪，提升大规模 LLM RL 训练的稳定性

### 公式5: Wukong 低秩交互分解

$$
\text{Interaction}(F) = (F \cdot \mathbf{U}) \cdot (F \cdot \mathbf{V})^T, \quad \mathbf{U}, \mathbf{V} \in \mathbb{R}^{d \times r}
$$

- **意义**：将 $O(d^2)$ 交互矩阵降至 $O(dr)$，支持超大规模 CTR 模型训练

---

## 五、📚 参考文献列表

1. **HiGR** - Pang et al. "HiGR: Efficient Generative Slate Recommendation via Hierarchical Planning." arXiv:2512.24787 (2025).

2. **BannerAgency** - Wang et al. "BannerAgency: Advertising Banner Design with Multimodal LLM Agents." EMNLP 2025. arXiv:2503.11060.

3. **DHEN** - Zhang et al. "DHEN: A Deep and Hierarchical Ensemble Network for Large-Scale CTR Prediction." arXiv:2203.11014 (2022).

4. **GR2** - Liang et al. "Generative Reasoning Re-ranker." arXiv:2602.07774 (2026).

5. **Meta Lattice** - Luo et al. "Meta Lattice: Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations." KDD 2026. arXiv:2512.09200.

6. **LLM Reranker** - Sun et al. "LLM as Explainable Re-Ranker for Recommendation System." arXiv:2512.03439 (2025).

7. **DAPO** - Yu et al. "DAPO: An Open-Source LLM Reinforcement Learning System at Scale." arXiv:2503.14476 (2025).

8. **Wukong CTR** - "Wukong CTR: Scalable Deep CTR Prediction via Massive Parallel Training." (2023).

9. **MaskNet** - Zhang et al. "MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask." DLP-KDD 2021. arXiv:2102.07619.

10. **Hierarchy Policy Optimization** - "Hierarchy Enhanced Policy Optimization for Ad Ranking." arXiv:2603.xxxx (2026).

### 相关基础工作
- **DeepFM** - Guo et al. (2017). Combined FM + DNN for CTR.
- **xDeepFM** - Lian et al. (2018). Compressed Interaction Network.
- **DCN** - Wang et al. (2017). Deep & Cross Network.
- **BERT4Rec** - Sun et al. (2019). Self-attentive sequential recommendation.
- **OneRec** - 字节跳动. 统一检索与精排的生成式推荐框架.
- **TIGER** - Rajput et al. (2023). Semantic ID-based generative recommendation.

---

## 六、🎓 面试 Q&A（≥10道）

### Q1: CTR 预估中，加性特征交互和乘性特征交互有什么本质区别？各自的代表方法是什么？

**A**: 
- **加性交互**：特征通过加权求和组合，如 $o = \sigma(\sum_i w_i \cdot h_i)$。代表方法：MLP/DNN（全连接层本质是加性操作）
- **乘性交互**：特征通过逐元素或矩阵乘法组合，如 FM 的 $\langle v_i, v_j \rangle x_i x_j$。代表方法：FM、MaskNet（element-wise mask）、DCN（cross product）
- **本质区别**：加性操作从数学角度无法高效表达两个特征的乘性关系（需要无限多个隐层单元），乘性操作直接建模特征间的交叉影响

---

### Q2: 工业 CTR 系统为什么用 NE（Normalized Entropy）而不是 AUC 来评估模型？

**A**: 
- **AUC** 衡量排序能力，与绝对概率值无关（量级无关）
- **NE** 衡量概率校准精度，归一化后消除不同数据集背景 CTR 不同的影响
- 工业广告系统的出价（bidding）依赖精确的 CTR 预估值（$\text{bid} = \text{eCPM} / \text{CTR}$），AUC 好但 NE 差会导致出价严重失真，影响收入
- 因此：**NE** 更能反映工业系统的真实价值，0.1% NE 改进通常对应可观收入提升

---

### Q3: Multi-Domain Multi-Objective（MDMO）推荐系统的主要挑战是什么？Meta Lattice 如何解决？

**A**: 
主要挑战：①数据碎片化（跨产品线信号无法共享）②负迁移（强域压制弱域）③参数爆炸（N×M个独立模型）④系统运维复杂度
Meta Lattice 解决方案：①跨域知识共享（统一 User Tower）②Domain-adaptive 损失权重（防负迁移）③统一模型+路由（从N×M缩至1）④蒸馏+系统优化（保效果降成本）
结果：一个模型同时服务所有产品线，+10%收入，-20%算力。

---

### Q4: 生成式推荐（Generative Recommendation）相比判别式推荐（Discriminative Recommendation）有什么优势和劣势？

**A**: 
| | 生成式 | 判别式 |
|--|--------|--------|
| **优势** | ①端到端优化整个列表 ②天然支持多样性控制 ③可利用 LLM 世界知识 ④扩展能力强（Scaling Law） | ①延迟低 ②对超大候选集高效 ③协同过滤信号更充分 |
| **劣势** | ①推理延迟高 ②ID 空间大时难处理 ③冷启动依赖语义 | ①缺乏全局列表规划 ②多样性需要额外 post-processing ③可解释性差 |

**工业实践**：两者结合，判别式做粗排，生成式做精排/重排（如 GR2、HiGR）。

---

### Q5: 广告重排阶段使用 LLM 的核心挑战有哪些？业界有哪些解决方案？

**A**: 
核心挑战：①**延迟**：LLM 推理 100ms~1s vs 重排 SLA <10ms ②**ID 空间**：工业系统数十亿非语义 ID，LLM 无法处理 ③**Reward Hacking**：模型发现保持原始顺序可获高奖励

解决方案：
- **延迟**：Speculative Decoding + KV Cache + 只对 Top-K 候选做 LLM 重排
- **ID 空间**：语义 ID（Semantic ID）转换（GR2 方案，确保≥99%唯一性）
- **Reward Hacking**：条件可验证奖励（GR2），只有实际重排发生才给满分

---

### Q6: DAPO 算法的四大关键技术是什么？分别解决什么问题？

**A**: 
1. **解耦裁剪**：正负样本不同 clip 阈值 → 加快正样本学习，稳定负样本惩罚
2. **动态采样**：监控有效样本比例，动态调整难度 → 避免无效梯度，保持探索
3. **Token 级策略梯度**：每个 token 计算优势函数 → 解决长序列信用分配问题
4. **过长惩罚**：超长推理链给负奖励 → 防止冗长 reward hacking

---

### Q7: HiGR 的 Contrastive RQ-VAE（CRQ-VAE）为什么要做 Prefix-level 对比学习？排除最后一层的原因？

**A**: 
- **为什么做 Prefix-level 对比**：确保相似物品的 Semantic ID 共享相同前缀（语义层次化），使 Slate Planner 可以直接在 ID 前缀空间做多样性控制，避免代价高昂的 embedding 比较
- **排除最后一层的原因**：前 D-1 层共享粗粒度语义，最后一层负责区分具体物品（细粒度）。若对最后层也做对比约束，会强制不同物品的最终 ID 也相似，破坏 ID 的唯一性和区分度

---

### Q8: 广告系统的 A/B 实验设计中，有哪些常见的实验偏差和规避方法？

**A**: 
常见偏差：
- **网络效应（Interference）**：用户 A 看到广告影响用户 B 的行为，对照组污染
- **幸存者偏差**：只分析有点击的样本
- **长期效应遗漏**：短期 A/B 无法捕获用户长期行为变化
- **新奇效应**：新功能短期内因新鲜感获得额外收益

规避方法：
- **受众分组**（user bucketing）而非请求分组
- **Holdout 实验**：长期维持对照组
- **分层实验**：按设备/地区分层，控制干扰变量
- **Switchback 实验**：时间粒度轮换，适用于市场级变化

---

### Q9: 广告 CTR 模型的"特征工程"在深度学习时代还重要吗？应该重点关注哪些特征？

**A**: 
深度学习时代特征工程仍然重要，方向转变：
- **不再重要**：手工组合特征（模型自动学交叉）
- **仍然关键**：①**特征选择**：去除噪声特征，防止过拟合 ②**时序特征**：用户行为序列的时间衰减设计 ③**统计特征**：位置偏差、历史 CTR（需谨慎处理穿越） ④**跨域特征**：多产品线的行为信号融合 ⑤**业务先验**：广告主出价/预算特征

重点关注：**用户实时行为序列**（DIN/DIEN 验证其价值最大）、**场景上下文特征**（设备、时段、位置）。

---

### Q10: 在广告推荐系统中，如何平衡"短期收入最大化"和"用户长期留存"？

**A**: 
核心矛盾：高频广告展示短期提升 RPM，但降低用户体验，长期降低 DAU。

平衡方法：
- **约束优化**：将用户满意度设为约束条件（Lagrangian），在满足用户体验下限的前提下最大化收入
- **层次化策略**（Hierarchy Policy Opt）：高层策略优化长期 DAU，低层策略执行短期 RPM
- **多目标 Pareto 优化**：维护 RPM-满意度 Pareto 前沿，业务方选择目标操作点
- **长期指标设计**：引入"7日留存""用户满意度调研"等长期指标加入奖励函数
- **广告密度控制**：设置每个 session 的广告曝光上限，避免用户被"轰炸"

---

### Q11: MaskNet 中的 Instance-Guided Mask 和 Transformer 的 Self-Attention 有什么区别？

**A**: 
| | Instance-Guided Mask | Self-Attention |
|--|---------------------|---------------|
| **权重来源** | 基于输入实例 MLP 生成固定掩码 | 基于 Query-Key 相似度动态计算 |
| **操作方式** | Element-wise product（逐元素乘） | 加权求和（softmax + value） |
| **交互范围** | 特征内部（同一特征的不同维度） | 特征间（不同特征之间） |
| **复杂度** | O(d) | O(n²d)（n=特征数） |
| **适合场景** | 单个特征的自适应重加权 | 建模特征间的相关性 |

两者互补：MaskNet 做特征内部乘性增强，Attention 做特征间关系建模，可以组合使用。

---

### Q12: 广告创意（Banner）自动生成与 CTR 预估如何协同优化？

**A**: 
当前多为独立优化，前沿方向是联合优化：
- **BannerAgency 的局限**：生成美观的广告素材，但不直接优化 CTR
- **理想的协同框架**：
  1. CTR 模型预测不同素材版本的 CTR
  2. 生成模型根据 CTR 反馈调整设计方向
  3. 闭环：高 CTR 设计风格 → 生成模型学习 → 新素材 CTR 更高

- **实际挑战**：素材 CTR 的反馈延迟（需要足够曝光），难以直接作为生成目标
- **近期进展**：使用 CLIP 等视觉语言模型作为 CTR 代理指标指导素材生成

---

## 七、技术趋势与展望

1. **生成式 Ranking 全面兴起**：从判别式 CTR 到生成式 Slate，端到端优化列表质量
2. **LLM 与传统模型深度融合**：LLM 不是替代者，而是传统模型的上层"理解与对齐"模块
3. **大规模 RL 普及**：DAPO 等算法使工业 RL 从实验室走向大规模生产部署
4. **多模态广告创意**：从文字优化到图文联合优化，素材自动生成成为核心竞争力
5. **系统级协同设计**：模型架构与训练/服务系统协同设计（DHEN、Wukong）成为主流范式
6. **统一模型架构**：从N×M个专门模型到统一架构（Meta Lattice），降低运维成本同时提升效果


## 📐 核心公式直观理解

### DCN v2 的交叉层

$$
\mathbf{x}_{l+1} = \mathbf{x}_0 \odot (W_l \mathbf{x}_l + \mathbf{b}_l) + \mathbf{x}_l
$$

- $\mathbf{x}_0$：原始特征
- $W_l$：可学习的交叉权重矩阵

**直观理解**：每一层都和原始输入 $\mathbf{x}_0$ 做 element-wise 乘法，实现显式的特征交叉。$L$ 层交叉网络能建模 $L+1$ 阶特征交互——比 DNN 隐式交叉更可控，比手动构造特征更自动化。

### 位置偏差矫正

$$
P(\text{click}) = P(\text{examine} | \text{pos}) \times P(\text{relevant} | \text{query, ad})
$$

**直观理解**：用户点不点击 = "有没有看到" × "看到了觉得相不相关"。排第一的广告被看到的概率远高于排第十的，如果不修正位置偏差，模型会认为排第一的广告质量更好——形成"越排前面越被点→越被点越排前面"的死循环。

### 知识蒸馏的 soft label

$$
\mathcal{L}_{\text{KD}} = \alpha \cdot \text{CE}(y, \hat{y}_s) + (1-\alpha) \cdot \tau^2 \cdot \text{KL}\left(\frac{\hat{y}_t}{\tau} \| \frac{\hat{y}_s}{\tau}\right)
$$

**直观理解**：大模型（teacher）的"犹豫"包含丰富信息——如果 teacher 对某个样本说"70% 会点击"（而非 100%），这个 30% 的不确定性反映了样本的难度。小模型（student）学习这种 soft distribution 比只学 hard label（0/1）效果更好。

