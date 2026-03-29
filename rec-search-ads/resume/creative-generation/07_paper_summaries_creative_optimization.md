# 广告创意优化核心论文精读笔记

> 6篇覆盖创意选择/排序/组合优化的关键论文，面向DCO系统工程实现

---

## 目录
1. [AutoCO — 自动化创意交互函数搜索](#1-autoco)
2. [AES — 树结构创意组合选择](#2-aes)
3. [VAM-HBM — 视觉先验混合Bandit创意排序](#3-vam-hbm)
4. [Peri-CR — 广告与创意并行排序](#4-peri-cr)
5. [Amazon MVT — 实时多变量Bandit优化](#5-amazon-mvt)
6. [TTTS — Top-Two Thompson Sampling创意选择](#6-ttts)

---

<a id="1-autoco"></a>
# 1. AutoCO: Automated Creative Optimization for E-Commerce Advertising

> 来源: WWW 2021 | arXiv: 2103.00436
> 作者: Jin Chen, Ju Xu, Gangwei Jiang, Tiezheng Ge, Zhiqiang Zhang, Defu Lian, Kai Zheng (阿里妈妈)
> 核心贡献: 将AutoML的one-shot搜索引入创意元素交互建模，结合变分推断Thompson Sampling实现自动化创意优化

## 问题背景

电商广告创意由多个元素组成（标题、图片、价格标签、促销信息等），不同元素之间的交互对CTR有重要影响。传统方法（如FM）假设元素间交互为内积形式，但实际交互可能更复杂。同时，新创意的反馈数据有限，CTR估计方差大，需要在探索（exploration）和利用（exploitation）之间平衡。

**核心挑战：**
- 创意元素交互函数未知，简单内积不够表达
- 新创意缺乏曝光数据，CTR估计不准
- 组合空间大，需要高效探索策略

## 核心方法

### 1. 交互函数搜索空间（AutoML启发）

借鉴NAS（Neural Architecture Search）思想，定义创意元素间交互的搜索空间。对每对元素 (i, j)，候选交互操作包括：

**候选操作集合 O：**
- **Plus**: e_i + e_j（加法）
- **Minus**: e_i - e_j（减法）
- **Multiply**: e_i * e_j（逐元素乘法/Hadamard积）
- **Max**: max(e_i, e_j)（逐元素取最大值）
- **Concat**: [e_i; e_j]（拼接）
- **Inner Product**: <e_i, e_j>（标准内积，FM基线）

### 2. One-Shot搜索算法

不像传统NAS那样逐个评估每个架构，AutoCO使用one-shot方法：

**混合操作（Mixed Operation）：**
```
f(e_i, e_j) = Σ_k α_k · o_k(e_i, e_j)
```
其中 α_k 是操作 o_k 的架构参数（权重），通过softmax归一化。

**Operation-Aware Embedding (OAE)：**
关键创新 — 不同操作使用独立的embedding，而非共享embedding：
```
f(e_i, e_j) = Σ_k α_k · o_k(e_i^(k), e_j^(k))
```
每个操作 o_k 有自己的embedding e_i^(k)，避免不同操作对embedding的梯度冲突。

### 3. 随机变分推断（Stochastic Variational Inference）

为实现Thompson Sampling，需要参数的后验分布。AutoCO使用变分推断：

**CTR预测模型：**
```
p(click | creative) = σ(w^T · h(creative))
```
其中 h(creative) 是通过搜索到的交互函数得到的特征表示。

**变分后验：**
```
q(w) = N(μ, diag(σ²))
```
使用重参数化技巧（Reparameterization Trick）：
```
w = μ + σ ⊙ ε,  ε ~ N(0, I)
```

**ELBO目标函数：**
```
L = E_q[log p(D|w)] - KL(q(w) || p(w))
```

### 4. Thompson Sampling探索

每次曝光时：
1. 从变分后验 q(w) 中采样参数 w_sample
2. 用采样参数预测每个候选创意的CTR
3. 选择预测CTR最高的创意展示
4. 收集反馈，更新变分后验

**优势：** 参数不确定性大的创意有更大概率被探索到（方差大 → 采样值可能很高）

## 系统架构

```
广告请求 → 候选创意生成（元素组合）
        → AutoCO模型：
            1. 各元素embedding查询
            2. 搜索到的交互函数计算特征
            3. 变分采样 → CTR预测
            4. 选择CTR最高的创意
        → 展示 & 收集反馈
        → 在线更新变分后验
```

## 关键实验结果

**离线实验（公开数据集 + 合成数据）：**
- AutoCO在累积遗憾（cumulative regret）上显著优于单一交互函数
- OAE模块相比共享embedding带来明显CTR提升
- 变分推断+TS优于贪心策略（greedy）

**在线A/B测试：**
- **CTR提升7%**（对比baseline方法）
- 部署在阿里妈妈电商广告平台

## 与我们系统的关联

- **创意元素交互建模：** 我们的DCO系统同样面临多元素组合问题，AutoCO的操作搜索思路可用于自动发现最优的元素组合方式
- **变分推断+TS：** 这是解决新创意冷启动探索的有效方案，可直接应用于我们的创意选择模块
- **OAE思想：** 不同的融合方式应使用独立embedding，这个设计原则对我们的特征工程有指导意义

## 核心公式

```
# 混合交互操作（含OAE）
f(e_i, e_j) = Σ_k softmax(α_k) · o_k(e_i^(k), e_j^(k))

# 变分后验 + 重参数化
q(w) = N(μ, diag(σ²))
w = μ + σ ⊙ ε,  ε ~ N(0, I)

# ELBO
L = E_q[log p(D|w)] - KL(q(w) || p(w))

# Thompson Sampling选择
creative* = argmax_c  σ(w_sample^T · h(c)),  w_sample ~ q(w)
```

---

<a id="2-aes"></a>
# 2. AES: Efficient Optimal Selection for Composited Advertising Creatives with Tree Structure

> 来源: AAAI 2021 | arXiv: 2103.01453
> 作者: Jin Chen, Tiezheng Ge, Gangwei Jiang, Zhiqiang Zhang, Defu Lian, Kai Zheng (阿里妈妈)
> 核心贡献: 利用创意元素的树状层级结构 + 动态规划实现Thompson Sampling的高效探索，大幅降低组合空间的搜索复杂度

## 问题背景

广告平台可以实时将广告主提供的基础素材（标题、图片、背景、按钮等）组合成完整创意。这产生了巨大的组合空间：如果有K种素材类型，每种有n_k个选择，总组合数为 ∏n_k，可能达到数百万级。

**核心挑战：**
- 组合空间指数级增长，无法逐一估计CTR
- 反馈数据稀疏，每个组合的曝光极少
- 需要在海量组合中快速找到最优创意

## 核心方法

### 1. 树结构建模

将创意的组成元素组织为树状结构：

```
         创意(root)
        /     |     \
    标题    图片    背景
    / \     / \    / | \
  t1  t2  i1  i2  b1 b2 b3
```

**关键假设：** 创意的CTR可以分解为沿树路径的各节点贡献之和：
```
CTR(creative) = Σ_{node ∈ path} θ_node
```
或更一般地，CTR由树上各节点的参数决定，且具有可分解性。

### 2. 动态规划选择最优创意

给定树结构的可分解性，最优创意选择可通过动态规划在O(N)时间完成（N为节点总数），而非枚举所有∏n_k种组合：

**DP递推：**
```
V(node) = θ_node + max_{child ∈ children(node)} V(child)
```
自底向上计算，最终 V(root) 即为最优创意的预估CTR。

### 3. Thompson Sampling + 动态规划

**传统TS的问题：** 对每个组合独立维护后验分布，组合数太多不可行。

**AES的创新：** 在树节点层面维护后验分布，而非组合层面：
- 每个节点 node 维护参数后验：θ_node ~ N(μ_node, σ²_node)
- Thompson Sampling时，从各节点后验中采样 θ̃_node
- 用动态规划在采样参数上找最优组合

**算法流程：**
```
for each round t:
    1. 对每个节点，采样 θ̃_node ~ N(μ_node, σ²_node)
    2. 用DP找到采样CTR最大的创意组合
    3. 展示该创意，观察点击反馈
    4. 更新路径上各节点的后验参数
```

### 4. 方差缩减

由于反馈稀疏，直接估计节点参数方差大。AES使用以下技巧：
- **共享统计信息：** 同一节点出现在多个组合中，汇聚所有包含该节点的组合反馈
- **层次先验：** 同类型节点共享先验信息
- **逐步收敛：** 随着曝光增加，后验方差自然缩小

## 系统架构

```
广告请求 → 可用素材集合
        → AES模型：
            1. 构建/更新素材树结构
            2. 各节点后验采样
            3. DP求解最优组合
            4. 返回最优创意
        → 展示 & 收集反馈
        → 更新路径节点后验
```

**复杂度优势：**
- 暴力TS：O(∏n_k) 每轮
- AES (TS+DP)：O(N) 每轮，N = Σn_k（节点总数）

## 关键实验结果

**合成数据：**
- AES在收敛速度上显著优于独立bandit和全组合枚举
- 方差缩减技术使得在有限样本下即可接近最优

**真实数据（阿里妈妈广告平台）：**
- 在CTR和累积遗憾上均优于competing baselines
- 收敛速度更快，更快锁定最优创意组合

## 与我们系统的关联

- **组合爆炸问题的优雅解法：** 我们的创意系统也面临多元素组合的空间爆炸，树结构+DP是工程可落地的方案
- **节点级别的参数共享：** 同一标题/图片在不同组合中共享统计信息，大幅加速学习——这对素材复用场景非常有价值
- **实时选择的低延迟：** O(N) 的DP计算可以满足在线服务的延迟要求

## 核心公式

```
# 树结构CTR分解
CTR(creative) = Σ_{node ∈ path(creative)} θ_node

# DP递推
V(node) = θ_node + max_{child} V(child)

# 节点后验 (Gaussian)
θ_node ~ N(μ_node, σ²_node)

# TS采样 + DP选择
θ̃_node ~ N(μ_node, σ²_node), ∀ node
creative* = DP_argmax(θ̃)

# 后验更新 (观察到点击y, 路径节点集合S)
μ_node ← (n·μ_node + y) / (n+1),  ∀ node ∈ S
σ²_node ← σ²_node / (n+1),  ∀ node ∈ S
```

---

<a id="3-vam-hbm"></a>
# 3. VAM-HBM: A Hybrid Bandit Model with Visual Priors for Creative Ranking in Display Advertising

> 来源: WWW 2021 | arXiv: 2102.04033
> 作者: Shiyao Wang, Qi Liu, Tiezheng Ge, Defu Lian, Zhiqiang Zhang (阿里妈妈)
> 核心贡献: 用视觉模型为创意排序提供冷启动先验，再通过混合Bandit模型在线演化，解决创意冷启动问题

## 问题背景

电商展示广告中，卖家会为同一商品设计多个创意（不同的商品主图、排版、配色等）。系统需要动态选择展示哪个创意以最大化CTR。

**核心挑战：**
- 创意的冷启动问题比商品推荐更严重 — 创意更新频率极高
- 新创意没有历史数据，传统bandit方法需要大量探索才能收敛
- 创意的视觉外观包含丰富的CTR相关信息，但传统bandit忽略了这些信号

## 核心方法

### 1. Visual-Aware Model (VAM) — 视觉感知排序模型

**目标：** 根据创意的视觉外观预测其相对吸引力，为新创意提供CTR先验。

**架构：**
- 骨干网络：ResNet-18（也可用ResNet-50/101获得更好效果）
- 输入：创意图片
- 输出：视觉得分（用于排序）

**训练损失 — List-wise Ranking Loss：**
不是预测绝对CTR，而是学习创意之间的相对排序：
```
L_rank = -Σ_i [y_i · log(softmax(s_i))]
```
其中 s_i 是模型对创意i的打分，y_i 是基于真实CTR的排序标签。

**点击率回归正则化：**
为使视觉模型输出接近真实CTR（稳定bandit学习），加入point-wise回归损失：
```
L_total = L_rank + λ · L_regression
L_regression = Σ_i (s_i - CTR_i)²
```

### 2. Hybrid Bandit Model (HBM) — 混合Bandit模型

**核心思想：** 用VAM的视觉评估作为先验，通过贝叶斯线性回归在线更新。

**特征表示：**
从VAM中提取创意的特征表示 φ(c)（ResNet中间层特征）。

**贝叶斯线性回归：**
```
CTR(c) = w^T · φ(c) + ε,  ε ~ N(0, σ²)
```

**先验设置（来自VAM）：**
```
w ~ N(μ_0, Σ_0)
```
其中 μ_0, Σ_0 由VAM的预训练结果决定 — 这就是"视觉先验"。

**后验更新（在线观察后）：**
```
Σ_n = (Σ_0^{-1} + (1/σ²) · Φ^T Φ)^{-1}
μ_n = Σ_n · (Σ_0^{-1} · μ_0 + (1/σ²) · Φ^T y)
```

**Thompson Sampling决策：**
```
w̃ ~ N(μ_n, Σ_n)
creative* = argmax_c  w̃^T · φ(c)
```

### 3. 在线演化流程

```
阶段1（冷启动）：VAM视觉先验主导 → 已能给出合理排序
阶段2（有少量数据）：HBM在VAM先验基础上用观察数据更新后验
阶段3（数据充足）：后验趋近真实CTR分布，bandit高效利用
```

## 数据集 — CreativeRanking

**首个大规模创意排序数据集：**
- 170万+ 创意，涵盖50万商品
- 包含真实曝光和点击数据
- 每个商品有多个创意候选
- 已开源（阿里天池平台）

## 关键实验结果

**离线评估指标：**
- **sCTR（simulated CTR）：** 模拟在线选择过程的CTR
- **累积遗憾（Cumulative Regret）：** 与最优策略的差距

**关键发现：**
- VAM-Greedy（仅用视觉先验的贪心策略）在初期就能带来约**5% CTR提升** — 证明视觉先验有效
- VAM-HBM（加入bandit探索后）显著超越所有baseline
- 在CreativeRanking数据集和Mushroom公开数据集上均表现最优

**在线部署（阿里展示广告）：**
- 在生产系统中验证了有效性
- 视觉先验大幅缩短了冷启动时间

## 与我们系统的关联

- **冷启动解决方案：** 我们的创意生成系统同样面临新创意缺乏数据的问题，VAM提供了"看一眼就知道好不好"的能力
- **视觉特征的价值：** 证明了创意的视觉质量可以被模型学习，并作为有效先验——我们可以在创意生成后用类似方法预筛选
- **Bandit框架的工程化：** 贝叶斯线性回归的后验更新计算量小，适合在线服务
- **评估指标设计：** sCTR这种模拟评估方法对我们的离线评估体系有参考价值

## 核心公式

```
# VAM训练目标
L_total = L_listwise_rank + λ · L_pointwise_regression

# HBM贝叶斯线性回归
CTR(c) = w^T · φ(c) + ε
先验: w ~ N(μ_0, Σ_0)  # 来自VAM
后验: w|D ~ N(μ_n, Σ_n)
Σ_n = (Σ_0^{-1} + (1/σ²)Φ^TΦ)^{-1}
μ_n = Σ_n(Σ_0^{-1}μ_0 + (1/σ²)Φ^Ty)

# Thompson Sampling
w̃ ~ N(μ_n, Σ_n)
creative* = argmax_c w̃^T · φ(c)
```

---

<a id="4-peri-cr"></a>
# 4. Peri-CR: Parallel Ranking of Ads and Creatives in Real-Time Advertising Systems

> 来源: AAAI 2024 | arXiv: 2312.12750
> 作者: Zhiguang Yang, Lu Wang, Chun Gan, et al.
> 核心贡献: 提出广告排序与创意排序并行执行的架构（Peri-CR），消除串行依赖，在不增加延迟的前提下大幅提升CTR

## 问题背景

实时广告系统需要同时完成两个任务：
1. **广告排序（Ad Ranking, AR）：** 从候选广告中选出Top-L
2. **创意排序（Creative Ranking, CR）：** 为每个广告选择最优创意

传统架构有两种：
- **Post-CR（串行后置）：** 先排广告，再选创意 — 创意模块时间预算不足
- **Pre-CR（串行前置）：** 先选创意，再排广告 — 总延迟增加18.9%

## 核心方法

### 1. Peri-CR架构 — 并行执行

**核心思想：** 广告排序和创意排序并行执行，互不等待。

```
请求到达 ─┬─→ [Ad Ranking Module]  ─┬─→ 合并结果 → 返回
           └─→ [Creative Ranking Module] ─┘
```

**关键问题：** CR模块需要知道哪些广告被选中（依赖AR结果），如何并行？

**解决方案：** CR模块使用**历史统计CTR**代替实时pctr_ad作为输入，消除对AR模块的依赖。

### 2. JAC — 离线联合优化模型

虽然在线并行，但离线训练时通过联合模型传递信息：

**两塔级联架构：**

**AR塔（广告排序）：**
- 输入：30+ 用户特征, 40+ 广告特征, 300+ 交叉特征
- 模型：2层Transformer (2 attention heads) + DCN (4层: 512-512-256-128)
- 输出：pctr_ad

**CR塔（创意排序）：**
- 输入：11 用户特征, 5 广告特征, 创意ID + 内容特征
- 模型：3层MLP (128-64-32)
- 额外输入：pctr_ad 的量化embedding
- 输出：pctr_c

**pctr_ad量化公式：**
```
bucket = ⌊K · log_{r+1}(1 + r · pctr_ad)⌋
```
K=8192（embedding大小），r为信息增益超参数。对数量化保留非线性分布特性。

**梯度传递：** 量化操作不可微，使用**梯度直通（Straight-Through Estimator）**：
将CR的梯度直接复制到AR，实现联合优化。

### 3. 核心数学推导

**联合概率分解：**
```
p(y=1|x) · p(c|x,y=1) = p(c|x) · p(y=1|x,c)
```

**创意分布建模：**
```
p(c|x) = softmax(p(y|x,c) / Σ_{c'} p(y|x,c'))
```

### 4. NSCTR — 新评估指标

传统sCTR与线上效果相关性差。提出Normalized sCTR：
```
NSCTR: 通过广告分布比 (Imp_A_m / Imp^s_A_m) 对每个广告的创意样本重新加权
```
**NSCTR与线上CTR的相关系数：0.988**（vs sCTR: 0.636, AUC: 0.741, GAUC: -0.152）

## 系统架构

```
┌───────────────────────────────────────────────┐
│              在线推理 (90ms总延迟)               │
│                                                │
│  请求 ──┬── AR模块 (30+用户, 40+广告, 300+交叉)  │
│         │   2层Transformer + 4层DCN             │
│         │   输出: pctr_ad, Top-L广告             │
│         │                                        │
│         └── CR模块 (11用户, 5广告, 创意特征)      │
│             3层MLP (128-64-32)                   │
│             输入: 历史统计CTR (非实时pctr_ad)      │
│             输出: 每个广告的最优创意               │
│         ──→ 合并 ──→ 返回                        │
└───────────────────────────────────────────────┘

┌───────────────────────────────────────────────┐
│              离线训练 (JAC联合优化)               │
│                                                │
│  AR塔 ─── pctr_ad ──→ 量化 ──→ CR塔            │
│    ↑                              │              │
│    └──── 梯度直通 (STE) ←─────────┘              │
└───────────────────────────────────────────────┘
```

## 关键实验结果

### 在线A/B测试（5天连续测试）

**架构对比：**
| 方法 | CTR | RPM | 响应时间 |
|------|-----|-----|----------|
| no-CR（无创意排序） | baseline | baseline | 90ms |
| Post-CR（串行后置） | +6.25% | +5.63% | 94ms (+4.4%) |
| Pre-CR（串行前置） | +8.54% | +6.81% | 107ms (+18.9%) |
| **Peri-CR+（并行）** | **+10.12%** | **+7.67%** | **90ms (无增加)** |

**JAC联合优化效果：**
| 方法 | CTR | RPM |
|------|-----|-----|
| Peri-CR → Peri-CR+ | +0.92% | +0.52% |
| AR → AR+ | +0.78% | +0.55% |

### 离线实验

**数据规模：**
- 训练: 180亿样本（5月1日-6月30日）
- 测试: 3亿样本，5300万用户，1600万广告，5400万创意
- 平均每个创意仅6次曝光
- 整体CTR: 2.4%

**创意排序效果（NSCTR）：**
| 方法 | NSCTR |
|------|-------|
| VAM-HBM | 0.2267 |
| CR (Peri-CR的CR模块) | 0.2392 (+5.5%) |
| JAC | 0.2427 (+7.0%) |

## 与我们系统的关联

- **并行架构核心参考：** Peri-CR直接解决了我们面临的"创意选择增加延迟"问题，是系统设计的关键参考
- **量化embedding传递：** 对数量化+STE梯度直通的方法可用于我们的多模块联合优化
- **NSCTR评估指标：** 解决了离线评估与在线效果不一致的问题，我们应该采用类似的归一化评估方法
- **工程落地细节丰富：** embedding维度(AR:16, CR:4)、批次大小(512)、优化器(Adagrad 0.05)等直接可参考

## 核心公式

```
# 联合概率分解
p(y=1|x) · p(c|x,y=1) = p(c|x) · p(y=1|x,c)

# pctr量化
bucket = ⌊K · log_{r+1}(1 + r · pctr_ad)⌋,  K=8192

# 创意分布
p(c|x) = exp(f(x,c)) / Σ_{c'} exp(f(x,c'))

# 梯度直通 (STE)
前向: bucket = quantize(pctr_ad)  # 不可微
反向: ∂L/∂pctr_ad = ∂L/∂bucket    # 直接复制梯度
```

---

<a id="5-amazon-mvt"></a>
# 5. Amazon MVT: An Efficient Bandit Algorithm for Realtime Multivariate Optimization

> 来源: KDD 2017 (Audience Appreciation Award) | arXiv: 1810.09558
> 作者: Daniel N. Hill, Houssam Nassif, Yi Liu, Anand Iyer, S.V.N. Vishwanathan (Amazon)
> 核心贡献: 将多变量页面优化建模为带交互项的线性Bandit，用爬山法高效实时求解，在Amazon获得21%转化率提升

## 问题背景

优化网页布局涉及多个组件的联合决策（标题、图片、按钮、超链接等）。传统A/B测试需要枚举所有组合（如48种布局需要66天），效率极低。

**核心挑战：**
- D个组件，每个有N个选项 → N^D种布局，组合爆炸
- 组件之间存在交互效应（如标题+图片的配合）
- 需要实时决策，延迟 < 10ms
- 传统MVT（多变量测试）样本效率太低

## 核心方法

### 1. 带交互的线性奖励模型

**奖励模型：**
```
E[reward | layout A, context X] = μ^T · B(A, X)
```
其中：
- A = (a_1, ..., a_D) 是布局向量，D维，每个a_d是组件d的选择
- X = 上下文（用户/session信息）
- B(A, X) ∈ R^M 是特征向量，包含：
  - 主效应：每个组件选择的one-hot编码
  - **二阶交互：** 组件对之间的交互特征
  - （忽略高阶效应以避免组合爆炸）
- μ ∈ R^M 是未知权重

### 2. Thompson Sampling

**后验维护（贝叶斯线性回归）：**
```
μ ~ N(μ_n, Σ_n)
Σ_n = (Σ_0^{-1} + (1/σ²) · Φ^T Φ)^{-1}
μ_n = Σ_n · (Σ_0^{-1} · μ_0 + (1/σ²) · Φ^T y)
```

每轮：
1. 采样 μ̃ ~ N(μ_n, Σ_n)
2. 选择布局 A* = argmax_A μ̃^T · B(A, X)
3. 观察奖励，更新后验

### 3. 爬山法（Hill Climbing）求解

**问题：** 步骤2中的 argmax 是NP-hard（最大边权重团问题）。

**解决方案 — 多次随机爬山：**
```
for s = 1 to S:
    A ← random_layout()          # 随机初始化
    while improving:
        for d = 1 to D:            # 遍历每个组件
            a_d ← argmax_{a} μ̃^T · B(A[d←a], X)  # 贪心替换
    candidates[s] ← A
return argmax candidates          # S次中取最优
```

**计算复杂度：** O(S · D · N · M)，远小于 O(N^D)

**实验验证：** 爬山法相比base Thompson Sampling额外带来10-15%提升。

### 4. 个性化扩展

- 上下文特征X可以包含用户画像（人口统计、行为特征等）
- 为每个用户segment维护独立后验（或层次化结构）
- 不同用户群可以收敛到不同的最优布局

## 系统架构

```
用户请求(含context X)
    → Thompson Sampling采样 μ̃
    → S次爬山法求解最优布局 A*
    → 展示布局
    → 收集转化/点击反馈
    → 更新贝叶斯后验

模型更新: 每天批量更新后验（非实时）
在线决策: ~10ms延迟
```

## 关键实验结果

### Amazon真实部署

**实验设置：**
- 5个组件（widget）：标题、图片、要点、按钮、超链接
- 48种布局组合（2×2×2×2×3）
- 12天在线实验，每天数万次曝光
- 对比方法：N^D-MAB（48臂独立bandit）、D-MAB（每组件独立bandit）

**核心结果：**
- **较中位布局提升21%** 转化率
- 较最差布局提升44%
- MVT2（本文方法）显著优于N^D-MAB和D-MAB
- 仅1周即收敛 vs A/B测试需66天

**移动端实验（32种布局）：**
- 发现显著的二阶交互效应，证明了建模交互的必要性

### 模拟实验

- Thompson Sampling较UCB和ε-greedy减少20-30%累积遗憾
- 爬山法额外提供10-15%增益

**理论遗憾界：**
```
Regret = Õ(d√T)
```
d为特征维度，T为总轮次。

## 与我们系统的关联

- **最直接的工程参考：** 我们的创意优化本质上就是多变量优化（标题×图片×CTA×配色），这篇论文提供了完整的工业级方案
- **交互建模的必要性：** 实验证明忽略元素间交互会显著降低效果，我们的系统需要建模元素组合效应
- **爬山法的实用性：** 提供了在NP-hard组合空间中的实用近似解法，延迟可控
- **收敛速度：** 1周收敛 vs 66天A/B测试，这对快速迭代的广告场景极有价值
- **个性化：** 不同用户可以看到不同的最优创意组合

## 核心公式

```
# 线性奖励模型（含交互）
E[r|A,X] = μ^T · B(A,X)
B(A,X) = [主效应; 二阶交互; 上下文交叉]

# 贝叶斯线性回归后验
Σ_n = (Σ_0^{-1} + (1/σ²)Φ^TΦ)^{-1}
μ_n = Σ_n(Σ_0^{-1}μ_0 + (1/σ²)Φ^Ty)

# Thompson Sampling + Hill Climbing
μ̃ ~ N(μ_n, Σ_n)
A* = HillClimb(μ̃, X, S_restarts)

# 遗憾界
Regret = Õ(d√T)
```

---

<a id="6-ttts"></a>
# 6. TTTS: Efficient Creative Selection in Online Advertising using Top-Two Thompson Sampling

> 来源: WSDM 2025 | DOI: 10.1145/3701551.3706128
> 作者: Daiki Katsuragawa, Yusuke Kaneko, Kaito Ariu, Kenshi Abe (CyberAgent, Tokyo)
> 核心贡献: 将Top-Two Thompson Sampling应用于广告创意选择，相比传统A/B测试在识别最优创意的准确性和实验成本上均有显著优势

## 问题背景

在线广告平台需要从多个候选创意中选出最优的一个。传统做法是A/B测试（均匀流量分配），但存在两个问题：
1. **效率低：** 均匀分配大量流量给明显较差的创意
2. **成本高：** 需要大量样本才能达到统计显著性

**核心需求：** 这是一个**Best Arm Identification (BAI)** 问题 — 目标不是最大化累积奖励，而是用最少的样本准确识别最优臂。

## 核心方法

### 1. Top-Two Thompson Sampling (TTTS) 算法

TTTS是标准Thompson Sampling的变体，专为Best Arm Identification设计：

**标准TS vs TTTS：**
- **标准TS：** 从后验采样，选采样值最高的臂 → 倾向利用（exploitation）
- **TTTS：** 从后验采样，但在前两名之间随机选择 → 更多探索有竞争力的臂

**算法流程：**
```
for each round t:
    1. 对每个创意arm i, 从后验采样: θ̃_i ~ p(θ_i | D_t)
    2. 找到前两名:
       I_1 = argmax_i θ̃_i        # 采样值最高
       I_2 = argmax_{i≠I_1} θ̃_i   # 采样值次高
    3. 以概率 β 选择 I_1，以概率 (1-β) 选择 I_2
    4. 展示选中创意，观察反馈
    5. 更新该创意的后验分布
```

### 2. β 参数

- β 通常取 **0.5**（前两名等概率被选中）
- β = 1.0 退化为标准Thompson Sampling（更贪心）
- β = 0.5 时，理论保证最优指数收敛率的因子2以内

**理论性质：**
```
指数收敛: P(最优臂识别错误) → 0, 速率为 exp(-Ω(t))
```
当β=0.5时，渐近最优（在一定条件下）。

### 3. 后验更新（Bernoulli奖励）

对于点击/未点击的Bernoulli反馈，使用Beta分布：
```
先验: θ_i ~ Beta(α_i, β_i), 初始化 α_i = β_i = 1 (均匀先验)
观察点击(1): α_i ← α_i + 1
观察未点击(0): β_i ← β_i + 1
后验: θ_i | D ~ Beta(α_i, β_i)
```

### 4. 停止准则

当对最优创意的置信度足够高时停止实验：
```
停止条件: P(arm i* 是最优 | D) > 1 - δ
```
其中 δ 是预设的错误容忍度（如0.05）。

## TTTS vs 传统A/B测试对比

| 特性 | A/B测试 | TTTS |
|------|---------|------|
| 流量分配 | 均匀 | 自适应 |
| 探索效率 | 低（大量流量给差创意） | 高（聚焦有竞争力的创意） |
| 样本需求 | 固定，通常很大 | 自适应，通常更少 |
| 停止条件 | 预定义样本量/时间 | 基于后验置信度 |
| 目标 | 统计假设检验 | Best Arm Identification |

## 关键实验结果

**在CyberAgent在线广告平台上的实验：**
- TTTS在准确识别最优创意方面优于A/B测试和标准Thompson Sampling
- 显著降低了实验成本（所需样本量更少）
- 在多种创意差异程度的场景下均有效

**模拟实验对比：**

| 场景 | 均匀随机 | 标准TS | TTTS |
|------|---------|--------|------|
| 明显最优 | 0.99 | 1.00 | 1.00 |
| 差距不大 | 0.92 | 0.99 | 1.00 |
| 竞争激烈 | 0.82 | 0.92 | 0.92 |

TTTS在差距不大的场景（最常见的实际情况）优势最明显。

## 系统架构

```
创意候选池 [c1, c2, ..., cn]
    → 维护每个创意的Beta后验: Beta(α_i, β_i)
    → 每次曝光:
        1. 从各后验采样
        2. TTTS: 前两名coin flip
        3. 展示 & 收集反馈
        4. 更新后验
    → 达到停止条件时:
        选择后验均值最大的创意作为winner
        全量流量切换到winner
```

## 与我们系统的关联

- **最简洁的创意A/B测试替代方案：** 实现极其简单（Beta分布+采样+coin flip），几乎无工程成本
- **适用场景：** 当我们需要在少量候选创意中选最优时（如LLM生成了5个候选标题），TTTS是最佳选择
- **与Bandit方法互补：** TTTS适合"选出最好的然后全量上"，而TS/UCB适合"持续探索和利用"——两者覆盖不同场景
- **实验成本优化：** 对广告主来说，减少在差创意上的花费直接提升ROI
- **快速决策：** 在CyberAgent的实际部署中证明了快速收敛的能力

## 核心公式

```
# TTTS算法
θ̃_i ~ Beta(α_i, β_i),  ∀i
I_1 = argmax_i θ̃_i
I_2 = argmax_{i≠I_1} θ̃_i
选择 I_1 w.p. β, I_2 w.p. (1-β),  β=0.5

# Beta后验更新
点击: α_i ← α_i + 1
未点击: β_i ← β_i + 1

# 停止条件
P(i* = argmax_j E[θ_j] | D) > 1 - δ

# 理论保证
P(错误) ≤ exp(-Ω(t)),  渐近最优(β=0.5)
```

---

# 论文间关系与技术演进总结

```
                    创意优化技术谱系

[问题类型]        [方法]              [论文]

单创意选择         A/B Testing替代     → TTTS (WSDM'25)
(BAI问题)         Best Arm Id

多创意排序         视觉先验+Bandit     → VAM-HBM (WWW'21)
(Ranking问题)     贝叶斯线性回归

元素交互建模       AutoML搜索          → AutoCO (WWW'21)
(交互发现)        变分推断+TS

组合空间优化       树结构+DP           → AES (AAAI'21)
(组合爆炸)        节点级TS

多变量布局优化     线性Bandit+爬山     → Amazon MVT (KDD'17)
(页面优化)        交互建模+个性化

系统架构           并行排序+联合训练   → Peri-CR (AAAI'24)
(延迟优化)        量化传递+STE
```

## 面试核心问答框架

**Q: 如何为DCO系统选择创意优化策略？**

A: 根据场景分层：
1. **少量候选选最优** → TTTS（简单高效的BAI）
2. **新创意冷启动** → VAM-HBM（视觉先验快速给出合理排序）
3. **元素组合优化** → AES树结构DP（解决组合爆炸）或Amazon MVT（带交互的线性Bandit）
4. **交互函数未知** → AutoCO（自动搜索最优交互方式）
5. **系统延迟受限** → Peri-CR并行架构（创意排序不增加延迟）

**Q: Thompson Sampling在这些论文中的不同变体？**

| 论文 | TS变体 | 后验分布 | 特点 |
|------|--------|----------|------|
| TTTS | Top-Two TS | Beta | BAI专用，前两名coin flip |
| AutoCO | 变分TS | Gaussian (VI) | 用变分推断近似后验 |
| AES | TS + DP | Gaussian | 节点级采样+DP求解 |
| VAM-HBM | TS + 贝叶斯LR | Gaussian | 视觉先验初始化 |
| Amazon MVT | TS + Hill Climbing | Gaussian (BLR) | 爬山法近似argmax |
| Peri-CR | - | - | 使用DNN直接预测，非bandit |
