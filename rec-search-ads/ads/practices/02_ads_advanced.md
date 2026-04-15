# 广告系统进阶技术笔记

> 涵盖CVR/CTCVR预估、出价策略、多目标优化、探索利用、工业级架构五大核心模块

---

## 一、CVR/CTCVR预估

### 1.1 基础定义

| 指标 | 定义 | 公式 | 说明 |
|------|------|------|------|
| **CTR** | 点击率 | $\text{CTR} = \frac{\text{点击数}}{\text{曝光数}}$ | 预估用户点击广告的概率 |
| **CVR** | 转化率 | $\text{CVR} = \frac{\text{转化数}}{\text{点击数}}$ | 预估用户点击后转化的概率 |
| **CTCVR** | 点击且转化率 | $\text{CTCVR} = \frac{\text{转化数}}{\text{曝光数}}$ | 从曝光到转化的整体概率 |

**核心关系式：**

$$
\text{CTCVR} = \text{CTR} \times \text{CVR}
$$

### 1.2 样本空间偏差（SSB - Sample Selection Bias）

#### 问题定义

CVR建模面临**双重样本选择偏差**：

```
┌─────────────────────────────────────────────────────────┐
│                    曝光样本空间                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │              点击样本空间（CTR>0）               │   │
│  │  ┌─────────────────────────────────────────┐   │   │
│  │  │        转化样本空间（CVR正样本）         │   │   │
│  │  │         （只占点击的1-5%）               │   │   │
│  │  └─────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**偏差来源：**
1. **Training-Serving 不一致**：训练用点击样本，推理用曝光样本
2. **稀疏性问题**：CVR正样本仅占曝光量的 0.01%-0.1%
3. **选择偏差**：点击样本本身是高偏差的（有偏用户群体）

#### 数学表达

设 $x$ 为特征，$y \in \{0,1\}$ 为转化标签，$s \in \{0,1\}$ 表示是否点击

$$
\underbrace{P(y=1|x)}_{\text{期望建模}} \neq \underbrace{P(y=1|x, s=1)}_{\text{实际建模}}
$$

### 1.3 ESMM 模型（Entire Space Multi-Task Model）

#### 核心思想

**多任务联合学习**：用 CTR 和 CTCVR 任务辅助学习 CVR

```
┌─────────────────────────────────────────────────────────────┐
│                        特征输入                               │
│                    (User + Ad + Context)                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
    ┌─────────────────┐         ┌─────────────────┐
    │   Embedding层    │         │   Embedding层    │
    └────────┬────────┘         └────────┬────────┘
             │                           │
             ▼                           ▼
    ┌─────────────────┐         ┌─────────────────┐
    │    CTR塔        │         │   CVR塔（隐藏）   │
    │  (独立Embedding) │         │  (共享Embedding) │
    └────────┬────────┘         └────────┬────────┘
             │                           │
             ▼                           ▼
        pCTR = ŷ₁                   pCVR = ŷ₂
             │                           │
             └───────────┬───────────────┘
                         ▼
              ┌─────────────────┐
              │  pCTCVR = ŷ₁ × ŷ₂ │  ← 乘积形式
              └─────────────────┘
```

#### 损失函数

**CTR 任务损失**（曝光样本空间）：

$$
L_{CTR} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i^{CTR} \log(\hat{y}_i^{CTR}) + (1-y_i^{CTR}) \log(1-\hat{y}_i^{CTR}) \right]
$$

**CTCVR 任务损失**（曝光样本空间）：

$$
L_{CTCVR} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i^{CVR} \log(\hat{y}_i^{CTR} \cdot \hat{y}_i^{CVR}) + (1-y_i^{CVR}) \log(1-\hat{y}_i^{CTR} \cdot \hat{y}_i^{CVR}) \right]
$$

**总损失**（加权）：

$$
L_{ESMM} = L_{CTR} + \alpha \cdot L_{CTCVR} + \lambda \cdot \Omega(\theta)
$$

其中 $\alpha$ 通常取 1，$\Omega(\theta)$ 为正则项。

#### ESMM 优势

| 优势 | 说明 |
|------|------|
| **全空间建模** | CTCVR 在曝光空间学习，解决 SSB |
| **隐式迁移** | CTR 任务为 CVR 提供丰富监督信号 |
| **特征共享** | 底层 Embedding 共享，缓解稀疏问题 |
| **值域保证** | 乘积形式天然保证 $pCTCVR \leq pCTR$ |

### 1.4 延迟转化问题（Delayed Feedback）

#### 问题背景

转化发生在点击后的**任意时间**（几秒到30天），导致训练标签不确定：

```
时间线：
────────────────────────────────────────────────────────►
曝光    点击    可能转化窗口               最终标签确定
  │      │    ═══════════════════════               │
  ▼      ▼           ▼          ▼                   ▼
  ○──────○──────────●──────────●────────────────────■
       ↑          ↑          ↑
     训练时      转化1      转化2
     看到的    （3天后）  （7天后）
     状态
```

#### 解决方案

**方案1：等待窗口（Waiting Window）**

```python
# 伪代码：等待固定窗口期
WAITING_DAYS = 7  # 等待7天

def get_label(event_time, current_time, has_conversion):
    if has_conversion and conversion_time - event_time <= WAITING_DAYS:
        return 1
    elif current_time - event_time > WAITING_DAYS:
        return 0 if not has_conversion else 1
    else:
        return None  # 样本未ready，不入库
```

- **缺点**：数据新鲜度下降（延迟7天）

**方案2：FFM（Faked-Feedback Model）/ 延迟反馈模型**

引入**转化时间分布**建模：

$$
P(y=1, d | x) = \underbrace{P(y=1|x)}_{\text{转化概率}} \cdot \underbrace{P(d|y=1, x)}_{\text{延迟分布}}
$$

其中 $d$ 为延迟天数，$P(d|y=1, x)$ 可用指数分布建模：

$$
P(d|y=1, x) = \lambda(x) \exp(-\lambda(x) \cdot d)
$$

**方案3：Importance Sampling（ESDF）**

利用重要性采样纠正延迟偏差：

$$
L_{real} = \mathbb{E}_{D_{real}}[\ell(f(x), y)] = \mathbb{E}_{D_{obs}}\left[ \frac{P_{real}(x,y)}{P_{obs}(x,y)} \ell(f(x), y) \right]
$$

### 1.5 常见考点

| 考点 | 答案要点 |
|------|----------|
| **ESMM 为什么能缓解SSB？** | CTCVR 在全曝光空间学习，CVR 塔通过共享隐式学习，推理时在全空间 |
| **ESMM 的 CVR 塔标签从哪来？** | 点击样本的转化标签（0/1），但损失通过 CTCVR 传播 |
| **为什么 pCTCVR = pCTR × pCVR？** | 数学上 $P(cv|impression) = P(cv|click) \cdot P(click|impression)$ |
| **延迟转化怎么处理？** | 等待窗口、时间分布建模、重要性采样、Online Learning |
| **CVR 样本稀疏怎么办？** | ESMM 共享学习、负采样、对 CVR 塔单独加权、Focal Loss |

---

## 二、广告出价策略

### 2.1 oCPM 定义

**oCPM（Optimized Cost Per Mille）**：优化千次展现成本，按曝光计费，但优化目标是转化

```
计费点 vs 出价点 vs 优化点
┌────────────┬────────────────────────────────────────┐
│   计费点    │ 实际扣费的事件（曝光/点击）              │
├────────────┼────────────────────────────────────────┤
│   出价点    │ 广告主表达意愿的事件（转化/付费）         │
├────────────┼────────────────────────────────────────┤
│   优化点    │ 系统算法优化的目标（ROI/留存）            │
└────────────┴────────────────────────────────────────┘

oCPM: 计费=曝光, 出价=转化, 优化=转化成本
```

### 2.2 eCPM 计算公式

**标准 eCPM 公式**：

$$
\text{eCPM} = \text{bid} \times pCTR \times pCVR \times \text{智能调整因子} \times 1000
$$

**展开形式**（以 oCPM 为例）：

$$
\text{eCPM} = \underbrace{\text{Bid}_{	ext{tCPA}}}_{\text{广告主目标转化出价}} \times \underbrace{pCTR}_{\text{预估CTR}} \times \underbrace{pCVR}_{\text{预估CVR}} \times \underbrace{\beta}_{\text{成本控制系数}} \times 1000
$$

**广义第二价格（GSP）计费**：

$$
\text{Charge} = \frac{\text{eCPM}_{	ext{第二名}} + 1}{pCTR_{当前} \times 1000}
$$

### 2.3 tCPA / tROAS 智能出价

#### tCPA（target Cost Per Acquisition）

目标：让实际转化成本接近广告主设定的目标

** pacing 调整公式**：

$$
\beta_t = \frac{\text{Target CPA}}{\text{Actual CPA}_{	ext{t-1}}} \cdot \alpha
$$

其中 $\alpha$ 为平滑因子（如0.8），防止剧烈波动。

**出价调整**：

$$
\text{Bid}_{	ext{adjusted}} = \text{Bid}_{	ext{original}} \times \min(\max(\beta_t, 0.5), 2.0)
$$

限制在 [0.5x, 2.0x] 防止过度调整。

#### tROAS（target Return On Ad Spend）

目标广告支出回报率 = 广告主收入 / 广告成本

$$
\text{ROAS} = \frac{\text{广告带来的收入}}{\text{广告花费}} \times 100\%
$$

**动态出价公式**：

$$
\text{Bid}_{	ext{ROAS}} = \text{Bid}_{	ext{base}} \times \frac{\text{预估订单金额}}{\text{Target ROAS} \times \text{目标CPA}}
$$

### 2.4 智能出价优化（KKT条件约束）

#### 问题建模

最大化平台收益，满足广告主成本约束：

$$
\max_{b_i} \sum_{i} \text{Revenue}_{\text{i(b}_{\text{i)
$$

约束条件：

$$
\text{s.t.}} \quad \text{CPA}_{\text{i(b}_{\text{i) \leq \text{Target CPA}_i, \quad \forall i
$$

#### KKT 条件应用

构建拉格朗日函数：

$$
\mathcal{L}(b, \lambda) = \sum_{i} R_i(b_i) - \sum_{i} \lambda_i \left( CPA_i(b_i) - \overline{CPA}_i \right)
$$

最优条件：

$$
\frac{\partial \mathcal{L}}{\partial b_i} = 0 \Rightarrow \frac{\partial R_i}{\partial b_i} = \lambda_i \frac{\partial CPA_i}{\partial b_i}
$$

**工程意义**：
- $\lambda_i$ 为广告主 $i$ 的影子价格（约束松紧度）
- 约束紧的（频繁超成本）广告主，$\lambda_i$ 大，出价下调幅度大

### 2.5 预算 Pacing

#### 目标

平滑消耗预算，避免早期快速花完或后期花不出去：

```
理想消耗曲线：
花费
 │    ╭──────────────────────────────╮
 │   ╱ 平滑消耗                      ╲
 │  ╱                                  ╲
 │ ╱    ✗ 早期burst    ✗ 后期花不完      ╲
 │╱                                        ╲
 └───────────────────────────────────────────► 时间
    0                                    24h
```

#### 预算控制公式

**剩余预算率**：

$$
r_t = \frac{\text{Remaining Budget}}{\text{Total Budget}}
$$

**剩余时间率**：

$$
s_t = \frac{\text{Remaining Time}}{\text{Total Campaign Duration}}
$$

**Pacing 系数**：

$$
\gamma_t = \left( \frac{r_t}{s_t} \right)^k
$$

其中 $k$ 为调节参数（通常 1-2）：
- $\gamma_t > 1$：剩余预算多，加速投放
- $\gamma_t < 1$：剩余预算少，减速投放

#### PID 控制器

更精细的 pacing 可用 PID 控制：

$$
\gamma_t = K_p \cdot e_t + K_i \cdot \int_0^t e_\tau d\tau + K_d \cdot \frac{de_t}{dt}
$$

其中误差 $e_t = \text{目标消耗速率} - \text{实际消耗速率}$

### 2.6 常见考点

| 考点 | 答案要点 |
|------|----------|
| **eCPM 为什么要×1000？** | 转化为千次展现成本，数值更合理（避免小数） |
| **GSP 机制有什么好处？** | 鼓励广告主出真实价值，降低博弈复杂度，平台收益稳定 |
| **tCPA 怎么保证不超成本？** |  pacing 调节系数 + KKT 约束优化 + 历史CPA反馈调整 |
| **预算 pacing 的核心挑战？** | 流量波动、竞争变化、冷启动期预估不准 |
| **为什么用 KKT 而不是直接梯度下降？** | 需要显式处理广告主成本约束，KKT 提供带约束优化框架 |

---

## 三、多目标优化

### 3.1 帕累托前沿（Pareto Frontier）

#### 定义

在多目标优化中，**帕累托最优**指：无法在不损害其他目标的情况下改进任一目标。

```
目标2（如用户满意度）
    │
    │      ● 方案A（帕累托最优）
    │     ╱
    │    ● 方案B（帕累托最优）
    │   ╱
    │  ● 方案C（帕累托最优） ←── 帕累托前沿
    │ ╱
    │● 方案D（被支配）
    │
    └─────────────────────► 目标1（如平台收入）
```

**数学定义**：

解 $x^*$ 是帕累托最优，当且仅当不存在 $x$ 使得：

$$
\begin{cases}
f_i(x) \leq f_i(x^*), & \forall i \in \{1,2,...,m\} \\
f_j(x) < f_j(x^*), & \exists j \in \{1,2,...,m\}
\end{cases}
$$

### 3.2 多目标加权方法

#### 线性加权（Weighted Sum）

$$
L_{total} = \sum_{i=1}^{m} w_i \cdot L_i
$$

**问题**：
- 需要调参 $w_i$
- 不同任务量纲不同，简单相加不合理
- 梯度冲突：一个任务梯度大，主导更新

#### 梯度归一化（GradNorm）

动态调整权重，使各任务梯度范数相近：

$$
w_i(t) = \frac{\bar{G}}{G_i(t)} \cdot \tilde{L}_i(t)^{\alpha}
$$

其中：
- $G_i(t) = \|\nabla_\theta w_i(t) L_i(t)\|$ 为任务 $i$ 的梯度范数
- $\bar{G}$ 为平均梯度范数
- $\tilde{L}_i(t)$ 为相对损失（当前/初始）

#### 不确定性加权（Homoscedastic Uncertainty）

基于任务不确定性自动学习权重：

$$
L = \sum_{i} \frac{1}{2\sigma_i^2} L_i + \log \sigma_i
$$

其中 $\sigma_i$ 为可学习参数，表示任务 $i$ 的不确定性。

### 3.3 MMoE 架构（Multi-gate Mixture-of-Experts）

#### 结构

```
                 输入特征
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   ┌────────┐  ┌────────┐  ┌────────┐
   │Expert 1│  │Expert 2│  │Expert N│
   │ (DNN)  │  │ (DNN)  │  │ (DNN)  │
   └───┬────┘  └───┬────┘  └───┬────┘
       │           │           │
       └───────────┬───────────┘
                   ▼
   ┌─────────────────────────────────────┐
   │           Gating Network             │
   │  ┌─────────┐    ┌─────────┐         │
   │  │Gate A   │    │Gate B   │  ...    │
   │  │(Softmax)│    │(Softmax)│         │
   │  └───┬─────┘    └───┬─────┘         │
   └──────┼──────────────┼───────────────┘
          ▼              ▼
     任务A输出        任务B输出
```

#### 公式

**专家输出**：

$$
f_k(x) = \text{Expert}_{\text{k(x), \quad k = 1,2,...,K
$$

**门控权重**：

$$
g^k(x) = \text{Softmax}}(W_g^k \cdot x + b_g^k)
$$

**任务塔输入**（加权融合）：

$$
f^k(x) = \sum_{i=1}^{K} g_i^k(x) \cdot f_i(x)
$$

**优势**：
- 任务共享专家参数（减少参数量）
- 门控网络实现自适应路由
- 缓解负迁移（任务间冲突）

### 3.4 PLE 架构（Progressive Layered Extraction）

#### 核心思想

**显式分离共享专家和任务专属专家**，从浅层到深层渐进提取：

```
Layer 1:                    Layer 2:                   Layer 3:
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│ ┌───────────┐   │        │ ┌───────────┐   │        │ ┌───────────┐   │
│ │Shared Exp │───┼────────│→│Shared Exp │───┼────────│→│Shared Exp │   │
│ └───────────┘   │        │ └───────────┘   │        │ └───────────┘   │
│ ┌───────────┐   │        │ ┌───────────┐   │        │ ┌───────────┐   │
│ │Task A Exp │───┼────────│→│Task A Exp │───┼────────│→│Task A Exp │───┼──► Tower A
│ └───────────┘   │        │ └───────────┘   │        │ └───────────┘   │
│ ┌───────────┐   │        │ ┌───────────┐   │        │ ┌───────────┐   │
│ │Task B Exp │───┼────────│→│Task B Exp │───┼────────│→│Task B Exp │───┼──► Tower B
│ └───────────┘   │        │ └───────────┘   │        │ └───────────┘   │
└─────────────────┘        └─────────────────┘        └─────────────────┘
```

#### CGC（Customized Gate Control）

每一层门控只选择**共享专家 + 当前任务专家**：

$$
g^k(x) = \text{Softmax}(W_g^k \cdot [f_{shared}; f_k] + b_g^k)
$$

区别于 MMoE 选择所有专家，PLE 避免任务间干扰。

#### 损失函数

**多任务联合损失**：

$$
L = \sum_{k=1}^{K} w_k \cdot L_k + \lambda \cdot \Omega(\theta)
$$

**技巧**：
- 主任务（如CTR）给高权重
- 辅助任务（如观看时长）给低权重
- 使用 L2 或 Dropout 正则

### 3.5 常见考点

| 考点 | 答案要点 |
|------|----------|
| **MMoE vs 共享底 vs 独立模型？** | 共享底：参数量小但负迁移；独立：无负迁移但数据稀疏；MMoE：折中，自适应共享 |
| **PLE 为什么比 MMoE 好？** | 显式分离共享/专属专家，避免无效参数共享，渐进式特征提取 |
| **多目标权重怎么调？** | GradNorm动态调整、不确定性加权、人工网格搜索、帕累托采样 |
| **梯度冲突怎么处理？** | PCGrad（投影冲突梯度）、GradVac、或者PLE架构隔离 |
| **什么时候用多目标？** | 目标间有相关性（0.3-0.8），数据分布相似，有共享特征 |

---

## 四、E&E 探索利用

### 4.1 问题定义

**Exploration vs Exploitation 困境**：

```
利用（Exploit）：选当前预估最高的广告（贪婪）
    ↓
  可能错过更好的新广告
    ↓
探索（Explore）：尝试未知广告获取反馈
    ↓
  短期收益可能下降
```

**广告场景应用**：
- 新广告冷启动（无历史数据）
- 用户兴趣漂移
- 流量环境变化

### 4.2 Epsilon-Greedy

#### 算法

```python
def select_ad(ads, epsilon=0.1):
    if random.random() < epsilon:
        # 探索：随机选择
        return random.choice(ads)
    else:
        # 利用：选预估CTR最高的
        return max(ads, key=lambda ad: ad.pCTR)
```

**公式**：

$$
a_t = \begin{cases}
\text{random}(\mathcal{A}), & \text{with prob } \epsilon \\
\arg\max_a Q(a), & \text{with prob } 1-\epsilon
\end{cases}
$$

**变体**：

- **Decay Epsilon-Greedy**：$\epsilon_t = \epsilon_0 \cdot \gamma^t$，随时间衰减
- **Adaptive Epsilon**：根据估计方差调整 $\epsilon$

### 4.3 UCB（Upper Confidence Bound）

#### 核心思想

**乐观面对不确定性**：选择"可能最好"的臂（预估 + 置信区间上界）

$$
a_t = \arg\max_a \left[ \hat{\mu}_a + c \cdot \sqrt{\frac{\ln t}{N_a(t)}} \right]
$$

其中：
- $\hat{\mu}_a$：臂 $a$ 的平均收益（利用项）
- $\sqrt{\frac{\ln t}{N_a(t)}}$：置信区间宽度（探索项）
- $c$：调节参数
- $N_a(t)$：臂 $a$ 被选择次数

#### 直观解释

```
收益
 │    ┌─────┐
 │    │  B  │ ← 高不确定性，高UCB（探索）
 │    └─────┘
 │  ╭─────────────╮
 │  │      A      │ ← 高平均收益（利用）
 │  ╰─────────────╯
 │      ┌───┐
 │      │ C │ ← 低收益，少探索
 │      └───┘
 └───────────────────►
```

### 4.4 Thompson Sampling

#### 核心思想

**贝叶斯方法**：维护奖励分布的后验，采样而非点估计

```
先验：奖励 r ~ Bernoulli(θ)，θ ~ Beta(α, β)
观测：得到 n 次成功，m 次失败
后验：θ ~ Beta(α + n, β + m)
选择：从每个臂的后验采样 θ，选最大的
```

**算法**：

```python
def thompson_sampling(ads):
    samples = []
    for ad in ads:
        # Beta分布采样
        theta = np.random.beta(ad.alpha, ad.beta)
        samples.append((theta, ad))
    return max(samples)[1]

# 更新（观测到点击或未点击）
if clicked:
    ad.alpha += 1
else:
    ad.beta += 1
```

**优势**：
- 天然平衡探索利用
- 天然处理延迟反馈
- 可扩展到上下文场景（Contextual Bandit）

### 4.5 广告冷启动策略

#### 多层级冷启动

```
新广告上线
    │
    ▼
┌─────────────────┐
│   第1阶段：探索期 │  ← 强制小流量曝光（如5%）
│  (0-100曝光)     │
└────────┬────────┘
         ▼
┌─────────────────┐
│   第2阶段：学习期 │  ← 中等流量，TS/UCB决策
│  (100-1000曝光)  │
└────────┬────────┘
         ▼
┌─────────────────┐
│   第3阶段：正常期 │  ← 进入正常排序
│  (>1000曝光)     │
└─────────────────┘
```

#### 冷启动特征补充

**元学习（Meta Learning）**：

$$
\theta_{new} = f(\theta_{base}, x_{ad}) = \theta_{base} + g(x_{ad}; \phi)
$$

利用相似广告的参数快速初始化新广告。

**对比学习**：

$$
L_{contrastive} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_k \exp(\text{sim}(z_i, z_k)/\tau)}
$$

将相似广告聚集，不同广告推开。

### 4.6 常见考点

| 考点 | 答案要点 |
|------|----------|
| **UCB 公式第二项含义？** | 置信区间宽度，被选择少的臂更大，鼓励探索 |
| **Thompson Sampling vs UCB？** | TS用采样，UCB用边界；TS更鲁棒，UCB理论分析更清晰 |
| **为什么冷启动不能用纯探索？** | 平台收益损失大，广告主体验差，需要渐进式流量 |
| **Contextual Bandit 是什么？** | 考虑上下文的Bandit，如LinUCB、NeuralUCB，推荐系统常用 |
| **探索过度/不足怎么检测？** | 监控整体CTR波动、新广告收敛速度、A/B测试对比 |

---

## 五、工业级架构痛点

### 5.1 特征穿越（Feature Leakage / Data Leakage）

#### 定义

训练时使用了**不应该知道**的信息（未来信息、标签泄露）。

**典型场景**：

```
错误示例：用"用户是否转化"构造特征

特征向量：
[用户年龄, 广告类型, 用户历史点击率, 用户是否转化] ← 泄露！
                                            
模型学到：用户是否转化=1 → 必然转化
结果：离线AUC=0.99，在线效果极差
```

#### 常见泄露来源

| 泄露类型 | 示例 | 解决方案 |
|----------|------|----------|
| **标签泄露** | 用转化行为构造特征 | 严格时间窗口控制 |
| **未来信息** | 用"未来7天点击数"作为特征 | 只使用当前时刻之前的统计 |
| **全局统计泄露** | 用包含当前样本的全局CTR | 排除当前样本后再计算 |
| **ID泄露** | 用户ID、广告ID直接入模 | 哈希/脱敏，或仅用做Embedding索引 |

#### 工程实践

**时间戳严格检查**：

```python
# 特征时间戳必须 <= 样本生成时间戳
assert feature_timestamp <= sample_timestamp

# 滑动窗口统计：只用 [t-30d, t-1d] 的数据
window_features = agg(start=t-30d, end=t-1d)  # 不含当天
```

**离线/在线一致性校验**：

```python
def check_consistency(offline_features, online_features, tolerance=1e-6):
    diff = np.abs(offline_features - online_features)
    if np.any(diff > tolerance):
        raise FeatureInconsistencyError(f"Diff max: {np.max(diff)}")
```

### 5.2 延迟反馈归因（Delayed Feedback Attribution）

#### 问题

转化归因链路长，涉及多个触点：

```
用户旅程：
曝光1 → 曝光2 → 点击A → 曝光3 → 点击B → 转化
   │       │        │       │        │      │
   ▼       ▼        ▼       ▼        ▼      ▼
  助攻    助攻     归因？  助攻     归因？   最终转化

问题：转化功劳归给谁？
```

#### 归因模型

| 模型 | 规则 | 公式 |
|------|------|------|
| **Last Click** | 功劳全给最后一次点击 | $Credit = \delta_{i, last}$ |
| **First Click** | 功劳全给第一次点击 | $Credit = \delta_{i, first}$ |
| **Linear** | 平均分配给所有触点 | $Credit_i = \frac{1}{N}$ |
| **Time Decay** | 时间越近权重越高 | $Credit_i \propto e^{-\lambda(t_{end} - t_i)}$ |
| **Data-Driven** | 用Shapley值计算 | $Credit_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [v(S \cup \{i\}) - v(S)]$ |

#### 工程实践

**归因窗口期**：

```python
ATTRIBUTION_WINDOW = 7 * 24 * 3600  # 7天

def attribute_conversion(events, conversion_time):
    """
    只归因窗口期内的触点
    """
    valid_events = [
        e for e in events 
        if conversion_time - e.time <= ATTRIBUTION_WINDOW
    ]
    return distribute_credit(valid_events)
```

### 5.3 训练-服务一致性（Training-Serving Skew）

#### 问题来源

```
训练                    服务                    问题
─────────────────────────────────────────────────────────
离线数据                实时请求                 数据分布不同
批处理特征              实时计算特征              计算方式不同
Python代码              C++服务                  实现差异
历史样本                在线流量                  时间漂移
```

#### 一致性检查清单

| 检查项 | 方法 |
|--------|------|
| **特征计算一致性** | 同一份代码（如用TF Transform） |
| **分布一致性** | PSI（Population Stability Index）< 0.1 |
| **模型输出一致性** | 采样请求，对比离线/在线打分差异 |
| **Embedding一致性** | 检查ID映射是否一致 |

#### PSI 计算

$$
PSI = \sum_{i} (Actual_i - Expected_i) \times \ln\left(\frac{Actual_i}{Expected_i}\right)
$$

- PSI < 0.1：分布基本不变
- 0.1 ≤ PSI < 0.25：轻微变化，需关注
- PSI ≥ 0.25：显著变化，需重新训练

### 5.4 广告系统完整链路

#### 漏斗架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         请求进入                                │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  召回层（Recall）                                                │
│  ├─ 定向过滤：用户画像 ∩ 广告定向                                │
│  ├─ 触发策略：协同过滤、向量检索（Annoy/Faiss）                  │
│  └─ 输出：~1000 候选广告                                         │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  粗排层（Coarse Ranking）                                        │
│  ├─ 轻量模型：LR、浅层DNN、双塔模型                              │
│  ├─ 快速打分：Golang/C++ 实现，<5ms                             │
│  └─ 输出：~200 候选广告                                          │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  精排层（Fine Ranking）                                          │
│  ├─ 深度模型：DeepFM、DIN、MMoE、Transformer                     │
│  ├─ 完整特征：实时+离线特征，百维级                              │
│  ├─ 多目标：CTR、CVR、深度转化等                                 │
│  └─ 输出：~50 候选广告，含eCPM分数                               │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  重排层（Re-ranking / Rank）                                     │
│  ├─ 多样性控制：MMR、DPP                                         │
│  ├─ 频控：用户频次、广告主频次                                   │
│  ├─ 刷新控制：广告位置打散                                       │
│  └─ 输出：~10 候选广告                                           │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  混排层（Blending）                                              │
│  ├─ 多源融合：广告 + 自然内容 + 运营位                           │
│  ├─ 统一排序：按综合价值（如信息流用Willness）                   │
│  └─ 输出：最终展示列表                                           │
└─────────────────────────────────────────────────────────────────┘
```

#### 各层职责对比

| 层级 | 候选量 | 延迟要求 | 模型复杂度 | 主要目标 |
|------|--------|----------|------------|----------|
| 召回 | 10^6 → 10^3 | <50ms | 极低 | 相关性过滤 |
| 粗排 | 10^3 → 10^2 | <5ms | 低 | 快速筛选 |
| 精排 | 10^2 → 10^1 | <50ms | 高 | 精准预估 |
| 重排 | 10^1 → 10^1 | <10ms | 中 | 业务规则 |
| 混排 | 多源 → 最终 | <10ms | 低 | 生态平衡 |

### 5.5 常见考点

| 考点 | 答案要点 |
|------|----------|
| **特征穿越怎么排查？** | 检查特征时间戳、看离线AUC是否虚高、分析特征重要性是否异常 |
| **为什么召回不能用精排模型？** | 延迟要求不同，精排太慢；候选量太大，需要快速过滤 |
| **粗排和精排怎么保持一致性？** | 粗排蒸馏精排、同一份训练数据、特征对齐 |
| **PSI 怎么算？什么阈值？** | 分箱后计算分布差异，<0.1正常，>0.25需重训 |
| **混排为什么重要？** | 平台长期价值，避免广告过度挤压自然内容，影响用户体验 |

---

## 附录：核心公式速查表

| 公式 | 用途 |
|------|------|
| $\text{eCPM} = \text{bid} \times pCTR \times pCVR \times 1000$ | 竞价排序 |
| $\text{CTCVR} = \text{CTR} \times \text{CVR}$ | 全链路转化 |
| $L_{ESMM} = L_{CTR} + \alpha \cdot L_{CTCVR}$ | 多任务联合训练 |
| $UCB_a = \hat{\mu}_a + c\sqrt{\frac{\ln t}{N_a}}$ | 探索利用平衡 |
| $\gamma_t = (r_t / s_t)^k$ | 预算 pacing |
| $PSI = \sum (A_i - E_i) \ln(A_i/E_i)$ | 分布稳定性 |

---

> **编写说明**：本笔记涵盖广告系统核心技术栈，建议结合具体业务场景深入学习。面试准备时重点关注：公式推导、工程权衡、实际案例。
