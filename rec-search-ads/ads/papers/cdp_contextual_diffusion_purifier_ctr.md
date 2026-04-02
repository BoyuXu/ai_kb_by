# Adaptive User Interest Modeling via Conditioned Denoising Diffusion For CTR Prediction
> 来源：https://arxiv.org/abs/2509.19876 | 领域：ads | 学习日期：20260401

## 问题定义

在搜索/广告系统中，用户行为序列被用来建模用户兴趣。然而现有方法面临两大核心问题：

**问题1：噪声污染（Interest Fossil Problem）**
用户历史行为序列就像"兴趣化石"——记录了真实意图，但被多种噪声侵蚀：
- **曝光偏差（Exposure Bias）**：用户只能点击被推荐的内容，历史行为受平台算法扭曲
- **品类漂移（Category Drift）**：用户在不同时期兴趣不同，历史序列混杂了多种意图
- **上下文噪声（Contextual Noise）**：无意点击、误触等导致序列中有噪声行为

**问题2：静态表示（Static Context-Agnostic Representation）**
现有方法（DIN、DIEN、SIM等）输出的用户兴趣是固定的，不能根据当前的 Query-User-Item-Context 动态适应。
- 例如：同一用户搜"手机壳"和"手机"时，应该激活不同的历史兴趣
- 现有方法通过 Attention 做一定程度的适应，但受限于 Attention 的上限

## 核心方法与创新点

### Contextual Diffusion Purifier（CDP）

CDP 将扩散模型（Diffusion Model）引入用户兴趣建模，实现：
1. **噪声去除**：通过前向加噪 + 条件反向去噪，过滤行为序列中的噪声
2. **上下文感知**：以 Query × User × Item × Context 的交叉特征作为条件，生成动态适应当前场景的纯净兴趣表示

**前向过程（Forward Noising）**：
将经过品类过滤的用户行为序列视为"含噪观测"（contaminated observations），通过马尔可夫链逐步添加高斯噪声：

$$
q(z_t | z_{t-1}) = \mathcal{N}(z_t; \sqrt{1-\beta_t} z_{t-1}, \beta_t I)
$$

**条件反向过程（Conditional Reverse Denoising）**：
以 Query × User × Item × Context 的交叉特征作为条件信号，引导去噪过程：

$$
p_\theta(z_{t-1} | z_t, c) = \mathcal{N}(z_{t-1}; \mu_\theta(z_t, c, t), \Sigma_\theta(z_t, t))
$$

其中条件 $c = f_\text{cross}(\text{Query}, \text{User}, \text{Item}, \text{Context})$ 包含了当前请求的完整上下文信息。

**直觉理解**：
- 前向过程：把用户历史（本身就含噪）进一步"加噪"到纯高斯分布
- 反向过程：从纯高斯噪声开始，在当前 Query-Context 条件下"去噪"生成纯净兴趣
- 最终得到的表示是：**既过滤了历史噪声，又适应了当前搜索/点击场景**的兴趣向量

### 与现有方法的对比

| 方法 | 噪声过滤 | 上下文感知 | 动态适应 |
|------|----------|------------|----------|
| DIN  | ❌ | 部分（Attention） | 静态 |
| DIEN | ❌ | 部分 | GRU动态 |
| SIM  | ❌ | 通过检索 | 静态 |
| CDP  | ✅ 扩散去噪 | ✅ 全交叉条件 | ✅ 动态生成 |

### 品类过滤（Category-filtered Behaviors）

CDP 对行为序列做品类过滤作为预处理，保留与当前 Item 相关品类的历史行为，降低无关噪声后再输入扩散模型。

## 实验结论

**离线实验**：
- 在多个公开 CTR 预测数据集上，CDP 超越 DIN、DIEN、SIM、CAN 等 SOTA 方法
- AUC 提升 +0.2%~+0.5%（在广告系统中属于显著提升）

**在线 A/B 实验**：
- 在真实搜索广告系统部署，核心 CTR 指标提升（具体数值见论文）
- 论文为 5 页短文，处于 under review 状态，在线结果简洁

**核心发现**：条件去噪过程是关键，仅做前向去噪（无条件反向）效果明显差于 CDP。

## 工程落地要点

1. **扩散步数权衡**：扩散步数越多效果越好，但推理耗时增加；工业场景通常使用 DDIM 加速采样（10~20步代替1000步）
2. **条件特征维度控制**：Query × User × Item × Context 的交叉维度很高，需要降维（MLP + BN）
3. **在线推理延迟**：扩散模型推理较慢，建议 distill 到轻量模型或使用少步 DDIM
4. **训练稳定性**：扩散模型训练需要调整噪声调度（noise schedule），避免过早或过晚收敛
5. **与主模型集成**：CDP 输出的纯净兴趣表示可以作为 embedding 输入主 CTR 模型，保持模块化
6. **冷启动用户**：行为序列很短的用户，扩散过程退化，需要特殊处理（如 default prior）

## 常见考点

**Q1: 为什么要用扩散模型来建模用户兴趣？**
A: 扩散模型的核心优势是生成能力——它不是直接从噪声行为序列中 Attend，而是通过学到的去噪过程，生成一个比原始序列更"干净"且上下文感知的兴趣向量。传统 Attention 只能做加权组合，无法去除个别噪声行为的影响。

**Q2: CDP 中的"条件"是什么？为什么设计为 Query × User × Item × Context 交叉？**
A: 条件是当前请求的完整上下文。交叉设计确保兴趣表示能同时感知用户身份、被推荐的 Item、当前 Query 意图以及情境因素，从而生成精准匹配当前场景的兴趣表示，而不是一个"平均"表示。

**Q3: 扩散模型在推荐系统中有哪些应用场景？**
A: （1）用户兴趣去噪（CDP）；（2）增强数据生成（数据增强）；（3）序列生成式推荐（DiffRec）；（4）多模态特征生成；（5）负样本增强。核心价值在于扩散模型的强大生成能力。

**Q4: 如何解决扩散模型推理延迟高的问题？**
A: 使用 DDIM（去噪扩散隐式模型）将采样步骤从 1000 步降低到 10-20 步；或者使用 Consistency Model 一步生成；还可以通过 Knowledge Distillation 将扩散模型的知识蒸馏到轻量判别模型。
