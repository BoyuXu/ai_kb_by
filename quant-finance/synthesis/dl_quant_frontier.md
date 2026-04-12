# 深度学习量化前沿 — Transformer、GNN与生成模型

> DL 在量化投资中的前沿方向：Transformer 时序预测、GNN 选股、RL 组合优化、生成模型数据增强。本文从搜广推工程师视角出发，重点对比与搜广推的异同，帮助快速建立直觉。

相关文档：[[ml_in_quant]] | [[factor_investing]] | [[strategy_development]]
搜广推交叉：[[concepts/attention_in_recsys]] | [[concepts/sequence_modeling_evolution]] | [[concepts/embedding_everywhere]]

---

## 1. Transformer 时序预测

### 1.1 为什么 Transformer 进入金融时序？

搜广推领域 Transformer 已经统治了序列建模（SASRec、BST、BERT4Rec），金融时序自然也想借力。但金融序列与搜广推序列有本质差异：

| 维度 | 搜广推序列 | 金融时序 |
|------|-----------|---------|
| **数据类型** | 离散事件（点击、购买） | 连续值（价格、收益率） |
| **时间间隔** | 不规则（用户随时来） | 等间隔（日频/分钟频） |
| **序列长度** | 通常 50-200 | 可达数千（日频多年） |
| **信号强度** | 强（行为=兴趣） | 极弱（噪声主导） |
| **目标** | 预测下一个交互 | 预测收益率/波动率 |

🔄 **核心差异**：搜广推中 Attention 捕捉的是「用户对不同历史行为的兴趣衰减」，金融中 Attention 需要捕捉的是「不同时间尺度的价格模式」。搜广推 token 是离散 item_id，金融 token 是连续数值向量。

### 1.2 Temporal Fusion Transformer (TFT)

Google 提出的多尺度时序融合架构，**量化中最被广泛引用的 Transformer 变体**。

**核心设计**：
- **Variable Selection Network**：自动筛选重要特征（类似搜广推中的 Gate/Attention 特征选择）
- **Gated Residual Network (GRN)**：控制信息流，避免过拟合
- **Multi-head Attention**：跨时间步的长程依赖
- **分位数回归输出**：不只预测均值，输出分位数区间（金融必须有不确定性估计）
- **可解释性**：attention weight 可视化 → 哪些时间步/特征对预测贡献大

**量化适配**：TFT 的分位数输出天然适配风险管理。可以输出 5%/50%/95% 分位，直接用于 VaR 估计。

### 1.3 PatchTST — 把时序当图像切 Patch

**核心思想**：将长时序切成固定长度的 patch（类比 ViT 把图像切 patch），每个 patch 作为一个 token。

**关键设计**：
- **Patching**：将序列 `[x_1, ..., x_L]` 切成 `[p_1, ..., p_N]`，每个 patch 长度 P，步长 S
- **Channel-independence**：每个变量（股票/因子）独立建模，不做跨变量 attention
- **为什么 channel-independence 有效**：金融变量之间的关系是非平稳的（行业轮动、风格切换），强行建模跨变量关系容易过拟合

```python
import torch
import torch.nn as nn

class SimplePatchTST(nn.Module):
    """简化版 PatchTST：Patch + Transformer Encoder"""

    def __init__(self, seq_len=252, patch_len=21, d_model=64,
                 nhead=4, num_layers=2, pred_len=5):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len  # non-overlapping patches
        self.num_patches = (seq_len - patch_len) // self.stride + 1

        # Patch embedding: 将每个 patch 映射到 d_model
        self.patch_embed = nn.Linear(patch_len, d_model)

        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # 预测头
        self.head = nn.Linear(d_model * self.num_patches, pred_len)

    def forward(self, x):
        """
        x: (batch, seq_len) — 单变量时序（channel-independence）
        return: (batch, pred_len) — 未来预测
        """
        B = x.shape[0]
        # 切 patch: (B, num_patches, patch_len)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)

        # Patch embedding + position
        z = self.patch_embed(patches) + self.pos_embed  # (B, num_patches, d_model)

        # Transformer encode
        z = self.encoder(z)  # (B, num_patches, d_model)

        # Flatten + predict
        z = z.reshape(B, -1)
        return self.head(z)

# 使用示例：252 日（1年）预测未来 5 日收益
model = SimplePatchTST(seq_len=252, patch_len=21, pred_len=5)
x = torch.randn(32, 252)  # batch=32, 一年日频数据
pred = model(x)  # (32, 5) — 未来5日预测
```

### 1.4 iTransformer — 反转维度

**核心创新**：传统 Transformer 中 token = 时间步，iTransformer 中 **token = 变量**。

- 输入矩阵 `(T, N)` 中，T 是时间步，N 是变量数
- 传统做法：N 个特征拼成一个 token，沿 T 做 attention
- iTransformer：T 个时间步拼成一个 token，沿 N 做 attention → 学习变量间关系

**量化意义**：直接在「股票维度」做 attention → 自动学习股票间的相关结构，类似 GNN 但更灵活。

### 1.5 Autoformer / FEDformer — 频域分解

- **Autoformer**：Auto-Correlation 替代 self-attention，在频域计算周期性依赖
- **FEDformer**：在傅里叶域做 attention，天然适合捕捉金融的季节性模式（月末效应、季报周期）

> **面试常问**：Transformer 做时序预测和做 NLP/推荐有什么本质不同？
> 答：NLP/推荐的 token 是离散的、语义丰富的；时序 token 是连续数值、信噪比低。时序需要额外处理趋势/季节性分解、长程依赖（用 patch 降低序列长度）、以及不确定性估计（分位数回归）。

---

## 2. GNN 选股

### 2.1 股票关系图构建

GNN 选股的核心不是模型，而是**如何构建图**。图的质量决定了 GNN 的上限。

| 关系类型 | 构建方法 | 边权重 | 稳定性 |
|---------|---------|--------|--------|
| **行业链** | 申万/中信行业分类 | 二值（同/异） | 高（年度调整） |
| **供应链** | 财报/公告中提取 | 供应占比 | 中（季度变化） |
| **持仓重叠** | 基金持仓数据 | 共同持仓比例 | 低（季度调仓） |
| **收益率相关性** | 滚动窗口相关系数 | 相关系数值 | 低（动态变化） |
| **知识图谱** | NLP 提取实体关系 | 关系强度 | 中 |

🔄 **与搜广推图的对比**：

| 维度 | 搜广推（用户-物品图） | 量化（股票关系图） |
|------|---------------------|-------------------|
| **图类型** | 异构二部图 | 同构/异构（股票-股票或股票-事件） |
| **节点数** | 亿级用户+百万物品 | ~5000 只 A 股 |
| **边含义** | 用户行为（点击/购买） | 关联关系（行业/供应链） |
| **动态性** | 实时更新 | 日频/周频更新 |
| **核心挑战** | 可扩展性（规模大） | 图构建质量（规模小但关系复杂） |

搜广推用 GNN 是为了在海量稀疏交互中传播信息（LightGCN、PinSage），量化用 GNN 是为了在有限节点间捕捉关联结构。

### 2.2 GAT/GCN 选股模型

**基本流程**：
1. 节点特征：每只股票的因子值（价量因子、基本面因子等）
2. 图结构：行业链 + 供应链 + 相关性
3. GNN 传播：聚合邻居信息，获得 graph-aware embedding
4. 预测头：排序打分 → 多空组合

**GAT vs GCN**：
- GCN：固定权重聚合（度归一化），简单稳定
- GAT：attention 加权聚合，可学习不同邻居的重要性 → 金融中更常用（不同关联关系重要性不同）

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class StockGAT(nn.Module):
    """用 GAT 做股票关系建模"""

    def __init__(self, in_dim=30, hidden_dim=64, out_dim=1, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)  # 预测收益率排序分
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.gat1(x, edge_index))
        x = torch.relu(self.gat2(x, edge_index))
        return self.pred_head(x)  # (num_stocks, 1)

def build_stock_graph(factor_matrix, corr_matrix, threshold=0.6):
    """
    构建股票关系图
    factor_matrix: (num_stocks, num_factors) — 节点特征
    corr_matrix: (num_stocks, num_stocks) — 相关性矩阵
    threshold: 相关性阈值，超过则连边
    """
    num_stocks = factor_matrix.shape[0]

    # 从相关性矩阵构建边：相关性 > threshold 的股票对连边
    edges = []
    for i in range(num_stocks):
        for j in range(i + 1, num_stocks):
            if abs(corr_matrix[i, j]) > threshold:
                edges.append([i, j])
                edges.append([j, i])  # 无向图

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(factor_matrix, dtype=torch.float)

    return Data(x=x, edge_index=edge_index)

# 使用示例
num_stocks, num_factors = 500, 30
factor_matrix = torch.randn(num_stocks, num_factors).numpy()
corr_matrix = torch.randn(num_stocks, num_stocks).numpy()

graph = build_stock_graph(factor_matrix, corr_matrix, threshold=0.6)
model = StockGAT(in_dim=num_factors)
scores = model(graph)  # (500, 1) — 每只股票的预测排序分
```

### 2.3 知识图谱 + GNN

将财报数据、新闻事件、分析师报告构建成知识图谱，与股票关系图融合：
- **实体**：股票、行业、概念、事件、人物
- **关系**：属于行业、供应给、受益于事件、管理层变动
- **融合方式**：异构图 GNN（HAN、HGT）或先做 KG embedding 再拼接节点特征

> **面试常问**：GNN 选股相比传统多因子模型的优势是什么？
> 答：传统多因子模型把每只股票独立打分，忽略了股票间的关联结构。GNN 通过信息传播捕捉关联效应（如行业龙头带动、供应链传导），但代价是引入了图构建的不确定性和更高的过拟合风险。实践中 GNN 通常作为补充因子而非替代传统因子。

---

## 3. 强化学习做 Portfolio Optimization

### 3.1 问题建模

将组合优化建模为 Markov Decision Process：

| MDP 元素 | 量化定义 | 典型维度 |
|---------|---------|---------|
| **State** | 市场状态（价格、因子值、持仓、现金） | ~100-1000 维 |
| **Action** | 资产权重分配 `[w_1, ..., w_N]`，sum=1 | N（资产数） |
| **Reward** | 风险调整收益（Sharpe ratio、Sortino） | 标量 |
| **转移** | 市场随机演化（agent 无法控制） | — |

### 3.2 常用算法

| 算法 | 特点 | 适用场景 |
|------|------|---------|
| **PPO** | 稳定，限制策略更新幅度 | 通用首选 |
| **SAC** | 最大熵框架，鼓励探索 | 连续动作空间（权重分配） |
| **A2C** | Actor-Critic，同步更新 | 简单基线 |
| **TD3** | 双 Q 网络，解决 Q 值高估 | 低延迟决策 |

### 3.3 FinRL 框架

开源的金融强化学习框架（Columbia 团队），提供：
- 标准化环境（StockTradingEnv）
- 数据接口（Yahoo Finance、Alpaca、WRDS）
- 预置算法（PPO/A2C/SAC/TD3/DDPG）
- 回测模块

### 3.4 RL 做量化的核心挑战

1. **环境非平稳**：金融市场的转移概率不断变化（regime switching），RL agent 学到的策略可能很快失效
2. **Reward 设计困难**：用 Sharpe ratio 做 reward → 稀疏（月度才有意义）；用日收益做 reward → 噪声大
3. **过拟合**：RL 在有限历史数据上训练，极易过拟合特定市场环境
4. **Action 空间约束**：权重之和=1、做空限制、交易成本 → 需要 constrained RL
5. **样本效率低**：金融数据无法像 Atari 一样无限生成

> **面试常问**：RL 做量化和 RL 做游戏/机器人的核心差异？
> 答：游戏/机器人的环境是平稳的（物理定律不变），可以无限模拟采样；金融环境非平稳且数据有限，无法无限采样。这导致 RL 在金融中的样本效率和泛化能力远不如在游戏中。

---

## 4. 生成模型在量化中的应用

### 4.1 GAN 生成合成市场数据

**动机**：金融数据量有限，且尾部事件（暴跌/熔断）极少 → 用 GAN 生成合成数据做增强。

**典型架构**：
- **TimeGAN**：结合自回归和 GAN，保留时序动态特性
- **Quant GAN**：用 Temporal CNN 做 Generator，生成逼真的价格路径
- **条件 GAN**：生成特定 regime 下的市场数据（如：生成"类似 2008 危机"的价格路径）

**评估指标**：
- 分布一致性：生成数据的收益率分布是否匹配真实数据（肥尾、偏度）
- 时序特性：自相关、波动率聚集是否保留
- 下游任务表现：用合成数据训练的模型在真实数据上的效果

### 4.2 Diffusion Model 生成市场 Scenario

Diffusion Model（DDPM）在图像生成大获成功后，开始进入金融场景生成：

- **Scenario Generation**：生成未来 N 天的多条市场路径 → 替代 Monte Carlo 模拟
- **优势**：比 GAN 更稳定（无 mode collapse），生成多样性更好
- **应用**：压力测试、风险管理（生成极端市场情景）

### 4.3 VAE 做异常检测（Regime 变化）

**思路**：用 VAE 学习「正常市场状态」的潜在分布，当市场进入异常 regime 时，重建误差显著增大。

- **训练**：在正常行情数据上训练 VAE
- **推理**：实时计算重建误差，超过阈值 → 触发风险预警
- **应用**：检测市场 regime 变化（牛→熊、低波→高波）

🔄 **与搜广推异常检测的对比**：搜广推中 VAE/AE 常用于检测异常流量（刷量、爬虫），方法论完全一致，只是特征和阈值不同。参见 [[concepts/embedding_everywhere]]。

---

## 5. 为什么 DL 在金融落地难？

这是搜广推工程师转量化最需要深刻理解的问题。

### 5.1 信噪比极低

| 领域 | 典型 R^2 | 含义 |
|------|---------|------|
| 搜广推 CTR | 20%-40% | 模型可解释较大比例的方差 |
| 量化日频 | 1%-5% | 绝大部分变动是噪声 |
| 量化高频 | 5%-15% | 微观结构信号稍强 |

R^2 < 5% 意味着：即使你的模型完美捕捉了所有可预测信号，95% 以上的变动仍是随机的。DL 的强大拟合能力在这种场景下更多是拟合噪声。

### 5.2 数据量有限

- A 股：~5000 只股票 x ~20 年日频 = ~2500 万条（看似不少，但独立截面只有 ~5000 个交易日）
- 搜广推：日均亿级样本，DL 可以尽情吃数据
- **有效样本量**：金融的 5000 个交易日中，很多日期高度相关（市场整体涨跌），独立有效样本远少于名义样本

### 5.3 非平稳性

金融数据的分布在持续漂移：
- 市场 regime 切换（牛/熊/震荡）
- 监管政策变化（涨跌停、注册制）
- 参与者结构变化（散户→机构→量化）
- 因子衰减（一个因子被发现后，alpha 逐渐消失）

搜广推也有分布漂移（用户兴趣变化），但远没有金融剧烈。

### 5.4 过拟合

**参数量 vs 有效样本量** 的矛盾：
- 一个简单的 3 层 MLP 可能有数万参数
- 有效独立样本可能只有数千个交易日
- DL 参数量轻松超过有效样本量 → 必然过拟合

**应对策略**：
- 强正则化（dropout、weight decay、early stopping）
- 小模型优先（参数量远小于有效样本量）
- 集成学习（多个弱模型投票）
- 更激进的交叉验证（时序 K-fold、Purged CV）

### 5.5 可解释性要求

- 金融监管（如 MiFID II）要求模型决策可解释
- 风控需要理解模型在什么条件下会失效
- 基金经理不会仅凭一个黑箱模型做投资决策

> **面试高频题**：为什么简单模型在量化里往往比 DL 好？
>
> **标准答案**：三个原因叠加——
> 1. **信噪比低**：可预测信号本身就弱（R^2 < 5%），DL 的额外拟合能力更多拟合了噪声
> 2. **数据量少**：有效独立样本有限（~5000 个交易日），DL 参数量容易超过有效样本量
> 3. **非平稳性**：分布持续漂移，复杂模型学到的历史模式更容易在未来失效
>
> 简单模型（线性、XGBoost）的归纳偏置更强，相当于自带正则化，在低信噪比+少数据+非平稳的环境下泛化更好。这和搜广推形成鲜明对比——搜广推数据充足、信噪比高，DL 的拟合能力是优势而非负担。

---

## 6. 总结与实践建议

| 方向 | 成熟度 | 实用性 | 建议 |
|------|--------|--------|------|
| Transformer 时序 | 学术热门 | 中（需大量调参） | 先用 PatchTST/TFT 做 baseline |
| GNN 选股 | 有实盘案例 | 中高（图构建是关键） | 从行业图开始，逐步加边 |
| RL 组合优化 | 学术探索 | 低（非平稳+过拟合） | 了解即可，实盘谨慎 |
| GAN/Diffusion 数据增强 | 早期 | 低中（评估困难） | 用于压力测试场景 |
| VAE 异常检测 | 有落地 | 中高 | 作为 regime 变化预警补充 |

**搜广推工程师的量化 DL 路线建议**：
1. 先搞清楚为什么简单模型在量化里更好（上面第 5 节）
2. 从 GNN 选股入手（与搜广推图技术最接近）
3. Transformer 时序预测做研究储备（PatchTST 代码最简洁）
4. RL 和生成模型了解概念即可，不要一开始就投入

---

*最后更新：2026-04*
*交叉引用：[[ml_in_quant]] | [[factor_investing]] | [[concepts/attention_in_recsys]] | [[concepts/sequence_modeling_evolution]]*
