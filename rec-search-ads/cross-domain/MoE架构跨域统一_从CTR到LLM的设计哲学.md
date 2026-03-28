# MoE 架构跨域统一：从 CTR 到 LLM 的设计哲学

> 📅 创建：20260328 | 类型：深度整合 | 领域：cross-domain
> 🔗 串联领域：ads × rec-sys × llm-infra

---

## 📋 一句话洞察

**混合专家（MoE）是2025-2026年"参数扩展而计算不扩展"的唯一经济方案**——从广告CTR特征交叉（DHEN）、多任务推荐（HoME）、排序Scaling（RankMixer），到大语言模型（Qwen3），MoE的核心哲学始终如一：**让不同"专家"专注于不同的知识空间，通过稀疏激活解耦容量与计算**。

---

## 📚 参考文献

> - [DHEN](../../ads/papers/dhen_deep_hierarchical_ensemble_ctr.md) — Meta广告，层次Ensemble多种交叉模块，每层多专家输出聚合
> - [HoME](../../rec-sys/papers/home_hierarchy_mulitgate_experts_kuaishou.md) — 快手多任务推荐，修复Expert Collapse/Degradation/Underfitting三大问题
> - [RankMixer](../../rec-sys/papers/rankmixer_scaling_ranking_models.md) — 字节排序模型，稀疏MoE实现千亿参数Scaling，AUC持续提升无plateau
> - [Wukong CTR](../../ads/papers/wukong_ctr_scalable_deep_parallel_training.md) — 快手CTR，Pre-LN深层Transformer-like架构，可理解为参数共享的"单专家"极端
> - [Qwen3 Technical Report](../../llm-infra/papers/qwen3_technical_report.md) — LLM MoE的天花板：Thinking/Non-thinking双模式统一，MoE层交替Dense层

---

## 📐 核心公式与原理

### 1. 专家路由：Top-K稀疏激活（通用形式）

$$\mathbf{y} = \sum_{k \in \text{TopK}(g(\mathbf{x}))} g_k(\mathbf{x}) \cdot E_k(\mathbf{x})$$

其中：
- $g(\mathbf{x}) = \text{softmax}(\mathbf{W}_g \mathbf{x})$：门控网络，输出各专家的路由概率
- $\text{TopK}$：只激活概率最高的K个专家（K=1或2）
- $E_k(\mathbf{x})$：第k个专家的变换
- **稀疏性保证**：总参数量=N×单专家参数，但推理FLOPs=K/N×总参数（仅K/N被激活）

### 2. DHEN：层次Ensemble（每层多模块聚合）

$$h^{(l+1)} = LayerNorm\left(\sum_{k=1}^{K} \alpha_k \cdot f_k(h^{(l)})\right)$$

其中 $f_k$ 可以是Cross Network、Bilinear Interaction、Self-Attention、MLP等不同结构。
- **"水平专家"**：同层并行，表征多种特征交叉类型
- 权重 $\alpha_k$ 可学习（类MoE软路由），也可等权
- 每层叠加后深入：浅层学简单交叉，深层学复杂组合

### 3. HoME：同质化正则（防止Expert Collapse）

$$\mathcal{L}_{homo} = \lambda \sum_{e_i \neq e_j} \text{KL}(P_{e_i} \| P_{e_j})$$

惩罚专家输出分布过度分化，强制所有专家在同一激活分布上运作，避免某些专家"坍塌"（>90%零值）。

### 4. RankMixer：双路径MoE Block（Scaling的核心）

$$\mathbf{h}^{(l+1)} = \underbrace{\text{FFN}(\text{MHA}(\mathbf{h}^{(l)}))}_{\text{Dense路径（全局交叉）}} + \underbrace{\text{MoE-FFN}(\mathbf{h}^{(l)})}_{\text{Sparse路径（容量Scaling）}}$$

- Dense路径：参数共享，处理全局特征交叉，保基础表达能力
- MoE路径：稀疏激活，专家数可无限扩展而推理延迟不变

### 5. MoE负载均衡损失（工程必备）

$$\mathcal{L}_{balance} = \alpha \cdot N \sum_{e=1}^{N} f_e \cdot P_e$$

其中 $f_e$ 是专家e在该batch中的实际激活比例，$P_e$ 是路由网络给出的概率。
最小化此损失使所有专家的负载趋于均匀，防止Expert Collapse和部分专家不收敛。

---

## 🗺️ 技术演进脉络：MoE设计的历史纵轴

```
MoE演进史（推荐/广告 → LLM）：

2017: Sparsely-Gated MoE (Shazeer)
  └─ 首次提出Token-level稀疏路由
      ↓
2021: Switch Transformer (Google)  
  └─ LLM场景的大规模MoE验证（Top-1路由）
      ↓
2022: MMoE → CGC → PLE（推荐系统多任务）
  └─ 不同任务共享部分专家，独占部分专家
      ↓
2023: DeepSeek MoE（专家级精细化路由）
  └─ 细粒度Expert + 全局Shared Expert分离
      ↓
2024: DHEN（CTR特征交叉的层次Ensemble）
  └─ 不同交叉类型 = 不同"专家"并行
      HoME（多任务MoE三大问题系统修复）
  └─ 工业落地的教科书级实践
      ↓
2025: RankMixer（排序Scaling的MoE方案）
  └─ 千亿参数推荐排序，持续Scaling无plateau
      Qwen3 MoE（LLM Thinking/Non-thinking统一）
  └─ 共享Expert处理通用任务，路由Expert处理专业任务
      ↓
2026: 跨域统一——推荐、广告、LLM的MoE设计趋于收敛
```

---

## 🔍 横向对比：五个MoE系统的设计选择

| 系统 | 领域 | 专家粒度 | 路由策略 | 专家数 | 核心解决的问题 |
|------|------|---------|---------|-------|--------------|
| DHEN | 广告CTR | 交叉模块粒度 | 可学习软权重 | 3-6种 | 单一交叉类型的表达局限 |
| HoME | 多任务推荐 | 参数块粒度 | 层次化硬/软路由 | 10-50 | Collapse/Degradation/Underfitting |
| RankMixer | 排序Scaling | FFN粒度 | Top-K稀疏路由 | 16-128 | 参数规模与推理延迟的解耦 |
| Wukong | CTR训练 | 无MoE（全密集） | N/A | N/A | 深层网络训练稳定性（对照组） |
| Qwen3 | LLM | FFN粒度 | Top-K + Shared | 64-128 | 计算效率与模型能力的极限扩展 |

**设计哲学差异的根因**：
- DHEN/HoME：**功能多样性需求**——不同交叉类型/任务需要不同专家
- RankMixer：**容量Scaling需求**——突破Dense网络的参数上限
- Qwen3：**兼顾推理与效率**——Thinking模式需要深度推理，Non-thinking模式需要低延迟

---

## 🎯 核心洞察（老师视角）

### 洞察1：MoE的"本质"是条件计算（Conditional Computation）
不论是CTR的特征交叉专家还是LLM的FFN专家，本质都是**"根据输入决定激活哪些计算路径"**。推荐系统早在2019年就用MMoE实现了条件计算，LLM的MoE只是在更大规模上验证了同一思想。

### 洞察2：Expert Collapse是MoE的"通病"，但原因不同
- 推荐多任务（HoME）：dominant task梯度劫持 → 用同质化正则
- 排序Scaling（RankMixer）：路由网络初始不稳定 → 用负载均衡辅助loss
- LLM MoE（DeepSeek）：细粒度专家过多 → 引入Shared Expert保证基础覆盖
三种场景，同一问题，解法殊途同归——都是"强制所有专家有效利用"。

### 洞察3：Wukong（全Dense）是重要的"对照实验"
Wukong选择不用MoE，而是用Pre-LN + 深层堆叠（10+层）做CTR，验证了**单一结构的深层化也能达到很强效果**。这提示工程师：MoE不是必须的，深层Dense在中等规模（100亿以内）是更简单的替代方案。

### 洞察4：MoE的推理延迟瓶颈从"计算"转移到"内存带宽"
稀疏激活解决了FLOPs问题，但Expert Parallel引入All-to-All通信，大量专家的参数无法全部驻留GPU显存（需从HBM/DDR动态加载）。未来的MoE优化方向：专家缓存预取、Expert Co-location优化、量化+稀疏联合压缩。

### 洞察5：推荐系统MoE比LLM MoE更早工程化，反哺LLM工程实践
MMoE/CGC (2018-2020) → Switch Transformer (2021)，推荐系统的多任务MoE**早于**LLM MoE成熟落地，积累了丰富的负载均衡、专家崩溃、梯度干扰等工程经验，实际上为LLM MoE的工业化提供了参考。

---

## 🏭 工程落地桥梁：从论文到生产

### 常见陷阱与解决方案

| 陷阱 | 论文中的理想假设 | 工业生产中的现实 | 解决方案 |
|------|---------------|----------------|---------|
| Expert Collapse | 均匀路由分布 | 少数专家被过度选择 | 负载均衡Loss + 容量限制 |
| 通信瓶颈 | 专家并行理论上线性加速 | All-to-All通信延迟显著 | Expert Co-location + 梯度压缩 |
| 内存爆炸 | 千亿参数MoE | 单GPU内存8-80GB | ZeRO-3 + Offloading + 量化 |
| 增量更新 | 全量重训练 | 日级增量训练 | 只更新激活专家，冻结未激活专家 |
| 冷启动 | 均匀初始化 | 路由网络不稳定 | 随机路由Warmup前N步 |

### 部署延迟优化路径

```
生产MoE延迟优化策略（按优先级排序）：

1. 专家量化：INT4/INT8 Expert权重（内存减半，带宽瓶颈改善）
2. 专家蒸馏：MoE Teacher → Dense Student（推理时用更小Dense）
3. 专家缓存：预测下一batch会用哪些专家，提前加载GPU
4. Top-1替代Top-2路由：牺牲少量精度换2倍速度
5. 专家并行 + 流水线并行：跨机器分布式推理
```

---

## 🎓 面试考点（≥10题）

**Q1：MoE和Ensemble的区别是什么？**
A：Ensemble通常是**固定权重组合**多个独立训练的模型（如随机森林），计算量随模型数线性增长。MoE是**条件计算**：路由网络根据输入动态选择专家，推理时只激活K/N的专家，总计算量与单模型相当。DHEN的层次Ensemble更接近"软MoE"——每层对所有交叉模块求加权和，而非稀疏激活。

**Q2：为什么推荐系统的MoE专家数通常比LLM少很多？**
A：LLM MoE（如DeepSeek：256专家）的专家粒度是FFN子网络（参数量小），总专家数多。推荐系统（如MMoE：3-10专家）的专家粒度通常是完整的Transformer/MLP块（参数量大），加上推荐特征维度低（千维 vs LLM的万维），太多专家会导致路由学习困难和负载不均。

**Q3：HoME中的Expert Degradation（退化）和Expert Collapse（坍塌）有何不同？**
A：Collapse：专家本身输出分布退化（如>90%零值），整体失效，被门控网络忽略。Degradation：专家功能退化为单任务专属，不再通用，仍然被激活但失去了"共享"的价值。前者是"专家死了"，后者是"共享专家变成私有专家"。

**Q4：RankMixer中，Dense路径和MoE路径各自负责什么？为什么要并行而非串行？**
A：Dense路径（MHA+FFN）：全局特征交叉，参数被所有输入共享，处理通用模式。MoE路径：稀疏激活，专家分别处理不同特征子空间（如不同用户群体的行为模式），提供差异化表达。并行而非串行：串行会让MoE路径依赖Dense的输出，失去独立表达能力；并行让两路各自学习，最终残差连接融合，表达能力更强。

**Q5：Qwen3中的Shared Expert和Routed Expert分别是什么角色？**
A：Shared Expert：所有token都会激活，学习通用语言知识（类似MMoE的shared expert），保证基础能力不因路由失败而退化。Routed Expert：Top-K稀疏激活，学习领域专业知识（数学、代码、特定语言）。两者结合解决了早期MoE的"基础能力退化"问题（有时路由所有token到专业专家，通用能力反而下降）。

**Q6：MoE的负载均衡Loss为什么重要？不加会怎样？**
A：不加负载均衡Loss，路由网络会在训练早期随机倾向于某些专家（随机初始化的轻微偏差），这些专家得到更多梯度更新 → 变得更强 → 被更多选中 → 正反馈循环。最终2-3个专家处理99%的token，其他专家几乎不训练，MoE退化为Dense模型（浪费了大量参数）。

**Q7：在推荐系统中，什么时候应该选择MoE而不是深层Dense（如Wukong）？**
A：选MoE：(1) 需要超过百亿参数的模型但推理延迟有严格约束；(2) 有明确的子任务/用户群体分化（专家能学到真正的功能差异）；(3) 计算资源充足可以承受Expert Parallel的通信开销。选深层Dense（Wukong-like）：(1) 百亿以内参数量；(2) 工程团队没有MoE优化经验；(3) 流量规模不需要千亿参数的记忆容量。

**Q8：如何量化一个MoE系统的"专家利用率"？**
A：指标体系：①专家激活频率（每个专家在batch中被激活的token比例，理想=均匀=1/N）；②专家负载熵（$H = -\sum_e p_e \log p_e$，越高越均匀，越低越集中）；③专家输出方差（各专家输出向量的L2范数分布，Collapse时某些专家接近0）；④门控权重熵（某专家权重过高时说明路由退化）。

**Q9：跨任务MoE（多任务推荐）和跨输入MoE（稀疏路由推荐排序）有何本质区别？**
A：跨任务MoE（MMoE/HoME）：路由信号是**任务标识**（哪个任务的预测头），专家按任务功能分化，是"知道在做什么任务就走什么专家"。跨输入MoE（RankMixer/DeepSeek）：路由信号是**当前输入token**，专家按输入内容分化，是"当前输入特征决定走哪个专家"。前者的路由更可预测，后者的路由更自适应但更难解释。

**Q10：MoE在广告系统工业落地时，最常见的失败模式是什么？**
A：最常见：(1) **推理延迟超标**——Expert Parallel引入All-to-All通信，实测P99延迟比Dense高30-50%，超过SLA → 解决：Expert缓存预取 + INT8量化；(2) **Expert Collapse**——前N天在线学习后大量专家不激活 → 解决：负载均衡Loss + 监控报警；(3) **离线-在线不一致**——离线MoE模型的专家激活分布与在线serving环境不一致 → 解决：用线上实时请求数据做专家warmup。

---

## 🔑 总结：MoE设计的统一框架

```
MoE工程三要素（无论哪个领域）：

①设计专家：专家的粒度和边界
  - 功能边界（不同交叉类型/任务）→ DHEN, HoME
  - 内容边界（不同输入子空间）→ RankMixer, Qwen3

②设计路由：如何决定激活哪些专家
  - 软路由（可学习加权和）→ DHEN, 早期MMoE
  - Top-K硬路由（稀疏激活）→ RankMixer, Qwen3
  - 层次化路由（先粗后细）→ HoME

③保障专家多样性：防止退化
  - 负载均衡Loss → 通用方案
  - 同质化正则（KL约束）→ HoME特有
  - Shared Expert保底 → DeepSeek/Qwen3
```
