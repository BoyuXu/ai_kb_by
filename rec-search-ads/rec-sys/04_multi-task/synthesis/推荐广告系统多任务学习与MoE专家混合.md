# 推荐广告系统多任务学习与 MoE 专家混合：从 MMoE 到层次化专家系统

> 📚 参考文献
> - [HoME_hierarchy_multi_gate_experts_multi_task_learning_kuaishou](../papers/HoME_hierarchy_multi_gate_experts_multi_task_learning_kuaishou.md) — HoME: 层级化 MoE MTL，解决快手 Expert Collapse/Degradation/Underfitting
> - [Meta_Lattice_model_space_redesign_industry_scale_ads](../../../ads/03_rerank/papers/Meta_Lattice_model_space_redesign_industry_scale_ads.md) — Meta Lattice: 统一 MDMO 模型，多产品×多目标 →1个模型，收入 +10%，算力 -20%
> - [DHEN_deep_hierarchical_ensemble_network_CTR_prediction](../../../ads/02_rank/papers/DHEN_deep_hierarchical_ensemble_network_CTR_prediction.md) — DHEN: 深层异构专家集成，NE -0.27%，Facebook 工业验证

> 知识卡片 | 创建：2026-03-29 | 领域：rec-sys / ads | 类型：综合分析

---

## 📐 核心公式与原理

### 1. MMoE 基础门控机制

$$
\text{MMoE}(x) = \sum_{k=1}^{K} g_k(x) \cdot E_k(x), \quad g_k(x) = \text{softmax}(W_k x)
$$

- 每个任务有独立 Gate，动态路由到 K 个共享专家

### 2. DHEN 层次集成聚合

$$
h_l = \text{Aggregate}\left(\text{Module}_1(h_{l-1}), \text{Module}_2(h_{l-1}), ..., \text{Module}_K(h_{l-1})\right)
$$

- 同层内并行多个异构交互模块（FM/DCN/CIN），层输出作为下层输入

### 3. Meta Lattice 联合训练目标

$$
\mathcal{L}_{Lattice} = \mathcal{L}_{task} + \lambda_{distill} \mathcal{L}_{KD} + \lambda_{domain} \mathcal{L}_{domain\_adapt}
$$

- 任务损失 + 知识蒸馏 + 域自适应，三损失联合优化

### 4. MoE 稀疏 Top-K 路由（RankMixer）

$$
\text{MoE}(x) = \sum_{k \in \text{TopK}(g(x))} g_k(x) \cdot E_k(x)
$$

- 每个 token 只路由到 Top-K（通常 K=2）专家，实现参数 100× 扩展但推理 FLOPs 基本不变

### 5. 辅助均衡损失（防 Expert Collapse）

$$
\mathcal{L}_{aux} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i
$$

- $f_i$ = 样本被路由到专家 $i$ 的比例，$P_i$ = Gate 概率均值；乘积鼓励均匀路由

---

## 🎯 核心洞察

1. **工业 MoE 的三大实证病症（快手 HoME 的诊断贡献）**：专家坍塌（某些专家 90%+ 神经元死亡）、专家退化（共享专家被单一任务占据）、专家欠拟合（稀疏任务特定专家得不到充分训练）。这三个问题在学术界论文中几乎看不到，只有超大规模工业系统才能暴露。

2. **DHEN 的信息互补性发现**：实验证明不同特征交互模块（FM、DCN、CIN）即使声称捕获相同阶数的交互，实际捕获的信息仍存在显著差异。这个发现是异构集成系统的理论基础——不是"越大越好"，而是"越多样越互补"。

3. **Meta Lattice 的模型空间压缩哲学**：从 N产品 × M目标 个独立模型压缩为 1 个统一模型，不是简单的参数共享，而是通过知识蒸馏 + 域自适应实现"统一但不失差异化"。收入 +10% 的同时算力 -20%，证明多模型冗余的巨大浪费。

4. **多任务学习的数据不均衡是核心矛盾**：快手有数十个行为预测任务（点击、完播、点赞、评论等），任务间数据量差异 10×-100×。数据稀疏任务倾向于"蹭"数据密集任务训练的共享专家，导致自身专家得不到有效更新。

5. **层次化 MoE 比平面 MoE 更稳健**：层次化（HoME）将专家组织为树状结构，高层专家捕捉通用知识，低层专家处理任务差异，天然避免"所有任务争抢同一池专家"的竞争冲突。

6. **DHEN 的协同设计原则**：模型架构扩展（更深/更宽）必须与训练系统协同设计（模型并行、梯度检查点、混合精度）。架构创新 + 系统创新缺一不可，这是工业模型区别于学术 toy 实验的核心。

7. **MoE 是推荐系统 Scaling 的正确路径**：固定推理 FLOPs 约束下，MoE 稀疏激活允许参数量 100× 扩展（RankMixer），对比 Dense 模型的参数/效果 trade-off 更优。推荐广告的服务 SLA 约束决定了 MoE 是 scaling 唯一可行路径。

---

## 📈 技术演进脉络

```
早期多任务 (2018 以前)
  ├── 共享底层 MLP（ESMM 等）
  └── 问题：任务冲突，负迁移

MMoE (Google, 2018)
  ├── 引入专家网络 + 多门控
  ├── 每任务独立 Gate，动态路由专家
  └── 实践验证：YouTube 推荐首次大规模工业 MTL

CGC / PLE (腾讯, 2020)
  ├── Customized Gate Control
  ├── 任务专有专家 + 共享专家分离
  └── 更好处理任务相关性差异

DHEN (Meta/Facebook, 2022)
  ├── 异构交互模块集成（FM+DCN+DIN等）
  ├── 层次堆叠：层输出→下层输入
  └── NE改善 0.27%，工业 SOTA 验证

HoME (快手, 2024)
  ├── 诊断三大工业病症（Collapse/Degradation/Underfitting）
  ├── 层次化专家组织（树状结构）
  └── 快手数十任务同时优化

Meta Lattice (Meta, 2025)
  ├── N×M 模型 → 1 个统一 Lattice 模型
  ├── 知识蒸馏 + 域自适应
  └── 收入 +10%, 算力 -20%，生产验证

RankMixer MoE (2025-2026)
  ├── 精排 MoE，参数 100×
  ├── Token Mixing 替代 Self-Attention（MFU 4.5%→45%）
  └── 1B Dense 参数全量上线，时长 +1.08%
```

---

## 🏗️ 从论文到工业落地的工程鸿沟

### 挑战1：专家健康监控（快手 HoME 的教训）
- **问题**：Expert Collapse 在线上悄然发生（90%+ 激活值为 0），但离线 Benchmark 没有告警
- **解法**：生产系统必须监控每个专家的：激活率分布、被 Gate 选中频次、梯度大小、输出分布方差

### 挑战2：共享专家的任务归属问题
- **问题**：共享专家理论上服务所有任务，实际被某个数据密集任务"占据"
- **解法**：HoME 的层次化分离（高层共享/低层专有）；或者设计正则化损失惩罚单一任务对共享专家的过度依赖

### 挑战3：MDMO 场景的负迁移
- **问题**：多产品/多目标联合训练时，某些任务/目标会相互干扰降低效果
- **解法**：Meta Lattice 的域自适应损失 $\mathcal{L}_{domain\_adapt}$；通过梯度手术（Gradient Surgery）或 PCGrad 减少任务间梯度冲突

### 挑战4：MoE 的推理系统工程
- **问题**：稀疏 MoE 的 Expert 分布在多卡上，All-to-All 通信是推理瓶颈
- **解法**：Expert 并行（Expert Parallelism）专门的通信库；DeepSeek MoE 的 Expert 共置策略减少跨机通信

### 挑战5：多产品统一模型的 AB 实验设计
- **问题**：Lattice 是单一模型但服务多产品，传统单产品 AB 实验框架不适用
- **解法**：产品级实验桶划分；domain-specific 指标独立监控；统一模型的风险隔离（某产品指标下降时快速 fallback）

---

## 🎓 常见考点

**Q1：MMoE 和 PLE（CGC）的核心区别是什么？各自适用什么场景？**
> A：MMoE 所有专家完全共享，每个任务通过 Gate 动态选择；任务相关性较强时效果好，但任务差异大时负迁移严重。PLE（CGC）增加任务专有专家层，共享专家和任务专家分离；适合任务相关性差异较大的场景（如点击和购买可能有较大差异）。快手 HoME 发现 PLE 的共享专家在超大规模下仍然发生 Degradation，需要更强的层次化约束。

**Q2：Expert Collapse 是什么现象？如何在训练中预防？**
> A：Expert Collapse 指部分专家的激活值大量为 0（ReLU 死亡），或者专家输出分布极度偏斜，Gate 网络无法公平分配工作。预防：① 激活函数改为 GELU/SiLU（避免 ReLU 死神经元）；② 辅助均衡损失（Load Balancing Loss）鼓励专家均匀被使用；③ Expert 归一化（对专家输出做 LayerNorm）稳定梯度；④ 监控每个 Expert 的激活率，设置报警阈值。

**Q3：Meta Lattice 如何将 N×M 个模型压缩为 1 个模型？关键技术是什么？**
> A：五大关键技术：① 跨域知识共享：统一 User Tower 捕获通用偏好；② 数据整合：打破产品线孤岛；③ 模型统一：动态路由/门控区分不同 domain/objective；④ 知识蒸馏：统一大模型→轻量服务模型；⑤ 系统优化：减少存储冗余和推理路径复杂度。核心是"统一但不失差异化"——共享主干 + 轻量 domain-specific 头。

**Q4：DHEN 的信息互补性实验发现了什么？对系统设计有何启示？**
> A：DHEN 实验发现：FM、DCN、CIN 等模块即使声称捕获相同阶数（如二阶）的特征交互，实际在信息空间的覆盖仍然存在显著差异（互补而非冗余）。设计启示：① 集成优于单一：不同交互模块捕获不同 "维度" 的特征关联；② 层次集成优于简单平均：深层堆叠让高层模块学习更复杂的综合交互；③ 多样性 > 深度：增加模块类型多样性往往比单纯堆叠更多相同类型层效益更高。

**Q5：工业 MoE 的路由负载均衡问题如何解决？**
> A：两大方向：① 辅助损失（Auxiliary Loss）：在主任务损失外加 $\mathcal{L}_{aux} = \alpha N \sum f_i P_i$，惩罚路由不均衡（GShard/Switch Transformer 方案）；② 无辅助损失方案：DeepSeek 的 Auxiliary-Loss-Free 路由（通过动态偏置调整 Expert 容量上限，无需额外损失项）。实践中两者各有优劣：辅助损失影响主任务精度，无辅助损失方案调试复杂度更高。

**Q6：为什么快手 HoME 要设计层次化结构而不是直接增大 Expert 数量？**
> A：直接增大 Expert 数量有三个问题：① Expert Collapse 概率随数量增加而上升（更多专家竞争相同梯度信号）；② 稀疏任务的专有专家更难训练（数据更少）；③ Gate 网络复杂度 O(K²) 增长，推理延迟上升。层次化结构的优势：高层专家专注通用知识（梯度信号丰富），低层专家专注任务差异（减少竞争），天然实现"分级管理"，避免所有任务争抢同一个专家池。

**Q7：统一多产品多目标模型（Lattice）的主要风险是什么？**
> A：① 任务间负迁移：某产品/目标表现不好会影响整体模型参数；② 风险耦合：某产品出现数据质量问题可能"污染"整个统一模型；③ 迭代速度变慢：任何变更都需全产品回归测试，而独立模型只需测试对应产品；④ 新产品接入复杂：需要设计 domain-specific 头 + 微调流程，不如独立模型直接。Meta 的解法：统一大模型 + 产品特定蒸馏小模型（大模型离线，服务时各产品用自己的蒸馏版）。

**Q8：多任务学习中如何平衡高频任务（点击）和低频任务（购买）的梯度？**
> A：三种方案：① 梯度手术（Gradient Surgery）/PCGrad：检测任务间梯度冲突（夹角 > 90°），对冲突梯度投影以消除干扰方向；② 不确定性加权（Kendall et al.）：根据任务不确定性自动调整权重 $w_k \propto 1/\sigma_k^2$；③ 动态权重调整：监控各任务 loss 下降速度，对收敛慢的任务动态增大权重（GradNorm）。快手 HoME 的实践：数据稀疏任务引入辅助监督（蒸馏高频任务知识），变相增加低频任务的有效训练信号。

**Q9：为什么 RankMixer 用 Token Mixing 替代 Self-Attention 能提升 MFU？**
> A：Self-Attention 的 O(n²) 计算对推荐场景（特征数通常 50-200）虽然可接受，但 Attention 计算需要 Q/K/V 矩阵乘法 + Softmax，存在大量内存访问跳跃（随机访问 KV Cache），GPU 流水线效率低。Token Mixing（MLP-Mixer 风格）用两个 MLP 分别做 token 间混合和 channel 内变换，计算模式规则、内存访问连续，GPU 矩阵并行效率高，MFU 从 4.5% 提升到 45%（10×）。

**Q10：如何设计一个能服务快手数十个行为预测任务的 MoE 系统？**
> A：借鉴 HoME 设计：① 任务分组：按任务相关性/数据量聚类（点击/完播/长播组，互动组，新颖组），组内专家共享，组间专家分离；② 层次化架构：底层共享基础专家（特征抽取），中层组内专家，顶层任务特定头；③ 稀疏任务处理：设计辅助监督信号（如用高频任务蒸馏低频任务专家）；④ Expert 监控：实时监控每个专家的激活率、门控分布；⑤ 渐进接入：新任务先 fine-tune 独立头，验证效果后再融入统一 MoE 系统。

## 📐 核心公式直观理解

### Shared-Bottom 多任务

$$
\hat{y}_k = \text{Tower}_k(\text{SharedBottom}(x))
$$

**直观理解**：底层共享 + 顶层分叉。所有任务共享同一个特征提取器，顶层各自学任务特定的映射。简单高效但有"跷跷板"问题——优化 CTR 可能损害 CVR（梯度方向冲突时底层被"拉扯"）。

### MoE 门控机制

$$
G(x) = \text{softmax}(W_g \cdot x), \quad \text{output} = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)
$$

**直观理解**：多个 expert 像"顾问团"，门控网络根据输入决定"这次听谁的"。不同类型的输入（如不同品类）可能激活不同的 expert 组合。比 shared-bottom 灵活——不同任务可以用不同的 expert 组合（通过 MMoE 的多门控）。

### 梯度冲突检测

$$
\cos(\nabla_{\theta} \mathcal{L}_1, \nabla_{\theta} \mathcal{L}_2) < 0 \Rightarrow \text{任务冲突}
$$

**直观理解**：如果两个任务的梯度方向夹角>90°（余弦<0），说明它们在"拉扯"共享参数——优化一个会恶化另一个。此时需要梯度调和（如 PCGrad 投影掉冲突分量）或增加任务特定参数（如 PLE 的私有 expert）。

---

## 相关概念

- [[concepts/multi_objective_optimization|多目标优化]]
- [[concepts/attention_in_recsys|Attention 在搜广推中的演进]]

---

## 记忆助手 💡

### 类比法

- **Shared-Bottom = 一个人干所有活**：所有任务共用一个底层网络，像一个员工同时负责销售、客服、财务，任务多了顾不过来
- **MMOE = 多个顾问团**：每个任务有自己的"老板"（Gate），根据输入决定听哪些顾问（Expert）的建议
- **PLE = 私人顾问 + 公共智库**：每个任务有专属顾问不被抢，同时共享公共资源
- **HoME层次化 = 公司组织架构**：高层管理（通用Expert）负责战略，中层（组内Expert）负责部门协调，基层（任务Expert）负责执行
- **Expert Collapse = 员工躺平**：某些 Expert 90%+ 神经元死亡（ReLU 死区），形同虚设，需要监控和干预
- **Meta Lattice = 集团统一管理**：N个产品×M个目标从各自为政变成一个统一模型，减少冗余但保留差异化

### 对比表

| 架构 | 核心机制 | 任务隔离度 | 参数效率 | 适用场景 |
|------|---------|-----------|---------|---------|
| Shared-Bottom | 底层全共享 | 无 | 最高 | 任务高度相关 |
| MMOE | 多Expert+多Gate | 低（软分离） | 中 | 任务中等相关 |
| PLE | 私有+共享Expert | 中（硬分离） | 较低 | 任务差异大 |
| HoME | 层次化Expert树 | 高 | 较低 | 数十个任务 |
| Meta Lattice | 统一模型+域自适应 | 中 | 高 | 多产品多目标 |

### 口诀/助记

- **多任务四代记**："全共享→软分离→硬分离→层次化" — Shared-Bottom → MMOE → PLE → HoME
- **Expert 三大病**："坍塌（死神经元）、退化（被单任务占据）、欠拟合（稀疏任务学不到）"
- **MoE 扩展金律**："稀疏激活扩参数，推理 FLOPs 不变，100× 参数但推理成本可控"
- **梯度冲突检测**："余弦小于零就冲突，PCGrad 投影消干扰"

### 面试一句话

- **MMOE vs PLE**："MMOE 所有 Expert 对所有任务可见只是权重不同，PLE 增加任务私有 Expert 其他任务不可见，从根本上隔离干扰，适合任务差异大的场景"
- **Expert Collapse**："工业 MoE 中某些 Expert 90%+ 神经元死亡，Gate 无法均匀分配。预防：用 GELU 替代 ReLU + 辅助均衡损失 + 激活率监控"
- **多任务梯度平衡**："GradNorm 监控各任务 loss 下降速度动态调权，PCGrad 对冲突梯度投影消除干扰方向，不确定性加权按 1/σ² 自动调权"
