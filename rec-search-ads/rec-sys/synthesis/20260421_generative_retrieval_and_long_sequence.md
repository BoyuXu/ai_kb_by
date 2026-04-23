# 生成式检索与长序列建模前沿 (2026-04-21)

> 本期聚焦：生成式检索 (Generative Retrieval) 新范式、长序列用户兴趣建模、多任务/多场景统一学习、以及 LLM 知识蒸馏在序列推荐中的落地。涵盖 10 篇论文，横跨快手、抖音、字节、Meta 等工业实践。

---

## 1. 总览表

| # | 论文 | 核心贡献 | 部署/验证 |
|---|------|----------|-----------|
| 1 | **DualGR** (WWW 2026) | 双分支长短期兴趣路由 + 搜索式 SID 解码 + 曝光感知损失 | 快手在线 +0.527% 播放 |
| 2 | **Next-User Retrieval** | 逆向视角：生成式预测"下一个用户"解决物品冷启动 | 抖音 +0.0142% DAU |
| 3 | **LIGER** (Meta) | 统一生成式与稠密检索，语义 ID + 文本嵌入混合 | 公开基准 SOTA |
| 4 | **DTN** | 多任务特征交互网络，替代 MMoE/PLE 的专家模块 | 电商 +3.28% CTR |
| 5 | **MDL** | 多场景多任务统一 Tokenization，Prompting 范式 | 抖音搜索 +0.0626% LT30 |
| 6 | **LONGER** (RecSys 2025) | GPU 友好长序列 Transformer，全局 token + token 合并 | 字节 10+ 场景全量上线 |
| 7 | **Efficient Dataset Selection** (ICLR 2026 Workshop) | 梯度表示 + 分布匹配的持续适应数据选择 | Spotify 十年数据验证 |
| 8 | **DLLM2Rec** (RecSys 2024) | LLM→轻量序列模型知识蒸馏，平均提升 47.97% | 三种经典序列模型 |
| 9 | **PerSRec** (ICDM 2025, Meta) | 长历史压缩为可学习 token，兼容 HSTU/HLLM | Meta 内部验证 |
| 10 | **GAMER** | 层级行为建模生成式推荐 + ShortVideoAD 数据集 | 短视频广告场景 |

---

## 2. 技术深度解析

### 2.1 DualGR: 双分支长短期兴趣生成式检索

**论文:** Zhongchao Yi, Kai Feng, Xiaojian Ma et al. (WWW 2026)

**问题定义:**
传统生成式检索 (GR) 将物品量化为有限 token 空间，通过自回归解码候选集。但现有方法面临三个关键挑战：
1. 长短期兴趣干扰——同一模型同时建模导致信号冲突
2. 层级 SID 生成中的上下文噪声——细粒度解码受粗粒度错误传播影响
3. 缺乏对曝光未点击反馈的显式学习

**架构创新:**

**(a) Dual-Branch Router (DBR):**
- 双分支选择性激活机制，分别建模长期稳定偏好和短期即时兴趣
- 路由网络动态决定每个请求应侧重哪个分支
- 避免了单一 Transformer 中长短期信号的互相干扰

**(b) Search-based SID Decoding (S2D):**
- 约束细粒度 token 解码仅在当前粗粒度桶 (bucket) 内进行
- 类似"先搜索再生成"的两阶段策略，显著降低噪声传播
- 解码效率提升的同时保持召回质量

**(c) Exposure-aware Next-Token Prediction Loss (ENTP-Loss):**
- 将曝光未点击样本作为粗粒度困难负样本
- 促进兴趣衰减的及时学习（用户看了但没点 → 兴趣下降信号）
- 公式：在标准 NTP loss 基础上增加曝光负样本的对比项

**实验结果:**
- 快手在线 A/B: +0.527% 视频播放量, +0.432% 观看时长
- 离线指标在多个 Recall@K 上显著优于 TIGER, DR, HSTU 等基线

**关键洞察:** ENTP-Loss 的设计非常巧妙——曝光未点击在传统模型中常被忽略或简单当作负样本，DualGR 将其作为"粗粒度困难负样本"处理，这个粒度选择比 item 级负样本更合理，因为用户可能只是对该类别暂时不感兴趣。

---

### 2.2 Next-User Retrieval: 逆向生成式模型解决冷启动

**论文:** Yu-Ting Lan, Yang Huo, Yi Shen, Xiao Yang, Zuotao Liu

**核心思路翻转:**
传统推荐是"给用户找物品"(next-item)，本文反过来"给物品找用户"(next-user)。这一视角转换对冷启动物品尤其有效——新物品没有足够的交互历史，但可以利用与之交互过的少量用户的序列来预测下一个潜在用户。

**架构设计:**
- Transformer-based 序列模型捕捉最近交互用户之间的单向关系
- 物品特征作为 prefix prompt embeddings 注入，引导 next-user 生成
- 灵感来自 lookalike 算法，但用生成式模型替代了传统的相似度匹配

**技术细节:**
1. 输入：物品的最近 N 个交互用户序列 $[u_1, u_2, \ldots, u_N]$
2. Prefix：物品特征嵌入（类别、标签、内容特征）
3. 输出：自回归预测下一个最可能交互的用户 $u_{N+1}$

**实验结果:**
- 抖音在线 A/B: +0.0142% DAU, +0.1144% 发布量
- 虽然绝对数字小，但在抖音这个量级下非常显著

**面试关注点:** 这篇论文的核心贡献不在于模型架构（标准 Transformer），而在于问题建模的创新——将冷启动问题从"物品表示不够好"转化为"用户序列预测"，巧妙绕开了新物品缺少交互的根本难题。

---

### 2.3 LIGER: 统一生成式与稠密检索

**论文:** Liu Yang, Fabian Paischer, Kaveh Hassani et al. (Meta, Facebook Research)

**动机:**
- 稠密检索 (Dense Retrieval): 为每个物品存储独立表示，内积排序，但内存开销随物品数线性增长
- 生成式检索 (Generative Retrieval): 用语义 ID 编码物品，自回归生成，内存高效但冷启动差
- 两者互补，但此前没有统一框架

**LIGER 架构:**
1. **语义 ID 输入分支:** 将物品编码为语义 ID 序列，通过 Transformer 解码生成候选
2. **文本嵌入分支:** 利用物品文本描述计算稠密向量表示
3. **混合候选生成:**
   - 第一步：生成式检索产生 K 个候选
   - 第二步：将冷启动物品集合并入候选池
   - 第三步：稠密检索对扩展候选池做精排

**关键创新:**
- 生成式检索负责高效缩小候选范围（从百万级到千级）
- 稠密检索补充冷启动物品的表示能力（文本嵌入不依赖交互历史）
- 联合训练使两个分支相互增强

**实验结果:**
- 在 Amazon 和 MovieLens 等公开数据集上实现 SOTA
- 冷启动场景提升尤为显著
- 代码开源: github.com/facebookresearch/liger

---

### 2.4 DTN: 深层多任务特征交互网络

**论文:** Yaowen Bi et al. (17 位作者)

**核心观察:**
MMoE 和 PLE 等经典多任务模型的专家模块本质上是 MLP，**没有显式的特征交互**。而在单任务排序模型中，特征交互（如 DeepFM, DCN）已被证明至关重要。

**"特征重要性分歧现象":**
- 同一特征在不同任务中重要性差异巨大
- 例如：用户年龄对 CTR 很重要，但对 CVR 影响较小
- 这要求模型能为每个任务学习独立的特征交互模式

**DTN 架构:**
1. **Task-specific Feature Interaction Modules:** 替代传统的 MLP 专家模块，每个任务拥有独立的特征交互层
2. **Diversified Interaction Methods:** 同一任务内使用多种不同的交互方法（如内积、外积、注意力），捕获不同粒度的交叉信息
3. **Task-sensitive Network:** 根据任务上下文自适应调节特征重要性权重

**实验结果:**
- 工业电商数据（63 亿样本）: +3.28% 点击, +3.10% 订单, +2.70% GMV
- 公开数据集上全面超越 MMoE, PLE, AITM 等基线

**与经典方法的关系:**

| 方法 | 专家类型 | 特征交互 | 任务敏感 |
|------|----------|----------|----------|
| MMoE | MLP | 无 | Gate |
| PLE | MLP | 无 | Progressive Gate |
| **DTN** | **Feature Interaction** | **多样化** | **Task-sensitive** |

---

### 2.5 MDL: 多分布统一学习器

**论文:** Shanlei Mu, Yuchen Jiang, Shikang Wu et al.

**问题背景:**
工业推荐系统通常需要同时处理多场景 (MSL) 和多任务 (MTL)。例如抖音搜索中，不同搜索场景（主搜、频道搜索等）× 不同任务（CTR、CVR、停留时长）形成高维分布空间。

**核心创新——Tokenization 范式:**
受 LLM prompting 启发，MDL 将场景和任务信息视为**专门的 token**，而非传统的辅助输入或门控信号。

**三层协同机制:**
1. **Feature Token Self-Attention:** token 化的特征之间的自注意力交互
2. **Domain-Feature Attention:** 场景/任务 token 作为 query，激活相关特征子集
3. **Domain-Fused Aggregation:** 融合多个场景×任务组合的预测

**逐层激活机制:**
场景 token 和任务 token 从底层到顶层逐步"激活"模型参数空间，实现从粗粒度到细粒度的分布建模。

**实验结果:**
- 抖音搜索在线 A/B (1 个月): +0.0626% LT30, -0.3267% 换 query 率
- 服务数亿日活用户
- 全面超越 STAR, PEPNet, AdaSparse 等 MSL/MTL 基线

---

### 2.6 LONGER: GPU 高效长序列推荐 Transformer

**论文:** Zheng Chai, Qin Ren, Xijun Xiao et al. (ByteDance, RecSys 2025)

**核心挑战:**
长行为序列（$L \geq 2000$）对捕获长短期偏好至关重要，但 vanilla Transformer 的 $O(L^2d)$ 注意力复杂度使其在工业场景中不可行。

**LONGER 架构三大创新:**

**(a) Global Token 机制:**
- 引入少量可学习的全局 token，与所有位置做注意力
- 稳定长上下文中的注意力分布，防止注意力稀释
- 类似 Longformer 的全局注意力思想，但针对推荐场景优化

**(b) Token Merge + InnerTransformer:**
- 将长序列分段，每段内用轻量级 InnerTransformer 处理
- 段间用混合注意力策略聚合
- 将复杂度从 $O(L^2)$ 降低到接近 $O(L \sqrt{L})$

**(c) 工程优化:**
- 混合精度训练 + 激活重计算：降低显存占用
- KV Cache 推理：加速在线 serving
- 全同步训练-推理框架：GPU 统一处理稠密和稀疏参数

**实验结果:**
- 字节跳动广告和电商双场景在线 A/B 均显著提升
- 已全量部署至 10+ 场景，服务数十亿用户

**工程启示:** LONGER 的价值不仅在于模型设计，更在于端到端的工程方案——从训练的混合精度到推理的 KV Cache，再到统一的 GPU 训练框架，解决了长序列在工业环境中"不可用"的根本问题。

---

### 2.7 Efficient Dataset Selection: 生成式推荐的持续适应

**论文:** Cathy Jiao, Juan Elenter, Praveen Ravichandran et al. (ICLR 2026 CAO Workshop, Oral)

**问题:**
大规模流式推荐系统中，用户行为分布持续漂移 (temporal drift)。全量重训成本高昂，需要高效的数据子集选择策略。

**方法论:**
1. **梯度表示 (Gradient-based Representations):**
   - 用模型梯度作为数据点的特征表示
   - 梯度向量编码了数据点对模型更新的"信息量"
2. **分布匹配 (Distribution Matching):**
   - 选择子集使其梯度分布匹配目标分布（最新数据）
   - 等价于最大化子集对当前分布的代表性
3. **评估维度:**
   - 对比多种表示选择（梯度 vs. 嵌入 vs. 随机）
   - 对比多种采样策略（分布匹配 vs. 不确定性 vs. 均匀）

**实验设置:**
- Spotify 十年真实音乐+播客交互数据
- 系统性评估时间漂移对性能的影响
- 证明小而精心选择的子集可恢复全量重训的大部分收益

**关键发现:**
- 梯度表示 + 分布匹配 = 最优组合
- 仅用 10-20% 数据即可接近全量重训效果
- 时间越近的数据价值越高，但不能只用最新数据（多样性重要）

---

### 2.8 DLLM2Rec: LLM 到序列模型的知识蒸馏

**论文:** Yu Cui, Feng Liu, Pengbo Wang et al. (RecSys 2024)

**动机:**
LLM-based 推荐模型效果好但推理延迟高，无法直接部署。能否将 LLM 的知识蒸馏到轻量级序列模型？

**三大挑战:**
1. 教师知识不一定可靠——LLM 在某些样本上也会犯错
2. 容量差距——学生模型难以完全吸收教师知识
3. 语义空间差异——LLM 的文本嵌入和序列模型的 ID 嵌入空间不同

**DLLM2Rec 架构:**

**(a) Importance-aware Ranking Distillation:**
- 根据教师置信度和师生一致性加权实例
- 教师越确定 + 学生已部分掌握的知识 → 高权重
- 过滤掉教师不确定的"噪声知识"

**(b) Collaborative Embedding Distillation:**
- 从数据中挖掘协同过滤信号
- 将其与教师嵌入知识融合
- 弥合文本语义空间和协同过滤空间的差距

**实验结果:**
- 对 SASRec, GRU4Rec, Caser 三种经典序列模型进行蒸馏
- 平均提升 47.97%
- 部分场景下蒸馏后的学生模型**反超 LLM 教师**

**面试重点:** "学生反超教师"现象的解释——LLM 擅长捕捉语义和世界知识，但对协同过滤信号的利用不如专用序列模型。蒸馏后的学生同时获得了两种信号，因此能超越仅有语义知识的教师。

---

### 2.9 PerSRec: 长历史压缩的通用框架

**论文:** Qiang Zhang, Hanchao Yu, Ivan Ji et al. (Meta, ICDM 2025)

**核心思想:**
将用户的长期交互历史压缩为少量**可学习的 token** (Peripatetic tokens)，然后与近期交互拼接送入序列推荐模型。

**设计动机:**
- Transformer 对序列长度 $L$ 的计算复杂度为 $O(L^2)$
- 用户历史可能有数千次交互，直接输入不现实
- 但长期历史包含重要的稳定偏好信号

**压缩机制:**
1. 长期历史（如前 1000 次交互）→ 独立编码器 → K 个可学习 token（$K \ll 1000$）
2. 近期历史（如最近 50 次交互）→ 保持原始 token
3. 拼接：$[\text{compressed}_1, \ldots, \text{compressed}_K, \text{recent}_1, \ldots, \text{recent}_{50}]$
4. 送入下游序列模型（HSTU 或 HLLM）

**与 LONGER 的对比:**

| 维度 | LONGER | PerSRec |
|------|--------|---------|
| 策略 | 改进注意力机制处理长序列 | 压缩长序列为少量 token |
| 复杂度 | $O(L\sqrt{L})$ | $O((K+L_{\text{recent}})^2)$，$K$ 很小 |
| 兼容性 | 专用架构 | 通用插件，兼容 HSTU/HLLM |
| 信息保留 | 高（全序列参与注意力） | 有损（压缩必然丢失细节） |
| 工程难度 | 高（需要定制 GPU 优化） | 低（即插即用） |

**实验结果:**
- 在 HSTU 和 HLLM 上均验证有效
- 计算成本显著下降，推荐精度基本持平
- 代码开源: github.com/facebookresearch/PerSRec

---

### 2.10 GAMER: 层级行为建模的生成式推荐

**论文:** Zhefan Wang, Guokai Yan, Jinbei Yu et al.

**问题:**
现有生成式推荐方法存在两个缺陷：
1. 序列建模不够充分——无法捕捉用户行为序列中的跨层级依赖
2. 缺少合适的数据集——公开多行为数据集几乎全来自电商

**GAMER 架构:**

**(a) Decoder-only Backbone:**
- 采用 GPT 风格的自回归生成架构
- 输入为物品的语义 ID 序列

**(b) Cross-level Interaction Layer:**
- 建模不同行为层级之间的依赖关系
- 例如：浏览 → 点击 → 点赞 → 转发 构成行为层级
- 跨层级注意力捕捉"浏览后点击"、"点击后转发"等转化模式

**(c) Sequential Augmentation Strategy:**
- 针对多行为序列的数据增强
- 提升训练鲁棒性，缓解数据稀疏

**ShortVideoAD 数据集:**
- 来源：主流短视频广告平台
- 特点：包含多种转瞬即逝的广告行为（曝光、点击、完播、转化等）
- 提供预训练语义 ID
- 填补了短视频广告场景多行为数据集的空白

**实验结果:**
- 在 ShortVideoAD 和两个公开电商多行为数据集上均优于判别式和生成式基线
- 短视频广告场景提升尤为显著

---

## 3. 跨论文洞察

### 3.1 生成式检索的三大演进方向

本批论文揭示了生成式检索从"能用"到"好用"的三条路径：

| 方向 | 代表论文 | 核心思路 |
|------|----------|----------|
| **兴趣解耦** | DualGR | 长短期分支 + 曝光感知 |
| **视角翻转** | Next-User Retrieval | 从 next-item 到 next-user |
| **范式统一** | LIGER | 生成式 + 稠密检索混合 |

DualGR 深耕"next-item"范式内的优化；Next-User Retrieval 开辟全新问题定义；LIGER 试图统一两大检索范式。三者并不冲突，可以在不同阶段组合使用。

### 3.2 长序列建模：两条技术路线

处理长用户行为序列目前形成了两条清晰的技术路线：

- **改进注意力 (LONGER):** 保留全序列信息，通过全局 token + token 合并降低复杂度
- **压缩历史 (PerSRec):** 将长历史压缩为可学习 token，即插即用

LONGER 适合有 GPU 资源且对精度要求极高的场景（字节级别），PerSRec 适合资源有限但需要快速迭代的场景（Meta 的通用插件方案）。

### 3.3 Tokenization 成为统一语言

从 MDL 的场景/任务 Tokenization，到 GAMER 的语义 ID，再到 DualGR 的 SID 体系，**Tokenization 正在成为推荐系统的统一表示语言**。这与 LLM 领域的发展趋势高度一致——将异构信息（特征、场景、任务、物品）都映射到统一的 token 空间，用 Transformer 架构统一处理。

### 3.4 LLM 知识的高效利用

DLLM2Rec 和 Efficient Dataset Selection 从不同角度回答了同一个问题：**如何在计算受限下最大化利用知识？**

- DLLM2Rec：从 LLM 教师模型中提取知识到轻量学生
- Efficient Dataset Selection：从海量数据中选择最有信息量的子集

两者的共同启示是：**不是越多越好，而是越精越好**。

### 3.5 工业部署的共同模式

| 模式 | 论文实例 |
|------|----------|
| 在线 A/B 验证 | DualGR (快手), MDL (抖音), LONGER (字节), Next-User (抖音) |
| 全量上线 | LONGER (10+ 场景), MDL (数亿用户) |
| 开源代码 | LIGER (Meta), PerSRec (Meta) |
| 新数据集贡献 | GAMER (ShortVideoAD) |

---

## 4. 面试 Q&A

### Q1: 生成式检索相比传统稠密检索有什么优劣势？DualGR 和 LIGER 分别如何解决其局限性？

**A:** 生成式检索 (GR) 用语义 ID 编码物品，自回归生成候选。优势：(1) 不需要为每个物品存储独立向量，内存高效；(2) 通过 cross-attention 显式建模目标-历史交互。劣势：(1) 冷启动物品缺少语义 ID 训练数据；(2) 长短期兴趣在单一解码器中容易冲突。

DualGR 用双分支路由 (DBR) 分离长短期兴趣，S2D 约束解码空间降噪，ENTP-Loss 利用曝光未点击信号。LIGER 则走统一路线，将生成式检索先缩小候选范围，再用稠密检索（基于文本嵌入）做精排，从而在冷启动场景也能工作。

### Q2: 如何在工业级推荐系统中处理超长用户行为序列（$L > 2000$）？对比 LONGER 和 PerSRec 的方案。

**A:** 两条路线：

**LONGER 路线（改进注意力）：** (1) 全局 token 稳定长程注意力；(2) token 合并 + InnerTransformer 降低复杂度到亚二次方；(3) 配套混合精度训练、KV Cache 推理等工程优化。优点是信息无损，缺点是需要大量工程投入。

**PerSRec 路线（压缩历史）：** 将长期历史编码为 K 个可学习 token（$K \ll L$），与近期交互拼接。复杂度从 $O(L^2)$ 降到 $O((K+L_{\text{recent}})^2)$。优点是即插即用、兼容 HSTU/HLLM，缺点是压缩有信息损失。

实际选择取决于场景：高价值场景（如广告竞价）用 LONGER 追求精度；通用推荐用 PerSRec 平衡效率。

### Q3: DLLM2Rec 中"学生反超教师"现象如何解释？这对 LLM 在推荐系统中的应用有何启示？

**A:** LLM 教师擅长语义理解和世界知识，但对协同过滤信号（用户-物品共现模式）的利用不如专用序列模型。DLLM2Rec 的学生模型通过 Collaborative Embedding Distillation 同时获得了 LLM 的语义知识和从数据中挖掘的协同信号，两种信息互补使学生超越教师。

启示：(1) LLM 不是推荐的终极方案，而是知识来源之一；(2) 最优策略可能是"LLM 提供知识 + 专用模型落地"的蒸馏范式；(3) 协同过滤信号在推荐中仍然不可替代。

### Q4: MDL 的 Tokenization 范式相比传统的多任务门控机制（MMoE/PLE）有何本质区别？

**A:** 传统 MMoE/PLE 用门控网络 (gate) 加权混合多个专家的输出，场景/任务信息作为辅助输入或门控条件。MDL 的 Tokenization 范式将场景和任务本身视为 token，与特征 token 平等参与注意力交互。

本质区别在于**信息融合的位置和方式**：
- MMoE/PLE：场景信息在**顶层**通过门控融合 → 底层表示与场景无关
- MDL：场景 token 从**底层**开始参与交互 → 每一层都是场景感知的

MDL 的三层机制（Self-Attention → Domain-Feature Attention → Domain-Fused Aggregation）实现了"从底到顶的逐层激活"，更充分地利用了模型参数空间。实验中 MDL 在抖音搜索上全面超越 PEPNet、STAR 等方法。

### Q5: 在推荐系统的持续训练 (continual training) 中，如何高效进行数据选择？梯度表示为什么比嵌入表示更好？

**A:** Efficient Dataset Selection 的核心发现是"梯度表示 + 分布匹配"是最优的数据选择策略。

梯度表示优于嵌入表示的原因：
- **嵌入表示**反映数据点"是什么"（语义信息），但不告诉模型"需要从中学什么"
- **梯度表示**直接编码数据点对模型参数更新的方向和幅度，反映"模型从这个样本能学到什么"
- 对于持续适应场景，我们关心的恰好是"哪些数据能最有效地更新模型"

分布匹配的直觉：选择子集使其梯度分布近似目标分布（最新数据的梯度分布），等价于用小数据集模拟全量训练的效果。实验表明，仅用 10-20% 数据即可恢复全量重训的大部分收益，对计算资源受限的场景极具实用价值。

---

## 5. 相关概念页链接

- [[attention_in_recsys]] — DualGR 的双分支注意力、LONGER 的全局 token 注意力、MDL 的 Domain-Feature Attention
- [[embedding_everywhere]] — LIGER 的语义 ID + 文本嵌入混合、PerSRec 的可学习压缩 token
- [[sequence_modeling_evolution]] — LONGER/PerSRec 的长序列建模、GAMER 的层级行为序列
- [[generative_recsys]] — DualGR/LIGER/GAMER/Next-User 的生成式推荐新范式
- [[multi_objective_optimization]] — DTN 的多任务特征交互、MDL 的多场景多任务统一

---

*Synthesized: 2026-04-21 | 10 papers | Topics: Generative Retrieval, Long Sequence Modeling, Multi-Task Learning, Knowledge Distillation*
