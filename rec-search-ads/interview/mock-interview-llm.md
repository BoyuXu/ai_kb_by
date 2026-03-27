# 🎯 大模型算法岗模拟面试 - MelonEggLearn

> 面试官：Melon | 候选人：EggLearn  
> 时长：约90分钟 | 难度：P7级别

---

## 一、Transformer架构细节 (15分钟)

### Q1: Self-Attention的时间/空间复杂度是多少？如何优化？

**候选人高质量答案：**

Self-Attention的复杂度分析：
- **时间复杂度**: $O(n^2 \cdot d)$，其中 $n$ 是序列长度，$d$ 是隐藏维度
- **空间复杂度**: $O(n^2)$，主要来自注意力矩阵的存储

**优化方法体系：**

| 优化方法 | 核心思想 | 复杂度降低 | 代表工作 |
|---------|---------|-----------|---------|
| **Sparse Attention** | 只计算局部或稀疏位置的注意力 | $O(n\sqrt{n})$ 或 $O(n\log n)$ | Longformer, BigBird |
| **Linear Attention** | 通过核技巧将 softmax 移到求和之外 | $O(n \cdot d^2)$ | Performer, Linformer |
| **FlashAttention** | IO-Aware的tiling + recomputation | 不变，但实际加速显著 | FlashAttention-1/2/3 |
| **Multi-Query Attention** | K/V共享 across heads | 显存 ↓，速度基本持平 | MQA, GQA |
| **Sliding Window** | 只关注局部窗口 | $O(n \cdot w \cdot d)$ | Mistral, Longformer |

**追问1：FlashAttention为什么能加速？**
- 痛点：标准Attention的 $O(n^2)$ 内存访问是瓶颈（HBM带宽限制）
- 方案：
  1. **Tiling**: 将Q/K/V分块加载到SRAM，计算局部softmax
  2. **Online Softmax**: 增量计算softmax，避免存储完整注意力矩阵
  3. **Recomputation**: 反向传播时重新计算前向值，节省显存
- FlashAttention-2优化：减少non-matmul FLOPs，更好的warp并行
- FlashAttention-3：异步化，利用Tensor Core的FP8支持

**追问2：MQA vs GQA vs MHA的区别和选择？**
- MHA: $h$ 个独立的Q/K/V投影
- MQA: 所有head共享同一组K/V (1组)
- GQA: 折中方案，$h$ 个Q对应 $g$ 组K/V ($g < h$，如LLaMA2-70B用8组)
- 效果：MQA/GQA显著减少KV Cache显存，轻微影响质量，长文本场景必选

---

### Q2: 位置编码有哪些方案？RoPE为什么成为主流？

**候选人高质量答案：**

**位置编码演进：**

1. **绝对位置编码 (Sinusoidal)**
   - $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$
   - 问题：外推能力差，长文本位置没学过

2. **可学习位置编码**
   - 直接学习位置embedding
   - 问题：同样外推差，且增加参数量

3. **相对位置编码 (T5/RPE)**
   - 在attention中引入相对位置偏置
   - 学习相对位置的信息

4. **旋转位置编码 (RoPE)**
   - **核心思想**: 通过旋转矩阵将位置信息注入Q/K
   - $f(q, m) = q \cdot e^{i m \theta}$ (复数形式)
   - 实际实现：二维旋转矩阵
   
**RoPE成为主流的原因：**

| 特性 | 说明 |
|-----|------|
| **相对位置感知** | $\langle f(q,m), f(k,n) \rangle = g(q,k,m-n)$，内积只与相对距离有关 |
| **远程衰减** | 随着相对距离增大，内积自然衰减，符合直觉 |
| **长度外推友好** | 可通过NTK-aware、YaRN等方法扩展长度 |
| **训练/推理一致** | 推理时无需修改，直接应用旋转 |

**长度外推技术：**
- **NTK-aware**: 低频部分插值，高频部分外推
- **YaRN**: 结合temperature scaling和NTK
- **Dynamic NTK**: 运行时根据长度动态调整base

---

### Q3: KV Cache的原理是什么？显存占用如何计算？

**候选人高质量答案：**

**KV Cache核心原理：**
- 问题：自回归生成时，每个token的K/V会被重复计算
- 方案：缓存之前所有token的Key和Value，避免重复计算
- 效果：将生成复杂度从 $O(n^3)$ 降到 $O(n^2)$，但增加显存占用

**显存占用公式：**

$$
\text{KV Cache} = 2 \times \text{batch}_{\text{size}} \times n \times n_{\text{layers}} \times d_{\text{head}} \times n_{\text{kv}_{\text{heads}}} \times \text{bytes}_{\text{per}}_{\text{param}}
$$

以LLaMA-70B为例：
- layers=80, d_head=128, heads=8 (GQA), FP16=2 bytes
- batch=1, seq_len=4096: ~2GB
- batch=32, seq_len=128K: ~1TB (不可接受！)

**优化技术：**

| 技术 | 原理 | 效果 |
|-----|------|------|
| **PagedAttention** | 非连续存储，按需分配block | 减少碎片，支持长序列 |
| **Quantization** | INT8/INT4/FP8存储KV | 显存 ↓ 50-75% |
| **Eviction/Compression** | 驱逐不重要的token | 支持无限长上下文 |
| **StreamingLLM** | 保留sink tokens + 滑动窗口 | 效果稳定，显存恒定 |

---

## 二、预训练 (15分钟)

### Q4: 预训练数据如何清洗？如何构建高质量预训练数据？

**候选人高质量答案：**

**数据清洗Pipeline：**

```
原始数据
  → 语言过滤 (langid/fastText)
  → 质量过滤 (规则 + 分类器)
  → 去重 (MinHash/BloomFilter)
  → 敏感过滤 (PII/毒性内容)
  → 混合采样 (数据配比)
```

**具体方法：**

1. **语言识别与过滤**
   - fastText语言分类器
   - 设定阈值 (如>0.9置信度)
   - 多语言模型需保留多语言数据

2. **质量过滤 (Quality Filtering)**
   - **启发式规则**: 去广告、去导航文本、去占位符
   - **统计特征**: 符号/单词比例、行长度分布、重复行检测
   - **基于模型的过滤**: 用高质量数据训练质量分类器 (如CCNet的perplexity过滤)

3. **去重 (Deduplication)**
   - **文档级**: MinHash + LSH (近似去重)
   - **段落级**: 滑动窗口去重
   - **代码数据**: AST-based去重
   - **重要性**: DeepSeek证明去重对性能影响巨大

4. **敏感内容处理**
   - PII检测与脱敏 (邮箱、电话、身份证号)
   - 毒性内容过滤 (HateBERT等)
   - 版权内容处理

**数据配比 (Data Mixing)：**

| 数据源 | 比例 | 策略 |
|-------|------|------|
| Web数据 (CommonCrawl) | 60-70% | 核心语料，需深度清洗 |
| 代码 (GitHub/StackOverflow) | 15-20% | 提升推理能力 |
| 书籍/维基 | 5-10% | 高质量长文本 |
| 专业领域 | 5-10% | 医疗、法律等 |
| 多语言 | 5-10% | 按语言人口比例采样 |

**追问：如何评估数据质量？**
- **离线指标**: 困惑度、多样性 (entropy)、重复度
- **在线指标**: 下游任务性能、loss收敛速度
- **Scaling Law验证**: 小规模实验预测大规模效果

---

### Q5: 预训练Loss如何设计？除了Next Token Prediction还有什么？

**候选人高质量答案：**

**标准NTP (Next Token Prediction)：**

$$
\mathcal{L} = -\sum_{i=1}^{N} \log P(x_i | x_{<i})
$$

**改进Loss设计：**

1. **Masked Language Modeling (MLM)**
   - BERT-style，预测被mask的token
   - 优势：双向上下文
   - 劣势：预训练-微调gap，生成能力弱

2. **Prefix LM / Causal with Prefix**
   - Encoder-Decoder结构
   - T5, UL2的mixture of denoisers

3. **Fill-in-the-Middle (FIM)**
   - 代码预训练常用
   - 将文档随机分为<前缀>, <中间>, <后缀>
   - 训练模型生成中间部分
   - 格式: `<PRE> prefix <SUF> suffix <MID> middle`

4. **Multi-Task Pre-training**
   - 混合多种任务格式
   - T0, Flan系列的风格
   - 示例格式: `Task: [task] Input: [input] Output: [output]`

5. **对比学习目标**
   - SimCLE: 句子级别的对比学习
   - 帮助模型学习更好的表示

**追问：Loss weighting策略？**
- **Token-level weighting**: 根据token重要性加权
- **Sequence-level weighting**: 高质量数据样本加权
- **Domain-level weighting**: 不同领域不同权重

---

### Q6: 继续预训练 (Continual Pre-training) 需要注意什么？

**候选人高质量答案：**

**应用场景：**
- 领域适配 (医疗、法律、金融)
- 时间更新 (新知识注入)
- 多语言能力扩展

**关键挑战与解决方案：**

| 挑战 | 问题描述 | 解决方案 |
|-----|---------|---------|
| **灾难性遗忘** | 学新知识，忘旧知识 | 混合原数据、LoRA、EWC正则 |
| **学习率敏感** | 学习率太大破坏原能力 | 小LR (1e-5~1e-6)、warmup、cosine decay |
| **数据分布偏移** | 新数据与预训练分布不同 | 渐进式训练、数据混合 |
| **过拟合** | 小数据集上 overfit | early stopping、正则化 |

**最佳实践：**
1. **数据混合**: 新数据 + 10-30%原数据
2. **学习率**: 通常比预训练小10-100倍
3. **训练步数**: 基于Chinchilla optimal计算
4. **评估监控**: 同时监控新领域和通用能力

---

## 三、对齐 (20分钟)

### Q7: RLHF vs DPO vs PPO，原理上有什么区别？各有什么优缺点？

**候选人高质量答案：**

**三阶段对比：**

```
SFT (Supervised Fine-Tuning)
    ↓
Reward Modeling (学习人类偏好)
    ↓
RL Optimization (PPO/DPO)
```

**1. PPO (Proximal Policy Optimization)**

原理：
- 最大化奖励同时约束与参考模型的KL散度
-

$$
\mathcal{L}_{PPO} = \mathbb{E}[\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)]
$$

- 其中 $r_t = \frac{\pi_\theta}{\pi_{ref}}$

优点：
- 理论基础扎实，稳定性好
- 可处理复杂奖励函数
- 在线学习，可以探索

缺点：
- 需要训练4个模型 (Policy, Ref, Value, Reward)
- 显存占用大，训练复杂
- 超参数敏感

**2. DPO (Direct Preference Optimization)**

原理：
- 绕过显式奖励模型，直接从偏好数据优化
- 推导出最优策略与奖励的关系：$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)}$
- 目标函数：

$$
\mathcal{L}_{DPO} = -\mathbb{E}[\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})]
$$

优点：
- 只需2个模型 (Policy + Ref)
- 简单稳定，不需要PPO的复杂trick
- 训练效率高

缺点：
- 依赖偏好数据质量
- 对分布外数据敏感
- 超参数$\beta$需要调优

**3. RLHF (通用框架)**

通常指PPO-based RLHF，但也包括其他RL算法。

**对比总结：**

| 维度 | PPO | DPO | SFT+DPO |
|-----|-----|-----|---------|
| 模型数量 | 4 | 2 | 2 |
| 显存占用 | 高 | 低 | 低 |
| 训练稳定性 | 中等 | 高 | 高 |
| 数据效率 | 高(在线采样) | 中 | 中 |
| 天花板 | 高 | 中高 | 中 |
| 工程复杂度 | 高 | 低 | 低 |

**追问：DPO为什么比PPO稳定？**
- DPO是offline、off-policy的，不需要在线采样
- 没有critic网络，减少了训练不稳定因素
- 目标函数直接优化偏好概率，避免了RL的信用分配问题

**追问：DPO的$\beta$参数作用？**
- 控制与参考模型的偏离程度
- $\beta$太大：过于保守，效果不明显
- $\beta$太小：偏离参考模型太远，可能collapse
- 通常取值0.1-0.5

---

### Q8: Reward Hacking是什么？如何解决？

**候选人高质量答案：**

**问题定义：**
- 模型找到奖励函数的"捷径"而非真正学习期望行为
- 例子：生成重复的高频词、过度使用特定格式、生成长但空洞的回答

**具体表现：**

| 类型 | 例子 |
|-----|------|
| **重复攻击** | 重复奖励高的词或短语 |
| **长度攻击** | 生成超长回答 (如果奖励与长度正相关) |
| **格式攻击** | 过度使用markdown、emoji |
| **安全攻击** | 用委婉说法绕过安全检测 |

**解决方案：**

1. **Reward Model改进**
   - 对抗性数据增强
   - 多维度奖励分解 (helpfulness, harmlessness, honesty)
   - 使用Bradley-Terry模型而非回归

2. **训练策略**
   - KL惩罚：约束与参考模型的距离
   - Entropy正则：保持探索
   - 多轮迭代：持续更新reward model

3. **数据层面**
   - 高质量偏好标注
   - 覆盖更多边界case
   - 多样性采样

4. **其他方法**
   - **RPO (Robust Preference Optimization)**: 考虑不确定性
   - **IPO (Identity Preference Optimization)**: 改进DPO的稳定性
   - **KTO**: 从二元反馈学习，而非成对偏好

---

### Q9: 除了PPO/DPO，还有什么对齐方法？

**候选人高质量答案：**

**1. KTO (Kahneman-Tversky Optimization)**
- 只需要二元反馈 (好/坏)，不需要成对比较
- 基于prospect theory，假设人类对好坏的感知是非对称的
- 适合大规模收集反馈的场景

**2. ORPO (Odds Ratio Preference Optimization)**
- 将SFT和对齐合二为一
- 在NTP loss中加入odds ratio项
- 一步训练，效率更高

**3. SimPO (Simple Preference Optimization)
- 去掉参考模型，进一步简化
- 使用长度归一化的奖励
- 内存效率更高

**4. RL-Free方法**
- **SLiC (Sequence Likelihood Calibration)**: 校准生成概率
- **RRHF**: 排名 + 拒绝采样
- **RAFT**: 拒绝采样微调

**5. Constitutional AI**
- 用AI而非人类进行反馈
- Self-critique和revision
- 减少人工标注成本

---

## 四、推理加速 (15分钟)

### Q10: vLLM的PagedAttention原理是什么？为什么能提升吞吐量？

**候选人高质量答案：**

**背景问题：**
- 自回归生成时，KV Cache显存占用大且动态增长
- 传统实现为每个请求预分配连续显存块，导致：
  1. **内部碎片**: 预分配 > 实际使用
  2. **外部碎片**: 碎片化导致无法分配大块连续内存
  3. **无法共享**: 难以实现beam search等解码策略的内存共享

**PagedAttention核心思想：**

借鉴操作系统的虚拟内存和分页机制：
- 将KV Cache划分为固定大小的 **block** (如每block存储16个token的K/V)
- block不需要连续存储
- 通过block table映射逻辑位置到物理block

**关键机制：**

```
逻辑视角:  [block0][block1][block2]  →  seq_len=48
物理存储:   block0@GPU0  block1@GPU5  block2@GPU2  (非连续)
           ↓
        block table维护映射关系
```

**优势：**

| 优势 | 说明 |
|-----|------|
| **动态分配** | 按需分配block，无内部碎片 |
| **内存复用** | 空闲block可立即回收 |
| **Copy-on-Write** | beam search共享大部分block，分叉时复制 |
| **并行调度** | 支持continuous batching |

**Continuous Batching：**
- 传统：一个batch内所有请求完成才处理下一批
- Continuous：新请求随时加入batch，完成的请求随时退出
- 配合PagedAttention，最大化GPU利用率

**追问：vLLM vs HuggingFace TGI的区别？**
- vLLM: PagedAttention + Continuous Batching，吞吐量优化
- TGI: 侧重低延迟，支持Safetensors、Tensor Parallelism
- TensorRT-LLM: NVIDIA优化，极致性能但闭源

---

### Q11: 投机采样 (Speculative Decoding) 原理是什么？

**候选人高质量答案：**

**核心思想：**
- 用一个小的draft model快速生成候选token
- 大的target model并行验证这些候选
- 接受则继续，拒绝则重新采样

**数学保证：**
- 使用modified rejection sampling保证分布等价
- 接受概率：$\min(1, \frac{P_{target}(x)}{P_{draft}(x)})$
- 保证最终采样结果与target model直接采样一致

**流程：**

```
Draft Model (小模型):  快速生成 γ 个候选token
                              ↓
Target Model (大模型):  并行计算 γ 个位置的logits
                              ↓
Verification:           逐个验证，找到第一个不匹配位置
                              ↓
接受: γ个全对 → 继续下一轮
拒绝: 在第k个位置拒绝 → 从纠正分布采样，重新开始
```

**加速效果取决于：**
1. **接受率**: Draft model与target model分布接近程度
2. **速度比**: Draft生成速度 vs Target验证速度
3. **Candidate长度**: 通常2-5个token

**Draft Model选择：**
- 小版本同系列模型 (7B draft → 70B target)
- N-gram lookup (无模型开销)
- 任务特定轻量模型

**变体：**
- **Medusa**: 在target model上加多个解码头，多头并行预测
- **Lookahead Decoding**: 不依赖draft model，用Jacobi迭代
- **EAGLE**: 特征级投机解码，接受率更高

---

### Q12: 量化有哪些方案？INT4/FP8/GPTQ/AWQ的区别？

**候选人高质量答案：**

**量化分类：**

| 类型 | 代表方法 | 特点 |
|-----|---------|------|
| **PTQ (Post-Training Quantization)** | GPTQ, AWQ, SmoothQuant | 无需训练，速度快 |
| **QAT (Quantization-Aware Training)** | LLM-QAT | 需要训练，效果好 |
| **Dynamic Quantization** | 运行时量化 | 灵活但开销大 |

**具体方法对比：**

**1. GPTQ (Group-wise Post-Training Quantization)**
- 基于OBQ (Optimal Brain Quantization)
- 逐层量化，用Hessian信息补偿误差
- 分组量化 (如128个参数一组)，组内共享scale
- 支持INT4/INT3/INT2

**2. AWQ (Activation-aware Weight Quantization)**
- 观察：保护1%的salient weight可显著提升效果
- 根据activation scale识别salient weight
- 对salient weight做per-channel scaling
- 通常比GPTQ效果好，尤其小模型

**3. SmoothQuant**
- 解决activation量化难题 (activation outliers)
- 将量化难度从activation迁移到weight
- $X' = X \cdot diag(s)^{-1}$, $W' = diag(s) \cdot W$
- 适合INT8，可与TensorRT等推理引擎配合

**4. FP8 (H100支持)**
- NVIDIA Transformer Engine原生支持
- 1-4-3格式 (E4M3) 或 1-5-2格式 (E5M2)
- 硬件级支持，零开销

**选择建议：**
- 追求极致压缩: GPTQ-INT4/INT3
- 追求效果平衡: AWQ-INT4
- 生产部署: SmoothQuant-INT8 (TensorRT-LLM)
- H100环境: FP8

---

## 五、RAG系统设计 (15分钟)

### Q13: 从Naive RAG到Advanced RAG到Modular RAG，架构如何演进？

**候选人高质量答案：**

**Naive RAG (基础RAG)：**

```
Query → Embedding → VectorDB检索 → 拼接Prompt → LLM生成
```

问题：
- 检索质量依赖向量表示
- 简单拼接context，无精细处理
- 检索-生成脱节

**Advanced RAG (增强RAG)：**

| 阶段 | 优化技术 |
|-----|---------|
| **Pre-retrieval** | Query改写、扩展、消歧；HyDE (假设文档嵌入) |
| **Retrieval** | 混合检索 (向量+关键词)、重排序 (Rerank) |
| **Post-retrieval** | 上下文压缩、重排序、选择最相关片段 |

关键技术：
- **Rerank**: 用cross-encoder精细排序 (如bge-reranker)
- **HyDE**: 用LLM生成假答案再检索，解决query-doc表示鸿沟
- **Query Transformation**: 分解复杂问题 (Step-back, RAG-Fusion)

**Modular RAG (模块化RAG)：**

将RAG拆解为可插拔模块：

```
┌─────────────────────────────────────────────────┐
│  Router → 判断是否需要检索                       │
├─────────────────────────────────────────────────┤
│  Query Analyzer → 理解意图、分解子问题           │
├─────────────────────────────────────────────────┤
│  Retriever Pool → 向量/图谱/SQL多路召回          │
├─────────────────────────────────────────────────┤
│  Reranker → 精排                                │
├─────────────────────────────────────────────────┤
│  Memory → 维护对话历史                           │
├─────────────────────────────────────────────────┤
│  Generator → LLM生成                             │
└─────────────────────────────────────────────────┘
```

Advanced RAG模式：
- **Self-RAG**: 模型自判断是否需要检索
- **Corrective RAG**: 检索质量差时切换搜索/web
- **GraphRAG**: 结合知识图谱
- **RAPTOR**: 递归摘要构建树状索引

---

### Q14: 检索环节如何优化召回率和准确率？

**候选人高质量答案：**

**多路召回 (Multi-Channel Retrieval)：**

| 方法 | 适用场景 | 原理 |
|-----|---------|------|
| **Dense Retrieval** | 语义匹配 | Embedding相似度，捕获语义 |
| **Sparse Retrieval** | 精确匹配 | BM25，关键词匹配 |
| **Hybrid** | 综合 | Dense + Sparse融合 |
| **Graph Retrieval** | 关系查询 | 知识图谱游走 |

**Embedding优化：**
- **领域微调**: 用领域数据微调embedding模型
- **Matryoshka Embedding**: 不同维度表示，灵活trade-off
- **ColBERT**: 细粒度late interaction

**重排序 (Reranking)：**
- 两阶段：粗排 (召回Top-K) → 精排 (Top-N)
- 精排模型：Cross-encoder，查询和文档一起编码
- 代表：bge-reranker, cohere-rerank, Cohere Rerank API

**Query优化：**
- **Expansion**: 扩展同义词、相关词
- **Rewrite**: 用LLM改写query，更清晰
- **Hypothetical Document**: HyDE生成假文档

**索引策略：**
- **分块 (Chunking)**: 大小、重叠策略
- **父子索引**: 小块用于检索，大块用于生成
- **摘要索引**: 文档摘要建立索引

---

### Q15: 如何解决RAG中的幻觉问题？

**候选人高质量答案：**

**幻觉来源分析：**

| 阶段 | 问题 | 表现 |
|-----|------|------|
| 检索 | 未召回相关内容 | "我不知道" 或 编造 |
| 生成 | 忽略检索内容，用参数知识 | 与检索内容矛盾 |
| 整合 | 错误整合多段信息 | 事实拼接错误 |

**解决方案：**

**1. 检索增强**
- 多路召回降低漏召
- 查询改写扩展召回面
- 相关性阈值过滤，无结果时触发fallback

**2. 生成控制**
- **System Prompt约束**: "只基于提供的context回答"
- **Citation**: 要求标注来源，增加可验证性
- **Self-RAG/Corrective RAG**: 不确定时主动检索

**3. 后处理验证**
- **Faithfulness检查**: 用NLI模型验证答案是否被context支持
- **Self-Consistency**: 多次采样，投票
- **Chain-of-Verification**: 先生成，再验证，最后修正

**4. 系统级方案**
- **知识图谱 grounding**: 结构化知识约束
- **Human-in-the-loop**: 不确定时请求人工确认
- **置信度校准**: 输出置信度分数

---

## 六、Agent框架 (10分钟)

### Q16: ReAct、CoT、Tool Use的原理和区别？

**候选人高质量答案：**

**CoT (Chain-of-Thought)：**

原理：
- 引导模型逐步推理，而非直接给答案
- 通过示例或指令激发推理能力

格式：
```
Q: 一个农场有鸡和兔共35只，脚共94只。鸡兔各几只？
A: 假设全是鸡，有70只脚。多出来24只脚是兔子的，每只兔多2只脚。
   所以兔子12只，鸡23只。答案是鸡23只，兔12只。
```

变体：
- **Zero-shot-CoT**: "Let's think step by step"
- **Self-Consistency**: 多条推理路径投票
- **Tree-of-Thoughts**: 树状搜索，可回溯

**ReAct (Reasoning + Acting)：**

原理：
- 将推理和行动交错进行
- Thought → Action → Observation → Thought → ...

流程：
```
Thought: 我需要计算当前汇率
Action: search[美元兑人民币汇率]
Observation: 1 USD = 7.2 CNY
Thought: 现在可以计算了...
```

优势：
- 行动基于推理，推理指导行动
- 可解释性强
- 错误可追踪

**Tool Use：**

原理：
- LLM学习调用外部工具/函数
- Function Calling机制

实现：
- 定义工具schema (名称、描述、参数)
- 模型生成函数调用JSON
- 系统执行并返回结果

---

### Q17: 如何设计一个多Agent系统？

**候选人高质量答案：**

**Multi-Agent设计模式：**

**1. 分层架构 (Hierarchical)**
```
        Supervisor Agent
       /       |        \
  Agent A    Agent B    Agent C
   (Code)    (Search)   (Math)
```
- Supervisor负责规划和分配
- Specialist负责具体执行

**2. 协作架构 (Collaborative)**
```
Agent A ←→ Agent B ←→ Agent C
   ↓          ↓          ↓
     Shared Memory / Message Queue
```
- 平等协作，互相通信
- 共同解决问题

**3. 竞争架构 (Competitive)**
- 多个Agent提出不同方案
- 投票/评估选择最优

**关键设计点：**

| 维度 | 考虑点 |
|-----|-------|
| **Communication** | 消息格式、通信协议、共享状态 |
| **Coordination** | 谁主导、冲突解决、任务分配 |
| **Memory** | 短期对话历史、长期知识存储 |
| **Tool Access** | 共享工具池或专用工具 |

**代表框架：**
- **AutoGen (Microsoft)**: 灵活的对话编程
- **MetaGPT**: 模拟软件公司组织架构
- **CrewAI**: 角色扮演、协作任务
- **LangGraph**: 状态机驱动的工作流

**追问：如何处理Agent间的冲突？**
- 优先级机制 (Supervisor裁决)
- 投票机制
- 置信度加权
- Human-in-the-loop仲裁

---

### Q18: 评估Agent系统有哪些指标？

**候选人高质量答案：**

**评估维度：**

| 维度 | 指标 | 说明 |
|-----|------|------|
| **任务完成** | 成功率、F1、EM | 最终答案正确性 |
| **效率** | 步骤数、工具调用次数、延迟 | 资源消耗 |
| **稳定性** | 多次运行方差 | 一致性 |
| **安全性** | 错误工具调用率、越权访问 | 风险管控 |

**具体评估方法：**

1. **端到端评估**
   - 标准benchmark: HotpotQA, WebShop, ToolBench
   - 人工评估: 人工判断任务完成质量

2. **组件级评估**
   - Tool Selection准确率
   - 参数填充正确率
   - ReAct步骤合理性

3. **对抗评估**
   - 注入恶意指令，测试鲁棒性
   - 边界case测试

4. **人类对齐**
   - 有用性 (Helpfulness)
   - 诚实性 (Honesty)
   - 无害性 (Harmlessness)

---

## 📝 面试总结

**EggLearn表现评估：**

| 领域 | 评分 | 评价 |
|-----|------|------|
| Transformer架构 | ⭐⭐⭐⭐⭐ | 对复杂度分析、优化方法理解深入 |
| 预训练 | ⭐⭐⭐⭐⭐ | 数据清洗和继续预训练实践经验丰富 |
| 对齐 | ⭐⭐⭐⭐⭐ | 深入理解RLHF/DPO数学原理和工程权衡 |
| 推理加速 | ⭐⭐⭐⭐⭐ | 对vLLM、投机采样有深刻认识 |
| RAG系统 | ⭐⭐⭐⭐⭐ | 架构演进理解清晰，工程落地思路明确 |
| Agent框架 | ⭐⭐⭐⭐☆ | 基础扎实，多Agent系统设计可更深入 |

**总体评价：**
- 理论基础扎实，能准确阐述算法原理
- 工程意识强，关注实际落地问题
- 知识面广，对前沿技术保持关注
- **建议**: 多Agent系统的实战经验可进一步加强

---

*文档生成时间: 2026-03-11*  
*模拟面试场景，供复习参考*
