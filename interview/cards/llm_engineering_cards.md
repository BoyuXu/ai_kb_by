# LLM工程 分层知识卡片

> 生成日期：2026-03-16 | MelonEggLearn
> 覆盖：预训练 · 微调 · 推理优化 · RAG · 部署工程 · Prompt Engineering

---

## 模块地图（ASCII知识树）

```
LLM工程
├── 预训练
│   ├── Transformer架构（Encoder-only / Decoder-only / Encoder-Decoder）
│   ├── Attention机制
│   │   ├── MHA（Multi-Head Attention）
│   │   ├── MQA（Multi-Query Attention）
│   │   └── GQA（Grouped-Query Attention）
│   └── 位置编码
│       ├── Sinusoidal（绝对位置编码）
│       ├── RoPE（旋转位置编码）
│       └── ALiBi（线性偏置）
│
├── 微调
│   ├── SFT（有监督微调）
│   ├── LoRA / QLoRA（低秩适配）
│   └── RLHF
│       ├── PPO（近端策略优化）
│       └── DPO（直接偏好优化）
│
├── 推理优化
│   ├── KV Cache + PagedAttention
│   ├── 量化（INT8/INT4/GPTQ/AWQ/SmoothQuant）
│   ├── 推测解码（Speculative Decoding）
│   └── Continuous Batching
│
├── RAG
│   ├── 检索增强生成架构
│   ├── Chunking策略
│   ├── 重排（Reranker）
│   └── HyDE（假设文档嵌入）
│
├── 部署工程
│   ├── vLLM / TGI
│   ├── Tensor Parallelism
│   └── Pipeline Parallelism
│
└── Prompt Engineering
    ├── CoT（思维链）
    ├── Few-shot
    ├── ReAct
    └── 结构化输出
```

---

## L1 概念卡（What）

### L1-01 | Transformer 三大架构变体

**Q：Transformer 三种架构各是什么？代表模型是哪些？**

**A：**
| 架构 | 代表模型 | 注意力方向 | 适合任务 |
|------|----------|-----------|---------|
| Encoder-only | BERT、RoBERTa | 双向 | NLU（分类、NER、问答抽取） |
| Decoder-only | GPT、LLaMA、Qwen | 单向（因果） | 文本生成、代码生成 |
| Encoder-Decoder | T5、BART | 双向+单向 | 翻译、摘要、问答生成 |

---

### L1-02 | Multi-Head Attention（MHA）

**Q：什么是 Multi-Head Attention？核心公式是什么？**

**A：**
- 并行多组 Attention，每组学习不同子空间的语义关系
- 核心公式：

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
MHA(Q,K,V) = Concat(head₁,...,headₕ) · Wₒ
```

- 缩放因子 `√d_k` 防止点积过大导致 softmax 梯度消失
- 时间复杂度：**O(n²d)**，n 为序列长度

---

### L1-03 | MQA 与 GQA

**Q：MQA 和 GQA 是什么？解决了什么问题？**

**A：**
- **MQA（Multi-Query Attention）**：多个 Q 共享单组 K/V → 大幅减少 KV Cache 显存，推理更快，但精度略降
- **GQA（Grouped-Query Attention）**：Query 分组，每组共享一对 K/V → MHA 与 MQA 的平衡方案
- **应用**：LLaMA2、Mistral、Gemma 使用 GQA；PaLM 使用 MQA

---

### L1-04 | 位置编码：RoPE vs ALiBi

**Q：RoPE 和 ALiBi 是什么？各有何特点？**

**A：**
- **RoPE（Rotary Position Embedding，旋转位置编码）**
  - 将位置信息编码为 Q/K 的旋转变换
  - 支持相对位置感知，外推能力强
  - 应用：LLaMA、Qwen、ChatGLM

- **ALiBi（Attention with Linear Biases）**
  - 在 Attention score 上添加线性偏置（距离越远偏置越负）
  - 无需额外参数，对超长序列外推友好
  - 应用：BLOOM

---

### L1-05 | LoRA（低秩适配）

**Q：什么是 LoRA？**

**A：**
- Low-Rank Adaptation：冻结预训练权重，只训练低秩分解矩阵
- 核心思想：权重更新量 ΔW 具有低秩结构，用 B·A 近似
- **参数减少 10~1000 倍**，训练显存大幅下降
- 推理时可合并（W = W₀ + BA），**零额外延迟**

---

### L1-06 | RLHF（人类反馈强化学习）

**Q：什么是 RLHF？包含哪些步骤？**

**A：**
1. **SFT**：用高质量示范数据做有监督微调
2. **奖励建模**：用人类偏好排序训练奖励模型（RM）
3. **RL 优化**：用 PPO/DPO 让模型最大化奖励，同时加 KL 散度约束防止偏离

- **PPO**：需要 Actor、Critic、RM、Reference 四个模型，训练复杂
- **DPO**：无需独立 RM，直接优化偏好对，更简洁

---

### L1-07 | KV Cache

**Q：KV Cache 是什么？解决了什么问题？**

**A：**
- 推理解码时，将每层 Attention 的 Key/Value 矩阵缓存起来
- 避免自回归生成每步重复计算所有历史 token 的 KV
- **代价**：显存占用随序列长度线性增长
- 传统 KV Cache 有严重碎片化（浪费 60%-80% 显存）

---

### L1-08 | 推测解码（Speculative Decoding）

**Q：什么是推测解码？**

**A：**
- 用小模型（Draft Model）快速生成多个候选 token
- 大模型（Target Model）并行验证，接受正确的、拒绝错误的并重采样
- **本质**：用小模型的速度换取大模型的并行验证，降低生成延迟
- 加速比取决于接受率（acceptance rate），通常 2-3x

---

### L1-09 | RAG（检索增强生成）

**Q：什么是 RAG？基本架构是什么？**

**A：**
- Retrieval-Augmented Generation：生成前先检索相关文档，拼入 prompt
- 基本架构：
  ```
  Query → Retriever（向量检索/BM25）→ Top-K 文档 → LLM 生成答案
  ```
- 解决 LLM 幻觉问题和知识时效性问题
- 三段式：**索引（Index）→ 检索（Retrieve）→ 生成（Generate）**

---

### L1-10 | Continuous Batching

**Q：什么是 Continuous Batching（连续批处理）？**

**A：**
- 传统 Static Batching：等一批请求全部完成才处理下一批，GPU 利用率低
- **Continuous Batching**：请求完成即移出批次，新请求立即插入
- 效果：GPU 利用率接近 100%，吞吐量提升 **3-10x**
- vLLM、TGI 均支持

---

## L2 原理卡（How）

### L2-01 | Attention 缩放原理

**Q：Self-Attention 为什么要除以 √d_k？推导思路？**

**A：**
- Q·Kᵀ 的每个元素是 d_k 维向量的点积，期望方差为 d_k
- 不缩放时方差随维度增大，softmax 输出接近 one-hot → 梯度消失
- 除以 √d_k 使方差归一化为 1，保持梯度流动
- 类比：Xavier 初始化也用 1/√n 缩放

---

### L2-02 | LoRA 核心公式

**Q：LoRA 的数学原理是什么？**

**A：**
```
W = W₀ + ΔW = W₀ + B·A

其中：
  W₀ ∈ ℝ^(d×d)  — 冻结的预训练权重
  A  ∈ ℝ^(r×d)  — 随机高斯初始化，训练
  B  ∈ ℝ^(d×r)  — 零初始化，训练
  r << d         — 秩（rank），通常 4~64

前向传播：h = W₀x + (B·A)x · (α/r)
  α 是缩放超参，通常 α = r
```
- 参数量：d×d → 2×d×r，节省比例 = d/(2r)
- 推理时合并：W_merged = W₀ + B·A，**无额外推理开销**

---

### L2-03 | KV Cache 内存计算

**Q：KV Cache 占用显存如何计算？**

**A：**
```
KV Cache 显存 = 2 × num_layers × num_heads × head_dim × seq_len × batch_size × dtype_bytes

例：LLaMA-7B（32层, 32头, head_dim=128, FP16=2B）
  单请求 seq_len=2048：
  = 2 × 32 × 32 × 128 × 2048 × 1 × 2 bytes
  = 2 × 32 × 32 × 128 × 2048 × 2
  ≈ 1 GB

批量 32 请求 ≈ 32 GB → 显存瓶颈！
```
- GQA/MQA 可将 KV Cache 缩小 num_heads/num_kv_groups 倍
- PagedAttention 解决碎片化，利用率从 <40% → >90%

---

### L2-04 | 推测解码加速原理

**Q：推测解码（Speculative Decoding）如何实现加速？**

**A：**
```
步骤：
1. Draft Model 快速生成 K 个候选 token：t₁, t₂, ..., tₖ
2. Target Model 一次前向，并行计算所有 K 个位置的概率分布
3. 接受/拒绝（基于概率比 p_target/p_draft）
   - 若接受：保留该 token
   - 若拒绝：从修正分布重采样，丢弃后续
4. 至少接受 1 个 token（安全保证），通常接受率 70-90%

加速比 ≈ 1 / (1 - acceptance_rate × K / 期望步数)
实际加速 2-3x，不改变输出分布（等价于 Target Model 直接生成）
```

---

### L2-05 | GPTQ 量化原理

**Q：GPTQ 量化的核心机制是什么？**

**A：**
- 逐层量化，每层独立处理
- 使用 **二阶信息（Hessian 矩阵）** 最小化量化误差
- "Lazy Batch-Updates"：逐列量化，误差传播到未量化列补偿
- 支持 2/3/4 bit，175B 模型可在 4×A100 上量化
- **量化误差公式**：
  ```
  min_Q ||WX - Q̂X||²_F   subject to Q̂ = quantize(W)
  ```

---

### L2-06 | AWQ 量化原理

**Q：AWQ 与 GPTQ 的区别？AWQ 如何保护重要权重？**

**A：**
- **AWQ（Activation-aware Weight Quantization）**
- 观察：激活值有少量"重要通道"（salient channels），量化这些通道损失大
- 方法：对重要权重通道乘以缩放因子 s，使其更难被量化损坏
  ```
  W_scaled = W · diag(s),  X_scaled = X · diag(s)⁻¹
  量化 W_scaled，推理时 s 与 X 的缩放互相抵消
  ```
- 效果：INT4 精度接近 FP16，且无需校准数据集

---

### L2-07 | PagedAttention 原理

**Q：vLLM 的 PagedAttention 如何解决 KV Cache 碎片化？**

**A：**
```
传统方式：为每个序列预分配最大长度连续显存 → 60-80% 浪费
PagedAttention：
  - KV Cache 切分为固定 block（默认 16 tokens）
  - Block Table 维护逻辑→物理映射（类似虚拟内存页表）
  - 不同请求的 block 可非连续存储
  - Prefix Caching：相同前缀的 KV block 可跨请求共享

效果：显存利用率 <40% → >90%，吞吐量 2-4x
```

---

### L2-08 | RoPE 旋转位置编码原理

**Q：RoPE 如何将位置信息编码到 Attention 中？**

**A：**
```
核心：将位置 m 编码为旋转矩阵 Rₘ，作用在 Q/K 上

qₘ = Rₘ · q,  kₙ = Rₙ · k
qₘᵀkₙ = qᵀRₘᵀRₙk = qᵀR(n-m)k  ← 只依赖相对位置 (n-m)

旋转角度：θᵢ = 1/10000^(2i/d)  （类似 Sinusoidal）

优点：
  1. 自然具有相对位置感知
  2. 支持长度外推（配合 YaRN/Dynamic NTK）
  3. 无额外参数
```

---

### L2-09 | DPO vs PPO 对比原理

**Q：DPO 如何绕过 PPO 的复杂性？**

**A：**
```
PPO（需要4个模型）：
  Actor → 生成输出
  Critic → 估计 Value
  RM → 评分
  Reference → KL 约束
  训练不稳定，超参敏感

DPO（直接偏好优化）：
  loss = -E[log σ(β·(log π_θ(y_w|x)/π_ref(y_w|x)
                    - log π_θ(y_l|x)/π_ref(y_l|x)))]
  其中 y_w = 偏好答案，y_l = 拒绝答案
  
  等价于隐式奖励建模，无需显式 RM
  只需 Actor + Reference（2个模型）
```
- DPO 更稳定，但对数据质量要求更高

---

### L2-10 | Tensor Parallelism vs Pipeline Parallelism

**Q：张量并行和流水线并行各如何切分模型？**

**A：**
```
Tensor Parallelism（层内切分）：
  - 将单层的权重矩阵按列/行切分到多卡
  - 每卡持有部分参数，需要 AllReduce 同步
  - 通信量：每层 O(batch_size × seq_len × hidden)
  - 延迟敏感，适合同一机器内 NVLink 高带宽

Pipeline Parallelism（层间切分）：
  - 不同层放在不同卡上，数据流水前进
  - 通信量：仅激活值（小）
  - 存在 Bubble（等待），用 Micro-batching 填充
  - 适合跨机器低带宽场景

混合：Megatron-LM = TP × PP × DP 三维并行
```

---

## L3 决策卡（Why）

### L3-01 | 为什么用 GQA 而不是 MHA？

**Q：现代 LLM 为什么普遍从 MHA 迁移到 GQA？**

**A：**
- MHA：每层每头都有独立 KV → **KV Cache 大**，显存成为瓶颈
- MQA：所有头共享 KV → KV Cache 最小，但表达能力下降，精度损失
- **GQA = 折中**：分 G 组，每组共享 KV
  - KV Cache 缩小 H/G 倍（H=head数，G=组数）
  - 性能接近 MHA，推理效率接近 MQA
- **结论**：序列长度增大时，KV Cache 压力 >> 计算压力，GQA 最优

---

### L3-02 | 为什么 LoRA 微调优于全参数微调？

**Q：LoRA 的核心优势是什么？什么时候优选？**

**A：**
| 对比维度 | 全参数微调 | LoRA |
|---------|-----------|------|
| 显存需求 | 全量梯度+优化器状态 | 仅低秩矩阵 |
| 灾难性遗忘 | 严重 | 轻微（权重冻结） |
| 多任务切换 | 需保存完整模型 | 只换 LoRA 权重（小） |
| 推理延迟 | 无额外 | 合并后零额外 |

- **优选 LoRA 场景**：资源受限、多任务、快速迭代
- **优选全参微调**：数据量极大（>10M）、任务与预训练差距极大

---

### L3-03 | 为什么选 DPO 还是 PPO？

**Q：RLHF 中应该选 PPO 还是 DPO？**

**A：**
| 场景 | 推荐算法 | 理由 |
|------|---------|------|
| 工程资源充足，需极致对齐 | PPO | 在线 RL，泛化更强 |
| 快速迭代，数据质量高 | DPO | 稳定简单，无需 RM |
| 数据噪声大 | PPO | RM 可过滤噪声 |
| 显存受限 | DPO | 只需 2 个模型 |

- Llama 2 Chat 用 PPO；Zephyr、Mistral Instruct 用 DPO

---

### L3-04 | RAG vs Fine-tuning 如何选择？

**Q：什么时候用 RAG，什么时候用 Fine-tuning？**

**A：**
| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| 知识实时更新（新闻/文档） | RAG | 无需重训，改向量库即可 |
| 私有知识库问答 | RAG | 精准检索，可溯源 |
| 风格/格式统一化 | Fine-tuning | 调整输出风格 |
| 特定领域专业能力 | Fine-tuning | 提升领域推理 |
| 知识注入 + 风格 | RAG + Fine-tuning | 混合策略 |

- **经验法则**：知识问题用 RAG，行为问题用 Fine-tuning

---

### L3-05 | 量化位宽如何选择？

**Q：INT8、INT4、FP8 各自适用什么场景？**

**A：**
| 量化方案 | 精度损失 | 显存节省 | 推荐场景 |
|---------|---------|---------|---------|
| FP16/BF16 | 无 | baseline | 对精度敏感，显存充足 |
| INT8（W8A8）| 极小 | 50% | 在线服务，对话系统 |
| INT4（GPTQ/AWQ）| 小 | 75% | 边缘/消费级 GPU 部署 |
| FP8 | 接近 FP16 | 50% | H100/H800 训推一体 |
| INT2/INT3 | 较大 | >80% | 极端资源受限，可接受精度损失 |

---

### L3-06 | 如何选择并行策略？

**Q：多卡部署时，何时用 TP，何时用 PP？**

**A：**
- **同机多卡（NVLink）**：优先 Tensor Parallelism，带宽高通信快
- **跨机多卡（InfiniBand）**：优先 Pipeline Parallelism，通信量小
- **超大模型（>100B）**：3D 并行 = TP × PP × DP
- **延迟优先（在线推理）**：TP 减少单请求延迟
- **吞吐优先（离线批处理）**：PP + DP 提高并发

---

## L4 场景卡（Scene）

### L4-01 | RAG 检索到的内容不相关怎么办？

**场景**：用户提问，RAG 检索到的 Top-K 文档相关性差，LLM 生成幻觉

**排查与优化步骤**：
1. **检查 Embedding 模型**：换专业领域 Embedding（如 BGE、E5）
2. **优化 Chunking 策略**：
   - 块太大 → 检索精度下降，改小（256-512 tokens）
   - 块太小 → 上下文断裂，改大或加 overlap
3. **引入 Reranker**：用 Cross-Encoder 重排（如 BGE-Reranker），代价大但精度高
4. **HyDE（假设文档嵌入）**：先让 LLM 生成假设答案，用假设答案做向量检索
5. **Query 改写**：分解复杂问题，多路检索后融合（RAG Fusion）
6. **兜底策略**：相关性分数低于阈值时，拒绝回答或提示"未找到相关信息"

---

### L4-02 | 模型推理延迟高如何优化？

**场景**：线上 LLM 服务 P99 延迟超标，用户体验差

**优化路径**（按收益排序）：
1. **KV Cache 优化**：启用 PagedAttention（vLLM），减少碎片化
2. **量化**：INT8/INT4 减少显存带宽压力（Decode 阶段 memory-bound）
3. **推测解码**：小模型辅助加速，适合对话类任务
4. **Continuous Batching**：提高 GPU 利用率，降低排队延迟
5. **模型蒸馏**：换用更小的高质量模型
6. **流式输出（Streaming）**：首 token 快速返回，改善用户感知
7. **Prefill 与 Decode 分离**：Prefill 集群 + Decode 集群，各自优化

---

### L4-03 | 如何选择 LoRA 的秩 r？

**场景**：做 LoRA 微调，不知道如何设置 rank 超参

**经验法则**：
| 数据量 | 任务复杂度 | 推荐 r |
|-------|-----------|-------|
| <1K 样本 | 简单（格式化） | r=4 ~ 8 |
| 1K-10K | 中等（风格、QA） | r=16 ~ 32 |
| >10K | 复杂（推理、代码） | r=64 ~ 128 |

- **r 越大**：表达能力越强，但显存增加，过拟合风险上升
- **α/r 通常保持为 1**（α=r），或尝试 α=2r（更大学习率效果）
- **实践**：从 r=16 开始，验证集收敛后决定是否增大

---

### L4-04 | 从零部署一个 LLM 推理服务的步骤？

**场景**：需要快速上线一个 7B 模型的在线推理服务

**推荐流程**：
```
1. 选框架：vLLM（高吞吐）或 TGI（易集成）
2. 量化（可选）：AWQ/GPTQ INT4，减少显存需求
3. 配置：
   - tensor_parallel_size = GPU数量
   - max_num_seqs = 并发请求数
   - gpu_memory_utilization = 0.9
4. 启动：
   vllm serve meta-llama/Llama-2-7b-chat-hf \
     --tensor-parallel-size 2 \
     --quantization awq
5. 监控：TTFT / TPOT / GPU利用率 / KV Cache命中率
6. 压测：逐步提升并发，找吞吐量 vs 延迟拐点
```

---

### L4-05 | 如何诊断 LLM 输出质量问题？

**场景**：微调后模型输出格式乱、幻觉增多、拒绝率高

**诊断框架**：
```
输出格式乱 → 检查 SFT 数据格式一致性，加结构化 Prompt，用 JSON mode
幻觉增多   → 加 RAG 补充知识，降低 temperature，增加 repetition_penalty
拒绝率高   → RLHF 过度对齐，降低 KL 系数，或用 DPO 重新对齐
重复输出   → 调整 repetition_penalty / frequency_penalty
风格漂移   → 检查训练数据分布，加系统 prompt 约束
```

---

## L5 陷阱卡（Trap）

### L5-01 | 陷阱：量化一定会损失精度？

**❌ 误区**：量化 = 精度损失，不敢用

**✅ 真相**：
- **INT8（W8A8）精度损失极小**（<0.5% 性能下降），基本无感知
- **INT4（AWQ/GPTQ）** 在大模型上损失可控，7B以上模型量化效果更好
- **FP8** 在 H100 上接近 FP16，且速度更快
- 量化误差与模型大小负相关：越大的模型越不怕量化
- **真正有损的**是激进的 INT2/INT3 量化或不校准的 naïve 量化

**结论**：在线服务标准做法是 INT8/FP8，INT4 用于资源受限场景，不要因"精度损失"而拒绝量化。

---

### L5-02 | 陷阱：RAG 一定比 Fine-tuning 好？

**❌ 误区**：有了 RAG 就不需要微调了

**✅ 真相**：
- RAG 解决的是**知识检索**问题，Fine-tuning 解决的是**行为和能力**问题
- RAG 的上限受限于检索质量，检索不到就幻觉
- Fine-tuning 可以让模型学会特定推理方式、输出格式、领域术语
- 实际场景常见：RAG 检索到了，但模型不会用（需要 Fine-tuning 提升利用能力）
- **最佳实践**：复杂场景用 RAG + Fine-tuning 结合

---

### L5-03 | 陷阱：CoT 一定能提升推理？

**❌ 误区**：加上"Let's think step by step"就能提升所有任务

**✅ 真相**：
- CoT 对**复杂推理任务**（数学、逻辑）效果显著
- 对**简单分类任务**可能无效甚至有害（增加 token 消耗，引入错误步骤）
- CoT 效果依赖**模型规模**：< 100B 的模型 CoT 效果不稳定
- **错误的 CoT** 比无 CoT 更差（模型可能顺着错误步骤推理到错误答案）
- Few-shot CoT > Zero-shot CoT（需要示例引导推理路径）

**结论**：CoT 是条件有效的，勿盲目使用。

---

### L5-04 | 陷阱：更大的 LoRA rank 一定更好？

**❌ 误区**：r 越大，微调效果越好，应该尽量用大 r

**✅ 真相**：
- r 太大 → 过拟合小数据集，泛化性下降
- r 太大 → 接近全参微调，失去 LoRA 正则化优势
- 研究表明：r=8~16 在多数任务上已达到全参微调 90%+ 效果
- **LoRA 核心价值是效率**，不是追求最高秩
- 如果确实需要大秩，考虑直接全参微调（数据量充足时）

---

### L5-05 | 陷阱：Continuous Batching 没有代价？

**❌ 误区**：Continuous Batching 只有好处，无脑开启即可

**✅ 真相**：
- **长短请求混合**：短请求抢占资源，长请求延迟增大（优先级调度问题）
- **内存压力**：高并发时 KV Cache 撑满，触发 preemption（抢占重计算），延迟骤增
- **调优关键参数**：`max_num_seqs`（最大并发）和 `gpu_memory_utilization`（显存预留）
- 过高并发 → KV Cache OOM → 强制抢占 → 某些请求延迟 10x
- **最佳实践**：压测找到延迟 SLA 约束下的最大吞吐点，不要无限提升并发

---

## 快速总览

| 层级 | 主题 | 卡片数 |
|------|------|--------|
| L1 概念卡 | 架构/机制/方法定义 | 10 张 |
| L2 原理卡 | 公式推导/机制原理 | 10 张 |
| L3 决策卡 | 技术选型/权衡 | 6 张 |
| L4 场景卡 | 实战问题解决 | 5 张 |
| L5 陷阱卡 | 常见误区纠正 | 5 张 |
| **合计** | | **36 张** |

### 高频考点速查

```
预训练  → L1-01(架构) L1-02(MHA) L1-04(RoPE) L2-01(缩放原理) L2-08(RoPE原理)
微调    → L1-05(LoRA) L1-06(RLHF) L2-02(LoRA公式) L2-09(DPO vs PPO) L4-03(选r)
推理优化 → L1-07(KV Cache) L1-08(推测解码) L2-03(KV显存) L2-04(推测解码原理)
RAG    → L1-09(RAG) L3-04(RAG vs FT) L4-01(不相关) L5-02(陷阱)
部署    → L1-10(CB) L2-07(PA) L2-10(并行) L3-06(并行选择) L4-04(部署步骤)
量化    → L2-05(GPTQ) L2-06(AWQ) L3-05(选位宽) L5-01(陷阱)
```

---
*MelonEggLearn · 2026-03-16*
