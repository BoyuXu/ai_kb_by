# AI 核心技术速查手册

> 提炼自 ai-kb 全库 synthesis + fundamentals | 更新：2026-04-10
> 覆盖：rec-sys / ads / search / llm-infra / fundamentals（5域，25个子方向）

---

## 一、推荐系统核心技术

### 1.1 召回层

| 技术 | 核心思想 | 工程要点 | 代表作 |
|------|---------|---------|--------|
| 双塔（DSSM） | 用户/物品双编码，内积相似 | in-batch负样本+纠偏 | Youtube DNN, DSSM |
| GNN召回 | 高阶邻居传播，图协同信号 | Mini-batch邻居采样 | PinSAGE, LightGCN |
| 序列召回 | 自注意力建模兴趣序列 | Causal Mask防泄露 | SASRec, BERT4Rec |
| 多路融合 | 协同+内容+图多路合并 | 统一打分排序去重 | 微信看一看 |
| 生成式召回 | 自回归生成 Semantic ID | Beam Search层次解码 | TIGER, Spotify |

**双塔相似度**：$\text{score}(u,i) = \mathbf{e}_u^\top \mathbf{e}_i / \tau$  
**InfoNCE 损失**：$\mathcal{L} = -\log \frac{e^{\text{sim}(u,i^+)/\tau}}{\sum_j e^{\text{sim}(u,i_j)/\tau}}$

**召回负样本策略**：随机负样本（简单）→ Batch内负样本（高效）→ Hard Negative（被召回未转化）→ 混合采样（工业主流）

---

### 1.2 排序层（CTR预估）

| 技术 | 核心思想 | 工程要点 | 代表作 |
|------|---------|---------|--------|
| LR+GBDT | 人工特征+树特征编码 | GBDT叶节点当特征 | Facebook 2014 |
| FM / DeepFM | 二阶特征交叉+DNN | Field-aware变体FFM | 阿里DeepFM |
| Wide&Deep | 记忆(Wide)+泛化(Deep) | Wide用交叉特征 | Google Play |
| DCN-V2 | Cross Network显式交叉 | 参数矩阵替代向量 | Google 2021 |
| DIN | Target Attention动态兴趣 | DICE激活+稀疏正则 | 阿里巴巴 |
| DIEN | GRU建模兴趣演化 | AUGRU辅助损失 | 阿里巴巴 |

**DIN Target Attention**：$\alpha_t = \text{MLP}([\mathbf{e}_t; \mathbf{e}_a; \mathbf{e}_t \odot \mathbf{e}_a])$，$\mathbf{v}_u = \sum_t \alpha_t \mathbf{e}_t$  
**AUC vs 线上CTR不一致原因**：特征分布偏移、样本选择偏差、延迟问题、线上与离线环境差异

---

### 1.3 长序列用户行为建模

| 技术 | 核心思想 | 序列上限 | 工程挑战 |
|------|---------|---------|---------|
| DIN | Target Attention加权 | ~50 | 稀疏激活 |
| DIEN | GRU+辅助损失 | ~100 | 训练不稳 |
| SIM | 两阶段：检索再打分 | 10K+ | 检索准确性 |
| ETA | Hash近似检索 | 10K+ | Hash碰撞 |
| SparseCTR | 三分支稀疏注意力+个性化分块 | 10K+ | 端到端，Scaling Law |
| MAMBA/SSM | 线性复杂度序列建模 | 100K+ | 选择性遗忘机制 |

**SIM两阶段**：GSU（快速相似商品检索）→ ESU（精确Attention打分）  
**Mamba优势**：$O(n)$ vs Transformer $O(n^2)$，硬件感知并行扫描

---

### 1.4 重排与多样性

| 技术 | 核心思想 | 工程要点 |
|------|---------|---------|
| MMR | 相关性-多样性权衡 | $\text{score} = \lambda \cdot \text{rel} - (1-\lambda) \cdot \text{sim}$ |
| DPP | 行列式点过程多样性 | 核矩阵计算近似 |
| 滑动窗口重排 | 窗口内序列排序 | SetRank / DLCM |
| 生成式重排 | LLM直接输出排序 | 延迟是核心挑战 |

---

### 1.5 多任务学习

| 技术 | 核心思想 | 解决问题 | 代表作 |
|------|---------|---------|--------|
| Shared-Bottom | 底层全共享 | 基础多任务 | - |
| MMoE | 多专家+门控路由 | 任务差异大 | Google 2018 |
| PLE | 私有+共享专家分离 | 跷跷板效应 | 腾讯 2020 |
| SMES | 稀疏专家门控 | 20+任务可扩展 | 美团 2024 |
| ESMM | 联合估计pCTR×pCVR | CVR样本偏差 | 阿里巴巴 |

**MMoE 门控**：$g_k^{(t)}(x) = \text{softmax}(W_g^{(t)} x)$，$h^{(t)}(x) = \sum_k g_k^{(t)} f_k(x)$  
**PLE vs MMoE**：PLE 显式区分私有/共享专家，任务干扰最小；MMoE 是软分离

---

### 1.6 生成式推荐（前沿）

| 技术 | 核心思想 | 关键数字 | 代表作 |
|------|---------|---------|--------|
| HSTU | ReLU替代softmax，Scaling Law | 1.5T参数 Meta | Meta 2024 |
| TIGER | RQ-VAE离散化+T5生成 | 冷启动+25% | Google 2023 |
| MTGR | 双流：序列流+特征交叉流 | GMV+2.1% | 美团 2024 |
| PROMISE | PRM过程奖励测试时扩展 | Recall@10+9.1% | - 2025 |
| UniGRec | Soft ID连续向量，碰撞率0 | Recall+5.8% | - 2024 |
| Mender | 自然语言偏好条件生成 | SOTA Preference Steering | TMLR 2025 |
| SID Prefix Ngram | 语义碰撞替代随机碰撞 | 长尾AUC↑，稳定性↑ | Meta 2025 |

---

## 二、广告系统核心技术

### 2.1 CTR/CVR 预估

| 技术 | 核心思想 | 工程要点 | 代表作 |
|------|---------|---------|--------|
| LR | 线性模型，在线学习 | FTRL+L1稀疏 | Google 2013 |
| GBDT+LR | 树特征+线性模型 | 叶节点One-hot | Facebook 2014 |
| DeepFM | FM二阶+DNN高阶 | Embedding共享 | 阿里 2017 |
| ESMM | 整体空间CVR估计 | pCTR×pCVR联合 | 阿里 2018 |
| ESCM² | IPW/DR去CVR偏差 | 倾向分纠偏 | 阿里 2022 |

**ESMM CVR整体空间**：$\text{pCTCVR} = \text{pCTR} \times \text{pCVR}$，在全曝光空间联合训练  
**IPW CVR纠偏**：$\hat{\theta}_{CVR} = \frac{1}{N}\sum_i \frac{y_i \cdot z_i}{p_i}$（倾向分加权）

---

### 2.2 出价策略

| 技术 | 核心思想 | 工程要点 | 代表作 |
|------|---------|---------|--------|
| 手动出价 | 广告主自定CPC/CPM | 无 | 早期RTB |
| oCPC/oCPA | 目标转化出价 | PID预算控制 | 百度凤巢 |
| KKT最优出价 | Lagrange乘子对偶 | $b^* = v \cdot (1+\lambda)^{-1}$ | 理论最优 |
| RL自动出价 | CMDP约束优化 | Lagrangian RL | 阿里、字节 |
| AutoBidding | 端到端竞价策略 | MARL多智能体 | 2022+ |

**最优出价公式**：$b_i^* = v_i / (1 + \lambda)$，$\lambda$ 为预算对偶变量（"每元预算边际价值倒数"）  
**PID Pacing**：$\Delta b = K_p e + K_i \int e\, dt + K_d \dot{e}$，$e = \text{目标消费速率} - \text{实际消费速率}$

---

### 2.3 预算控制与 Pacing

| 技术 | 核心思想 | 工程要点 |
|------|---------|---------|
| 匀速Pacing | 按时间比例预算分配 | 简单，不适应流量波动 |
| PID控制 | 闭环反馈控制消费速率 | 参数调优关键 |
| 预测型Pacing | 预测流量分布+动态分配 | 需实时CTR预测 |
| 双层优化 | 预算+ROI联合约束 | CMDP对偶 |

---

### 2.4 广告偏差治理

| 偏差类型 | 产生原因 | 解决方法 |
|---------|---------|---------|
| 曝光偏差 | 只有曝光物品有标签 | IPS倾向分加权 |
| 选择偏差 | 展示受排序策略影响 | 反事实估计 |
| 位置偏差 | 位置影响点击率 | Position-aware模型 |
| 幸存者偏差 | CVR只在点击样本计算 | ESMM整体空间 |
| 数据飞轮偏差 | 已推的内容越推越多 | Uplift/探索策略 |

---

### 2.5 LTV 与效果预测

| 技术 | 核心思想 | 适用场景 |
|------|---------|---------|
| BG/NBD | 购买与流失双泊松过程 | 电商LTV |
| ZILN | 零膨胀对数正态建模 | 含未转化样本 |
| Deep LTV | 神经网络分位数预测 | 大规模在线广告 |
| 生存分析 | Weibull/Cox延迟CVR | 延迟转化校正 |

**ZILN 损失**：$\mathcal{L} = \text{BinaryCE}(z) + z \cdot \text{LogNormalNLL}(y|\mu,\sigma)$  
**保守出价**：用 $\hat{y}_\alpha = \exp(\mu - \sigma^2/2)$ 第 $\alpha$ 分位数，降低不确定性风险

---

## 三、搜索核心技术

### 3.1 查询理解与扩展

| 技术 | 核心思想 | 工程要点 |
|------|---------|---------|
| 词法分析 | 分词+命名实体识别 | 领域词典+CRF |
| 意图识别 | 查询意图分类 | BERT微调 |
| 查询改写 | 同义词扩展/纠错 | Seq2Seq生成 |
| HyDE | 假设文档扩展查询 | 用LLM生成假设答案 |
| O1-Embedder | 内部推理增强编码 | 不改query，兼容现有索引 |

---

### 3.2 相关性检索

| 技术 | 核心思想 | 指标 | 特点 |
|------|---------|------|------|
| BM25 | TF-IDF词频统计 | BEIR ~47% | 无需训练，可解释 |
| DPR | 双编码器对比学习 | BEIR ~54% | 端到端训练 |
| ColBERT | 延迟交互Token级匹配 | ~60% | 精度高，存储大 |
| Qwen3-Emb | LLM backbone全量微调 | BEIR ~65% | 强语义理解 |
| 混合检索 | BM25+稠密RRF融合 | 最鲁棒 | 互补各自弱点 |

**RRF 融合**：$\text{score}(d) = \sum_r \frac{1}{k + \text{rank}_r(d)}$，$k=60$ 默认值

---

### 3.3 推理增强重排（前沿）

| 方案 | 参数量 | 延迟 | 工业可行性 | 核心方法 |
|------|--------|------|-----------|---------|
| Reasonrank | 7B | 100-200ms | ★★★ | CoT+RL直接优化 |
| LimRank | 7B | 50-80ms | ★★★★ | 压缩+Early Exit |
| DEAR | 1.5B | 20-40ms | ★★★★★ | 大模型推理链蒸馏 |

**DEAR蒸馏**：$\mathcal{L} = \text{KL}(P_{student}(\text{rank}) \| P_{teacher}(\text{rank}))$

---

## 四、LLM 基础设施核心技术

### 4.1 高效训练与微调

| 技术 | 核心思想 | 参数节省 | 代表方法 |
|------|---------|---------|---------|
| LoRA | 低秩分解适配器 | 99%冻结 | $W = W_0 + BA$，$r \ll d$ |
| QLoRA | 4bit量化+LoRA | 显存-75% | NF4量化+双重量化 |
| Prefix Tuning | 可训前缀token | 0.1%参数 | P-Tuning v2 |
| Adapter | 瓶颈层插入 | <1%参数 | AdapterFusion |
| BOFT | 正交微调，结构保持 | 参数少效果强 | 前沿方法 |

**LoRA公式**：$h = W_0 x + \frac{\alpha}{r} BA x$，$r$ 为秩，$\alpha$ 为缩放系数  
**QLoRA关键**：NF4量化（4bit正态分布分桶）+ Double Quantization（量化常数再量化）

---

### 4.2 推理加速

| 技术 | 优化维度 | 加速倍数 | 原理 |
|------|---------|---------|------|
| KV Cache | 避免重复KV计算 | 关键基础 | 空间换时间 |
| GQA/MQA | 减少KV头数 | 内存-4~8× | 多头共享KV |
| FlashAttention | HBM访问优化 | 1.5-2× | Tiling+Online Softmax |
| PagedAttention | 内存碎片消除 | 吞吐+2-3× | vLLM分页管理 |
| Speculative Decoding | 草稿模型并行验证 | 2-5× | Draft+Verify |
| Continuous Batching | 动态批处理 | 吞吐+2-3× | 请求级调度 |
| Prefill-Decode分离 | 两阶段解耦部署 | 延迟优化 | 阶段资源差异 |

**GQA KV压缩**：$G$ 组共享 $K/V$，$\text{KV大小} = G \times d_{kv}$（$G \ll h$）  
**Speculative Decoding接受率**：$\alpha = \min(1, P_t(x) / P_d(x))$，需 draft 和 target 分布接近

---

### 4.3 RAG 系统

| 阶段 | 技术 | 核心思想 |
|------|------|---------|
| Chunk | 语义分块+重叠 | 固定/语义/层次三种策略 |
| 索引 | HNSW+BM25混合 | 稠密+稀疏互补 |
| 检索 | 查询重写+HyDE | 扩展查询覆盖率 |
| 重排 | Cross-Encoder精排 | 全交互，Recall@K后精排 |
| 生成 | 上下文压缩 | 减少无关噪声 |
| 验证 | Self-RAG反思token | 自动判断是否需要检索 |

**RAG vs Finetune**：知识时效性强→RAG；任务特定格式/风格→Finetune；两者可组合

---

### 4.4 后训练对齐

| 技术 | 核心思想 | 适用场景 | 代表工作 |
|------|---------|---------|---------|
| SFT | 监督指令微调 | 基础对齐 | InstructGPT Phase1 |
| RLHF+PPO | 奖励模型+策略优化 | 主观偏好对齐 | InstructGPT |
| DPO | 隐式奖励，偏好对 | 省Reward Model | Llama-2-chat |
| GRPO | 组内均值替代Critic | 数学/代码推理 | DeepSeek-R1 |
| RLVR | 可验证奖励RL | 有标准答案任务 | DeepSeek-R1 |

**DPO损失**：$\mathcal{L} = -\log\sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)$  
**GRPO优势**：$A_i = (r_i - \bar{r}) / \text{std}(r)$，无需训练独立Critic模型，省50%显存

---

### 4.5 Agent 架构

| 组件 | 核心功能 | 工程要点 |
|------|---------|---------|
| 工具调用 | 函数路由+结果处理 | 格式严格+重试机制 |
| 记忆管理 | 工作记忆+长期记忆 | 分层索引+遗忘策略 |
| 规划 | CoT/ToT任务分解 | ReAct=推理+行动交叉 |
| 多Agent协作 | 角色分工+通信 | 避免过度通信开销 |
| 失败模式 | 幻觉/死循环/工具失败 | 超时+结果验证 |

---

## 五、通用基础算法

| 算法 | 适用场景 | 核心公式 | 注意事项 |
|------|---------|---------|---------|
| Attention | 序列建模/跨模态 | $\text{Attn}=\text{softmax}(QK^\top/\sqrt{d_k})V$ | 复杂度 $O(n^2d)$ |
| MHA→GQA | 推理KV压缩 | $G$ 组共享 $K/V$ | 效果接近MHA |
| 对比学习 | 无监督表示学习 | InfoNCE多正负对 | 温度τ=0.07-0.1 |
| BPR Loss | 排序学习召回 | $-\log\sigma(\hat{r}_{ui}-\hat{r}_{uj})$ | 需构建正负对 |
| LoRA | PEFT高效微调 | $\Delta W = BA$，秩 $r$ | 秩4-16通常足够 |
| Uplift/ITE | 因果效应估计 | $\tau(x)=E[Y(1)-Y(0)\|X=x]$ | 需随机实验数据 |
| Speculative Dec. | LLM推理加速 | $\alpha=\min(1, P_t/P_d)$ | Draft-Target分布相近 |
| GRPO | RL强化推理 | 组内归一化优势函数 | 全对/全错梯度消失 |

---

## 六、面试高频考点（按域）

### 推荐系统
1. **DIN vs DIEN vs BST vs SIM**：注意力机制演进，各自解决什么问题
2. **双塔负样本**：随机 / Batch内 / Hard Negative 的区别和配比
3. **MMoE vs PLE**：跷跷板效应定义，PLE如何解决，工业选型依据
4. **ESMM / CVR偏差**：整体空间训练原理，IPW纠偏公式推导
5. **冷启动策略**：用户冷/物品冷/系统冷，各自的解法
6. **AUC离线好线上差**：分布偏移、特征延迟、样本选择偏差等原因
7. **Mamba vs Transformer**：O(n) vs O(n²)，选择性状态空间机制
8. **Semantic ID**：RQ-VAE原理、STE梯度技巧、冷启动优势

### 广告系统
1. **oCPX出价**：oCPC/oCPA/oCPM区别，pCTR×pCVR×TargetCPA公式
2. **竞价机制**：GSP vs VCG，GSP为什么不Nash均衡
3. **最优出价推导**：KKT条件，$b^* = v/(1+\lambda)$
4. **预算Pacing**：PID三参数作用，欠投/过投的调参方向
5. **CVR偏差**：ESMM、ESCM²、延迟CVR（生存分析）
6. **LTV建模**：ZILN损失函数，零膨胀+对数正态两部分

### 搜索系统
1. **BM25 vs 稠密检索**：词法匹配 vs 语义理解，各自盲区
2. **混合检索RRF**：公式，$k=60$的意义，融合权重调节
3. **DEAR蒸馏**：大模型推理链→小模型，延迟从1s降到20ms
4. **O1-Embedder**：内部推理不改查询，兼容现有ANN索引
5. **查询改写 vs 文档扩展**：HyDE原理，适用场景

### LLM基础设施
1. **FlashAttention原理**：SRAM Tiling，IO-Bound vs Compute-Bound
2. **KV Cache压缩**：GQA/MQA内存节省，量化精度影响
3. **Speculative Decoding**：数学保证（draft接受率公式），适用条件
4. **DPO vs GRPO**：各自省掉什么，适用场景（主观偏好 vs 客观奖励）
5. **LoRA秩选择**：r=4 vs r=64的效果差异，合并权重的时机
6. **Prefill vs Decode**：两阶段资源需求差异，分离部署原因
7. **RAG vs Finetune**：时效性/风格/知识量的三角权衡

---

## 七、2026 前沿趋势速览

### 推荐系统
1. **Scaling Law验证**（来源：`生成式推荐范式统一_20260403.md`）：HSTU从1.5B→1.5T参数，推荐也有幂律，进入"scaling era"
2. **测试时扩展（Test-Time Compute）**（来源：`生成式推荐系统技术全景_2026.md`）：PROMISE用Best-of-N + PRM过程奖励，推理时多采样取最优
3. **端到端生成式架构**（来源：`生成式推荐范式统一_20260403.md`）：MTGR/OneTrans统一序列建模+特征交叉，消除多阶段信息损失

### 广告系统
1. **LLM能力渗透全链路**（来源：`工业广告系统生成式革命_20260403.md`）：IDProxy冷启动Embedding + ICL-Bandit数据标注，均选「离线蒸馏」而非在线调用
2. **生成式创意自动化**（来源：`广告创意优化.md`）：CoT驱动定向→画像→诉求→文案生成，人工介入从30%降至5%
3. **因果广告效果评估**（来源：`Uplift建模技术演进与工业实践.md`）：ECAD跨域倾向分 + Uplift建模，解决多触点归因和SSB偏差

### LLM推理与部署
1. **DeepSeek-R1蒸馏路径**（来源：`知识蒸馏技术整体总结.md`）：拒绝采样+CoT SFT蒸馏，R1-Distill-7B超越GPT-4o（AIME）
2. **Prefill-Decode物理分离**（来源：`LLM推理优化完整版.md`）：两阶段资源需求差异极大，拆成独立集群，GPU利用率+30%
3. **MoE架构主流化**（来源：`MoE架构设计与推理优化.md`）：Qwen3-235B-A22B（128专家/8激活），参数量大但计算量可控，成本效益最优

### 搜索系统
1. **推理增强检索（Reasoning Retrieval）**（来源：`推理增强检索技术综述.md`）：O1-Embedder/DEAR/CRE-T1，推理能力直接提升NDCG@10 ~10%
2. **Agentic Search**（来源：`端到端生成式搜索前沿_20260403.md`）：多轮自主检索+工具调用，Qagent等Agent架构取代固定pipeline
3. **生成式检索（DSI范式）**（来源：`端到端生成式搜索前沿_20260403.md`）：Transformer直接生成文档ID，DocID设计是核心挑战

---

## 附录：各域核心 synthesis 导航

| 域 | 入门综合 | 深度专题 |
|----|---------|---------|
| **rec-sys召回** | `召回系统工业界最佳实践.md` | `Semantic_ID演进知识图谱.md` |
| **rec-sys排序** | `CTR模型深度解析.md` | `用户行为序列建模.md` |
| **rec-sys架构** | `推荐系统全链路架构概览.md` | `精排模型进阶深度解析.md` |
| **rec-sys前沿** | `生成式推荐系统技术全景_2026.md` | `生成式推荐范式统一_20260403.md` |
| **ads CTR** | `CTR预估模型工业级实践进展.md` | `ESMM系列CVR估计演进.md` |
| **ads bidding** | `AutoBidding技术演进_从规则到RL.md` | `广告出价体系全景.md` |
| **ads系统** | `广告系统综合总结.md` | `LTV预测技术演进与工业实践.md` |
| **search** | `推理增强检索技术综述.md` | `端到端生成式搜索前沿_20260403.md` |
| **llm推理** | `LLM推理优化完整版.md` | `KVCache与LLM推理优化全景.md` |
| **llm训练** | `LoRA与PEFT高效微调技术进展.md` | `RLVR_vs_RLHF后训练路线.md` |
| **llm RAG** | `RAG系统全景.md` | `RAG_vs_Finetune决策框架.md` |
| **llm Agent** | `Agent失败模式与解法.md` | `多智能体工作流生产架构.md` |
| **fundamentals** | `attention_transformer.md` | `contrastive_learning.md` |

> 📝 本文档由 MelonEgg 自动提炼，来源于 ai-kb 全库 synthesis 文件  
> 更新：执行 `find ~/Documents/ai-kb -name "CORE_TECH_DIGEST.md" -exec touch {} \;` 触发重新生成
