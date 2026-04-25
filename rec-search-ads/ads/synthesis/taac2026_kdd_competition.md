# TAAC x KDD Cup 2026 -- 腾讯广告算法大赛学习笔记

> **赛题核心**：统一架构同时建模用户行为序列与多字段特征，做 CVR 预测，AUC 排名。禁止 Ensemble，推理延迟硬限制。
>
> **本队赛道**：工业赛道（Industrial Track），晋级线 Top 20，总奖池 $885,000。
>
> **相关概念页**：[[推荐中的注意力机制|Attention in RecSys]] | [[Embedding无处不在|Embedding全景]] | [[序列建模演进|序列建模演进]] | [[生成式推荐|生成式推荐]]

---

## 1. 比赛规则与关键约束

### 1.1 赛题定义

**Towards Unifying Sequence Modeling and Feature Interaction for Large-scale Recommendation**

目标：开发统一架构（USFIR），在单一模型中同时建模：
- 用户行为序列（曝光、点击、转化，带时间戳）
- 非序列多字段特征（用户属性、物品属性、上下文、交叉特征）

任务：CVR（转化率）预测，核心指标 AUC of ROC。

### 1.2 时间线

| 阶段 | 时间 | 关键信息 |
|------|------|---------|
| Phase 1 报名 | Mar.19 -- Apr.23 AOE | 组队截止，之后不能改队 |
| Phase 2 Round 1 | Apr.24 -- May.23 AOE | 隐藏测试集评估，每队每 AOE 日最多 3 次提交 |
| Phase 3 Round 2 | May.25 -- Jun.24 AOE | 约 10x 更大数据集，更严格延迟限制 |
| 获胜者公告 | Jul.15, 2026 | -- |
| KDD Workshop | Aug.9, 2026 | -- |

### 1.3 硬性约束

1. **禁止 Ensemble**：全程只能单模型推理
2. **推理延迟限制**：每个赛道、每轮有独立上限，超出延迟的提交无效（不论 AUC 多高）
3. **每日评估次数**：3 次/AOE 日（失败/停止不计入）
4. **模型存储上限**：20 个模型
5. **脚本上传大小**：训练和评估各 $\le$ 100 MB
6. **用户缓存空间**：20 GB（训练与评估共享）

### 1.4 平台规格（Angel ML Platform）

| 资源 | 规格 |
|------|------|
| GPU 计算力 | 单 GPU 的 20%（虚拟化分配） |
| GPU 显存 | 19 GiB |
| CPU 核数 | 9 核 |
| 内存 | 55 GiB |
| PyTorch | 2.7.1+cu126 |
| Python | 3.10.20 |

关键限制：Web UI 操作（无 SSH），通过上传脚本 + `run.sh` 入口训练，评估通过 `infer.py` 的无参 `main()` 函数。

### 1.5 工作流程

```
训练：上传脚本(含 run.sh) -> 创建训练任务 -> 运行 -> 查看 Logs/Output
                                                       |
导出模型：从 Output 选 checkpoint -> Publish (global_step* 格式)
                                                       |
评估：选已发布模型 -> 上传 infer.py(含无参 main()) -> 提交 -> 查看 AUC + Inference Time
```

### 1.6 晋级与奖项

- 工业赛道晋级：**Top 20**（Round 1 结束后排行榜冻结）
- 工业赛道冠军：$150,000；亚军：$75,000；季军：$30,000
- 额外独立奖项：Unified Block Innovation Award ($45,000) + Scaling Law Innovation Award ($45,000)

---

## 2. 数据格式与特征体系

### 2.1 数据来源

腾讯广告真实日志（完全匿名化处理），无原始内容（文本/图片/URL），无 PII。

### 2.2 特征分类

| 类型 | 内容 | 数据格式 |
|------|------|---------|
| 用户行为序列 | `user_id` + `seq[]`：每条含 `item_id`, `action_type`, `timestamp` | 多域（seq\_a/b/c/d），变长 list |
| 用户特征 | `feature_id` + 值（婚姻状态、性别、年龄等） | 稀疏整数 ID |
| 物品特征 | `feature_id` + 值（类型、品类、广告主类型等） | 稀疏整数 ID |
| 上下文特征 | `feature_id` + 值（设备品牌、OS 类型等） | 稀疏整数 ID |
| 交叉特征 | float\_array（如 User Embedding 高维向量） | 定长 float 向量 |

### 2.3 数据存储格式

- Parquet 格式，多文件，每文件含多个 Row Groups
- 配套 `schema.json` 描述特征布局：
  - `user_int`: `[[fid, vocab_size, dim], ...]`
  - `item_int`: `[[fid, vocab_size, dim], ...]`
  - `user_dense`: `[[fid, dim], ...]`
  - `seq`: `{domain: {prefix, ts_fid, features: [[fid, vocab_size], ...]}}`
- 标签：`label_type == 2` 表示转化（正样本），其余为负样本
- Demo 数据：`https://huggingface.co/datasets/TAAC2026/data_sample_1000`

### 2.4 序列特征细节

- 4 个序列域：seq\_a, seq\_b, seq\_c, seq\_d
- 每个域包含多个 side-info 特征（ID 类）+ 1 个时间戳特征
- 默认截断长度：seq\_a/b: 256, seq\_c/d: 512
- 时间差分桶化（Time Bucket）：64 个桶边界，将时间差映射为离散 bucket ID

---

## 3. Baseline 代码架构分析

### 3.1 整体架构：PCVRHyFormer

模型名 PCVRHyFormer（Post-Click CVR HyFormer），基于 HyFormer 论文思路实现。

```
输入特征
    |
    +-- NS Tokenizer (user_int + item_int -> NS tokens)
    |      GroupNSTokenizer: 按语义分组，每组一个 token
    |      RankMixerNSTokenizer: 全部拼接后均分切块投影
    |
    +-- Dense Projection (user_dense / item_dense -> dense tokens)
    |
    +-- Seq Tokenizer (seq_a/b/c/d -> embedded tokens per domain)
    |      per-fid Embedding -> concat -> Linear -> d_model
    |      + Time Bucket Embedding (optional)
    |
    v
MultiSeqQueryGenerator
    |  GlobalInfo_i = Concat(NS_flat, MeanPool(Seq_i))
    |  Q_i = [FFN_{i,1}(GlobalInfo_i), ..., FFN_{i,N}(GlobalInfo_i)]
    v
MultiSeqHyFormerBlock x N (stackable)
    |  Step 1: Sequence Evolution (per-domain encoder: SwiGLU / Transformer / Longer)
    |  Step 2: Query Decoding (per-domain cross-attention: Q -> Seq)
    |  Step 3: Token Fusion (concat all Q + NS tokens)
    |  Step 4: Query Boosting (RankMixerBlock: token mixing + FFN)
    v
Output Projection -> Classifier -> logit
```

### 3.2 核心模块详解

**3.2.1 NS Tokenizer（非序列特征 Token 化）**

两种变体：
- `GroupNSTokenizer`：按 `ns_groups.json` 语义分组，每组内 fid 做 Embedding + mean pooling，再投影到 $d\_model$，产生一个 token/组
- `RankMixerNSTokenizer`：所有 fid Embedding 拼接成一个长向量，均分成 $T$ 段，每段独立投影到 $d\_model$，$T$ 可自由指定

关键细节：
- 高基数特征可通过 `emb_skip_threshold` 跳过（输出零向量），节省显存
- 使用 `padding_idx=0`，值 $\le 0$ 统一映射为 0（填充）

**3.2.2 序列编码器（三种可选）**

| 编码器 | 结构 | 适用场景 |
|--------|------|---------|
| `SwiGLU` | LN + SwiGLU + Dropout + Residual | 快速，无 attention，适合短序列 |
| `Transformer` | Pre-LN Self-Attention + FFN + Residual | 标准方案，可选 RoPE |
| `Longer` | Top-K 压缩编码器（首层 Cross-Attn，后续层 Self-Attn） | 长序列友好，先压缩到 top\_k 再做 self-attn |

`LongerEncoder` 的关键设计：
- 首个 Block：L > top\_k 时做 Cross Attention（Q = latest top\_k tokens, KV = all tokens）
- 后续 Block：L <= top\_k 时退化为 Self Attention
- 支持 RoPE（Q 侧通过 gather 获取原始位置的 cos/sin）

**3.2.3 MultiSeqQueryGenerator**

每个序列域独立生成 $N_q$ 个 Query token：
- GlobalInfo = Concat(NS\_tokens\_flat, MeanPool(Seq\_i))
- 每个 query 由独立 FFN（SiLU 激活）生成

**3.2.4 MultiSeqHyFormerBlock**

可堆叠的核心 Block，每层执行：
1. **Sequence Evolution**：每个域独立编码
2. **Query Decoding**：每个域独立 cross-attention（Q tokens attend to seq tokens）
3. **Token Fusion**：concat decoded Q + NS tokens
4. **Query Boosting**：RankMixerBlock（parameter-free token mixing + shared FFN）

RankMixerBlock 的 Token Mixing：
$$Q \in \mathbb{R}^{B \times T \times D} \xrightarrow{\text{reshape}} \mathbb{R}^{B \times T \times T \times d_{\text{sub}}} \xrightarrow{\text{transpose}} \mathbb{R}^{B \times T \times D}$$

约束：$d\_model$ 必须能被 $T = N_q \times S + N_{ns}$ 整除（full 模式）。

**3.2.5 RoPEMultiheadAttention**

- 手动投影 Q/K/V，reshape 多头后注入 RoPE
- 使用 `F.scaled_dot_product_attention`（PyTorch 高效实现）
- 输出门控：$\text{out} = \text{out} \odot \sigma(W_g \cdot \text{query})$ + 线性投影
- NaN 保护：`torch.nan_to_num(out, nan=0.0)` 处理全 padding 的 softmax

### 3.3 训练流程

**双优化器策略**：
- Adagrad：稀疏参数（所有 Embedding 权重），lr=0.05
- AdamW：稠密参数（Linear, LayerNorm 等），lr=1e-4, betas=(0.9, 0.98)

**关键训练策略**：
- Early Stopping：基于 val AUC，patience=5
- 梯度裁剪：`clip_grad_norm_(max_norm=1.0)`
- 高基数 Embedding 冷重启：每 epoch 结束后，对 vocab > threshold 的 Embedding 重新 Xavier 初始化并重建 Adagrad 状态（参考 KuaiShou MultiEpoch 论文）
- 序列 ID 特征额外 Dropout：vocab > `seq_id_threshold` 的特征 dropout rate 翻倍
- 损失函数：BCE 或 Focal Loss（$\alpha=0.1, \gamma=2.0$）

**数据管道**：
- IterableDataset 逐 Row Group 读取 Parquet
- 预分配 numpy 缓冲区避免重复分配
- Shuffle Buffer：buffer\_batches 个 batch 合并后随机重排
- 多 Worker DataLoader + `file_system` sharing strategy

### 3.4 推理流程

`infer.py` 要求：
- 无参 `main()` 函数
- 读 `$EVAL_DATA_PATH` 的 Parquet + schema
- 读 `$MODEL_OUTPUT_PATH` 的 checkpoint（model.pt + train\_config.json + schema.json）
- 输出 `$EVAL_RESULT_PATH/predictions.json`：`{"predictions": {"user_id": prob}}`

### 3.5 默认超参

| 参数 | 默认值 | 说明 |
|------|--------|------|
| d\_model | 64 | backbone 隐维 |
| emb\_dim | 64 | Embedding 维度 |
| num\_queries | 1 | 每域 Q token 数 |
| num\_hyformer\_blocks | 2 | Block 堆叠层数 |
| num\_heads | 4 | 注意力头数 |
| seq\_encoder\_type | transformer | 序列编码器类型 |
| hidden\_mult | 4 | FFN 扩展倍数 |
| dropout\_rate | 0.01 | 主 dropout |
| batch\_size | 256 | -- |
| seq\_max\_lens | a:256, b:256, c:512, d:512 | 序列截断长度 |

---

## 4. 参考论文核心思想与启发

### 4.1 InterFormer（CIKM 2025）-- 异构信息双向交互

**核心**：在序列分支（Sequence Arch）与非序列分支（Interaction Arch）之间引入 Cross Arch，实现双向信息流。每个 Transformer 层内交替执行特征交互与序列建模，保留完整 token 级信息。

**启发**：baseline 的 Query Decoding 是单向的（NS -> Seq），InterFormer 的双向交互可作为改进方向。

### 4.2 OneTrans（WWW 2026）-- 单 Transformer 统一

**核心**：统一 Tokenizer 将序列属性与非序列属性映射为单一 token 序列。参数共享策略：序列 token 共享参数（通用序列模式），非序列 token 使用 token-specific 参数。Causal attention + KV cache 降低推理延迟。

**启发**：与赛题 USFIR 方向完全契合。统一 Tokenizer 是核心设计点。

### 4.3 HyFormer（arXiv 2026）-- 长序列 + 特征交互混合

**核心**：交替执行 Query Decoding（从长序列解码兴趣）和 Query Boosting（跨 query 特征交互）。baseline 代码已直接实现此架构（PCVRHyFormer = HyFormer 在 PCVR 场景的实现）。

**启发**：baseline 即 HyFormer 实现。改进空间在于 Query Decoding 的注意力模式和 Boosting 的交互方式。

### 4.4 DIN（KDD 2018）-- Target Attention

**核心**：Local Activation Unit 让候选物品对历史行为做加权求和。公式：

$$\alpha_t = \text{MLP}([\mathbf{e}_t;\, \mathbf{e}_a;\, \mathbf{e}_t \odot \mathbf{e}_a;\, \mathbf{e}_t - \mathbf{e}_a])$$

$$\mathbf{v}_u = \sum_{t=1}^{T} \alpha_t \cdot \mathbf{e}_t$$

**启发**：Target Attention 是 HyFormer Query Decoding 的前身。作为消融实验 baseline。

### 4.5 DCN V2（WWW 2021）-- 显式特征交叉

**核心**：矩阵化 Cross Layer 替代向量外积，支持混合低秩分解（Mixture of Low-Rank）。

$$\mathbf{x}_{l+1} = \mathbf{x}_0 \odot (W_l \mathbf{x}_l + \mathbf{b}_l) + \mathbf{x}_l$$

**启发**：可替换 baseline 的 RankMixerBlock 中的 FFN，作为非序列特征交互的替代方案。

---

## 5. pm-* 项目管理系统使用方法

### 5.1 系统架构

```
taac2026/
+-- main branch          <-- 代码（模型、数据、脚本）
+-- pm-state branch      <-- 项目状态（孤立分支，不与代码分支合并）
+-- .worktrees/
    +-- pm-state/        <-- pm-state 分支的 worktree
    +-- pm-<worker-A>/   <-- worker A 的代码 worktree
    +-- pm-<worker-B>/   <-- worker B 的代码 worktree
```

核心原则：**代码与状态完全分离**。

### 5.2 8 个 Skills

| Skill | 职责 |
|-------|------|
| `pm-init` | 初始化 `.pm/` 目录和 pm-state 孤立分支 |
| `pm-phase` | Phase CRUD，同步 GitHub Milestones |
| `pm-issue` | Issue CRUD，同步 GitHub Issues |
| `pm-task` | Task 全生命周期：create/claim/finish/suspend/resume/cancel |
| `pm-sync` | GitHub push/pull，唯一调用 `gh` 的 skill |
| `pm-eval` | 发起/解析 eval，路由 agent/human 评审 |
| `pm-worker-daemon` | 注册/注销/暂停 worker，管理 cron tick |
| `pm-worker` | Tick 主循环：自愈 -> 合并 -> refine -> 认领 -> 实现 |

### 5.3 日常工作流

```
tick 触发
  |
/pm-sync pull all          # 同步最新状态
  |
/pm-worker tick            # 自动认领任务 / 输出 refine 提示
  |
实现任务（在 pm/task-<id>-<short> 分支）
  |
/pm-worker finish-task <id>  # finish + push PR + 发起 eval
  |
等待 eval 结果
  -> approved -> tick 自动 squash merge
  -> rejected -> tick 输出 refine 提示 -> 修改 -> 重新 finish
```

### 5.4 Task 状态机

```
initial -> in_progress -> waiting_eval -> done
               |               |
           blocked         rejected -> in_progress (refine)
               |
           canceled
```

### 5.5 原子认领机制

`pm-task claim` 在本地写 `owner: agent:<id>` 后立即 `git push`。如果 push 被拒（non-fast-forward），说明另一个 worker 先写入，本次认领失败，重试下一个任务。利用 Git 线性历史保证原子性，不依赖外部锁服务。

### 5.6 常用命令速查

| 命令 | 说明 |
|------|------|
| `/pm-worker-daemon register <id> --repo=<path>` | 注册 worker |
| `/pm-worker-daemon list` | 查看所有 worker |
| `/pm-worker whoami` | 查看当前 worker 状态 |
| `/pm-worker tick` | 手动触发工作循环 |
| `/pm-task list` | 查看所有任务 |
| `/pm-task claim <id>` | 认领任务 |
| `/pm-task finish <id>` | 完成任务 |
| `/pm-eval raise <id>` | 发起评审 |
| `/pm-sync pull all` | 拉取最新状态 |

---

## 6. AI Agent 配合 Boyu 参赛操作流程

### 6.1 初始化（一次性）

```bash
# 1. 克隆 repo
git clone https://github.com/Lijinchi-pc/taac2026.git
cd taac2026

# 2. 安装依赖
pip3 install pyyaml

# 3. 安装 pm-* skills
bash scripts/install-pm-skills.sh

# 4. 注册 worker
/pm-worker-daemon register melon-worker --repo=/path/to/taac2026

# 5. 打开 worker session
cd .worktrees/pm-melon-worker
cc .
```

### 6.2 日常开发循环

1. **Boyu 创建 Issue/Task**：通过 pm-issue 和 pm-task 定义实验假设
2. **Agent tick**：`/pm-worker tick` 自动认领 initial 状态的任务
3. **实现**：在 worker worktree 的 task 分支上开发（修改模型/训练脚本/配置）
4. **完成**：`/pm-worker finish-task <id>` -- 提交 PR + 发起 eval
5. **评审**：
   - Agent eval：自动检查代码质量/测试通过
   - Human eval：Boyu 在 GitHub PR 上 Approve/Request Changes
6. **合并**：approved 后 tick 自动 squash merge 到 main

### 6.3 提交排行榜流程

1. 本地验证 val AUC 和推理延迟
2. 打包训练脚本（含 `run.sh` 入口）$\le$ 100 MB
3. 在 Angel ML Platform Web UI 创建训练任务
4. 训练完成后 Publish checkpoint（命名 `global_step*`）
5. 上传 `infer.py`（含无参 `main()`）
6. 提交评估，查看 AUC + Inference Time
7. 在 pm-issue 中记录：提交时间、AUC、延迟（ms/sample）

### 6.4 Agent 协作注意事项

- 每个 worker 同时只持有 1 个任务
- 遇阻时先 WebSearch，搜不到则 `pm-task suspend` + 自动创建 GitHub Issue
- 不直接编辑 `.pm/` 目录
- Checkpoint 存 `checkpoints/<phase>/<run_id>/`
- 实验日志存 `logs/<phase>/<run_id>.json`

---

## 7. 可能的技术方向与改进思路

### 7.1 Phase 路线图

| Phase | 目标 | 关键产出 |
|-------|------|---------|
| 01 data-baseline | 打通 pipeline，首次提交 | MLP baseline AUC > 0.70 |
| 02 usfir-architecture | 实现 USFIR 核心架构 | 统一架构超越 MLP baseline |
| 03 hyperparam-search | 系统超参搜索 | Optuna $\ge$ 30 组，锁定 base config |
| 04 iterative-refinement | Round 2 迭代冲分 | 冲击 Top 20 |

### 7.2 架构改进方向

1. **统一 Tokenizer（OneTrans 思路）**：将序列 token 和非序列 token 统一输入同一个 Transformer backbone，消除分离式处理的信息损失
2. **双向交互（InterFormer 思路）**：在 Query Decoding 阶段引入序列信息回传给 NS tokens，形成双向信息流
3. **DCN-V2 Cross Layer 替换 RankMixerBlock**：显式建模非序列特征的高阶交叉，可能比 token mixing 更有效
4. **LongerEncoder 优化**：
   - top\_k 值调优（当前默认 50）
   - 尝试多粒度压缩（不同域不同 top\_k）
   - 加入 causal mask 的消融实验

### 7.3 训练策略改进

1. **学习率调度**：当前无 warmup/cosine decay，可引入
2. **数据增强**：
   - 序列随机截断/掩码
   - 特征 dropout/noise
3. **样本加权**：CVR 预测正负比极端不平衡，Focal Loss 的 $\alpha/\gamma$ 需仔细调优
4. **多 epoch Embedding 冷重启策略微调**：threshold 和 epoch 时机的最优组合
5. **混合精度训练**：FP16/BF16 加速训练 + 减少显存

### 7.4 推理延迟优化

1. **KV Cache**（OneTrans 思路）：跨请求复用序列编码结果
2. **模型剪枝**：减少 Block 数/注意力头数，在 AUC-延迟 Pareto 前沿选点
3. **TorchScript / torch.compile**：静态图编译加速
4. **Embedding 量化**：INT8 量化高基数 Embedding 表
5. **序列截断优化**：更短的截断长度 = 更快推理，但要验证 AUC 损失可控

### 7.5 Scaling Law 探索（Innovation Award）

- 系统记录：参数量/数据量/计算量 vs AUC
- 固定架构增加层数/维度的 scaling 曲线
- Embedding 维度 vs AUC 的边际收益
- 训练数据量 vs AUC 的饱和点

---

## 8. 关键环境变量速查

### 训练环境

| 变量 | 说明 |
|------|------|
| `TRAIN_DATA_PATH` | 训练数据目录（Parquet + schema.json） |
| `TRAIN_CKPT_PATH` | Checkpoint 保存目录 |
| `TRAIN_TF_EVENTS_PATH` | TensorBoard 事件文件目录 |
| `USER_CACHE_PATH` | 用户缓存（训练/评估共享，20 GB） |

### 评估环境

| 变量 | 说明 |
|------|------|
| `MODEL_OUTPUT_PATH` | 已发布模型的输出路径 |
| `EVAL_DATA_PATH` | 测试数据目录 |
| `EVAL_RESULT_PATH` | predictions.json 输出目录 |
| `EVAL_INFER_PATH` | 用户上传的推理脚本目录 |
| `USER_CACHE_PATH` | 用户缓存（训练/评估共享） |

---

## 参考文献

1. InterFormer: Effective Heterogeneous Interaction Learning for CTR Prediction (CIKM 2025, arXiv:2411.09852)
2. OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer (WWW 2026, arXiv:2510.26104)
3. HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction (arXiv 2026, arXiv:2601.12681)
4. DIN: Deep Interest Network for Click-Through Rate Prediction (KDD 2018, arXiv:1706.06978)
5. DCN V2: Improved Deep and Cross Network (WWW 2021, arXiv:2008.13535)
