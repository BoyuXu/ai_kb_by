# ESMM / 全空间建模 / CVR 样本偏差 / 多目标融合

## 1. CVR 预估的核心难题

### 1.1 样本选择偏差 (Sample Selection Bias)

传统 CVR 模型的训练数据只来自「已点击」样本：
```
全量曝光样本 → 点击（~5%）→ 转化（~10% of 点击）
                ↑ 训练数据只用这部分
```

问题：
- 训练集和推理集分布不一致：训练在点击子集上，推理在全量曝光上
- 点击样本是有偏的子集（用户主动选择的结果），不能代表全量分布
- 导致 CVR 模型对未点击样本的预估不准

### 1.2 数据稀疏

转化事件远少于点击事件（通常只有点击的 1-10%），CVR 模型面临严重的正样本稀疏问题。

## 2. ESMM (Entire Space Multi-Task Model)

### 2.1 核心思路

不直接建模 P(conversion|click)，而是建模全空间的联合概率：

```
P(click & convert) = P(click) * P(convert | click)
即: pCTCVR = pCTR * pCVR
```

用全量曝光样本训练 pCTCVR，间接学习 pCVR。

### 2.2 模型结构

```
输入特征（全量曝光样本）
├→ 共享 Embedding 层
│
├→ CTR Tower → pCTR    (label: 是否点击, 全量样本)
└→ CVR Tower → pCVR    (不直接用 label 训练)

pCTCVR = pCTR * pCVR    (label: 是否转化, 全量样本)
```

训练 Loss:
```
L = L_ctr(pCTR, click_label) + L_ctcvr(pCTCVR, convert_label)
```

注意：CVR Tower 没有直接的监督信号，通过 pCTCVR 的梯度间接训练。

### 2.3 为什么有效

- 全空间训练：pCTCVR 和 pCTR 都在全量曝光样本上训练，消除了样本选择偏差
- 数据增强：CTR 任务的丰富数据通过共享 Embedding 帮助 CVR 学习更好的特征表示
- 数学一致性：pCVR = pCTCVR / pCTR 在数学上是严格成立的

### 2.4 样本构造

```
曝光但未点击: click=0, convert=0
曝光且点击但未转化: click=1, convert=0
曝光且点击且转化: click=1, convert=1
```

所有样本都参与 CTR loss 和 CTCVR loss 的计算。

### 2.5 ESMM 的局限

数值问题：
- pCVR = pCTCVR / pCTR，当 pCTR 极小时，pCVR 可能不稳定
- 实际中用 pCTR * pCVR 的乘法形式，避免除法

任务耦合：
- CVR 的学习完全依赖 CTR 的准确性
- CTR 不准 → pCTCVR 不准 → pCVR 也不准
- 解法：给 CTR Tower 更多容量和更多训练数据

## 3. ESMM 的后续改进

### 3.1 ESM2 (Extended ESMM)

扩展到更多阶段：曝光 → 点击 → 加购 → 转化

```
P(convert) = P(click) * P(cart|click) * P(convert|cart)
```

每个阶段一个 Tower，乘积关系保证全空间一致性。

### 3.2 AITM (Adaptive Information Transfer Multi-task)

- 不假设严格的概率乘积关系
- 用可学习的 Gate 控制信息从上游任务（CTR）向下游任务（CVR）的传递
- Adaptive Transfer Module：sigmoid gate 控制 CTR 表示对 CVR 的贡献比例
- 更灵活，适合任务关系不是严格概率链式分解的场景

```
h_cvr = gate * h_ctr + (1 - gate) * h_cvr_raw
gate = sigmoid(W * [h_ctr; h_cvr_raw])
```

### 3.3 其他方法处理样本偏差

IPS (Inverse Propensity Score)：
- 用 pCTR 作为倾向性得分，对 CVR 样本加权
- 权重 = 1/pCTR，放大不易被点击的样本的贡献
- 问题：高方差，pCTR 极小时权重爆炸
- 改进：裁剪权重、SNIPS 归一化

DR (Doubly Robust)：
- 结合 IPS 和直接建模，双重校正
- 即使 IPS 或直接模型之一有偏，另一个可以校正
- 实现复杂，工业中不如 ESMM 普及

## 4. 多目标在线融合

### 4.1 融合公式设计

排序阶段需要将多个预估分数融合为单一排序分：

加法融合：
```
score = w1*pCTR + w2*pCVR + w3*f(duration)
```
- 简单直接，可解释性强
- 缺陷：不同指标量级差异大，需要归一化

乘法融合（工业主流）：
```
score = pCTR^a * pCVR^b * duration^c * price^d
```
- a/b/c/d 为可调指数，通过在线 AB 实验确定
- 乘法天然处理量级差异（对数空间变加法）
- 物理意义更清晰：各因子独立贡献

混合融合：
```
score = pCTR^a * pCVR^b * (1 + w*bid)
```
- 广告场景：eCPM = pCTR * pCVR * bid
- 加入质量因子平衡收入和体验

### 4.2 融合权重调优

离线方法：
- 在历史日志上搜索权重空间，优化离线指标（如 NDCG@K of GMV）
- Grid search 或贝叶斯优化

在线方法（推荐）：
- AB 实验：直接在线上实验不同权重组合
- Bandit 自适应：用 Thompson Sampling 等方法自动探索最优权重
```
将权重空间离散化为若干 arm
每个 arm 对应一组 (a, b, c, d) 参数
用 Thompson Sampling 分配流量
根据业务目标（如 GMV、用户留存）的反馈更新各 arm 的后验分布
```

### 4.3 多目标定义

典型目标分类：
- 用户参与度：CTR、点击率
- 用户满意度：完播率、停留时长、点赞/收藏率
- 平台价值：CVR、GMV、广告收入
- 生态健康：多样性、新物品曝光率、创作者分发公平性

不同目标可能冲突（如广告 CTR vs 用户体验），需要 Pareto 分析确定权衡。

## 5. 在线部署与监控

### 5.1 服务架构

```
请求 → 特征服务 → 共享底层（一次推理）
                    ├→ CTR Tower → pCTR
                    ├→ CVR Tower → pCVR
                    └→ 时长 Tower → pred_dur

融合模块：score = f(pCTR, pCVR, pred_dur, ...)
→ 排序 → 返回 top-K
```

关键：共享底层只推理一次，各 Tower 并行出分，融合计算开销极小。

### 5.2 监控体系

实时监控（Flink 流处理）：
- 各任务预估值的分布：均值、分位数、极端值
- 预估值与真实标签的校准度（calibration）
- 融合分的排序质量

告警分级：
- P0（立即响应）：某任务预估值全为 0 或 1、模型加载失败
- P1（15 分钟内响应）：AUC 下降超 2%、预估偏差超阈值
- P2（次日处理）：指标轻微下降、长尾用户效果退化

### 5.3 AB 实验分析

多目标 AB 实验的分析要点：
- 不能只看单指标，需要看所有任务指标的联合变化
- Pareto 改进判定：至少一个任务提升，且没有任务显著下降
- OEC (Overall Evaluation Criterion)：将多指标加权为单一评价指标
- 分群分析：不同用户群的多目标表现可能截然不同

## 6. 面试高频问题

Q: ESMM 为什么不直接用转化 label 训练 CVR Tower？
A: 因为转化 label 只在点击样本上有定义，直接训练会引入样本选择偏差。ESMM 通过 pCTCVR = pCTR * pCVR 的乘积关系，用全量曝光样本间接训练 CVR Tower。

Q: ESMM 中 pCTR 接近 0 时怎么办？
A: 实际中不需要显式计算 pCVR = pCTCVR / pCTR。训练时用乘积形式 pCTCVR = pCTR * pCVR，梯度自然反传给 CVR Tower。推理时如果需要 pCVR 值，可以设置 pCTR 的下限阈值（如 0.01）。

Q: IPS 和 ESMM 解决样本偏差的思路有什么区别？
A: IPS 是统计学方法，通过逆倾向性加权在有偏样本上逼近无偏估计。ESMM 是建模方法，通过在全空间建模联合概率绕过了有偏子集的问题。ESMM 更稳定（不存在高方差问题），但假设了概率的乘积分解关系。

Q: 在线融合公式为什么通常用乘法而不是加法？
A: (1) 乘法天然处理量级差异；(2) 对数空间下乘法变加法，指数参数就是各目标的权重；(3) 物理意义更清晰——CTR=0 则分数为 0，符合直觉；(4) 加法中某个极大值可能主导排序，乘法更均衡。

Q: 如何用 AB 实验验证多目标优化的效果？
A: 定义 OEC（综合评估指标），如 GMV + 0.3*用户活跃度 - 0.1*退出率。实验组和对照组对比 OEC 及各分指标。关注 Pareto 改进：理想情况是所有指标均不降，至少一个提升。

Q: AITM 和 ESMM 的本质区别？
A: ESMM 假设严格的概率乘积分解（pCTCVR = pCTR * pCVR），AITM 用可学习的 Gate 控制上游任务向下游任务的信息传递。AITM 更灵活，不要求任务间是条件概率关系，适合复杂的任务依赖结构。
