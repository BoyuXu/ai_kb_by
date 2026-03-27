# 机器学习核心面试考点

> 整理自《三年面试五年模拟》之机器学习基础知识高频考点
> 涵盖：损失函数、优化方法、机器学习基础、模型评估四大模块

---

## 1. 损失函数

### 1.1 BCE (Binary Cross Entropy) / CE (Cross Entropy)

**数学定义：**

二分类交叉熵（BCE）：
$$
L_{BCE} = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]
$$

多分类交叉熵（CE）：
$$
L_{CE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)
$$

其中 $y$ 是真实标签，$\hat{y}$ 是预测概率。

**与KL散度的关系：**
$$
KL(P||Q) = H(P,Q) - H(P)
$$

交叉熵 $H(P,Q)$ 包含熵 $H(P)$ 和KL散度，当真实分布 $P$ 固定时，最小化交叉熵等价于最小化KL散度。

**推导过程：**
1. 从极大似然估计出发，最大化样本出现的概率
2. 取对数后转化为最小化负对数似然
3. 对于二分类，得到BCE形式；多分类得到CE形式

**适用场景：**
- BCE：二分类任务（CTR预估、欺诈检测）
- CE：多分类任务（图像分类、文本分类）

---

### 1.2 Focal Loss

**核心问题：** 解决类别不平衡和难易样本不平衡问题

**数学定义：**
$$
FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

其中：
- $p_t$：模型对正确类别的预测概率
- $\alpha_t$：类别权重因子，平衡正负样本
- $\gamma$：聚焦参数，控制对易样本的抑制程度（通常取2）

**关键设计：**
1. **$(1-p_t)^\gamma$ 调制因子**：当 $p_t \to 1$（易样本），因子趋近0，损失被抑制；当 $p_t \to 0$（难样本），因子趋近1，损失保持较大
2. **$\alpha_t$ 类别权重**：给少样本类别更高权重

**适用场景：**
- 目标检测（如RetinaNet）
- 极端类别不平衡场景（如医学图像分割）

---

### 1.3 MSE (Mean Squared Error)

**数学定义：**
$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

**梯度推导：**
$$
\frac{\partial MSE}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)
$$

**性质：**
- 凸函数，优化稳定
- 对大误差平方惩罚，对异常值敏感
- 适用于回归任务

**适用场景：**
- 房价预测、股票价格预测
- 图像超分辨率重建（PSNR优化目标）

---

### 1.4 Huber Loss

**数学定义（分段函数）：**
$$
L_\delta(a) = \begin{cases} \frac{1}{2}a^2 & \text{if } |a| \leq \delta \\ \delta(|a| - \frac{1}{2}\delta) & \text{if } |a| > \delta \end{cases}
$$

其中 $a = y - \hat{y}$ 是残差，$\delta$ 是阈值超参数。

**设计思想：**
- 小误差（$|a| \leq \delta$）：使用MSE，保证梯度平滑
- 大误差（$|a| > \delta$）：使用MAE（线性），抑制异常值影响

**梯度特性：**
$$
\frac{\partial L_\delta}{\partial a} = \begin{cases} a & \text{if } |a| \leq \delta \\ \delta \cdot \text{sign}(a) & \text{if } |a| > \delta \end{cases}
$$

**适用场景：**
- 含有异常值的回归问题
- 鲁棒优化场景（自动驾驶轨迹预测）

---

### 1.5 正负样本不均衡处理

**常用方法对比：**

| 方法 | 原理 | 适用场景 |
|------|------|----------|
| **Focal Loss** | 降低易样本权重，聚焦难样本 | 检测、分割任务 |
| **上采样（Oversampling）** | 复制少数类样本 | 数据量充足时 |
| **下采样（Undersampling）** | 减少多数类样本 | 数据量充足时 |
| **类别权重** | 给少样本类别更高损失权重 | 通用分类任务 |
| **SMOTE** | 合成少数类样本 | 结构化数据 |

**面试常考组合策略：**
- 推荐系统CTR预估：Focal Loss + 负采样（如Word2Vec的负采样）
- 目标检测：Focal Loss 替代 BCE

---

### 🔥 高频面试题 1

**Q：Focal Loss如何解决类别不平衡问题？与常规CE相比有什么改进？**

**答案要点：**
1. **问题背景**：常规CE对所有样本一视同仁，导致模型被易样本主导（如背景类样本多且易分类）
2. **Focal Loss核心**：增加调制因子 $(1-p_t)^\gamma$
   - 易样本（$p_t$大）：因子小，损失被抑制
   - 难样本（$p_t$小）：因子大，损失保持
3. **类别权重$\alpha_t$**：给少样本类别更高权重
4. **效果**：使模型更关注难分类的少数类样本

---

### 🔥 高频面试题 2

**Q：MSE和MAE的区别？Huber Loss的优势是什么？**

**答案要点：**
1. **MSE**：对大误差平方惩罚，梯度与误差成正比，对异常值敏感
2. **MAE**：线性惩罚，梯度恒定，训练不稳定（在0点不可导）
3. **Huber Loss优势**：
   - 小误差用MSE：梯度平滑，易优化
   - 大误差用MAE：抑制异常值影响，鲁棒性强
   - 整体可导，优化稳定

---

## 2. 优化方法

### 2.1 SGD / Momentum

**SGD更新公式：**
$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta J(\theta_t)
$$

**Momentum更新公式：**
$$
v_t = \beta v_{t-1} + \nabla_\theta J(\theta_t)
$$
$$
\theta_{t+1} = \theta_t - \eta \cdot v_t
$$

**核心思想：** 引入速度累积，加速收敛，抑制震荡

---

### 2.2 Adam (Adaptive Moment Estimation)

**核心思想：** 自适应学习率，结合Momentum和RMSProp

**更新公式：**
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \quad \text{(一阶矩估计)}
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \quad \text{(二阶矩估计)}
$$
$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad \text{(偏差修正)}
$$
$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

**超参数：**
- $\beta_1 = 0.9$（一阶矩衰减率）
- $\beta_2 = 0.999$（二阶矩衰减率）
- $\epsilon = 10^{-8}$（数值稳定性）

---

### 2.3 AdamW (Adam with Weight Decay)

**核心区别：** 将L2正则化与权重衰减解耦

**Adam的L2正则化问题：**
- 标准L2：$L = L_{data} + \lambda||w||^2$
- 在Adam中，$\frac{1}{\sqrt{\hat{v}_t}}$ 会缩放梯度，导致权重衰减效果被削弱

**AdamW正确做法：**
$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \cdot \lambda \theta_t
$$

**关键：** 权重衰减直接作用于参数，不经过自适应学习率缩放

**适用场景：**
- Transformer训练（BERT、GPT系列）
- 大模型微调

---

### 2.4 LAMB (Layer-wise Adaptive Moments optimizer)

**核心思想：** 层自适应学习率，支持大批量训练

**更新公式：**
$$
r_t = \frac{m_t}{\sqrt{v_t} + \epsilon}
$$
$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{r_t}{||r_t||} \cdot ||\theta_t||
$$

**关键设计：**
- 对参数更新进行归一化
- 支持超大batch size（如32k）训练而不损失精度

**适用场景：**
- 大模型预训练（BERT-large级别）
- TPU大规模训练

---

### 2.5 学习率调度策略

#### Warmup
**目的：** 训练初期避免过大学习率导致震荡

**实现：**
$$
\eta_t = \frac{t}{T_{warmup}} \cdot \eta_{max}, \quad t < T_{warmup}
$$

**原因：** 初始化参数远离最优，梯度较大，需要小步长稳定

#### Cosine Annealing
$$
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T_{max}}\pi))
$$

**特点：** 平滑衰减，有利于收敛到局部最优

#### OneCycle
**三个阶段：**
1. Warmup：线性增加到最大学习率
2. Cosine Decay：衰减到最小学习率
3. 可选：小学习率精调

**优势：** 更快的收敛速度，更好的泛化性能

---

### 2.6 梯度裁剪与权重衰减

**梯度裁剪（Gradient Clipping）：**
$$
g \leftarrow \min(1, \frac{clip\_value}{||g||}) \cdot g
$$

**用途：**
- 防止梯度爆炸（RNN/LSTM训练）
- 稳定大模型训练

**权重衰减（Weight Decay）：**
- L2正则化等价于SGD中的权重衰减
- 但在Adam中不等价，需用AdamW

---

### 🔥 高频面试题 3

**Q：Adam和SGD的区别？为什么大模型训练常用AdamW而不是Adam？**

**答案要点：**
1. **Adam优势**：自适应学习率，收敛快，对超参数不敏感
2. **SGD优势**：泛化性能好，最终精度可能更高（有动量时）
3. **AdamW改进**：
   - Adam的L2正则会被自适应学习率缩放，效果减弱
   - AdamW将权重衰减与梯度更新解耦，正则化效果更准确
4. **大模型选择**：AdamW + Warmup + Cosine Decay 是标准配置

---

### 🔥 高频面试题 4

**Q：学习率Warmup的作用是什么？**

**答案要点：**
1. **问题**：训练初期参数随机初始化，梯度较大，大学习率导致震荡或发散
2. **Warmup**：从0线性/指数增加到目标学习率
3. **效果**：
   - 稳定训练初期
   - 帮助模型快速适应数据分布
   - 在Transformer中尤其重要（自注意力机制敏感）
4. **变体**：Linear Warmup、Exponential Warmup

---

## 3. 机器学习基础

### 3.1 偏差-方差权衡

**误差分解：**
$$
Error = Bias^2 + Variance + Noise
$$

**偏差（Bias）：**
- 模型期望预测与真实值的差异
- 高偏差 → 欠拟合 → 模型过于简单

**方差（Variance）：**
- 模型对训练数据变化的敏感程度
- 高方差 → 过拟合 → 模型过于复杂

**权衡关系图示：**
```
Error
  |        /\
  |       /  \
  |      /    \
  |_____/      \_____ Model Complexity
       最优复杂度
```

**调优策略：**
- 高偏差：增加模型复杂度、减少正则化
- 高方差：增加数据、正则化、简化模型

---

### 3.2 L1/L2正则化的几何解释

**L1正则（Lasso）：**
$$
J(\theta) = L(\theta) + \lambda \sum_i |\theta_i|
$$

**几何解释：**
- 约束区域：$||\theta||_1 \leq C$ 形成菱形（高维为超八面体）
- 最优解倾向于在坐标轴上（某些维度为0）
- **产生稀疏性**

**L2正则（Ridge）：**
$$
J(\theta) = L(\theta) + \lambda \sum_i \theta_i^2
$$

**几何解释：**
- 约束区域：$||\theta||_2 \leq C$ 形成圆形（高维为超球体）
- 最优解在圆内部，各维度同时收缩
- **参数平滑，不产生稀疏性**

**对比图：**
```
L1 (菱形)          L2 (圆形)
     |                |
   _/\_              /\
  /    \            /  \
 /      \          |    |
 \      /          |    |
  \____/            \__/
```

**为什么L1产生稀疏解？**
- 菱形约束与损失函数等高线相交时，更容易在顶点处（某维度为0）达到最优

---

### 3.3 交叉验证

**K折交叉验证流程：**
1. 将数据集划分为K份
2. 每次取K-1份训练，1份验证
3. 重复K次，确保每份都作为验证集
4. 取K次结果的平均作为模型评估

**常见选择：** K=5或K=10

**分层抽样：** 确保每折中各类别比例与整体一致

**自助法（Bootstrap）：**
- 有放回抽样，约36.8%样本不会被选中
- 未选中样本可作为验证集（OOB）

---

### 3.4 AUC / GAUC / NDCG

#### AUC (Area Under ROC Curve)
**定义：** ROC曲线下面积，反映模型区分正负样本的能力

**物理意义：** 随机取一个正样本和一个负样本，正样本排在负样本前面的概率

**计算方式：**
$$
AUC = \frac{\sum_{i \in P} rank_i - \frac{|P|(|P|+1)}{2}}{|P| \times |N|}
$$

**优点：**
- 不受阈值影响
- 对类别不平衡不敏感

#### GAUC (Group AUC)
**定义：** 分组AUC，按用户/查询分组计算AUC后加权平均

$$
GAUC = \frac{\sum_i w_i \cdot AUC_i}{\sum_i w_i}
$$

**适用场景：**
- 推荐系统（每个用户的AUC）
- 搜索排序（每个查询的AUC）

#### NDCG (Normalized Discounted Cumulative Gain)
**定义：** 归一化折损累计增益

$$
DCG@k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i+1)}
$$
$$
NDCG@k = \frac{DCG@k}{IDCG@k}
$$

**特点：**
- 关注排序质量
- 高相关性文档排在前面得分更高
- 适用于搜索、推荐排序任务

---

### 3.5 树模型对比

#### XGBoost
**核心优化：**
- 目标函数二阶泰勒展开：$L^{(t)} \approx \sum_{i=1}^n [g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)] + \Omega(f_t)$
- 正则化项：$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$
- 列采样、行采样防止过拟合
- 近似算法加速（Weighted Quantile Sketch）

#### LightGBM
**核心优化：**
- **直方图算法**：连续特征离散化，减少计算量
- **Leaf-wise生长**：选择分裂增益最大的叶子优先分裂（vs Level-wise）
- **GOSS（Gradient-based One-Side Sampling）**：保留大梯度样本，采样小梯度样本
- **EFB（Exclusive Feature Bundling）**：互斥特征合并，减少特征数

**对比：**

| 特性 | XGBoost | LightGBM |
|------|---------|----------|
| 分裂算法 | Level-wise | Leaf-wise |
| 连续特征 | 预排序 | 直方图 |
| 速度 | 快 | 更快（约5-10倍）|
| 内存 | 大 | 小 |
| 处理大数据 | 良好 | 优秀 |
| 类别特征 | 需编码 | 原生支持 |

---

### 🔥 高频面试题 5

**Q：L1和L2正则化的区别？为什么L1会产生稀疏解？**

**答案要点：**
1. **数学形式**：L1是绝对值之和，L2是平方和
2. **几何解释**：
   - L1约束区域是菱形，与损失函数等高线相交于顶点概率大
   - L2约束区域是圆形，相交于边界内部
3. **稀疏性**：L1使部分参数精确为0，实现特征选择
4. **梯度特性**：L1在0点不可导，可用次梯度或近端梯度下降

---

### 🔥 高频面试题 6

**Q：XGBoost和LightGBM的主要区别？**

**答案要点：**
1. **生长策略**：
   - XGBoost：Level-wise（按层生长，平衡树）
   - LightGBM：Leaf-wise（按叶子生长，非平衡树，减少分裂次数）
2. **特征处理**：
   - XGBoost：预排序算法
   - LightGBM：直方图算法（内存小、速度快）
3. **采样策略**：
   - LightGBM有GOSS（基于梯度的单边采样）
4. **类别特征**：LightGBM原生支持，XGBoost需编码

---

## 4. 模型评估

### 4.1 混淆矩阵与基础指标

**混淆矩阵（二分类）：**

| 实际情况\预测 | 正类(1) | 负类(0) |
|--------------|---------|---------|
| 正类(1) | TP | FN |
| 负类(0) | FP | TN |

**基础指标：**

**准确率（Accuracy）：**
$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

**精准率/查准率（Precision）：**
$$
Precision = \frac{TP}{TP + FP}
$$

**召回率/查全率（Recall）：**
$$
Recall = \frac{TP}{TP + FN}
$$

**F1-Score：**
$$
F1 = \frac{2 \times Precision \times Recall}{Precision + Recall}
$$

**F-beta Score：**
$$
F_\beta = \frac{(1+\beta^2) \times Precision \times Recall}{\beta^2 \times Precision + Recall}
$$
- $\beta > 1$：更重视Recall
- $\beta < 1$：更重视Precision

---

### 4.2 ROC曲线与AUC

**ROC曲线绘制：**
- 横轴：FPR（假正率）$= \frac{FP}{FP + TN}$
- 纵轴：TPR（真正率/召回率）$= \frac{TP}{TP + FN}$

**绘制过程：**
1. 将样本按预测分数排序
2. 从最高分到最低分依次作为阈值
3. 计算每个阈值下的(FPR, TPR)
4. 连接各点形成ROC曲线

**AUC物理意义：**
- 随机取一对正负样本，正样本分数高于负样本的概率
- AUC = 0.5：随机猜测
- AUC = 1：完美分类

---

### 4.3 推荐系统离线 vs 在线指标

**离线指标：**
| 指标 | 说明 |
|------|------|
| AUC/GAUC | 排序能力 |
| NDCG@K | 排序质量 |
| HR@K (Hit Rate) | K内命中率 |
| MRR (Mean Reciprocal Rank) | 平均倒数排名 |
| Coverage | 覆盖度 |
| Diversity | 多样性 |

**在线指标：**
| 指标 | 说明 |
|------|------|
| CTR (Click-Through Rate) | 点击率 |
| CVR (Conversion Rate) | 转化率 |
| Dwell Time | 停留时长 |
| Session Duration | 会话时长 |
| Revenue | 营收 |

**离线-在线Gap原因：**
1. **数据分布差异**：离线数据是历史数据，在线遇到新用户/新物品
2. **位置偏差**：在线展示位置影响点击，离线难以完全模拟
3. **选择偏差**：用户只能看到推荐结果，无法看到未推荐的内容
4. **冷启动问题**：新用户/新物品在线表现差
5. **延迟反馈**：用户点击转化需要时间，离线难以准确归因

**缓解策略：**
- Debias技术（IPS、DR）
- 在线A/B测试
- 用户模拟仿真

---

### 🔥 高频面试题 7

**Q：Precision和Recall的区别？如何找到最优阈值？**

**答案要点：**
1. **定义**：
   - Precision：预测为正的中实际为正的比例（精确性）
   - Recall：实际为正的中被预测为正的比例（覆盖性）
2. **权衡关系**：提高阈值→Precision↑ Recall↓；降低阈值→Precision↓ Recall↑
3. **最优阈值选择**：
   - PR曲线找平衡点（Precision≈Recall）
   - 根据业务需求选择（如医疗诊断重视Recall）
   - 最大化F1-Score

---

### 🔥 高频面试题 8

**Q：为什么推荐系统离线AUC高，在线CTR可能不好？**

**答案要点：**
1. **选择偏差**：离线评估基于已曝光数据（用户只能点看到的），在线有曝光机会
2. **位置偏差**：排在第一位的点击率天然高于后面，离线难以准确建模
3. **数据分布**：离线是历史数据分布，在线遇到新用户/新物品（分布漂移）
4. **延迟反馈**：用户点击转化有延迟，离线标签可能不准确
5. **多样性/新颖性**：离线指标关注准确性，在线还需考虑用户体验

---

## 附录：面试速查表

### 损失函数速查
| 损失函数 | 公式 | 适用场景 |
|----------|------|----------|
| BCE | $-[y\log\hat{y}+(1-y)\log(1-\hat{y})]$ | 二分类 |
| CE | $-\sum y_i\log\hat{y}_i$ | 多分类 |
| Focal Loss | $-\alpha_t(1-p_t)^\gamma\log(p_t)$ | 类别不平衡 |
| MSE | $\frac{1}{n}\sum(y-\hat{y})^2$ | 回归 |
| Huber | 分段MSE/MAE | 鲁棒回归 |

### 优化器速查
| 优化器 | 核心特点 | 适用场景 |
|--------|----------|----------|
| SGD+Momentum | 动量累积 | 通用 |
| Adam | 自适应学习率 | 大部分场景 |
| AdamW | 解耦权重衰减 | 大模型训练 |
| LAMB | 层自适应 | 大批量训练 |

### 评估指标速查
| 指标 | 关注点 | 适用场景 |
|------|--------|----------|
| AUC | 排序能力 | 分类、CTR |
| GAUC | 分组排序 | 推荐 |
| NDCG | 排序质量 | 搜索、推荐 |
| F1 | 精确与召回平衡 | 分类 |
| MRR | 首位命中率 | 推荐 |

---

> 💡 **面试建议**：理解原理比背公式更重要，能画图解释（如L1/L2几何解释）是加分项！
