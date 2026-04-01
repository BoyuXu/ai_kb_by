# On the Practice of Deep Hierarchical Ensemble Network for Ad CVR Prediction
> 来源：https://arxiv.org/abs/2504.08169 | 领域：ads | 学习日期：20260401

## 问题定义

广告系统中 CVR（转化率）预估是广告推荐成功的关键环节。与 CTR 预估不同，CVR 预估面临更严峻的挑战：

1. **转化行为发生在广告点击之后的第三方网站/App**（如购买、加购、注册），数据稀疏，标签稀少
2. **样本选择偏差（SSB）**：只有被点击的广告才有CVR标签，样本分布偏移严重
3. **延迟反馈（Delayed Feedback）**：用户购买行为可能发生在点击数小时乃至数天后
4. **Model Architecture 选择困难**：DHEN 需要选择哪些 feature crossing 模块（MLP/DCN/Transformer），以及每个模块的深度、宽度和超参数

本文来自 Meta 广告团队（WWW 2025），系统研究 DHEN 在 CVR 预估（而非 CTR）场景下的工业实践。

## 核心方法与创新点

### DHEN 架构基础

DHEN（Deep Hierarchical Ensemble Network）将多个特征交叉模块以层次化方式组合：

$$
\text{DHEN}(x) = \text{Ensemble}\left[\text{MLP}(x), \text{DCN}(x), \text{Transformer}(x), \ldots\right]
$$

每层将不同类型的特征交叉模块的输出进行集成（concat + linear），形成层次化多模块结构。

### 贡献1：多任务学习框架（Multi-task DHEN）

将 DHEN 作为统一骨干网络预测所有 CVR 任务（多种转化类型）：
- 使用共享的 DHEN 主干 + 任务专属的输出头
- 不同转化任务共享底层特征表示，缓解标签稀疏问题
- 实验探索了不同 CVR 任务之间的共享深度

**关键工程问题研究**：
- Feature crossing 模块组合：如何选择 MLP/DCN/Transformer 的最优组合？
- 模型宽度 vs 深度：在相同参数量下，哪种配置更优？
- 超参数搜索策略：自动化 vs 专家经验

### 贡献2：行为序列特征工程

构建两类用户行为序列：
- **站内实时行为序列（On-site Real-time Sequence）**：用户在广告平台上的点击、停留等行为，实时更新
- **站外转化事件序列（Off-site Conversion Sequence）**：从第三方获取的转化事件（需延迟）

消融实验证明：
- 站外转化序列带来显著 CVR 提升（对"见过广告主产品"的用户进行个性化）
- 长序列建模比短序列更优，但超过某长度收益递减

### 贡献3：自监督辅助损失（Self-supervised Auxiliary Loss）

针对 CVR 标签稀疏问题，提出序列预测辅助任务：

$$
\mathcal{L}_{aux} = -\sum_{t} \log P(\text{future}}_{\text{{\text{action}}}_t | \text{context}}_{\text{{<t}})
$$

利用未来行为序列作为自监督信号，帮助模型在标签稀少的情况下仍能学到有效的用户表示，类似于 BERT 中的 MLM 策略。

**Why It Works**：转化是稀疏事件，但用户的中间行为（浏览、加购等）更密集，通过预测这些中间行为可以改善用户理解，进而提升稀疏转化的预估。

## 实验结论

**模型架构选择**：
- 在 CTR 模块中加入 Transformer 有显著帮助（长程依赖建模）
- DCN 对于捕捉显式特征交叉更有效
- 三者集成（MLP + DCN + Transformer）优于任意两者组合

**行为序列消融**：
- 站外转化序列：CVR AUC 提升 +0.3%（统计显著）
- 站内实时序列：提升更大，+0.8%
- 组合使用：累积提升明显

**辅助损失消融**：
- 自监督辅助损失在标签稀疏的 CVR 任务上提升 AUC +0.15%

**整体**：在 CVR 预估任务上显著优于前代单一特征交叉模块基线。

## 工程落地要点

1. **延迟标签处理**：需要建立 delayed feedback 机制，设置归因窗口（如 7 天）并处理回填
2. **多任务梯度冲突**：不同转化任务的梯度可能相互干扰，使用 GradNorm 或 PCGrad 等方法处理
3. **行为序列存储成本**：站外转化事件需要跨系统打通（广告主 API / Conversion API），存储和延迟问题需特殊处理
4. **模型推理效率**：DHEN 包含多个 Transformer，推理延迟较高，需要 quantization 或 knowledge distillation
5. **数据不均衡**：转化率远低于点击率，需要 focal loss 或负采样策略
6. **样本权重**：给有转化的正样本更高权重，平衡 SSB 问题

## 面试考点

**Q1: CVR 预估与 CTR 预估的核心区别是什么？**
A: CVR 的转化事件发生在点击后，存在严重的样本选择偏差（只有点击样本才有CVR标签），且标签极其稀疏（转化率通常<1%）。CTR 的点击相对密集，样本偏差较小。

**Q2: 什么是 Delayed Feedback 问题？如何解决？**
A: 用户的转化行为（如购买）可能在点击后数天才发生，导致训练时标签不完整。解决方案：（1）设定归因窗口（如7天），窗口内的转化才算正例；（2）使用 Expectation-Maximization 等方法建模延迟分布；（3）实时回填机制更新历史样本。

**Q3: DHEN 如何解决单一特征交叉模块的局限性？**
A: 不同特征交叉模块捕获不同类型的信息（MLP学非线性变换，DCN学显式高阶交叉，Transformer学序列依赖），层次化集成综合所有类型的信息，相互补充。

**Q4: 自监督辅助损失为什么能提升CVR预估？**
A: CVR标签稀疏导致模型欠拟合。通过预测序列中的未来行为（密集信号），模型能学到更丰富的用户表示，这些表示也有益于稀疏转化预估任务。类似于大模型预训练的思路。
