# A Comprehensive Survey on Advertising Click-Through Rate Prediction Algorithm
> 来源：综合调研 | 日期：20260319

## 问题定义
点击率（CTR）预测是计算广告系统的核心任务：预测用户在看到某广告时点击的概率。CTR预测需要处理高维稀疏特征（用户、广告、上下文），同时满足工业级低延迟（<10ms）要求。准确的CTR预测直接影响广告收入和用户体验。

## 核心方法与创新点

### 1. 特征交叉方法演进
- **LR（Logistic Regression）**：线性模型，需手工特征工程
- **FM（Factorization Machine）**：自动学习二阶特征交叉，参数效率高
- **FFM（Field-aware FM）**：引入field概念，不同field用不同向量交叉
- **DeepFM**：FM + DNN并联，同时捕获低阶和高阶特征交叉
- **DCN（Deep & Cross Network）**：显式cross层 + DNN，高效学习多阶交叉
- **DCN V2**：改用矩阵替代向量cross，提升表达能力

### 2. 注意力机制
- **DIN（Deep Interest Network）**：用目标广告对历史行为做注意力，捕获用户兴趣多样性
- **DIEN（Deep Interest Evolution Network）**：加入GRU建模兴趣演化
- **DIN/BST（Behavior Sequence Transformer）**：Transformer处理行为序列

### 3. 多任务学习
- **ESMM**：联合建模CTR和CVR，解决样本选择偏差
- **MMOE**：多专家混合，动态分配任务权重
- **PLE**：改进MMOE，区分共享和任务专属专家

## 实验结论
- DeepFM在Criteo等公开数据集上比LR提升约1-2% AUC
- DIN对有历史行为的用户AUC提升0.5-1.5%
- 多任务学习在CVR任务上提升显著（2-5% AUC），因解决了样本选择偏差

## 工程落地要点
1. **特征工程**：用户历史序列截断长度（通常50-200），ID类特征embedding维度（通常8-64）
2. **分布式训练**：PS（Parameter Server）架构处理万亿级参数
3. **特征哈希**：超大ID空间用Hash trick降维
4. **实时特征**：近实时更新用户行为特征（<1分钟延迟）
5. **模型蒸馏**：大模型Teacher→小模型Student，在线推理用Student

## 面试考点
Q1: FM和DNN结合的原理是什么（以DeepFM为例）？
> DeepFM并联FM层和DNN层：FM层负责学习精确的一阶和二阶特征交叉；DNN层负责学习高阶非线性特征交叉；两者共享embedding层，最终输出相加。优势：无需手工特征工程，同时捕获低阶精确交叉和高阶复杂交叉。

Q2: DIN（Deep Interest Network）的核心思想是什么？
> 传统模型将用户历史行为平均池化，丢失了用户兴趣多样性。DIN用目标广告作为query，对历史行为（keys）做注意力加权，相关的历史行为权重更高。这样同一用户面对不同广告时，激活不同的历史行为，更准确建模用户兴趣。

Q3: CTR预测中如何处理样本不均衡（CTR通常只有1-5%）？
> (1) 负采样：对负样本按1:n采样，但需校准预测值（CTR_real = CTR_sampled × sampling_rate）；(2) Focal Loss：对难分样本加大权重；(3) 损失权重：正样本赋予更高权重。
