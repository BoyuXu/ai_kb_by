# ICL-Bandit: Relevance Labeling in Advertisement Recommendation via LLM

> 来源：https://aclanthology.org/2025.findings-emnlp.1273/ | 领域：ads | 学习日期：20260403

## 问题定义

在广告推荐系统中，query-ad相关性标注(relevance labeling)是搜索广告质量保障的关键环节。传统方法依赖人工标注或基于BERT等模型的二分类器，前者成本高、产出慢，后者需要大量标注数据训练且泛化能力有限。

近年来，LLM通过in-context learning(ICL)展现了强大的few-shot能力，可以通过在prompt中提供少量示例(demonstrations)来完成query-ad相关性判断。然而，ICL的效果高度依赖于示例的选择——不同的示例组合可能导致截然不同的标注质量。如何自动化地为每个query-ad pair选择最优的in-context examples，成为关键的开放问题。

本文提出ICL-Bandit方法，将in-context example selection建模为contextual multi-armed bandit问题。每个候选示例是一个arm，query-ad pair是context，标注准确率是reward。通过bandit学习，系统自适应地为不同类型的query-ad pair选择最优示例组合，显著提升LLM相关性标注的准确率和稳定性。

## 核心方法与创新点

### Bandit问题建模

将ICL example selection形式化为contextual bandit。给定query $q$、广告 $a$ 以及候选示例池 $\mathcal{D} = \{d_1, d_2, ..., d_N\}$，目标是选择 $K$ 个示例组成prompt。定义上下文特征向量 $\mathbf{c} = \phi(q, a)$，每个arm $d_i$ 的期望奖励为：

$$
r_i(\mathbf{c}) = \mathbf{c}^T \boldsymbol{\theta}_i + \epsilon_i
$$

其中 $\boldsymbol{\theta}_i$ 是arm $d_i$ 的参数向量，$\epsilon_i$ 是噪声项。采用LinUCB策略选择top-K arms：

$$
d^* = \arg\max_{d_i \in \mathcal{D}} \left( \hat{\mathbf{c}}^T \hat{\boldsymbol{\theta}}_i + \alpha \sqrt{\mathbf{c}^T \mathbf{A}_i^{-1} \mathbf{c}} \right)
$$

其中 $\mathbf{A}_i$ 是arm $d_i$ 的特征协方差矩阵，$\alpha$ 控制exploration-exploitation平衡。UCB项 $\sqrt{\mathbf{c}^T \mathbf{A}_i^{-1} \mathbf{c}}$ 鼓励探索不确定性高的示例。

### 示例池构建与特征设计

候选示例池从历史标注数据中构建，每个示例包含一个query-ad pair及其人工标注的相关性等级(如 Excellent/Good/Fair/Bad)。上下文特征 $\phi(q, a)$ 包括：

- Query和ad的语义embedding相似度
- Query类别(导航型/信息型/交易型)
- Ad landing page质量分
- Query-ad token overlap率

### 关键创新

- **Bandit框架**：首次将ICL example selection建模为contextual bandit问题，实现自适应示例选择
- **动态探索**：UCB机制在标注准确率(exploitation)和示例多样性(exploration)间平衡
- **无需额外训练**：利用LLM的ICL能力，不需要fine-tune LLM本身
- **可组合性**：选出的K个示例通过组合排列进入prompt，支持示例间的协同效应

## 系统架构

```mermaid
graph TD
    A[Query-Ad Pair] --> B[特征提取 φ(q,a)]
    B --> C[LinUCB Bandit]
    D[候选示例池 D] --> C
    C --> E[Top-K示例选择]
    E --> F[Prompt构建]
    F --> G[LLM推理]
    G --> H[相关性标签]
    H --> I[奖励反馈]
    I --> C

    subgraph Prompt结构
        J[System Instruction]
        K[Selected Examples × K]
        L[Target Query-Ad]
    end
```

## 实验结论

- 在query-ad relevance四分类任务上，ICL-Bandit相比随机选择示例的ICL准确率提升 **+6.3%**
- 相比BM25 retrieval选择示例的方法提升 **+3.8%**
- 相比固定最优示例集(oracle fixed set)提升 **+2.1%**，说明动态选择的必要性
- 标注一致性(inter-annotator agreement与LLM的Cohen's Kappa)从0.62提升到 **0.74**
- 探索(exploration)阶段约需 **500-1000** 个样本即可收敛到近最优策略
- 不同LLM backbone(GPT-4, Claude, Llama-70B)上均有稳定提升，方法具有通用性

## 工程落地要点

- **示例池管理**：示例池需定期更新，移除过时示例、补充新领域示例，控制在200-500个
- **批量标注**：实际部署中将多个query-ad pair batch处理，共享相同的bandit策略更新
- **冷启动处理**：新加入的示例使用uniform exploration，积累足够数据后切换到UCB策略
- **成本控制**：通过bandit选择高质量示例减少prompt中的示例数量(从10个降到5个)，降低LLM API成本约 **40%**
- **质量监控**：持续对比bandit标注与人工标注的一致性，低于阈值时触发示例池重建

## 面试考点

1. **Q: 为什么用bandit而不是简单的retrieval方法选择ICL示例？** A: Retrieval方法(如BM25)只考虑表面相似性，bandit通过reward信号学习哪些示例真正有助于提升标注准确率，且能平衡探索和利用。
2. **Q: LinUCB中的UCB项起什么作用？** A: UCB项衡量对arm效果的不确定性，不确定性越高越倾向探索，确保不会错过潜在高质量示例。
3. **Q: 如何定义bandit的reward信号？** A: 以LLM标注结果与人工gold label的匹配程度作为reward，匹配给1分，不匹配给0分（或按等级差给部分分）。
4. **Q: ICL-Bandit的示例选择是per-query还是全局的？** A: 是per-query的contextual bandit，不同类型的query-ad pair会选择不同的最优示例组合。
5. **Q: 该方法如何处理候选示例池增长带来的计算开销？** A: 通过预筛选(如语义相似度top-50)缩小候选集，再在小候选集上运行bandit，兼顾效果与效率。
