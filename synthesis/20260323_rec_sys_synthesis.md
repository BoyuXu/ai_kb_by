# 推荐系统综合总结（20260323）

> 本批论文：13篇 | 主题：生成式推荐的工业化落地

## 本批核心主题（5个）

### 1. 生成式推荐的统一框架（OneRec / GPR / ETEGRec）
多篇论文聚焦于用单一生成式模型统一推荐的召回和排序阶段。核心动机是消除多阶段pipeline的信息损失和目标不一致。快手OneRec是目前工业落地最完整的案例，线上用户留存率+1.8%。**关键insight：生成式统一比分阶段pipeline在序列依赖建模上有本质优势。**

### 2. 物品Tokenization的进化（ETEGRec / COBRA / Sparse Meets Dense）
从固定的RQ-VAE tokenizer演进到端到端可学习tokenization（ETEGRec），再到稀疏-稠密混合表示（COBRA）。**关键insight：tokenizer和推荐模型的联合优化是下一个重要方向，端到端训练比分离训练带来约3%的NDCG提升。**

### 3. 工业级大模型推荐（Actions Speak Louder / Wukong / MTGRBoost）
Meta和美团的工作共同验证：推荐系统也存在Scaling Law，但形式不同（Embedding Table scaling比MLP更有价值）。万亿参数推荐模型在Meta视频推荐中带来显著的时长增益。**关键insight：工业推荐的主要参数应优先扩展在Embedding，而非MLP深度。**

### 4. 推理增强推荐（REG4Rec / Act-With-Think / OneRec-Think）
将LLM的推理能力（CoT/思维链）引入推荐，通过显式推理步骤提升复杂意图理解。"轻量推理"（Act-With-Think）在效果和效率间取得最佳平衡，32-64 token的推理链性价比最高。**关键insight：推理对长尾和冷启动用户收益最大（行为数据少，推理补充语义先验）。**

### 5. 多目标个性化优化（Pantheon / 淘宝Re-ranking）
从固定权重多目标加权演进到Pareto优化和个性化权重推断。生成式重排序（淘宝）考虑列表级物品交互，相比pointwise重排有本质优势。**关键insight：个性化多目标权重比全局权重提升约1.5%，用户满意度综合指标更优。**

## 技术演进趋势

**从"搜索空间缩小"到"生成式扩展"**：传统推荐是在候选集中过滤，生成式推荐是从token空间自回归生成，根本范式不同。工业界正在从验证生成式可行性（2023-2024）转向规模化落地优化（2025+）。

**大模型能力向推荐迁移的三条路径**：(1) LLM知识蒸馏到推荐模型（LEARN/PLUM）；(2) 推荐模型引入LLM推理能力（REG4Rec）；(3) 推荐直接用LLM作为backbone（HLLM）。三条路径适用不同的资源和业务场景。

**效率是工业落地的第一约束**：生成式推理比pointwise慢10-100x，几乎所有工作都在研究加速（prefix cache/speculative decoding/beam search剪枝）。latency是工业推荐最硬的约束。

## 面试高频考点（8条）

1. **Q: 生成式推荐和传统两阶段推荐的根本区别？** A: 生成式直接从token空间生成物品序列，联合优化召回排序；传统方式分阶段优化，存在目标不一致和信息损失

2. **Q: RQ-VAE在生成式推荐中的作用？** A: 将连续的物品embedding量化为离散的层级token序列，让自回归生成模型可以用有限词表表示海量物品

3. **Q: 推荐系统的Scaling Law与LLM有何不同？** A: 推荐系统参数主要在Embedding Table（稀疏），优先扩展Embedding比扩展MLP收益更高；LLM主要扩展稠密的注意力和FFN

4. **Q: 生成式推荐的工业落地主要挑战？** A: (1)推理latency（beam search慢）；(2)长尾物品覆盖率；(3)新物品冷启动；(4)物品动态增减的索引更新

5. **Q: 列表级重排（listwise reranking）比pointwise的优势？** A: 考虑物品间的互补性、多样性和竞争关系；pointwise独立打分忽略列表内上下文，无法建模位置偏差

6. **Q: 推荐系统中"偏好对齐"（Preference Alignment）的方法？** A: DPO（Direct Preference Optimization）、PPO（RLHF）、迭代偏好对齐（IPA），用用户正负反馈构建偏好数据对

7. **Q: 推理链（CoT）在推荐中的最优长度？** A: 实验表明32-64 token的轻量推理性价比最高；过短无法深入分析，过长引入噪声；通常比LLM推理短10-20x

8. **Q: Outcome-Conditioned生成（PinRec）的意义？** A: 根据期望的用户行为（保存/点击/深度参与）条件化生成，不同场景可切换不同目标条件，比无条件生成更可控灵活

## 与已有知识的连接

- **与TIGER/P5等早期生成式推荐**的连接：本批论文是工业界对学术界生成式推荐的工程落地验证，快手OneRec是最完整的工业案例
- **与推荐Scaling Law的关联**：Wukong、Actions Speak Louder共同验证推荐领域Scaling有效，为后续大规模投入提供理论依据
- **与LLM对齐（RLHF/DPO）的关联**：推荐系统的"偏好对齐"借鉴了LLM对齐的方法论，迭代式偏好优化已在快手验证可行
