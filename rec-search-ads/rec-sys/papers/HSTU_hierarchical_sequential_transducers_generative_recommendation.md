# HSTU: Hierarchical Sequential Transducers for Generative Recommendation (Meta)
> 来源：arXiv:2402.17152 | 领域：rec-sys | 学习日期：20260330

## 问题定义
大规模工业推荐系统面临两大挑战：(1) 用户行为序列超长（数万次），传统 Transformer 的 O(n²) 复杂度无法承受；(2) 需要统一召回和排序以减少误差传播。Meta 提出 HSTU（Hierarchical Sequential Transducer），用分层序列建模替代标准 Transformer，实现万亿参数规模的生成式推荐。

## 核心方法与创新点
1. **Hierarchical Transducer**：多层级序列建模，底层 token 级（action level）、中层 session 级、高层 user 级，分别捕捉短期行为、会话意图、长期兴趣。
2. **线性复杂度注意力**：用 Mamba-style 状态空间模型（SSM）替换 softmax attention，将序列建模从 O(n²) 降为 O(n)，支持数万步历史。
3. **Trillion-Parameter Scale**：模型规模达万亿参数（稀疏 MoE 形式），embedding table 占主体，dense 部分 ~10B。
4. **生成式统一**：同一模型既做召回（生成候选 ID 集合）又做排序（logit 作为打分），替代传统多阶段 pipeline。
5. **Context Parallelism**：长序列分片到多 GPU，配合 ring-attention 通信，训练效率线性扩展。

## 实验结论
- Meta Reels 推荐线上实验：用户观看时长 +1.8%，参与度 +2.1%
- 相比 DLRM 类模型，序列建模质量大幅提升（特别是用户新兴趣捕捉）
- 训练 FLOPs 增加 3x，但线上推理因 SSM 缓存可做 O(1) 递推，延迟可控

## 工程落地要点
- SSM 推理需维护隐状态缓存（$h_t$），在 serving 层需设计状态管理系统
- Embedding 更新频率高（实时行为），需配合 PSS（Parameter Server System）异步更新
- 万亿参数模型需 MoE + Expert Parallelism，通信开销是瓶颈
- 长序列分片训练需解决 padding/masking 跨分片对齐问题

## 常见考点
- Q: Transformer 在推荐系统中的瓶颈是什么？
  - A: O(n²) 注意力复杂度导致长序列无法处理；大批次 embedding lookup 导致访存带宽成为瓶颈
- Q: SSM（Mamba）和 Transformer attention 的本质区别？
  - A: SSM 用递推状态 $h_t = A h_{t-1} + B x_t$ 压缩历史，O(n) 推理；Transformer 全局 attention，O(n²) 但并行度更高
- Q: Meta HSTU 为什么选择 Hierarchical 结构？
  - A: 用户行为有天然层次（点击→会话→兴趣圈），层次结构既降低序列长度又保留多粒度语义
