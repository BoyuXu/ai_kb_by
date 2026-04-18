# LLM 基础设施论文笔记 — 2026-04-17

## 1. LoRA-based Fine-tuning for Domain-Specific LLM Recommendation Systems

**领域：** 参数高效微调 × 领域适配
**核心贡献：** LoRA 在推荐系统领域的高效适配方案

**关键公式：**
- 权重更新: W = W_0 + (α/r) × BA
- W_A ∈ R^{d×r}, W_B ∈ R^{r×k}，秩 r << min(d,k)
- 参数效率比: 1 - (2rk)/(dk)

**核心优势：**
- 更新参数量减少 10,000x
- GPU 内存降低 3x（相比全量微调）
- 单基座模型派生多个领域特化版本
- Multi-scale LoRA：跨层分布式低秩更新

**工业影响：** HuggingFace 上已部署 143,920+ LoRA adapters（截至 2024.10）

**面试考点：** 推导 LoRA 权重更新公式并解释低秩矩阵保持模型行为的原因、Multi-scale LoRA 跨语义尺度分布更新、LoRA 在推荐系统频繁重训场景的内存-计算 trade-off

---

## 2. Speculative Decoding for 10x Faster LLM Inference

**来源：** Google Research
**领域：** LLM 推理加速 × 投机解码
**核心贡献：** 用小模型并行生成多 token，大模型单次验证，保证输出分布完全一致

**Draft-and-Verify 框架：**
1. 小模型（draft model）提出 K 个 draft tokens
2. 大模型（target model）并行验证所有 K 个 tokens
3. 接受的 token 保留大模型预测；拒绝触发重采样
4. **P-EAGLE** 变体：单次前向传播生成所有 K 个 drafts

**关键保证：** 输出分布与原始大模型完全一致（无质量损失）

**工业影响：**
- Google Search AI Overviews 生产部署
- 集成于 vLLM 和 SGLang 框架
- 可实现 2-3x 推理加速

**面试考点：** Draft-and-Verify 算法如何保证输出分布不变、Draft model 质量对接受率和整体加速的影响、生产环境基准测试和 draft model 选择策略

---

## 3. Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG

**来源：** https://arxiv.org/abs/2501.09136
**领域：** Agentic RAG × 综合综述
**核心贡献：** 系统梳理从静态 RAG 到 Agentic RAG 的演进

**分类法维度：**
- **Agent 基数：** 单 Agent vs 多 Agent
- **控制结构：** 集中式 vs 分布式
- **自主程度：** 工具调用 vs 自主规划
- **知识表示：** 向量存储 vs 知识图谱

**Agentic 模式：**
- 自主决策检索时机和策略
- 多步推理 + 反思循环
- 工具编排实现知识获取
- 多 Agent 协作框架
- 知识图谱管理记忆

**应用场景：** 医疗、金融、教育、企业文档处理

**面试考点：** 分类法四维度及设计 trade-off、Agentic 模式如何改进静态 RAG 管线、企业应用的 Agentic 需求分析
