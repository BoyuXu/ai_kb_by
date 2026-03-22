
---

## 📅 2026-03-20 自我评估

### 今日学习领域覆盖
- ✅ 广告 Pacing（预算控制、PID、对偶梯度下降、FPA Shading）
- ✅ 混合检索（BM25 vs Dense、RRF、LeSeR、BGE-M3）
- ✅ MoE 解耦推理（MegaScale-Infer、乒乓流水线、M2N 通信）
- ✅ LLM 推荐召回（URM、Q' Recall、生成式检索）
- ✅ KV Cache 压缩（ZSMerge、KVSharer，今日未出题，需补强）

### 掌握不够的知识点

1. **KV Cache 压缩（ZSMerge / KVSharer）**：今日未出题，说明掌握深度不够自信。具体弱点：ZSMerge 的 Zero-Shot 压缩判断准则、KVSharer 的层间共享选择机制，需要再深读一遍合成卡片后独立出题。

2. **FPA Shading 的非平稳扩展**：Q1 考察了基础 shading，但 Wasserstein 距离追踪的理论推导（遗憾界 O(√T + W_T) 的证明思路）仍感模糊，是面试高频难点。

3. **LeSeR 具体实现细节**：只了解"语义召回 → BM25 精排"的框架，但 Top-K 怎么定、BM25 重排用什么分数函数，细节层面还需巩固。

4. **LLM 召回的延迟量化**：开放题中提到"100ms+"，但没有给出具体的 benchmark 数字（GPT-4o vs 蒸馏 7B vs 量化 INT4 的典型 latency）。

### 明天补充计划
- [ ] 精读 KV Cache 综合卡，独立出 2 道题（ZSMerge 原理 + KVSharer 层选择）
- [ ] 查 Wasserstein 遗憾界的直觉证明，写到 Pacing 卡片的附注里
- [ ] 搜索 "LLM inference latency benchmark 2025"，补充延迟数字到 LLM 召回笔记

### 今日亮点
- 系统设计题（Q4 预算管控系统）回答较完整，分层架构 + 容量估算都到位
- 开放题的"三步走决策框架"思路清晰，能体现工业化落地判断


---
## 自我评估 - 2026-03-21

### 今日题目覆盖
5题：SPLADE vs BM25（初）、生成式检索 Constrained Decoding（中）、GRPO vs PPO（中）、电商搜索系统设计（高）、跨领域 RL 迁移开放题（高）

### 掌握较好的点
- GRPO 的组内相对 reward 机制和 advantage 归一化原理，能清楚说出"节省 50% 显存"的来源
- SPLADE 的 MLM Head + log-ReLU 权重计算，以及和 BM25 底座相同（倒排索引）的工程价值
- 生成式检索的端到端流程（RQ-VAE → Trie 约束解码 → ANN 融合）

### 掌握不够深的点
1. **GRPO 的数学细节**：KL penalty 的具体公式（是 forward KL 还是 reverse KL？系数怎么设？）— 需要精读 DeepSeek-R1 原始论文
2. **SPLADE 的 INT8 量化感知训练**：知道有这个技术，但不清楚在倒排索引场景如何量化权重（浮点权重 → 整数，精度损失来自哪里）
3. **Constrained Decoding 的工程实现细节**：Trie 如何高效并行支持 batch beam search？GPU 上 Trie 遍历的内存访问模式？

### 明日补充计划
- [ ] 精读 GRPO 原文（arxiv 2402.03300）的 KL 惩罚项公式
- [ ] 查一下 SPLADE-v3 的 QAT（量化感知训练）怎么处理稀疏向量的
- [ ] 复习 LLM structured output（如 Outlines 库）的 Constrained Decoding 实现，和推荐场景的 Trie 有什么共同之处

### 整体评估
今日理论掌握度：★★★★☆（GRPO/SPLADE 原理清楚，系统设计能给出完整架构）
开放题质量：★★★★☆（跨域迁移角度新颖，但"Lagrange 乘子迁移到 KL 惩罚"这个点需要用数学验证是否成立）


---

## 2026-03-22 自我评估

### 今日覆盖内容
- 广告偏差治理三部曲（Position Bias / ESMM / DIN）
- 生成式推荐范式对比（自回归 vs 扩散）
- Semantic ID 完整画像（RQ-VAE / 可变长度 / Spotify落地）
- LLM推理效率三角（GRPO / MoE / Speculative Decoding）
- 稀疏 vs 密集检索决策（SPLADE-v3 / RRF融合）
- 统一搜推模型（Spotify ULM）

### 薄弱点识别

**❌ 需要加强（优先级 High）**：
1. **Speculative Decoding 工程细节**：接受率验证的具体算法（rejection sampling with correction）还不够熟练。背后的数学推导（如何保证目标模型分布不变）有模糊地带。
2. **ESMM 延迟反馈建模**：ESMM 解决了空间偏差，但转化延迟（用户3天后才购买）的处理细节没学到——ESM² 或 DEFER 等方法需要补充。
3. **MoE 路由的负载均衡损失**：知道"辅助损失防止路由坍塌"，但具体的 load balance loss 公式和调参经验不掌握。

**⚠️ 需要巩固（优先级 Medium）**：
4. **SPLADE-v3 的 DeBERTa 升级细节**：知道用了 DeBERTa 基座，但 v2→v3 的具体改进（蒸馏策略、量化方案）模糊。
5. **DiffGRM 课程扩散（Curriculum Diffusion）**：大致知道先学粗粒度后学细粒度，但具体实现（噪声调度 vs 课程调度如何结合）不熟。

**✅ 掌握较好**：
- ESMM 核心思想（全空间乘法分解）
- RRF 融合公式和 k=60 的含义
- Semantic ID / RQ-VAE 核心机制
- GRPO 的组相对优势替代 Critic 的思路

### 明天补充计划
1. **DEFER/ESM² 延迟转化建模**：搜索最新的延迟反馈广告系统论文，补充到 `ads/` 笔记
2. **Speculative Decoding rejection sampling 数学推导**：读原论文 (Leviathan et al., 2023) 的 Algorithm 1
3. **MoE 负载均衡 loss 实现**：找 Mixtral 或 Switch Transformer 代码，看 auxiliary loss 具体实现

### 面试准备状态
- qa-bank.md 总计 4134 行（增加 240 行今日内容）
- 今日5题质量评估：Q4系统设计题最能体现综合水平，Q3关于Speculative Decoding加速上限的数学推导是高频考点

