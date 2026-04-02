# Speculative Decoding for 10x Faster LLM Inference in Production
> 来源：arxiv/2211.17192 | 领域：llm-infra | 学习日期：20260326

## 问题定义
LLM 推理（解码）速度是生产部署的关键瓶颈：
- 自回归解码：每次生成一个 token，需要大模型完整前向传播
- 内存带宽瓶颈：LLM 推理受限于内存带宽（不是计算），GPU 利用率低（<30%）
- 延迟不可接受：GPT-4 级别模型每秒 20-50 tokens，用户体验差
- 成本高：延迟高 = GPU 时间长 = 推理成本高

## 核心方法与创新点
**Speculative Decoding（投机解码）**：用小模型"猜"，用大模型"验证"。

**核心算法：**
```python
def speculative_decode(prompt, target_model, draft_model, gamma=4):
    """
    gamma: 每轮猜测的 token 数量
    """
    tokens = list(prompt)
    
    while not done(tokens):
        # Step 1: 小模型（Draft）连续猜 gamma 个 token
        draft_tokens = []
        draft_probs = []
        for i in range(gamma):
            p_draft = draft_model(tokens + draft_tokens)
            t = sample(p_draft)
            draft_tokens.append(t)
            draft_probs.append(p_draft[t])
        
        # Step 2: 大模型（Target）并行验证所有 gamma+1 个位置
        # 一次前向传播同时计算 gamma+1 个位置的概率
        target_probs = target_model(tokens + draft_tokens)  # 并行！
        
        # Step 3: 逐位验证（接受/拒绝）
        accepted = 0
        for i, (t, p_d, p_t) in enumerate(zip(draft_tokens, draft_probs, target_probs[:-1])):
            accept_prob = min(1, p_t[t] / p_d[t])  # 重要性采样
            if random() < accept_prob:
                tokens.append(t)
                accepted += 1
            else:
                # 拒绝：从修正后的分布采样一个替代 token
                p_corrected = max(0, p_t - p_d) / sum(max(0, p_t - p_d))
                tokens.append(sample(p_corrected))
                break
        
        # Step 4: 如果所有都接受，用大模型生成第 gamma+1 个
        if accepted == gamma:
            tokens.append(sample(target_probs[-1]))
    
    return tokens
```

**加速原理：**
```
传统解码：每个 token 需要 1 次大模型前向 → N 个 token = N 次
投机解码：每轮 1 次大模型前向 → 期望接受 E[accepted_per_step] > 1 个
加速比 = E[accepted_per_step] / (1/tokens_per_forward)
```

**数学保证（输出分布完全一致）：**
```
接受率 α_i = min(1, P_target(x_i) / P_draft(x_i))
证明：最终生成的 token 序列的分布 = 直接用 target_model 生成
→ 投机解码是无损加速（不降低输出质量）
```

## 实验结论
- T5-XXL（11B）+ T5-Small（60M）作为 Draft：
  - 加速比：2.2-3.1x（取决于任务）
- GPT-4 级别模型（理论）：
  - 代码生成任务：4-6x 加速（代码可预测性高，接受率高）
  - 随机对话：1.5-2x 加速（不可预测，接受率低）
- Google Bard 生产部署（2023）：~3x 加速

## 工程落地要点
1. **Draft 模型选择**：需要与 Target 共享 Tokenizer 和 Vocabulary；通常是同系列小模型（如 Llama-7B → Llama-70B）
2. **Gamma（猜测长度）**：通常 γ=4-8；过大则大模型并行计算成本高；过小则加速不明显
3. **批量推理兼容**：投机解码对 Batch Size=1 有效；大批量时效果下降（需要对齐不同请求的接受长度）
4. **vLLM 集成**：vLLM 已内置 Speculative Decoding，开启方式：`--speculative-model draft_model --num-speculative-tokens 5`
5. **动态 gamma**：根据历史接受率动态调整 gamma（高接受率时增大 gamma）

## 常见考点
**Q1: 投机解码为什么能在不降低质量的情况下加速？**
A: 关键是接受/拒绝机制（重要性采样）：只有当小模型的预测分布 "覆盖" 大模型分布时才接受，否则拒绝并从修正分布采样。数学上可证明：最终生成的 token 序列分布与直接用大模型采样完全相同，加速不以牺牲输出质量为代价。

**Q2: 投机解码的加速比取决于哪些因素？**
A: ①Draft 模型与 Target 模型的分布相似度（接受率 α）②Gamma 值（每轮猜测数）③任务可预测性（代码/固定格式 > 随机对话）④大模型推理延迟（大模型越大，每次前向越贵，加速收益越大）。加速比 ≈ (1-α^γ)/(1-α) 的简化估算。

**Q3: 投机解码与 Beam Search 的本质区别？**
A: Beam Search：在解码时保留多条候选序列，提高输出质量（确定性）。投机解码：用小模型预猜 + 大模型批量验证，提高解码速度（不改变输出分布）。两者目标不同：Beam Search 提质，投机解码提速。

**Q4: 在工业推理服务中，投机解码的主要工程挑战？**
A: ①大批量并发：投机解码对 batch=1 最有效，大批量时不同请求接受的 token 数不同，需要变长 padding 处理 ②显存：需要同时加载 Draft + Target 两个模型 ③Draft 模型维护：需要与 Target 模型同步更新（模型升级时两者要对齐）。

**Q5: 除了 Speculative Decoding，还有哪些 LLM 推理加速方法？**
A: ①KV Cache：缓存历史 token 的 KV，避免重复计算（标准优化，必须用）②Flash Attention：内存高效的 attention 计算（减少内存带宽瓶颈）③量化（INT8/INT4）：减少模型大小和计算量（~2-4x）④Continuous Batching：动态 batch，提高 GPU 利用率 ⑤vLLM PagedAttention：优化 KV Cache 内存碎片，提高吞吐量。
