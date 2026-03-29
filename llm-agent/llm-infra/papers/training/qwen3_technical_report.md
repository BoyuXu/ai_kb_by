# Qwen3 Technical Report

> 来源：https://arxiv.org/abs/2505.09388 | 领域：llm-infra | 学习日期：20260329

## 问题定义

### 核心挑战

1. **模型分裂问题**：传统范式需要在不同模型间切换——聊天优化模型（如GPT-4o）用于快速响应，专用推理模型（如QwQ-32B）用于复杂任务，导致部署成本高、用户体验割裂。

2. **推理成本不可控**：现有推理模型无法让用户灵活控制计算资源消耗，简单问题和复杂问题消耗相同算力。

3. **小模型训练效率低**：轻量级模型（<14B）直接进行RL训练成本高、效果差，需要更高效的蒸馏方法。

4. **多语言覆盖不足**：前代Qwen2.5仅支持29种语言，全球化应用场景受限。

---

## 核心方法与创新点

### 模型架构设计

| 模型类型 | 规模范围 | 关键架构特点 |
|---------|---------|-------------|
| Dense 模型 | 0.6B/1.7B/4B/8B/14B/32B | GQA + SwiGLU + RoPE + RMSNorm，去除QKV-bias，引入**QK-Norm** |
| MoE 模型 | 30B-A3B / 235B-A22B | 128专家/8激活，**无共享专家**，全局批次负载均衡损失 |

**Dense模型架构配置**：
```
Qwen3-4B/8B:   36层, 32/8 Q/KV heads, 128K context
Qwen3-14B:     40层, 40/8 Q/KV heads, 128K context
Qwen3-32B:     64层, 64/8 Q/KV heads, 128K context
Qwen3-235B-A22B: 94层, 64/4 Q/KV heads, 128专家/8激活, 128K context
```

### 预训练三阶段策略

```
Stage 1 (通用阶段): 30T tokens, seq_len=4096, 119种语言
       ↓
Stage 2 (推理增强): 5T tokens, STEM/代码/合成数据比例提升
       ↓
Stage 3 (长上下文): 百亿级 tokens, seq_len=32768
                    ABF: RoPE base_freq 10,000 → 1,000,000
                    YARN + DCA: 4倍长度外推
```

**数据扩增策略**：
- Qwen2.5-VL 提取PDF文档文本
- Qwen2.5-Math 合成数学教材
- Qwen2.5-Coder 合成代码数据
- 实例级数据混合优化（vs 传统源/领域级）

### 混合思维模式（核心创新）

**四阶段后训练流程**：

```
Base Model
    ↓
Stage 1: Long-CoT Cold Start    ← 注入基础推理模式（小样本、短训练）
    ↓                             QwQ-32B生成候选 → 严格过滤
Stage 2: Reasoning RL           ← GRPO强化学习，3,995 query-verifier对
    ↓                             大batch + 高rollout数，熵控制
Stage 3: Thinking Mode Fusion   ← /think 与 /no_think 模式融合
    ↓                             SFT混合数据 + Chat Template设计
Stage 4: General RL             ← 20+任务，三种奖励类型
    ↓
Final Model (支持动态模式切换)
```

**Chat Template 设计**：

| 模式 | 用户输入 | 模型输出格式 |
|-----|---------|-------------|
| Thinking | `{query} /think` | `<think>推理过程</think>回答` |
| Non-thinking | `{query} /no_think` | `<think></think>直接回答` |

### Thinking Budget 机制（涌现能力）

**动态推理预算控制**：
- 用户可设置思考的token上限
- 当思考长度达到阈值时，强制插入：
  > "Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>"
- 模型基于已累积的推理生成最终答案
- **关键**：此能力**无需显式训练**，由模式融合自然涌现

### Strong-to-Weak 蒸馏（小模型优化）

**两阶段蒸馏**：

```
教师模型 (Qwen3-32B 或 235B-A22B)
       ↓
Phase 1: Off-policy Distillation  ← 使用 /think 和 /no_think 输出做SFT
        ↓
Phase 2: On-policy Distillation   ← 学生模型生成序列，对齐教师logits（最小化KL散度）
        ↓
轻量级模型 (0.6B-14B, 30B-A3B)
```

**效率提升**：仅需四阶段训练的 **1/10 GPU hours**

### General RL的三种奖励类型

1. **规则奖励**：指令遵循、格式验证（高精度，防止reward hacking）
2. **模型奖励（有参考答案）**：Qwen2.5-72B-Instruct评分（灵活处理多样任务）
3. **模型奖励（无参考答案）**：Reward Model基于人类偏好数据打分

---

## 实验结论

### 旗舰模型 Qwen3-235B-A22B（Thinking模式）

| 基准测试 | OpenAI-o1 | DeepSeek-R1 | Grok-3-Think | Gemini2.5-Pro | **Qwen3-235B-A22B** |
|---------|-----------|-------------|--------------|---------------|---------------------|
| GPQA-Diamond | 78.0 | 71.5 | **80.2** | **84.0** | 71.1 |
| AIME'24 | 74.3 | 79.8 | 83.9 | **92.0** | **85.7** |
| AIME'25 | 79.2 | 70.0 | 77.3 | **86.7** | **81.5** |
| LiveCodeBench v5 | 63.9 | 64.3 | **70.6** | 70.4 | **70.7** |
| CodeForces | 1891 | 2029 | - | 2001 | **2056** |
| BFCL v3 | 67.8 | 56.9 | - | 62.9 | **70.8** |
| Arena-Hard | 92.1 | 92.3 | - | **96.4** | 95.6 |
| MT-AIME2024 | 67.4 | 73.5 | - | **76.9** | **80.8** |

**关键结论**：
- ✅ **开源模型中全面领先DeepSeek-R1**（17/23项基准）
- ✅ **仅需DeepSeek-R1的60%激活参数、35%总参数量**
- ✅ **代码/Agent能力超越所有开源模型**

### Non-thinking模式对比GPT-4o

| 基准 | GPT-4o | DeepSeek-V3 | Qwen2.5-72B | Qwen3-235B-A22B |
|-----|--------|-------------|-------------|-----------------|
| Arena-Hard | 85.3 | 85.5 | 81.2 | **96.1** |
| AIME'24 | 11.1 | 39.2 | 18.9 | **40.1** |
| AutoLogi | 65.9 | 76.1 | 66.1 | **83.3** |

### Dense旗舰 Qwen3-32B

| 对比项 | 结果 |
|-------|------|
| vs QwQ-32B | **17/23项超越**，新32B级SOTA |
| vs OpenAI-o3-mini | 对齐和多语言更优，推理可比 |
| vs Qwen2.5-72B | 仅一半参数，10/15项超越 |
| AIME'24 | **81.4%** |
| LiveCodeBench v5 | **65.7%** |
| Arena-Hard | **93.8%** |

### 轻量级模型表现（蒸馏优势）

| 模型 | 激活参数 | AIME'24 | AIME'25 | LiveCodeBench |
|-----|---------|---------|---------|---------------|
| Qwen3-30B-A3B | 3B | 80.4% | 70.9% | 62.6% |
| Qwen3-14B | 14B | 79.3% | 70.4% | 63.5% |
| Qwen3-8B | 8B | **76.0%** | 67.3% | 57.5% |
| Qwen3-4B | 4B | 73.8% | 65.6% | 54.2% |
| Qwen3-1.7B | 1.7B | 48.3% | 36.8% | 33.2% |
| Qwen3-0.6B | 0.6B | 10.7% | 15.1% | 12.3% |

### 蒸馏 vs RL训练（Qwen3-8B对比）

| 方法 | AIME'24 (Pass@1/64) | GPU Hours |
|-----|---------------------|-----------|
| Off-policy蒸馏 | 55.0 (90.0) | - |
| + RL训练 | 67.6 (90.0) | 17,920 |
| + On-policy蒸馏 | **74.4 (93.3)** | **1,800** |

**结论**：蒸馏效率是RL的10倍，且Pass@64（探索能力）也更优。

### Thinking Budget效果

- 性能随预算增加**平滑提升**
- 32K token后仍有上升趋势
- 覆盖Math、Coding、STEM多个维度

---

## 工程落地要点

### 部署架构建议

```python
# 生产环境模式选择
class Qwen3Service:
    def query(self, question, complexity="auto"):
        if complexity == "simple":
            mode = "/no_think"
            budget = 0
        elif complexity == "medium":
            mode = "/think"
            budget = 1024
        else:  # complex
            mode = "/think"
            budget = 8192
        
        prompt = f"{question} {mode}"
        return self.model.generate(prompt, max_thinking_tokens=budget)
```

### Chat Template 使用

```python
# Hugging Face集成
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-32B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")

# Thinking模式（默认，适合复杂任务）
messages = [{"role": "user", "content": "解这道数学题：..."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Non-thinking模式（适合快速响应）
messages = [{"role": "user", "content": "解这道数学题：... /no_think"}]
```

### 推理预算配置推荐

| 场景 | Budget设置 | 理由 |
|-----|------------|------|
| 简单问答 | 0 (no_think) | 最低延迟 |
| 常规任务 | 512-1024 | 平衡质量与速度 |
| 数学/代码 | 2048-4096 | 充分推理 |
| 研究级难题 | 8192+ | 最大化性能 |
| 长文档检索 | 8192 (推荐上限) | 防止思考干扰检索 |

### 成本效益分析

| 模型 | 总参数 | 激活参数 | 相对推理成本 | 推荐场景 |
|-----|-------|---------|-------------|---------|
| Qwen3-235B-A22B | 235B | 22B | 1.0x | 核心生产服务 |
| Qwen3-32B | 32B | 32B | ~1.5x | 高性价比推理 |
| Qwen3-30B-A3B | 30B | 3B | **~0.15x** | 边缘部署 |
| Qwen3-8B | 8B | 8B | ~0.4x | 私有化部署 |

---

## 面试考点

### 考点1：Qwen3的核心创新是什么？与传统两模型方案有何不同？

**答案要点**：

1. **传统方案**：需维护两套独立模型（Chat模型 + Reasoning模型），部署成本高，切换延迟大

2. **Qwen3方案**：
   - **统一模型架构**：单模型支持两种模式，通过`/think`和`/no_think`动态切换
   - **Thinking Budget**：用户精确控制推理深度，实现性能-延迟的连续权衡
   - **模式融合训练**：Stage 3将推理与非推理能力整合到同一参数空间

3. **技术本质**：模型学会了"元控制"——生成`<think>`时深度推理，`</think>`后直接输出，两种模式共享底层表示但行为可控

---

### 考点2：解释Qwen3四阶段后训练流程各阶段目标

**答案要点**：

| 阶段 | 名称 | 核心目标 | 关键设计 |
|-----|------|---------|---------| 
| S1 | Long-CoT Cold Start | 注入基础推理模式 | QwQ-32B生成 + 严格过滤，小样本短训练 |
| S2 | Reasoning RL | 强化数学/代码推理 | GRPO，3,995 query-verifier对，熵控制 |
| S3 | Thinking Mode Fusion | 融合非推理能力 | SFT混合数据，Chat Template |
| S4 | General RL | 通用能力对齐 | 20+任务，三种奖励，工具调用训练 |

**设计精髓**：
- S3中保留空`<think></think>`块确保格式一致性
- S4 ThinkFollow任务确保98.9%模式切换准确率
- General RL后AIME等复杂任务性能略降是**可接受的trade-off**（换取通用性）

---

### 考点3：Thinking Budget机制是如何工作的？为什么无需显式训练？

**答案要点**：

1. **工作原理**：
   - 用户设定token上限（如2048）
   - 达到阈值时强制插入停止指令
   - 模型基于部分推理完成回答

2. **涌现原因**：
   - 模型经过Stage 3学会同时处理thinking和no-thinking
   - 本质上学会了"在任意思考深度处给出答案"
   - 不完整思考是两种模式的**连续中间态**

3. **实验验证**：性能随预算**平滑提升**，无断崖，证明连续性

---

### 考点4：Strong-to-Weak Distillation相比直接RL训练的优势？

**答案要点**：

| 方法 | AIME'24 | GPU Hours | Pass@64 |
|-----|---------|-----------|---------|
| + RL | 67.6% | 17,920 | 90.0（无提升） |
| + Distillation | **74.4%** | **1,800** | **93.3（提升）** |

1. **效率**：仅需1/10 GPU hours
2. **性能**：Pass@1 和 Pass@64 双提升
3. **核心原因**：教师logits包含完整的概率分布（暗知识），比硬标签更丰富
4. **Pass@64提升**说明蒸馏扩展了学生的**探索空间**，而RL未能做到

---

### 考点5：Qwen3 MoE架构相比Dense模型有何设计差异和优势？

**答案要点**：

1. **架构差异**：
   - 128个专家，每token激活8个（vs Dense全激活）
   - **去除共享专家**（不同于Qwen2.5-MoE）
   - 全局批次负载均衡损失

2. **性能效率**：
   - 1/5激活参数 ≈ 同等Dense模型性能
   - Qwen3-30B-A3B（3B激活）≈ Qwen3-14B（14B激活）

3. **旗舰对比**：
   - Qwen3-235B-A22B：235B总 / 22B激活
   - vs DeepSeek-V3：671B总 / 37B激活
   - **35%总参数、60%激活参数，17/23项超越**

4. **工程价值**：激活参数决定推理开销，MoE大幅降低serving成本，同样的算力可部署更大规模模型
