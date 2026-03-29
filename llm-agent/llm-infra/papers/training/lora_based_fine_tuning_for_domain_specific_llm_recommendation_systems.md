# LoRA-based Fine-tuning for Domain-Specific LLM Recommendation Systems
> 来源：arxiv/2310.xxxxx | 领域：llm-infra | 学习日期：20260326

## 问题定义
将通用 LLM 适配到特定推荐场景的挑战：
- 全量微调（Full Fine-tuning）成本极高：7B 模型需要 16 张 A100
- 推荐领域知识与通用知识差异大：协同过滤信号、用户行为序列
- 多任务适配：同一 LLM 需要支持多个推荐场景（电商/视频/音乐）
- 持续学习：用户偏好随时间变化，模型需要定期更新

## 核心方法与创新点
**LoRA（Low-Rank Adaptation）for Recommendation LLM**

**LoRA 核心原理：**
```python
# 原始权重矩阵（冻结）
W ∈ R^{d×k}  # 预训练权重，不更新

# LoRA 分解：低秩矩阵近似权重变化
ΔW = B · A
# B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k)

# 前向计算
h = W·x + ΔW·x = W·x + B·A·x
# α 为缩放超参数
h = W·x + (α/r)·B·A·x

# 参数量对比
Full FT: d × k（巨大）
LoRA:    d×r + r×k = r×(d+k)（小 100-1000x）
```

**推荐场景的 LoRA 适配点：**
```python
# 在哪些层加 LoRA？
lora_config = LoRAConfig(
    target_modules=["q_proj", "v_proj",  # Attention QV 最重要
                    "k_proj", "o_proj",  # 可选
                    "gate_proj", "up_proj"],  # FFN（推荐场景常加）
    r=16,           # 秩（推荐场景通常 8-64）
    lora_alpha=32,  # 缩放因子
    lora_dropout=0.1
)
```

**多场景 LoRA（MoLoRA）：**
```python
# 不同推荐场景使用不同 LoRA 权重，共享 Base Model
base_model = load_llm("Llama-7B")
lora_ecommerce = train_lora(base_model, ecommerce_data)
lora_video = train_lora(base_model, video_data)
lora_music = train_lora(base_model, music_data)

# 推理时根据场景动态加载 LoRA（基模型共享）
def serve(request, scene):
    lora_weights = load_lora(scene)
    return base_model(request, lora_adapter=lora_weights)
```

**增量 LoRA 更新（持续学习）：**
```python
# 每日新数据训练增量 LoRA，叠加到现有权重
delta_lora = train_lora(base_model + current_lora, new_daily_data, epochs=1)
current_lora = merge_lora(current_lora, delta_lora, rate=0.1)  # EMA 更新
```

## 实验结论
- 电商推荐（Amazon Review）：HR@10 +8.3%（vs 不微调的通用 LLM）
- 显存：Full FT 需要 80GB，LoRA(r=16) 只需 12GB（同样 7B 模型）
- 训练时间：LoRA 比 Full FT 快 4-6x
- 多场景（3个场景独立 LoRA vs 共享 LoRA）：独立 LoRA 每场景 +2.1%

## 工程落地要点
1. **秩选择**：推荐场景通常 r=8-32，更大 r 效果边际递减
2. **LoRA 合并**：推理时可将 LoRA 合并到 Base（无额外延迟），或保持分离（灵活切换）
3. **QLoRA**：Base Model 4-bit 量化 + LoRA FP16，4B 模型单卡 24GB 显存可训练
4. **场景路由**：根据请求类型路由到不同 LoRA，共享 Base Model 节省显存
5. **评估**：每次更新后在验证集上评估，确保增量更新无退化

## 面试考点
**Q1: LoRA 为什么有效？低秩假设的数学依据？**
A: 过参数化 LLM 在微调时的权重变化 ΔW 本质上是低秩的——这是 Aghajanyan 等人的研究发现。原因：语言任务的适配只需要调整少数几个"语义方向"，完整的权重矩阵有大量冗余。LoRA 通过 BA 分解只更新低维子空间，参数效率极高。

**Q2: LoRA 中 rank r 如何选择？**
A: 经验法则：一般任务 r=4-16 足够；复杂任务（代码/数学）r=32-64；推荐/对话 r=16 是好起点。判断标准：在验证集上评估 r=4/8/16/32，画 r-AUC 曲线，选边际收益显著下降处的 r。不需要 r>128，此时近似 Full FT，失去效率优势。

**Q3: LoRA 与 Prompt Tuning/Prefix Tuning 相比如何？**
A: Prompt Tuning：添加可训练的 soft token，推理时 context 变长（增加延迟），表达能力弱于 LoRA。Prefix Tuning：在每层 attention 的 KV 前添加可训练前缀，实现类似但控制更精细。LoRA 修改权重矩阵本身，可合并后零开销推理，更适合多场景生产部署。

**Q4: 推荐系统中 LoRA 微调的样本构建策略？**
A: 将推荐任务格式化为文本任务：①序列推荐："用户历史：item1, item2...→预测下一个 item"（自回归）②CTR 预测："用户画像[...]广告[...]是否点击？→是/否" ③解释生成："推荐这个商品的原因：..." 混合多种任务增强泛化。

**Q5: QLoRA 相比 LoRA 的额外优化是什么？**
A: QLoRA = NF4（4-bit NormalFloat量化）Base Model + BFloat16 LoRA + 双重量化 + 分页优化器。关键：Base Model 4-bit 量化（不参与梯度），LoRA 适配器保持 BF16 精度更新。65B 模型可在单张 48GB GPU 上训练，大幅降低硬件门槛。
