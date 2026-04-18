# Synthesis: 生成式推荐系统范式演进 (2024-2026)

**日期：** 2026-04-17
**涵盖论文：** HSTU (Meta), GPR (Tencent), OneRanker (Tencent), RecGPT (Alibaba), E2E Semantic ID (Snap/Spotify)

---

## 一、技术演进脉络

### 1.1 从级联到统一

传统推荐系统采用多阶段级联架构：召回 → 粗排 → 精排 → 重排。每个阶段独立优化，导致目标失配和误差累积。

**演进路线：**
- **Phase 1 — 独立模型优化：** 每阶段独立训练（DSSM/DIN/DCNv2）
- **Phase 2 — 端到端微调：** 跨阶段梯度传递（Joint Training）
- **Phase 3 — 统一生成模型：** 单模型完成全链路（GPR/OneRanker/OneTrans）
- **Phase 4 — 万亿参数生成式：** HSTU 在 Meta 规模验证万亿参数可行性

### 1.2 核心范式对比

| 维度 | 传统级联 | 统一生成式 |
|------|---------|-----------|
| 目标对齐 | 各阶段独立目标 | 全局端到端优化 |
| 误差传播 | 逐级累积 | 单模型无累积 |
| 表征效率 | 重复计算 | 共享表征 |
| 工程复杂度 | 多系统维护 | 单系统但模型复杂 |
| 实时性 | 阶段间延迟 | KV caching 优化 |

## 二、核心公式与技术

### 2.1 Pointwise Normalization (HSTU)

替代 softmax，解决非平稳词汇表问题：
- softmax 归一化在候选集动态变化时不稳定
- pointwise normalization 对每个位置独立归一化
- 实现 5.3x-15.2x 加速（vs FlashAttention2）

### 2.2 Value-Aware Fine-Tuning (GPR)

L_VAFT = Σ_i w_i × L_i

其中 w_i = f(action_type_i, normalize(eCPM_i))

将商业价值信号直接注入训练目标，解决 CTR 与 GMV 的目标失配。

### 2.3 Semantic ID Generation

嵌入 → 层级聚类 → 离散 ID 序列

特性：相似物品共享 ID 前缀（有意义碰撞），同时具备记忆性（离散精确）和泛化性（共享语义）。

### 2.4 Heterogeneous Hierarchical Decoder (GPR/OneRanker)

双解码器设计：
- Decoder-1（用户意图）：粗粒度兴趣建模
- Decoder-2（广告生成）：细粒度价值感知生成
- Key/Value 直通机制保持信息流

## 三、工业实践要点

### 3.1 部署规模与效果

| 系统 | 公司 | 参数量 | 在线指标 |
|------|------|--------|---------|
| HSTU | Meta | 万亿级 | +12.4% 核心指标 |
| GPR | Tencent | 未公开 | CTCVR/GMV 显著提升 |
| OneRanker | Tencent | 未公开 | GMV +1.34% |
| RecGPT | Alibaba | 百亿级 | CTR +4.68%, NER +11.46% |
| OneTrans | ByteDance | 330M | GMV +5.68% |

### 3.2 关键工程挑战

1. **推理延迟：** 生成式模型 autoregressive 解码慢 → KV caching + 投机解码
2. **训练稳定性：** 万亿参数训练不稳定 → 分阶段训练（MTP→VAFT→HEPO）
3. **在线服务：** 单模型替代多阶段需要 graceful migration → A/B 渐进切换
4. **冷启动：** 新物品无 Semantic ID → proxy embedding（IDProxy）

## 四、面试考点

1. **为什么生成式推荐优于传统级联？** 目标对齐、误差消除、表征共享
2. **Pointwise vs Softmax normalization？** 非平稳词汇表下的数值稳定性
3. **VAFT 的 eCPM 加权如何保证训练稳定？** 归一化 + 行为类型分组
4. **Semantic ID 如何实现冷启动泛化？** 有意义碰撞使新物品共享已知物品的语义前缀
5. **生成式模型的推理延迟瓶颈？** autoregressive 解码 + 解决方案（KV cache, speculative decoding）
6. **单模型统一的风险？** 单点故障、调试困难、无法分阶段灰度 → 工程对策

## 五、未来方向

- Agentic 推荐（RecGPT-V2 方向）：Agent 自主推理用户意图
- 多模态生成式推荐：融合视觉/文本/行为多模态到统一生成框架
- 实时流式生成：结合 HSTU 的流数据优化和 OneTrans 的 KV caching
