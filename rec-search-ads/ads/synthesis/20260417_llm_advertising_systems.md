# Synthesis: LLM 赋能广告系统 (2025-2026)

**日期：** 2026-04-17
**涵盖论文：** IDProxy (Xiaohongshu), HLLM-Creator (Douyin), RTBAgent (WWW 2025), LLM-Ads (工业研究)

---

## 一、技术演进脉络

### 1.1 LLM 在广告系统的四大角色

1. **特征生成器 (IDProxy)：** MLLM 生成 proxy embedding 解决冷启动
2. **创意生成器 (HLLM-Creator)：** 分层 LLM 个性化广告文案
3. **决策 Agent (RTBAgent)：** LLM 驱动实时竞价决策
4. **文案优化器 (LLM-Ads)：** LLM 生成超越人类的个性化广告

### 1.2 从工具到智能体

**传统方式：** LLM 作为离线特征提取 → 供下游模型使用
**当前趋势：** LLM 作为在线 Agent → 自主决策 + 反思优化

## 二、核心技术对比

| 论文 | LLM 角色 | 输入 | 输出 | 在线/离线 |
|------|---------|------|------|----------|
| IDProxy | 特征生成 | 图片+文本 | proxy embedding | 离线生成,在线使用 |
| HLLM-Creator | 创意生成 | 用户历史+广告 | 个性化标题 | 近线生成 |
| RTBAgent | 决策Agent | 竞价上下文 | 出价策略 | 在线决策 |
| LLM-Ads | 文案生成 | 人格+产品 | 说服力广告文案 | 离线生成 |

## 三、关键技术细节

### 3.1 IDProxy: 多模态冷启动

核心公式：
- 对齐损失: L_align = ||proxy_embed - id_embed||² (热门物品监督)
- CTR 损失: L_ctr = BCE(model(proxy_embed), label)
- 粗到精机制实现持续演化

### 3.2 HLLM-Creator: 三层解耦

Item LLM → User LLM → Creative LLM
- 解耦物品理解、用户建模、创意生成
- 用户聚类 + 匹配预测剪枝控制推理成本
- CoT 数据构建解决标注稀缺

### 3.3 RTBAgent: 工具增强推理

四工具：CTR 预估 + 策略库 + 出价器 + 历史检索
三记忆：历史决策 + 交易记录 + 专家知识
反思循环：每日总结 → 策略调整

## 四、工业实践要点

1. **延迟约束：** 广告系统 P99 < 50ms，LLM 在线推理需特殊优化
   - IDProxy: 离线预计算 proxy → 在线查表
   - HLLM-Creator: 用户聚类减少调用次数
   - RTBAgent: 规划-执行两步，规划可离线

2. **ROI 可控性：** LLM 输出不确定性 vs 广告主 ROI 要求
   - RTBAgent 的约束出价保证预算不超支
   - VAFT 将 eCPM 融入训练保证价值对齐

3. **可解释性：** 广告审核要求可解释
   - HLLM-Creator 的 CoT 链提供生成理由
   - RTBAgent 的策略库提供决策依据

## 五、面试考点

1. **LLM 在广告系统中的四种使用模式及各自的延迟-精度 trade-off？**
2. **IDProxy 如何在不微调 MLLM 的情况下实现 embedding 对齐？** 冻结 MLLM，训练轻量投影层
3. **RTBAgent vs RL-based RTB 的优劣？** LLM 可解释性+泛化 vs RL 精确优化+低延迟
4. **广告创意个性化的评估指标？** CTR/CVR + 人工评审 + 多样性指标
5. **LLM 在线推理如何满足广告系统的延迟 SLA？** 预计算/蒸馏/缓存策略
