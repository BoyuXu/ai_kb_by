# 生成式排序与重排技术演进 (2026-04-19)

## 核心趋势：从判别式到生成式排序范式转换

### 1. 技术演进路径

```
传统排序 → 深度CTR模型 → Transformer排序 → 生成式排序
(LR/GBDT)  (DIN/DIEN)    (HSTU/SASRec)   (GR2/HPGR/GenRank)
```

### 2. 今日学习论文矩阵

| 论文 | 核心创新 | 阶段 | 规模验证 |
|------|---------|------|---------|
| GR2 (2602.07774) | LLM推理+DAPO RL重排 | Reranking | 学术 |
| HPGR (2603.00980) | 层次化Session+偏好感知 | Ranking | ACM Web 2026 |
| Large-scale GenRank (2505.04180) | 工业规模生成式排序 | Ranking | 工业 |
| Foundation Model Survey (2504.16420) | 三范式全景 | Survey | N/A |
| Likelihood Trap (2510.10127) | 图结构打破似然陷阱 | Retrieval/Ranking | 学术 |

### 3. 核心技术对比

#### 3.1 GR2：LLM推理驱动重排
- **三阶段pipeline**：Semantic ID Mid-training → Reasoning Trace SFT → DAPO RL
- **关键突破**：首次将reasoning chain引入推荐重排
- **性能**：超越OneRec-Think (Recall@5 +2.4%, NDCG@5 +1.3%)

#### 3.2 HPGR：结构化序列建模
- **核心洞察**：扁平序列假设是生成式推荐的瓶颈
- **两阶段范式**：Session分割编码 → 偏好感知跨Session注意力
- **vs HSTU**：解决了dense attention在长序列中的噪声问题

#### 3.3 Large-scale GenRank：工业验证
- **核心结论**：生成式架构（而非训练技巧）是性能核心
- **Scaling行为**：模型规模和数据规模与推荐质量正相关

### 4. 面试核心考点

**Q1: 生成式推荐的三大范式是什么？**
- Feature-Based：LLM/VLM提取特征增强传统模型
- Generative：端到端生成推荐结果（Semantic ID → AR/NAR）
- Agentic：推荐作为自主Agent（规划+推理+工具调用）

**Q2: 生成式重排 vs 传统重排的核心优势？**
- 传统：Pointwise打分 → 独立评估每个候选
- 生成式：Listwise推理 → 考虑候选间上下文依赖 + 推理链提升可解释性

**Q3: 似然陷阱（Likelihood Trap）是什么？**
- 自回归模型倾向生成高似然（高频/热门）物品而非个性化推荐
- 解决方法：图结构约束、对比学习、DPO对齐

### 5. 工业实践总结

- Meta HSTU → 生成式排序标杆
- 快手 OneLoc → 生成式召回+排序统一
- 美团 MTGR → 多任务生成式排序
- GR2 → LLM推理驱动重排（下一代方向）
