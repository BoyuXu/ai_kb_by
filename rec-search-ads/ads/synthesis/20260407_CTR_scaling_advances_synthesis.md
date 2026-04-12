# 广告 CTR 预估前沿进展综合 - 2026-04-07

## 综合论文
- HeMix (2602.09387) - 异构兴趣提取
- DAES - 流式数值特征嵌入  
- GRAB (2602.01865) - LLM 启发序列优先范式
- EST (2602.10811) - 高效 Scaling Law
- RQ-GMM (2602.12593) - 多模态语义离散化

---

## 一、技术演进脉络

```
DNN + Embedding（YouTube, DNN for YouTube 2016）
  → Wide&Deep → DeepFM → DCN/DCNv2
    → Transformer-based CTR（BERT4Rec 等）
      → Sequence-First（GRAB）：自回归事件序列
      → Unified Modeling + Scaling（EST）：power-law scaling
  → 数值特征：分箱 → AutoDis → DAES（流式感知）
  → 多模态：原始特征 → VQ-VAE → RQ-VAE → RQ-GMM（GMM 软赋值）
  → 异构兴趣：DIEN → SIM → HeMix（context-aware + context-independent）
```

## 二、核心技术对比

| 问题 | 方法 | 核心创新 | 效果 |
|------|------|----------|------|
| 序列建模 | GRAB | CamA + 自回归范式 | Revenue +3.05%, CTR +3.49% |
| Scaling | EST | LCA + CSA + 统一序列 | Power-law scaling 验证 |
| 数值特征 | DAES | 水库采样 + 分布调制 | SOTA on短视频平台 |
| 多模态 | RQ-GMM | GMM 软赋值量化 | Adv Value +1.502% vs RQ-VAE |
| 兴趣提取 | HeMix | 动态+固定查询 + HeteroMixer | AMAP 数亿用户部署 |

## 三、核心公式

### EST 统一序列注意力（CSA）
基于内容相似度的稀疏注意力：
$$A_{ij} = \frac{\exp(q_i k_j / \sqrt{d}) \cdot \mathbb{1}[\text{sim}(c_i, c_j) > \tau]}{\sum_k \exp(q_i k_k / \sqrt{d}) \cdot \mathbb{1}[\text{sim}(c_i, c_k) > \tau]}$$

### RQ-GMM 高斯混合软赋值
$$P(z|x) = \sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$$
$$\text{SID} = \arg\max_k P(z_k | x)$$

### GRAB CamA 注意力
$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + \text{ActionMask}\right)V$$

## 四、工业实践经验

1. **Scaling Law 验证**（EST）：CTR 模型也具备 power-law scaling，统一建模是解锁 scaling 的关键。

2. **流式特征处理**（DAES）：水库采样高效估计分布，无需全量历史，适合实时系统。

3. **多模态 ID 化**（RQ-GMM）：直接使用连续 embedding 导致优化目标不对齐，离散化是工业落地必要步骤。

4. **序列优先设计**（GRAB）：将行为序列作为核心输入而非辅助特征，体现出与 LLM 设计理念的对齐。

## 五、面试高频考点

**Q：CTR 模型的 scaling law 为什么难以实现？**
A：异构输入、稀疏特征、早期聚合等导致信息瓶颈；统一序列建模（EST）是突破关键。

**Q：数值特征为什么不能直接用？**
A：数值特征分布可能随时间变化（概念漂移），且不同特征量纲不同；需要嵌入化处理。

**Q：多模态 embedding 如何与 CTR 模型对齐？**
A：离散化（SID）是主要方案；RQ-VAE→RQ-GMM 是演进路径；关键是 codebook 利用率和语义判别性。

**Tags:** #synthesis #ads #ctr #scaling #multimodal #sequence-modeling

---

## 相关概念

- [[concepts/vector_quantization_methods|向量量化方法]]
- [[concepts/attention_in_recsys|Attention 在搜广推中的演进]]
- [[concepts/sequence_modeling_evolution|序列建模演进]]
- [[concepts/embedding_everywhere|Embedding 技术全景]]
