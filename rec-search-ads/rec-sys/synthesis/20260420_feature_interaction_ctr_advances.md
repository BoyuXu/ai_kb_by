# CTR 特征交互前沿：InterFormer + INFNet (2026-04-20)

> 覆盖论文：InterFormer (CIKM 2025, Meta), INFNet (2508.11565, 工业广告)
> 主题：可扩展异构特征交互，从二阶交叉到全局信息流

---

## 一、技术演进脉络

```
特征交叉演进：
  手工交叉（LR） → 二阶交叉（FM/FFM） → 显式高阶交叉（DCN-V2/xDeepFM）
    → 隐式交叉（DNN） → 序列+特征统一（OneTrans/FuxiAlpha）
    → 异构交互学习（InterFormer, INFNet）  ← 2025-2026 新方向

核心转变：
  - 从"同质特征交叉"到"异构模态交互"（categorical + sequence + task）
  - 从"单向信息流"到"双向互益学习"
  - 从 O(n^2) 全局注意力到 O(n) 线性复杂度
```

## 二、核心技术对比

| 维度 | InterFormer (Meta) | INFNet |
|------|-------------------|--------|
| 核心问题 | 跨模态信息流不充分 + 信息过早聚合 | 特征交互复杂度二次方 + 任务无关 |
| 架构 | 交错式双向信息流 + Bridge Arch | Hub Token Aggregate-and-Broadcast |
| 复杂度 | 保留完整模态信息，Bridge 做选择性总结 | 线性复杂度（proxy cross-attention） |
| 创新点 | 双向交互 + 信息保留 + 分离汇总 | 三类 token（cat/seq/task）+ 双流设计 |
| 部署 | Meta Ads 主模型，topline +0.6% | 商业广告系统，收入 +1.587%，CTR +1.155% |
| 会议 | CIKM 2025 | arXiv 2025 |

## 三、核心公式

### InterFormer: 交错式双向信息流

**问题**：现有方法中不同数据模态（user features, item features, context, behavior sequence）间的信息流是单向的，导致互益学习不充分。同时，早期聚合（early summarization）导致过多信息丢失。

**解法**：

每个模态保留完整的 token 级表示，通过 Bridge Arch 做选择性信息交换：

$$\mathbf{h}_A^{(l+1)} = \text{SelfAttn}(\mathbf{h}_A^{(l)}) + \text{Bridge}(\mathbf{h}_B^{(l)} \to \mathbf{h}_A^{(l)})$$
$$\mathbf{h}_B^{(l+1)} = \text{SelfAttn}(\mathbf{h}_B^{(l)}) + \text{Bridge}(\mathbf{h}_A^{(l)} \to \mathbf{h}_B^{(l)})$$

Bridge 是独立的轻量注意力模块，负责信息选择和摘要，避免直接全连接带来的噪声。

### INFNet: Aggregate-and-Broadcast 信息流

**三类 Token**：
- Categorical tokens $\mathbf{T}_c$：用户/物品/上下文离散特征
- Sequence tokens $\mathbf{T}_s$：行为序列，保留 item 级信号
- Task tokens $\mathbf{T}_t$：任务标识（多任务场景）

**Hub Token 机制**：每组引入少量 hub token $\mathbf{H}$

**Aggregate**（跨组信息汇聚）：
$$\mathbf{H}^{(l)} = \text{CrossAttn}(\mathbf{H}^{(l-1)}, [\mathbf{T}_c; \mathbf{T}_s; \mathbf{T}_t])$$

**Broadcast**（信息回注）：
$$\mathbf{T}_x^{(l)} = \mathbf{T}_x^{(l-1)} + \text{PGU}(\mathbf{T}_x^{(l-1)}, \mathbf{H}^{(l)})$$

PGU (Proxy Gated Unit) 是门控广播单元，控制每个 token 接收多少全局信息。

**复杂度**：$O(K \cdot N)$ 其中 $K$ 是 hub token 数（远小于 $N$），实现线性交互。

## 四、工业实践经验

1. **双向信息流 > 单向**（InterFormer）：
   - 传统方法：user features → item features 单向注入
   - InterFormer：双向交互让 item 特征也能增强 user 表示
   - 信息保留策略：不做早期聚合，Bridge 做选择性总结

2. **Hub Token 是可扩展交互的关键**（INFNet）：
   - 直接全 token 交互 = O(n^2)，工业不可行
   - Hub token 做中介 = O(K*n)，K << n
   - 类似 Perceiver 的思想，但针对推荐场景做了任务感知设计

3. **Width-Preserving Stacking 保留序列信号**（INFNet）：
   - 传统做法：序列先 pooling 成一个向量再交叉 → 丢失 item 级信号
   - INFNet：每个序列 token 独立参与交互，保留完整时序信息
   - 这对长序列建模尤为关键

4. **异构 vs 同构交互需要分治**（INFNet）：
   - 异构交互（跨模态）用 proxy cross-attention
   - 同构交互（模态内）用 type-specific PGU
   - 分治策略比统一 attention 更高效

## 五、面试考点

**Q：为什么 CTR 模型需要异构交互？传统特征交叉不够吗？**
A：传统特征交叉（FM/DCN-V2）把所有特征视为同质的。但实际上，categorical 特征、行为序列和任务标识的语义空间完全不同。异构交互允许不同类型特征以不同方式交换信息，避免"一刀切"带来的信号噪声。InterFormer 的双向 Bridge 和 INFNet 的三类 token 都体现了这个思路。

**Q：InterFormer 的 Bridge Arch 和直接 Cross-Attention 有什么区别？**
A：直接 Cross-Attention 让所有 token 两两交互，信息量大但噪声也大（aggressive aggregation）。Bridge Arch 是独立的轻量模块，做选择性信息提取和摘要，类似"翻译官"而非"直接对话"。好处：(1) 保留各模态完整信息，(2) 只传递有用信号，(3) 计算量可控。

**Q：INFNet 的 hub token 机制和 Perceiver 有什么关系？**
A：思想类似——都用少量"代理 token"汇聚全局信息。区别：(1) INFNet 是双流设计（异构+同构分开），Perceiver 是统一处理；(2) INFNet 有任务感知（task token 参与路由），Perceiver 没有；(3) INFNet 的 Broadcast 用门控机制（PGU），而非简单加法。工业效果：收入 +1.587%。

**Q：特征交互从 FM 到 InterFormer 的演进主线是什么？**
A：四个阶段：(1) FM/FFM 二阶显式交叉 → (2) DCN-V2/xDeepFM 高阶显式交叉 → (3) DNN 隐式交叉 + Transformer 统一建模（OneTrans/FuxiAlpha）→ (4) 异构模态感知交互（InterFormer/INFNet）。核心趋势：从"所有特征一视同仁"到"尊重特征异质性"。

**Tags:** #synthesis #ctr #feature-interaction #interformer #infnet #heterogeneous-interaction #industrial

---

## 相关概念

- [[attention_in_recsys|Attention 在搜广推中的演进]]
- [[embedding_everywhere|Embedding 技术全景]]
- [[CTR模型深度解析|CTR 模型深度解析]]
- [[精排模型进阶深度解析|精排模型进阶深度解析]]
- [[20260407_GenRec_advances_synthesis|生成式推荐前沿进展]]
