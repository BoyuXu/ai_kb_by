# BiGEL: Behavior-informed Graph Embedding for Multi-Behavior Multi-Task Recommendation

> 来源：https://arxiv.org/abs/2601.07294 | 领域：rec-sys | 学习日期：20260403

## 问题定义

在实际推荐系统中，用户与物品的交互并非单一行为，而是多种行为并存：浏览、点击、加购、购买、评价等。这些行为之间存在层级关系（view -> click -> add-to-cart -> purchase）和信息互补关系。传统多任务推荐模型（MMoE、PLE）虽然能联合预测多个行为，但忽略了行为之间的依赖结构。

BiGEL（Behavior-informed Graph Embedding Learning）针对多行为多任务推荐场景，提出三个关键创新：(1) 级联门控反馈机制（Cascaded Gating Feedback）利用行为间的层级关系增强信息传递；(2) 全局上下文增强（Global Context Enhancement）通过图嵌入捕捉用户-物品交互的全局模式；(3) 对比偏好对齐（Contrastive Preference Alignment）确保不同行为的表示空间对齐。

该工作的核心问题是：如何在多行为多任务设置下，既利用行为间的层级关系，又防止噪声行为的负迁移？

## 核心方法与创新点

**Cascaded Gating Feedback（级联门控反馈）**：行为之间存在天然的转化漏斗关系（view -> click -> purchase），BiGEL 通过级联门控让高层行为（如 purchase）的信号反馈给低层行为（如 click）的表示学习：

$$
\mathbf{h}_k^{(l)} = \mathbf{h}_k^{(l-1)} + \sum_{j \in \text{parents}(k)} \sigma\left(\mathbf{W}_{jk} [\mathbf{h}_j^{(l-1)}; \mathbf{h}_k^{(l-1)}]\right) \odot \mathbf{h}_j^{(l-1)}
$$

其中 $\mathbf{h}_k^{(l)}$ 是行为 $k$ 在第 $l$ 层的表示，$\text{parents}(k)$ 是行为层级图中 $k$ 的父节点。门控 $\sigma(\mathbf{W}_{jk}[\cdot;\cdot])$ 控制来自高层行为的信息流入量，避免噪声传播。

**Global Context Enhancement（全局上下文增强）**：构建用户-物品-行为的异构图，通过 GNN 聚合全局交互模式：
- 节点：用户节点、物品节点
- 边：不同行为类型对应不同类型的边
- 通过 R-GCN（Relational GCN）在图上做多轮消息传递

**Contrastive Preference Alignment（对比偏好对齐）**：不同行为的表示空间可能不一致，BiGEL 通过对比学习将同一用户在不同行为下的表示拉近：

$$
\mathcal{L}_{\text{align}} = -\sum_{u} \sum_{k_1 \neq k_2} \log \frac{\exp(\text{sim}(\mathbf{z}_u^{k_1}, \mathbf{z}_u^{k_2}) / \tau)}{\sum_{v \neq u} \exp(\text{sim}(\mathbf{z}_u^{k_1}, \mathbf{z}_v^{k_2}) / \tau)}
$$

其中 $\mathbf{z}_u^{k}$ 是用户 $u$ 在行为 $k$ 下的表示，$\tau$ 为温度参数。正样本对是同一用户的不同行为表示，负样本对是不同用户的表示。

## 系统架构

```mermaid
graph TD
    subgraph 输入
        A1[用户-物品交互数据<br/>多种行为类型]
        A2[用户/物品特征]
    end

    subgraph 图嵌入模块
        A1 --> G1[异构交互图构建<br/>User-Item Bipartite + Behavior Edges]
        G1 --> G2[R-GCN<br/>Global Context Enhancement]
        G2 --> G3[全局用户/物品嵌入]
    end

    subgraph 级联门控模块
        A2 --> B1[Behavior-Specific<br/>Embedding Layers]
        G3 --> B1
        B1 --> C1[View Task Tower]
        B1 --> C2[Click Task Tower]
        B1 --> C3[Purchase Task Tower]
        C3 -->|Gating Feedback| C2
        C2 -->|Gating Feedback| C1
    end

    subgraph 对齐与输出
        C1 --> D[Contrastive<br/>Preference Alignment]
        C2 --> D
        C3 --> D
        D --> E1[P(view)]
        D --> E2[P(click)]
        D --> E3[P(purchase)]
    end
```

## 实验结论

- **公开数据集**（Tmall、CIKM2019 EComm）：
  - 相比 MBGMN（多行为图模型）：AUC +1.23%，NDCG@10 +2.15%
  - 相比 PLE + 行为特征：AUC +0.89%
  - 相比 CML（对比多行为学习）：AUC +0.67%
- **消融实验**：
  - 去掉 Cascaded Gating Feedback：AUC 下降 0.51%（最关键组件）
  - 去掉 Global Context Enhancement：AUC 下降 0.38%
  - 去掉 Contrastive Alignment：AUC 下降 0.29%
- **行为稀疏性分析**：对于购买行为（最稀疏），BiGEL 通过级联门控从点击行为借用信息，AUC 提升幅度最大（+1.8%）。

## 工程落地要点

1. **行为层级图定义**：需要领域专家根据业务特点定义行为间的层级关系（如电商：浏览→点击→加购→购买；内容：曝光→点击→阅读→分享）。
2. **图嵌入预计算**：全局图嵌入可以离线预计算并定期更新（如每小时），不需要在线实时计算 GNN，降低推理延迟。
3. **级联方向**：默认是高层行为反馈给低层行为，但也可以尝试双向级联，让低层行为的丰富数据辅助高层行为的稀疏数据学习。
4. **对比学习的采样策略**：负样本采样对对比偏好对齐的效果影响大，推荐使用 in-batch negative sampling 以提高效率。
5. **冷启动物品**：新物品缺少交互数据无法构建图嵌入，可以用物品 side features 的 embedding 作为初始化。

## 面试考点

1. **BiGEL 如何利用行为间的层级关系？** 通过 Cascaded Gating Feedback，高层行为（purchase）的信号通过门控网络反馈给低层行为（click）的表示学习，门控自适应控制信息流量避免噪声传播。
2. **为什么需要 Contrastive Preference Alignment？** 不同行为的表示空间独立学习可能会分裂——同一用户在 click 空间和 purchase 空间的表示距离很远，对比对齐确保用户在不同行为空间的一致性，有助于跨行为的知识迁移。
3. **全局图嵌入的作用是什么？** R-GCN 在用户-物品异构图上做消息传递，能够捕捉多跳邻居的协同信号（如"购买了 A 的用户也点击了 B"），为每个用户/物品提供全局上下文增强的嵌入。
4. **BiGEL 如何处理行为数据的严重不平衡？** 越高层行为越稀疏（purchase 远少于 click），级联门控让稀疏行为从丰富行为借用信息；对比对齐在不同行为空间之间迁移知识；GNN 的邻居聚合天然缓解数据稀疏。
5. **BiGEL 和 ESMM 的区别？** ESMM 通过 pCTCVR = pCTR * pCVR 的概率链式分解利用行为层级，但只支持两层行为且缺乏灵活性；BiGEL 通过图结构和门控机制支持任意多层行为，且通过对比学习实现更深层的表示对齐。
