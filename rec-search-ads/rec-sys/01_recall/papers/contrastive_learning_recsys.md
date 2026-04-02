# 对比学习在推荐系统中的应用

> 来源：技术综述 | 日期：20260316 | 领域：rec-sys

## 问题定义

推荐系统面临的核心挑战：**数据稀疏性** 和 **噪声标签**。传统监督学习仅依赖点击等隐式反馈，信号稀疏（点击率通常 <1%），模型泛化能力差。对比学习（Contrastive Learning）通过自监督方式从无标签数据中学习表征，提升推荐质量。

## 核心方法与创新点

### 1. SimCLR 在推荐中的应用

对 user/item 交互序列做两种 **数据增强**，要求增强后的两个视图相互吸引（正样本），与其他用户/item 排斥（负样本）：

**常见增强方式**：
- **Item masking**：随机遮盖序列中 15% 的 item
- **Item cropping**：截取序列的连续子段
- **Item reordering**：打乱序列中部分 item 顺序
- **Feature dropout**：随机 dropout 某些 side-info 特征

### 2. SGL（Self-supervised Graph Learning）

在图神经网络推荐中，通过 **节点/边 dropout** 生成图的两个增强视图，对比学习辅助主任务：

```
L_total = L_BPR + λ × L_CL
L_CL = -log(sim(z_u, z_u') / Σ sim(z_u, z_k))  # InfoNCE loss
```

### 3. CLRec / CL4Rec

专为序列推荐设计，在 **item embedding 层** 添加对比损失，防止 item 表征坍塌（collapse）。

### 4. 难负例挖掘（Hard Negative Mining）

随机负样本太容易，模型学不到精细区分。难负例策略：
- **In-batch 难负例**：用同 batch 中得分最高的负样本
- **语义难负例**：语义相似但不相关的 item
- **Curriculum Hard Negative**：训练初期用易负例，后期逐渐引入难负例

## 实验结论

- SGL 在 Amazon 数据集上相比 LightGCN 提升 **Recall@20: +7.5%, NDCG@20: +9.2%**。
- CL4Rec 在 MovieLens 数据集上 HR@10 提升 **5-8%**，特别是在稀疏用户群体。
- 数据增强策略影响显著：item masking + cropping 组合效果最佳。

## 工程落地要点

- 对比学习引入额外训练开销（需构建正负样本对），通常训练时间增加 30-50%。
- Temperature 超参 τ 影响显著（常用 0.1-0.5），需调优。
- In-batch 负采样简单高效，大 batch size（512-2048）能提供足够的负样本。
- 线上部署：对比学习只在训练阶段使用，推理时无额外开销。

## 常见考点

- Q: InfoNCE Loss 的形式是什么？
  A: L = -log(exp(sim(z, z+)/τ) / (exp(sim(z, z+)/τ) + Σ_k exp(sim(z, z_k-)/τ)))。其中 τ 是温度系数，控制分布的锐度。τ 小则梯度集中于难负例；τ 大则梯度更均匀。

- Q: 对比学习为什么能缓解数据稀疏问题？
  A: 对比学习是自监督的，不依赖人工标签。通过数据增强构造大量正负样本对，使模型在稀疏监督信号之外获得丰富的自监督训练信号，尤其对长尾用户/item 有显著帮助。

- Q: 对比学习中的 Collapse 问题是什么？如何避免？
  A: Collapse 指模型将所有输入映射到相同表征，对比损失退化为 0。避免方法：(1) 引入负样本（对比损失天然防 collapse）；(2) Asymmetric architecture（BYOL）；(3) Stop gradient；(4) Normalization + projection head。
