# IDProxy: Online Embedding via Identity Decomposition for Cold-Start Ads
> 来源：工业论文 | 领域：广告系统 | 学习日期：20260327

## 问题定义
新广告冷启动是广告系统的核心难题：
- 新广告没有历史ID embedding
- 随机初始化的embedding导致排序分数随机，展示机会少→数据少→学习慢的恶性循环
- 现有解决方案（内容特征直接映射）无法完全替代ID embedding的表达能力

**目标**：通过内容特征分解广告ID embedding，让新广告从上线第一天起就有有效的embedding，并随曝光积累逐渐过渡到真实ID embedding。

## 核心方法与创新点

### 1. ID Proxy Embedding
将ID embedding分解为内容部分和残差部分：

$$
e_{id} = e_{proxy} + \Delta e_{residual}
$$

- $e_{proxy}$：由内容特征（文本、图片）通过映射网络生成的代理embedding
- $\Delta e_{residual}$：个体差异项，通过历史数据学习

新广告只有 $e_{proxy}$，老广告两者都有。

### 2. Proxy Network训练
从老广告的(内容特征, ID embedding)对学习映射：

$$
e_{proxy} = f_\theta(\text{content}}_{\text{{\text{features}}})
$$

$$
\mathcal{L}_{proxy} = ||e_{proxy} - e_{id}^{target}||_2^2
$$

训练好后，新广告可以直接用内容特征生成 $e_{proxy}$。

### 3. 渐进式过渡机制
随着新广告积累历史数据，逐渐引入真实ID embedding：

$$
e_{final} = (1-w_t) \cdot e_{proxy} + w_t \cdot e_{id}^{learned}
$$

过渡权重 $w_t$ 随曝光次数增长：

$$
w_t = \min(1, \frac{\text{impression}}_{\text{{\text{count}}}}{N_{threshold}})
$$

$N_{threshold}$ 一般设为1000~5000次曝光。

### 4. 多模态内容特征
- **文本特征**：广告标题、描述（BERT编码）
- **图片特征**：广告主图（ViT/ResNet编码）
- **结构化特征**：类目、价格段、广告主历史表现

## 实验结论
- **CTR提升（冷启动期）**：+20%（相比随机初始化，冷启动前7天）
- **Bootstrap速度**：达到相同CTR效果所需曝光次数减少50%
- **长尾广告整体效果**：+8%（即使非完全新广告也受益）
- Proxy embedding与真实embedding的余弦相似度平均0.72（高度对齐）

## 工程落地要点
1. **Proxy Network更新频率**：建议每日更新，捕捉新的内容-行为关系
2. **特征对齐**：Proxy Network的输入特征必须与主模型的内容特征完全一致，防止分布偏移
3. **权重平滑**：过渡权重 $w_t$ 建议使用平滑曲线（sigmoid），避免跳变
4. **广告主视角**：新广告的冷启动质量直接影响广告主的留存意愿，需要在广告主面板展示冷启动进度
5. **离线评估**：用"将老广告伪装成新广告"的方式评估Proxy方案的效果（oracle知道真实embedding）

## 面试考点
Q1: IDProxy与Meta-Embedding（平均相似广告的embedding）的区别？
A: Meta-Embedding：找到与新广告内容相似的老广告，取其ID embedding的平均。简单有效，但：(1)依赖相似度计算（可能找不到足够好的相似广告）；(2)不能泛化到全新类目；(3)语义相似的广告CTR不一定相似（不同风格/出价的同类广告效果差异大）。IDProxy：直接学习内容→ID的映射函数，能捕获更细粒度的内容-行为关系，不依赖相似广告检索。

Q2: 渐进式过渡的意义是什么？为什么不直接从第一天就用真实ID embedding？
A: 新广告上线前几天数据极少（10-100次曝光），直接训练真实ID embedding会严重过拟合（只见过几次曝光，泛化性极差）。渐进式过渡让真实ID embedding在数据充足时才逐渐发挥作用，前期依靠Proxy提供稳定的先验，避免了"数据少→embedding差→曝光少→数据更少"的恶性循环。

Q3: 如何验证Proxy Embedding的质量？
A: (1)类内相似度：同类目广告的Proxy embedding应该聚集，不同类目应该分离；(2)下游任务验证：将Proxy替换真实ID embedding后，CTR模型的AUC变化（好的Proxy应该让AUC损失<1%）；(3)冷启动A/B测试：对比使用IDProxy vs 随机初始化的新广告CTR；(4)过渡速度：使用IDProxy的广告，其真实ID embedding收敛到稳定状态所需的曝光次数。
