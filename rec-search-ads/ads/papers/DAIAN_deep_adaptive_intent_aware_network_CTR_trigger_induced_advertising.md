# DAIAN: Deep Adaptive Intent-Aware Network for CTR Prediction in Trigger-Induced Advertising

> 来源：arXiv 2025 | 领域：ads | 学习日期：20260404

## 问题定义

**触发式广告（Trigger-Induced Advertising）**：用户搜索/浏览某商品后，在信息流中展示相关广告（类似搜索再营销）。

挑战：
- 触发词/触发物品与广告的语义对齐程度影响 CTR
- 用户当前意图（Intent）需从触发行为中动态推断
- 传统 CTR 模型不区分触发广告和普通广告，混合训练效果差

$$\text{CTR}(u, a | \text{trigger}) = P(\text{click} | u, a, \text{intent}(\text{trigger}))$$

## 核心方法与创新点

1. **触发意图编码器（Trigger Intent Encoder, TIE）**：
   - 输入：触发词 / 触发物品文本
   - 输出：用户当前意图向量 $h_{\text{intent}}$
   - 使用轻量 Bi-LSTM + Cross-Attention with user history

2. **自适应意图对齐（Adaptive Intent Alignment）**：
   - 动态计算触发意图与广告的对齐分数
   
$$s_{\text{align}} = \text{sim}(h_{\text{intent}}, e_{\text{ad}}) \cdot \text{MLP}([h_u, h_{\text{intent}}, e_{\text{ad}}])$$

3. **意图感知 Multi-Head Attention**：
   - 用意图向量作为额外 Key/Value，增强用户历史与当前意图的交互

$$\text{Attention}(Q_{user}, [K_{hist}; K_{intent}], [V_{hist}; V_{intent}])$$

4. **触发强度感知（Trigger Strength）**：
   - 估计触发事件与广告的语义匹配强度
   - 强匹配时放大意图信号，弱匹配时依赖用户历史偏好

## 实验结论

- 触发广告 CTR AUC: **+2.3‰** vs DIN
- 强匹配触发场景（触发词与广告完全相关）: **+5.1‰**
- 弱匹配场景（广泛匹配）: +1.2‰
- 触发广告转化率提升 **+8.4%**（线上 A/B）

## 工程落地要点

- 触发意图编码器延迟需 < 5ms（同步调用限制）
- 触发信号实时传入，不能离线缓存（意图是即时的）
- 强/弱匹配阈值需按广告类目调参（非标品 vs 标品不同）
- 与普通信息流广告分塔训练，避免样本不平衡

## 面试考点

1. **Q**: 触发式广告与普通展示广告的核心区别？  
   **A**: 触发广告有明确的意图信号（用户正在搜索/浏览相关内容），CTR 建模需要利用这个即时意图，而非仅靠历史兴趣。

2. **Q**: 如何度量触发意图与广告的对齐强度？  
   **A**: 语义相似度（Embedding 余弦相似）+ MLP 融合（考虑用户背景），两者组合比单一指标更准确。

3. **Q**: DAIAN 的对齐信号如何避免信息泄露（广告主优化触发词）？  
   **A**: 触发词匹配分数作为特征而非直接过滤，模型学习综合判断；广告主无法直接看到对齐分数（黑盒）。
