# IDProxy: Cold-Start CTR Prediction for Ads and Recommendation at Xiaohongshu

> 来源：arXiv 2025 | 领域：ads | 学习日期：20260404

## 问题定义

小红书广告冷启动场景（新广告/新商品）面临的核心挑战：
- 新广告 ID 在 Embedding Table 中随机初始化，CTR 预测误差大
- 冷启动广告曝光机会少（模型置信度低），形成恶性循环
- 内容平台特殊性：图文/视频内容信息丰富，但 ID 信号缺失

$$\text{CTR}_{\text{cold}} = f(\text{content}, \text{user}) \approx f(\text{proxy\_id}, \text{user})$$

## 核心方法与创新点

**IDProxy**：用代理 ID（Proxy ID）替代随机初始化的冷启动 ID：

1. **内容相似度匹配（Content Similarity Matching）**：
   - 用多模态 Encoder（文本 + 图像）提取内容特征
   - 在已有热门广告中检索最相似的 K 个广告作为代理
   
$$\text{ProxyIDs}(i_{\text{cold}}) = \text{TopK}_{j \in \mathcal{H}}[\text{sim}(e_i^{\text{content}}, e_j^{\text{content}})]$$

2. **代理 ID 聚合**：
   - 不直接用单个代理 ID，而是加权聚合多个代理的 Embedding
   
$$e_{\text{proxy}} = \sum_{k=1}^{K} \alpha_k \cdot e_{j_k}, \quad \alpha_k \propto \text{sim}(e_i, e_{j_k})$$

3. **渐进迁移（Progressive Transfer）**：
   - 初始：100% 代理 Embedding
   - 随曝光积累：混合比例逐渐向真实 ID Embedding 迁移
   
$$e_{\text{final}}(t) = (1 - \lambda(t)) \cdot e_{\text{proxy}} + \lambda(t) \cdot e_{\text{id}}$$

$$\lambda(t) = 1 - \exp(-\mu \cdot N_{\text{clicks}}(t))$$

4. **业务定制**：
   - 推荐场景：按内容语义相似
   - 广告场景：同时考虑内容相似 + 目标用户群相似

## 实验结论

- 新广告前 7 天 CTR AUC: **+3.1‰**
- 新广告 D+1 CTR: **+12.3%**（早期曝光质量提升最显著）
- 代理 K=5 vs K=1: 额外提升 +0.8‰
- 渐进迁移 vs 硬切换: 防止迁移抖动，性能更平滑

## 工程落地要点

- 内容 Embedding Index 需实时更新（新广告入库 5min 内完成检索）
- 代理查询放在广告入库流程中，离线完成（非在线计算）
- 曝光计数器需持久化（Redis），防止服务重启丢失
- 迁移阈值 μ 需按广告类型调参（高转化广告需更快迁移）

## 面试考点

1. **Q**: 冷启动广告 Embedding 为什么用代理 ID 而不是内容 Embedding？  
   **A**: 保持模型架构统一（仍用 ID Embedding 表），避免内容/ID 两套 Embedding 的融合复杂度；代理 ID 携带了热门广告的协同过滤信息（用户行为统计），纯内容 Embedding 没有。

2. **Q**: 渐进迁移中如何确定迁移速率？  
   **A**: 基于累计点击数（而非时间）：点击数反映模型对该广告 ID 的置信度。指数衰减函数控制迁移速率，μ 超参控制衰减速度。

3. **Q**: 这个方法的局限性？  
   **A**: 代理质量依赖内容检索准确性；如果新广告内容非常独特（无相似热门广告），代理 Embedding 可能引入噪声；多模态检索有额外计算成本。
