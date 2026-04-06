# Fidelity-Aware Recommendation Explanations via Stochastic Path Integration

**ArXiv:** 2511.18047 | **Date:** 2025-11 | **Authors:** Barkan et al.

## 核心问题
推荐系统解释的**保真度（fidelity）**严重未被探索——现有方法无法准确反映模型的真实推理过程。

## 方法：SPINRec（Stochastic Path Integration for Neural Recommender Explanations）

### 关键创新
**随机基线采样（Stochastic Baseline Sampling）**：
- 传统路径积分方法使用固定或不真实的基线
- SPINRec 从经验数据分布采样多个合理的用户画像
- 选择最忠实的归因路径，捕获已观测和未观测交互的影响
- 产生更稳定、个性化的解释

### 核心公式
基于 Integrated Gradients 扩展到推荐稀疏场景：
$$\text{Attr}(x_i) = (x_i - x'_i) \int_0^1 \frac{\partial F(x' + \alpha(x-x'))}{\partial x_i} d\alpha$$
其中 $x'$ 从经验分布随机采样。

## 评估
在 3 个模型（MF, VAE, NCF）× 3 个数据集（ML1M, Yahoo! 等）上进行了推荐领域最全面的保真度评估。

## 面试考点
- 推荐解释的保真度 vs 可理解性的权衡？
- 为什么固定基线不适合推荐场景？
- Integrated Gradients 在稀疏推荐数据上的挑战？

**Tags:** #rec-sys #explainability #fidelity #path-integration #attribution
