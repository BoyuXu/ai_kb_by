# No One Left Behind: How to Exploit Incomplete and Skewed Multi-Label Data for Conversion Rate Prediction

> 来源：arXiv 2025 | 领域：ads | 学习日期：20260404

## 问题定义

CVR（转化率）预估中，转化事件通常是多标签的（加购、收藏、支付、复购），但面临两大问题：

1. **不完整标签（Incomplete Labels）**：用户可能转化了但系统未记录（归因缺失、跨设备行为）
2. **倾斜标签（Skewed Labels）**：不同转化事件频率差异极大（展示 >> 点击 >> 加购 >> 支付）

$$P(y_{pay}=1) \approx 0.01\% \quad P(y_{add\_cart}=1) \approx 0.5\% \quad \text{极度不平衡}$$

## 核心方法与创新点

1. **互补标签学习（Complementary Label Learning）**：
   - 低频标签（支付）缺失时，用高频标签（加购）作为弱监督信号
   - 标签层次关系：$y_{pay} \leq y_{add\_cart} \leq y_{click}$（单调约束）

2. **标签依存网络（Label Dependency Network）**：
   - ESMM 范式扩展：多标签联合建模
   
$$P(pay) = P(pay | add\_cart) \cdot P(add\_cart | click) \cdot P(click)$$

3. **不确定性感知损失（Uncertainty-Aware Loss）**：
   - 对可能缺失的标签，用软目标（soft label）而非硬 0/1
   
$$\hat{y}_{\text{soft}} = \begin{cases} y & \text{if labeled} \\ P_{\text{impute}}(y=1) & \text{if missing} \end{cases}$$

4. **偏斜校正（Skew Correction）**：
   - 正样本重采样 + Focal Loss 组合
   - 不同标签分配不同 $\gamma$（难易程度调节）

$$\mathcal{L}_{\text{focal}} = -\alpha_t (1-p_t)^{\gamma} \log p_t$$

## 实验结论

- 支付 CVR AUC: **+2.8‰** vs 独立 CVR 模型
- 缺失标签处理提升长尾品类 CVR **+7.3%**
- 多标签联合 vs 独立训练：额外 AUC **+1.1‰**

## 工程落地要点

- 标签层次约束：训练时加入单调损失防止 P(pay) > P(add_cart)
- 缺失标签插补：用同品类历史转化率作为 soft label 初始值
- Focal Loss γ 按标签频率分配：支付 γ=2.5，加购 γ=1.5，点击 γ=0.5

## 面试考点

1. **Q**: 多任务 CVR 建模为什么优于独立模型？  
   **A**: 利用任务间关联（加购→支付的条件概率），共享表示减少过拟合，高频任务（加购）的信号辅助低频任务（支付）训练。

2. **Q**: ESMM 如何解决 CVR 的样本选择偏差（SSB）？  
   **A**: 在全展示空间建模 $P(CTR \times CVR)$，而非仅在点击空间训练 CVR，消除点击 → 曝光的样本偏差。

3. **Q**: 缺失标签如何处理？  
   **A**: 软标签插补（用先验概率替代硬 0）+ 不确定性损失权重（降低缺失样本对梯度的贡献）。
