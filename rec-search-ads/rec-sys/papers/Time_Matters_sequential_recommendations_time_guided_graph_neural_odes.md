# Time Matters: Enhancing Sequential Recommendations with Time-Guided Graph Neural ODEs

> 来源：arXiv 2025 | 领域：rec-sys | 学习日期：20260404

## 问题定义

序列推荐中时间信息通常被简化为时间戳 embedding 或位置编码，忽略了用户兴趣的**连续时间演化**特性：

- 兴趣衰减：长时间不访问某类别 → 兴趣下降
- 兴趣激活：某事件（节假日、促销）触发潜在兴趣
- 个体差异：不同用户的兴趣演化速率不同

$$\frac{dh_u(t)}{dt} = f(h_u(t), G_u(t), t)$$

## 核心方法与创新点

**Time-Guided Graph Neural ODE**：

1. **用户兴趣图构建**：
   - 节点：用户历史物品（按时间排序）
   - 边：时间相邻 + 语义相似（双重连接）
   - 动态图：随新交互增量更新

2. **时间引导 ODE（TG-ODE）**：
   - 图神经网络定义 ODE 的向量场
   - ODE Solver 在连续时间上积分，得到任意时刻的用户状态
   
$$h_u(t) = h_u(t_0) + \int_{t_0}^{t} \text{GNN}(h_u(\tau), \mathcal{G}_u) \, d\tau$$

3. **时间感知兴趣衰减**：
   - ODE 中加入指数衰减项：近期交互权重高
   
$$f(h, G, t) = \text{GNN}(h, G) \cdot e^{-\lambda(t - t_{\text{last}})}$$

4. **预测时刻插值**：
   - 推理时将 ODE 积分到预测时刻 $t_{\text{pred}}$
   - 不同用户个性化衰减率 λ 由辅助网络预测

## 实验结论

- HR@10: +6.8% vs SASRec on MovieLens-1M
- 时间敏感场景（促销期、节假日）提升 **+15%**
- ODE Solver 延迟：Euler（快）vs RK4（精），推荐 Euler 用于在线

## 工程落地要点

- ODE 积分步数 = 1（Euler）即可满足精度和延迟要求
- 用户状态 $h_u(t)$ 预计算并缓存，仅在新交互时触发更新
- 衰减率 λ 范围 [0.01, 0.5]，per-user 学习
- 动态图增量更新：新交互只添加新边，不重构整图

## 面试考点

1. **Q**: 为什么用 ODE 建模用户兴趣？  
   **A**: 兴趣是连续变化的，ODE 能自然建模任意时刻的兴趣状态，而非离散时间步跳跃。

2. **Q**: Graph Neural ODE 如何处理不规则时间间隔？  
   **A**: ODE 在连续时间积分，天然支持不规则间隔。离散方法需要填充/对齐，GN-ODE 无此问题。

3. **Q**: ODE 在工业推荐中的主要挑战？  
   **A**: 计算成本（ODE Solver 需多步积分）、训练稳定性（梯度通过 ODE Solver 反传）。实践中用 adjoint method 控制内存。
