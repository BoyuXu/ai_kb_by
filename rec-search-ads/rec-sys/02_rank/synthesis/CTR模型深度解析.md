> 📚 参考文献
> - [Din-Deep-Interest-Network](../../rec-sys/papers/DINDeep_Interest_Network.md) — DIN：深度兴趣网络（Deep Interest Network）
> - [A-Unified-Language-Model-For-Large-Scale-Search...](../../rec-sys/papers/A_Unified_Language_Model_for_Large_Scale_Search_Recommend.md) — A Unified Language Model for Large Scale Search, Recommen...
> - [A Generative Re-Ranking Model For List-Level Multi](../../rec-sys/papers/A_Generative_Re_ranking_Model_for_List_level_Multi_object.md) — A Generative Re-ranking Model for List-level Multi-object...
> - [Cobra Bridging Sparse And Dense Retrieval In Gene](../../rec-sys/papers/COBRA_Bridging_Sparse_and_Dense_Retrieval_in_Generative_R.md) — COBRA: Bridging Sparse and Dense Retrieval in Generative ...
> - [Mmoe-Multi-Task-Learning](../../rec-sys/papers/MMoEMulti_gate_Mixture_of_Experts.md) — MMoE：多门控混合专家（Multi-gate Mixture-of-Experts）
> - [Linear-Item-Item-Session-Rec](../../rec-sys/papers/Linear_Item_Item_Model_with_Neural_Knowledge_for_Session.md) — Linear Item-Item Model with Neural Knowledge for Session-...
> - [Deploying-Semantic-Id-Based-Generative-Retrieva...](../../rec-sys/papers/Deploying_Semantic_ID_based_Generative_Retrieval_for_Larg.md) — Deploying Semantic ID-based Generative Retrieval for Larg...
> - [Gems-Breaking-The-Long-Sequence-Barrier-In-Gene...](../../rec-sys/papers/GEMs_Breaking_the_Long_Sequence_Barrier_in_Generative_Rec.md) — GEMs: Breaking the Long-Sequence Barrier in Generative Re...

## 5. MMOE (Multi-gate Mixture-of-Experts) - Google 2018

### 5.1 解决的问题

多任务学习(MTL)在推荐系统中广泛应用（如同时预测点击率、收藏率、转化率），但面临核心挑战：

1. **任务冲突(Conflict)**：不同任务的最优特征表示可能矛盾。例如：点击任务偏好标题党，转化任务偏好质量内容

2. **负迁移(Negative Transfer)**：硬共享参数导致某任务的梯度损害其他任务性能

3. **跷跷板现象**：一个任务提升导致另一个下降，无法同时最优

传统解决方案Shared-Bottom的局限：
- 所有任务共享底层网络
- 表达能力受限，难以处理任务间复杂关系

MMOE的核心洞察：**不同任务应该自适应地选择不同的Expert组合，而非强制共享所有参数**。

### 5.2 核心创新点

#### 架构总览

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                           MMOE 架构图                                          │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│                        Input Features                                          │
│                               │                                                │
│                               ▼                                                │
│                     ┌─────────────────┐                                        │
│                     │  Shared Bottom   │                                        │
│                     │  (Embedding +    │                                        │
│                     │   Dense Layers)  │                                        │
│                     └────────┬────────┘                                        │
│                              │                                                 │
│                              ▼                                                 │
│               ┌──────────────────────────────┐                                 │
│               │     Expert Networks (N个)     │                                 │
│               │  ┌─────┐ ┌─────┐        ┌─────┐                               │
│               │  │E_1  │ │E_2  │  ...   │E_n  │  ← 每个Expert是独立网络        │
│               │  │ f_1 │ │ f_2 │        │ f_n │    学习不同特征表示             │
│               │  └──┬──┘ └──┬──┘        └──┬──┘                               │
│               └─────┼──────┼──────────────┼─────┘                             │
│                     │      │              │                                    │
│                     │      │              │                                    │
│                     ▼      ▼              ▼                                    │
│   ┌──────────────────────────────────────────────────────────────┐            │
│   │                    Gating Networks (门控网络)                  │            │
│   │                                                               │            │
│   │  Task A: [g_A1, g_A2, ..., g_An] ──┐                          │            │
│   │  Task B: [g_B1, g_B2, ..., g_Bn] ──┤                          │            │
│   │  Task C: [g_C1, g_C2, ..., g_Cn] ──┤                          │            │
│   │           ...                     │                          │            │
│   │                                    ▼                          │            │
│   │                        Weighted Combination                    │            │
│   │                        (每个任务独立加权)                       │            │
│   └──────────────────────────────────────────────────────────────┘            │
│                          │         │         │                                 │
│                          ▼         ▼         ▼                                 │
│                    ┌─────────┐ ┌─────────┐ ┌─────────┐                        │
│                    │ Task A  │ │ Task B  │ │ Task C  │                        │
│                    │  Tower  │ │  Tower  │ │  Tower  │                        │
│                    │ (CTR)   │ │ (CVR)   │ │ (Fav)   │                        │
│                    └────┬────┘ └────┬────┘ └────┬────┘                        │
│                         │           │           │                              │
│                         ▼           ▼           ▼                              │
│                      p(click)   p(convert)  p(favorite)                       │
└───────────────────────────────────────────────────────────────────────────────┘
```

#### 核心机制：Gating Network

```python
class MMOE(nn.Module):
    def __init__(self, input_dim, expert_num, expert_dim, task_num):
        super().__init__()
        
        # N个Expert网络（每个是独立的全连接网络）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, expert_dim)
            ) for _ in range(expert_num)
        ])
        
        # 每个任务有一个独立的Gating Network
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_num),
                nn.Softmax(dim=-1)  # 输出Expert权重，和为1
            ) for _ in range(task_num)
        ])
        
        # 每个任务的Tower网络
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ) for _ in range(task_num)
        ])
    
    def forward(self, x):
        """
        x: 输入特征向量
        """
        # 1. 所有Expert的输出 [batch_size, expert_num, expert_dim]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        task_outputs = []
        for task_id in range(self.task_num):
            # 2. Gating Network输出该任务的Expert权重
            # gating_weights: [batch_size, expert_num]
            gating_weights = self.gates[task_id](x)
            
            # 3. 加权融合Expert输出
            # gating_weights.unsqueeze(-1): [batch_size, expert_num, 1]
            # expert_outputs: [batch_size, expert_num, expert_dim]
            fused = torch.sum(
                gating_weights.unsqueeze(-1) * expert_outputs, 
                dim=1  # 沿expert_num维度求和
            )  # 结果: [batch_size, expert_dim]
            
            # 4. 输入Task Tower得到最终预测
            task_pred = self.towers[task_id](fused)
            task_outputs.append(task_pred)
        
        return task_outputs  # [task_num, batch_size, 1]
```

#### 与Shared-Bottom的对比

```
Shared-Bottom:
┌─────────────────────────────────────┐
│        Shared Bottom Network        │
│    (所有任务强制共享同一表示)        │
└───────────────┬─────────────────────┘
│               │               │
▼               ▼               ▼
Task A        Task B        Task C

问题：不同任务对Shared Bottom的梯度方向可能相反，导致优化冲突

MMOE:
┌─────────┐ ┌─────────┐ ┌─────────┐
│ Expert 1│ │ Expert 2│ │ Expert 3│
└────┬────┘ └────┬────┘ └────┬────┘
     │           │           │
     └─────┬─────┴─────┬─────┘
           │           │
    ┌──────┴───┐   ┌───┴──────┐
    │ Gate A   │   │ Gate B   │   ← 每个任务自适应选择Expert
    │[0.6,0.3,0.1]│  │[0.1,0.5,0.4]│
    └─────┬────┘   └────┬─────┘
          │             │
          ▼             ▼
       Task A        Task B

优势：任务A和B可以主要使用不同的Expert，避免直接冲突
```

### 5.3 关键公式

**Expert输出：**

$$
f_i(x) = \text{Expert}_i(x), \quad i = 1, 2, ..., N
$$

**Gating Network（任务k）：**

$$
g_k(x) = \text{Softmax}(W_{g_k} \cdot x + b_{g_k})
$$

$$
g_k(x) = [g_{k1}, g_{k2}, ..., g_{kN}], \quad \sum_{i=1}^{N} g_{ki} = 1
$$

**融合表示（任务k）：**

$$
f_k(x) = \sum_{i=1}^{N} g_{ki}(x) \cdot f_i(x)
$$

**任务输出：**

$$
y_k = \text{Tower}_k(f_k(x))
$$

**多任务损失（不确定性加权/动态加权）：**

$$
L = \sum_{k=1}^{K} w_k \cdot L_k + \lambda \sum_{k=1}^{K} \log(\sigma_k)
$$

### 5.4 与前代模型对比

| 维度 | Shared-Bottom | Cross-Stitch | MMOE |
|------|---------------|--------------|------|
| 参数共享 | 硬共享底层 | 软共享（线性组合） | 加权组合Expert |
| 表达能力 | 弱 | 中 | 强 |
| 任务关系建模 | 无 | 简单线性 | 自适应非线性 |
| 计算复杂度 | O(d) | O(K²d) | O(Nd + KNd) |
| 训练难度 | 简单 | 中 | 中（需调Expert数） |
| 任务冲突处理 | 差 | 一般 | 好 |

**MMOE vs OMoE (One-gate MoE)**：
- OMoE：所有任务共享同一个Gating Network
- MMOE：每个任务独立的Gating Network
- 关键区别：MMOE让每个任务选择不同的Expert组合，更灵活

### 5.5 工程落地要点

1. **Expert数量选择**：
   - 一般4-8个Expert足够
   - 过多：参数量大，训练慢，可能过拟合
   - 过少：表达能力不足，接近Shared-Bottom

2. **Expert结构**：
   - 可以是简单全连接（2-3层）
   - 也可以复杂化（如Transformer、DIN等）

3. **多任务损失平衡**：
   - 不同任务量级差异大时（如CTR 5% vs CVR 0.5%），需要加权
   - 方法：Uncertainty Weighting、GradNorm、Dynamic Weight Average

4. **Gating Network温度调节**：
   - Softmax加入温度参数$\tau$：Softmax($g/\tau$)
   - $\tau \to 0$：硬选择（one-hot）
   - $\tau \to \infty$：均匀分布

### 5.6 面试必答要点

1. **MMOE相比Shared-Bottom的核心优势？**
   - 避免任务冲突：不同任务可以选择不同的Expert子集
   - 灵活的任务关系建模：相关任务可共享Expert，无关任务使用不同Expert
   - 缓解负迁移：Gating可以将梯度引导到适合的Expert

2. **MMOE中为什么每个任务需要独立的Gating Network？**
   - 如果共享Gating，所有任务强制使用相同的Expert组合，退化为OMoE
   - 独立Gating让每个任务根据输入自适应选择最相关的Expert
   - 例如：Task A主要用Expert 1&2，Task B主要用Expert 2&3

3. **MMOE如何处理任务之间的相关性差异？**
   - 高度相关任务：Gating权重会收敛到相似的Expert组合
   - 弱相关任务：各自选择不同的Expert组合
   - 极端情况：任务完全不相关时，可退化为独立单任务模型

## 📐 核心公式与原理

### 1. 矩阵分解

$$
\hat{r}_{ui} = p_u^T q_i
$$

- 用户和物品的隐向量内积

### 2. BPR 损失

$$
L_{BPR} = -\sum_{(u,i,j)} \ln \sigma(\hat{r}_{ui} - \hat{r}_{uj})
$$

- 正样本得分 > 负样本得分

### 3. 序列推荐

$$
P(i_{t+1} | i_1, ..., i_t) = \text{softmax}(h_t^T E)
$$

- 基于历史序列预测下一次交互

---

## 6. PLE (Progressive Layered Extraction) - 腾讯2020

### 6.1 解决的问题

MMOE虽然通过MoE结构缓解了多任务学习的冲突问题，但在**任务差异较大**时仍有局限：

1. **专家分配冲突**：不同任务仍共享所有Expert，只是权重不同。当任务A和B差异很大时，Expert被迫学习兼顾两者的表示，可能两者都做不好

2. **跷跷板现象依然存在**：在任务相关性低时，一个任务的提升仍可能导致另一个下降

3. **没有显式分离公共知识和任务私有知识**：所有Expert都是共享的，无法明确区分哪些是跨任务通用的，哪些是任务特有的

PLE的核心洞察：**显式分离共享组件(Shared Experts)和任务私有组件(Task-specific Experts)，逐层提取和融合**。

### 6.2 核心创新点

#### 架构总览

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              PLE 架构图                                              │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Layer 1 (Bottom)                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                   Expert Layer 1                                            │    │
│  │  ┌──────────┐  ┌──────────┐     ┌──────────┐  ┌──────────┐                  │    │
│  │  │ Expert   │  │ Expert   │ ... │ Expert   │  │ Expert   │                  │    │
│  │  │ Task A   │  │ Task A   │     │ Shared   │  │ Shared   │                  │    │
│  │  │ (私有)    │  │ (私有)    │     │ (共享)   │  │ (共享)   │                  │    │
│  │  └────┬─────┘  └────┬─────┘     └────┬─────┘  └────┬─────┘                  │    │
│  │       │             │                │             │                        │    │
│  │       └─────────────┴────────────────┴─────────────┘                        │    │
│  │                        │                                                    │    │
│  │                        ▼                                                    │    │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │    │
│  │  │              Selector / Gating Network                              │   │    │
│  │  │  Task A: 从 Task A Experts + Shared Experts 中选择                  │   │    │
│  │  │  Task B: 从 Task B Experts + Shared Experts 中选择                  │   │    │
│  │  └─────────────────────────┬───────────────────────────────────────────┘   │    │
│  └────────────────────────────┼───────────────────────────────────────────────┘    │
│                               │                                                      │
│                               ▼                                                      │
│  Layer 2 ───────────────────────────────────────────────────────────               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                   Expert Layer 2 (结构与Layer 1类似)                          │    │
│  │  ┌──────────┐  ┌──────────┐     ┌──────────┐  ┌──────────┐                  │    │
│  │  │ Expert   │  │ Expert   │ ... │ Expert   │  │ Expert   │                  │    │
│  │  │ Task A   │  │ Task A   │     │ Shared   │  │ Shared   │                  │    │
│  │  └────┬─────┘  └────┬─────┘     └────┬─────┘  └────┬─────┘                  │    │
│  │       │             │                │             │                        │    │
│  │       └─────────────┴────────────────┴─────────────┘                        │    │
│  │                        │                                                    │    │
│  │                        ▼                                                    │    │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │    │
│  │  │              Selector / Gating Network                              │   │    │
│  │  │  (输入来自Layer 1对应任务的输出)                                      │   │    │
│  │  └─────────────────────────┬───────────────────────────────────────────┘   │    │
│  └────────────────────────────┼───────────────────────────────────────────────┘    │
│                               │                                                      │
│              ... (更多Layers，逐层提取和融合) ...                                   │
│                               │                                                      │
│                               ▼                                                      │
│  Layer L (Top)                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                   Expert Layer L                                            │    │
│  │  ┌──────────┐  ┌──────────┐     ┌──────────┐  ┌──────────┐                  │    │
│  │  │ Expert   │  │ Expert   │ ... │ Expert   │  │ Expert   │                  │    │
│  │  │ Task A   │  │ Task A   │     │ Shared   │  │ Shared   │                  │    │
│  │  └────┬─────┘  └────┬─────┘     └────┬─────┘  └────┬─────┘                  │    │
│  └───────┼─────────────┴────────────────┴─────────────┼──────────────────────────┘    │
│          │                                            │                               │
│          ▼                                            ▼                               │
│   ┌──────────────┐                           ┌──────────────┐                        │
│   │   Task A     │                           │   Task B     │                        │
│   │    Tower     │                           │    Tower     │                        │
│   │              │                           │              │                        │
│   │  p(click)   │                           │ p(convert)   │                        │
│   └──────────────┘                           └──────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

#### Progressive Extraction 机制

```python
class PLELayer(nn.Module):
    def __init__(self, task_num, experts_per_task, shared_experts, input_dim, output_dim):
        """
        PLE的一层
        """
        super().__init__()
        self.task_num = task_num
        
        # 每个任务的私有Expert
        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                Expert(input_dim, output_dim) 
                for _ in range(experts_per_task)
            ]) for _ in range(task_num)
        ])
        
        # 共享Expert
        self.shared_experts = nn.ModuleList([
            Expert(input_dim, output_dim)
            for _ in range(shared_experts)
        ])
        
        # 每个任务的Gating Network
        self.gates = nn.ModuleList([
            GatingNetwork(
                input_dim=input_dim,
                expert_num=experts_per_task + shared_experts  # 私有+共享
            ) for _ in range(task_num)
        ])
    
    def forward(self, task_inputs):
        """
        task_inputs: list of [input_A, input_B, ...]，每个任务上一层的输出
        
        返回: list of [output_A, output_B, ...]
        """
        outputs = []
        
        for task_id in range(self.task_num):
            # 1. 获取该任务的私有Expert输出
            task_exp_outs = [exp(task_inputs[task_id]) 
                           for exp in self.task_experts[task_id]]
            
            # 2. 获取共享Expert输出（每个任务输入共享Expert）
            shared_exp_outs = [exp(task_inputs[task_id]) 
                              for exp in self.shared_experts]
            
            # 3. 合并所有可选的Expert输出
            all_experts = task_exp_outs + shared_exp_outs
            
            # 4. Gating选择
            gate_weights = self.gates[task_id](task_inputs[task_id])
            
            # 5. 加权融合
            output = sum(w * exp_out for w, exp_out in zip(gate_weights, all_experts))
            outputs.append(output)
        
        return outputs

class PLE(nn.Module):
    def __init__(self, layer_num, ...):
        # 堆叠多层PLE Layer
        self.ple_layers = nn.ModuleList([
            PLELayer(...) for _ in range(layer_num)
        ])
    
    def forward(self, x):
        # 初始输入复制给所有任务
        task_inputs = [x] * self.task_num
        
        # 逐层Progressive Extraction
        for layer in self.ple_layers:
            task_inputs = layer(task_inputs)
        
        # 最终输出
        return [tower(out) for tower, out in zip(self.towers, task_inputs)]
```

#### 关键设计：分离与融合

```
分离的好处：
- Task A私有Expert：只学习对Task A重要的特征
- Shared Expert：学习跨任务通用的知识
- Task B无法直接看到Task A的私有Expert，避免干扰

融合的好处：
- 通过Gating Network，任务可以选择性地从共享Expert获取知识
- 逐层设计允许知识在不同抽象层次上融合
- 相关任务可以通过共享Expert自然协作
```

### 6.3 关键公式

**任务k在第l层的Expert输出：**

$$
E_{k,l} = \{E_{k,l}^1, ..., E_{k,l}^{m_k}\} \cup \{E_{S,l}^1, ..., E_{S,l}^{m_s}\}
$$

其中：
- $E_{k,l}^i$：任务k在第l层的第i个私有Expert
- $E_{S,l}^j$：第l层的第j个共享Expert

**Gating Network输出：**

$$
g_{k,l}(x) = \text{Softmax}(W_{k,l} \cdot x)
$$

**融合表示：**

$$
x_{k,l+1} = \sum_{i=1}^{m_k} g_{k,l}^i \cdot E_{k,l}^i(x_{k,l}) + \sum_{j=1}^{m_s} g_{k,l}^{m_k+j} \cdot E_{S,l}^j(x_{k,l})
$$

### 6.4 与前代模型对比

| 维度 | Shared-Bottom | MMOE | PLE |
|------|---------------|------|-----|
| 知识分离 | 无 | 无（软分离） | 有（显式私有/共享） |
| 渐进提取 | 无 | 无 | 有（多层逐层提取） |
| 任务干扰 | 严重 | 中等 | 最小 |
| 参数量 | 最小 | 中 | 较大 |
| 训练难度 | 易 | 中 | 较难（需调层数/专家数） |
| 任务差异适应性 | 差 | 一般 | 好 |

**PLE vs MMOE关键区别**：
- MMOE：所有Expert对所有任务可见，只是权重不同
- PLE：引入Task-specific Expert，其他任务不可见，从根本上隔离干扰

### 6.5 工程落地要点

1. **层数设计**：
   - 2-3层通常足够
   - 过多：训练困难，梯度消失
   - 过少：提取能力不足

2. **Expert分配**：
   - 每个任务2-3个私有Expert
   - 共享Expert数 = 任务数 或 2×任务数
   - 需要根据任务相关性调整

3. **与MMOE的选择**：
   - 任务相关性强（如CTR+CVR）：MMOE可能足够
   - 任务差异大（如点击+收藏+评论）：PLE更适合

4. **训练技巧**：
   - 渐进训练：先训练底层，再逐步解锁上层
   - 多任务损失需要仔细平衡

### 6.6 面试必答要点

1. **PLE相比MMOE的核心改进是什么？**
   - 显式分离私有Expert和共享Expert
   - MMOE所有任务共享所有Expert，只是权重不同
   - PLE中Task A的私有Expert对Task B不可见，从根本上避免干扰
   - 逐层提取设计允许在不同抽象层次融合知识

2. **为什么PLE能有效缓解跷跷板现象？**
   - 每个任务有独立的表示学习路径（私有Expert）
   - 任务A的优化不会直接影响任务B的私有参数
   - 共享Expert只负责公共知识，不承载任务特有细节

3. **PLE的层数如何选择？每层有什么不同？**
   - 通常2-3层，从具体到抽象逐层提取
   - 底层：学习原始特征级别的表示
   - 上层：学习高阶抽象表示
   - 层间通过Gating进行选择性信息传递

---

## 7. SIM (Search-based Interest Model) - 阿里2020

### 7.1 解决的问题

用户行为序列建模面临的**长序列困境**：

1. **DIN/DIEN的局限**：当用户行为序列长度>1000时，计算量和内存消耗爆炸
   - DIN：每预测一个候选物品，需要计算与所有历史行为的注意力
   - DIEN：GRU必须串行计算，长序列推理时延不可接受

2. **简单截断的问题**：只保留最近N个行为会丢失长期兴趣信号
   - 淘宝用户可能有数万历史行为
   - 用户买手机的周期可能是几个月，简单截断会丢失关键信息

3. **长序列中的噪音**：长期序列包含大量与当前候选无关的行为

SIM的核心洞察：**不需要对所有历史行为做精细建模，而是先"搜索"出与候选物品相关的子序列，再精细化建模**。

### 7.2 核心创新点

#### 架构总览：两阶段范式

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           SIM 两阶段架构图                                           │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │                      Stage 1: General Search Unit (GSU)                        │  │
│  │                      阶段1：通用搜索单元（粗筛）                                 │  │
│  │                                                                                │  │
│  │   用户长期行为序列 (可能10万+)                                                  │  │
│  │   ┌────────┐ ┌────────┐ ┌────────┐      ┌────────┐ ┌────────┐                 │  │
│  │   │Item_1  │ │Item_2  │ │Item_3  │ .... │Item_k  │ │Item_n  │                 │  │
│  │   └───┬────┘ └───┬────┘ └───┬────┘      └───┬────┘ └───┬────┘                 │  │
│  │       │          │          │               │          │                      │  │
│  │       └──────────┴──────────┴───────┬───────┴──────────┘                      │  │
│  │                                     │                                          │  │
│  │                                     ▼                                          │  │
│  │   ┌─────────────────────────────────────────────────────────────────────┐     │  │
│  │   │                   Similarity Search                                │     │  │
│  │   │                                                                     │     │  │
│  │   │  候选物品: Adidas跑鞋                                               │     │  │
│  │   │                                                                     │     │  │
│  │   │  ┌─────────────────────────────────────────────────────────────┐   │     │  │
│  │   │  │  Hard Search (硬搜索): 按Category/Brand匹配                    │   │     │  │
│  │   │  │  Soft Search (软搜索): 向量相似度检索                           │   │     │  │
│  │   │  └─────────────────────────────────────────────────────────────┘   │     │  │
│  │   │                                                                     │     │  │
│  │   │  筛选结果: 相关子序列 (如50-100个物品)                               │     │  │
│  │   └─────────────────────────────────────────────────────────────────────┘     │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                                 │
│                                    ▼                                                 │
│                                    50-100个相关历史行为                               │
│                                    │                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │                      Stage 2: Exact Search Unit (ESU)                          │  │
│  │                      阶段2：精确搜索单元（精排）                                 │  │
│  │                                                                                │  │
│  │   相关子序列                                                                    │  │
│  │   ┌────────┐ ┌────────┐ ┌────────┐      ┌────────┐                            │  │
│  │   │Item_i1 │ │Item_i2 │ │Item_i3 │ .... │Item_im │  (m=50-100)                 │  │
│  │   └───┬────┘ └───┬────┘ └───┬────┘      └───┬────┘                            │  │
│  │       │          │          │               │                                  │  │
│  │       └──────────┴──────────┴───────┬───────┘                                  │  │
│  │                                     │                                          │  │
│  │                                     ▼                                          │  │
│  │   ┌─────────────────────────────────────────────────────────────────────┐     │  │
│  │   │                   Multi-Head Attention 或 DIN/Transformer            │     │  │
│  │   │                                                                     │     │  │
│  │   │  精细化建模相关子序列与候选物品的关系                                  │     │  │
│  │   │  计算量可控：O(m) 而非 O(n)，m << n                                  │     │  │
│  │   └─────────────────────────────────────────────────────────────────────┘     │  │
│  │                                     │                                          │  │
│  │                                     ▼                                          │  │
│  │                           User Interest Representation                          │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                                 │
│                                    ▼                                                 │
│                           DNN → CTR Prediction                                       │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

#### Hard Search vs Soft Search

- **Hard Search**：基于属性匹配（Category、Brand），速度快（O(n)），可解释性强，但无法捕捉语义相似性
- **Soft Search**：基于向量相似度，语义匹配更全面，但需要预训练向量，构建ANN索引

**Hard Search示例**：
```python
def hard_search(candidate_item, user_long_sequence, top_k=100):
    candidate_category = candidate_item.category
    candidate_brand = candidate_item.brand
    
    related_items = []
    for item in user_long_sequence:
        if item.category == candidate_category or item.brand == candidate_brand:
            related_items.append(item)
    
    return related_items[:top_k]  # 按时间倒序取top_k
```

**Soft Search示例**：
```python
def soft_search(candidate_item, user_long_sequence_embedding_index, top_k=100):
    query_vec = item_embedding_model(candidate_item)
    # 使用FAISS/ScaNN进行ANN检索
    distances, indices = faiss_index.search(query_vec, top_k)
    return indices
```

### 7.3 关键公式

**Hard Search**：筛选同Category或同Brand的历史行为

**Soft Search**：基于余弦相似度的Top-K检索

**计算复杂度**：
- GSU：O(N) 线性扫描或使用ANN索引
- ESU：O(m * d²)，其中 m = 50-100 << N

### 7.4 与前代模型对比

| 维度 | DIN | DIEN | SIM |
|------|-----|------|-----|
| 序列长度 | < 100 | < 200 | > 10000 |
| 计算复杂度 | O(Nd²) | O(NH²) | O(N) + O(md²) |
| 长期兴趣 | 丢失 | 部分丢失 | 保留 |
| 在线时延 | 高 | 很高 | 可控 |

### 7.5 工程落地要点

1. GSU索引构建：使用FAISS、ScaNN构建ANN索引
2. Hard vs Soft选择：Hard Search做快速初筛 + Soft Search做语义召回
3. 子序列长度m：经验值50-200之间
4. 冷启动：新用户退化为普通DIN，新物品Hard Search可用

### 7.6 面试必答要点

1. **SIM为什么能解决长序列问题？**
   - 核心思想：两阶段范式（Coarse-to-Fine）
   - GSU阶段快速筛选相关子序列，降低问题规模从N到m（m<<N）
   - ESU阶段在可控规模上精细化建模

2. **SIM中Hard Search和Soft Search优缺点？**
   - Hard Search：速度快、简单、可解释，但无法捕捉语义相似性
   - Soft Search：召回更全面，但需要维护ANN索引，Embedding质量影响大

3. **ESU为什么选Multi-Head Attention？**
   - 并行计算，速度快
   - 能建模序列内部关系（Self-Attention）和与候选物品关系（Cross-Attention）
   - 长距离依赖能力强于RNN

---

## 8. DSSM (Deep Structured Semantic Model) - 微软2013

### 8.1 解决的问题

传统NLP和搜索推荐系统面临的**语义鸿沟(Semantic Gap)**问题：

1. **词汇不匹配**：用户搜索"apple"，传统基于关键词匹配的方法无法区分是水果还是公司
2. **高维稀疏表示**：传统One-hot或TF-IDF表示维度高、语义信息弱
3. **缺乏语义泛化能力**：无法处理同义词、近义词

DSSM的核心洞察：**将Query和Document映射到同一个低维语义空间，通过向量相似度衡量语义相关性**。

### 8.2 核心创新点

#### 架构总览：双塔结构

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            DSSM 双塔架构图                                           │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│            Query Tower (Query塔)              Document Tower (Document塔)            │
│                                                                                      │
│   Query: "深度学习推荐系统"                                                        │
│        │                                                                           │
│        ▼                                                                           │
│   ┌──────────────────────┐                                                ┌──────────────────────┐
│   │   Word Hashing       │         共享语义空间                            │   Word Hashing       │
│   │  (降维：50k→30k)     │◄──────────────────────────────────────────────►│  (降维：50k→30k)     │
│   └──────────┬───────────┘                                                └──────────┬───────────┘
│              │                                                                       │
│              ▼                                                                       │
│   ┌──────────────────────┐                                                ┌──────────▼───────────┐
│   │   Non-linear         │                                                │   Non-linear         │
│   │   Projection Layer 1 │                                                │   Projection Layer 1 │
│   │   (300 dims)         │                                                │   (300 dims)         │
│   └──────────┬───────────┘                                                └──────────┬───────────┘
│              │                                                                       │
│              ▼                                                                       │
│   ┌──────────────────────┐                                                ┌──────────▼───────────┐
│   │   Non-linear         │                                                │   Non-linear         │
│   │   Projection Layer 2 │                                                │   Projection Layer 2 │
│   │   (300 dims)         │                                                │   (300 dims)         │
│   └──────────┬───────────┘                                                └──────────┬───────────┘
│              │                                                                       │
│              ▼                                                                       │
│   ┌──────────────────────┐                                                ┌──────────▼───────────┐
│   │   Semantic Feature   │                                                │   Semantic Feature   │
│   │   (128 dims)         │                                                │   (128 dims)         │
│   │   y_Q = f(Q; W_Q)   │                                                │   y_D = f(D; W_D)   │
│   └──────────┬───────────┘                                                └──────────┬───────────┘
│              │                                                                       │
│              └───────────────────────┬───────────────────────────────────────────────┘
│                                      │
│                                      ▼
│                           ┌──────────────────────┐
│                           │   Similarity Score   │
│                           │  R(Q,D) = cosine(y_Q, y_D)
│                           └──────────┬───────────┘
│                                      │
│                                      ▼
│                           ┌──────────────────────┐
│                           │  P(D|Q) = softmax(γ·R)
│                           └──────────────────────┘
```

#### Word Hashing：降维技术

将单词转换为letter n-gram向量：
- 单词"good"的3-gram：#go, goo, ood, od#
- 优点：降维（50k词汇表→30k n-gram）、处理OOV、捕捉拼写相似性

#### 双塔网络核心代码

```python
class DSSM(nn.Module):
    def __init__(self, input_dim, hidden_dims=[300, 300, 128]):
        self.query_tower = self._build_tower(input_dim, hidden_dims)
        self.doc_tower = self._build_tower(input_dim, hidden_dims)
        self.gamma = 10  # 温度参数
    
    def forward(self, query_batch, doc_batch):
        query_vec = self.query_tower(query_batch)  # [batch, 128]
        doc_vec = self.doc_tower(doc_batch)        # [batch, 128]
        cosine_sim = F.cosine_similarity(query_vec, doc_vec, dim=1)
        return cosine_sim
```

#### 推荐系统双塔召回

```
User Tower                    Item Tower
   │                              │
   ▼                              ▼
┌──────────┐                ┌──────────┐
│   DNN    │                │   DNN    │
│  Layers  │                │  Layers  │
└────┬─────┘                └────┬─────┘
     │                           │
     ▼                           ▼
┌──────────┐                ┌──────────┐
│  User    │                │  Item    │
│  Vector  │                │  Vector  │
└────┬─────┘                └────┬─────┘
     │                           │
     │     离线构建Item索引       │
     │     (FAISS/Annoy)        │
     │◄──────── ANN搜索 ─────────┤
     │                           │
     ▼                           
  Top-K相似物品
```

### 8.3 关键公式

**语义向量**：

$$
y = f(x; W) = \tanh(W_L \cdot \tanh(...\tanh(W_1 x)))
$$

**余弦相似度**：

$$
R(Q, D) = \frac{y_Q^T y_D}{\|y_Q\| \cdot \|y_D\|}
$$

**后验概率**：

$$
P(D|Q) = \frac{\exp(\gamma R(Q, D))}{\sum_{D'} \exp(\gamma R(Q, D'))}
$$

### 8.4 与前代模型对比

| 维度 | TF-IDF | LDA | DSSM |
|------|--------|-----|------|
| 表示方式 | 高维稀疏 | 低维稠密 | 低维稠密 |
| 语义捕捉 | 词袋 | 概率主题 | 深度网络 |
| 训练方式 | 无监督 | 无监督 | 有监督 |

### 8.5 工程落地要点

1. 负样本采样：每正样本配4-10个负样本，使用Hard Negative Mining
2. 向量归一化：L2归一化后用内积近似余弦相似度
3. 温度参数γ：控制分布尖锐程度，通常取5-20
4. 定期重训练：保持向量空间一致性

### 8.6 面试必答要点

1. **DSSM为什么使用Word Hashing？**
   - 2013年时大规模Embedding学习不成熟
   - Word Hashing是有效的降维手段（50k→30k）
   - 能处理OOV，捕捉拼写相似性
   - 现代实现通常用预训练Embedding

2. **双塔为什么可以独立计算？**
   - 数学上：内积对称性允许分解 score(u,i) = f(u)^T · g(i)
   - 工程上：Item向量可离线预计算建立索引
   - 在线只需计算User向量 + ANN检索，毫秒级响应

3. **双塔模型vs精排模型的优缺点？**
   - 优点：计算高效、离线预计算、在线延迟低、易并行
   - 缺点：表达能力受限（只能内积）、无法User-Item交叉特征、信息压缩损失
   - 典型用法：双塔召回 + DIN/DCN精排

---

## 总结对比表

| 模型 | 年份 | 核心创新 | 适用场景 | 主要局限 |
|------|------|----------|----------|----------|
| **DSSM** | 2013 | 双塔向量召回 | 召回阶段 | 无法复杂交叉 |
| **DeepFM** | 2017 | FM+DNN并行 | 精排 | 无序列建模 |
| **DIN** | 2018 | Attention机制 | 精排(有序列) | 长序列计算量大 |
| **MMOE** | 2018 | 多任务MoE | 多目标学习 | 任务差异大时冲突 |
| **DIEN** | 2019 | GRU+AUGRU | 精排(有序列) | 长序列效率低 |
| **DCN-V2** | 2020 | 显式高阶交叉 | 精排 | 纯特征交叉 |
| **PLE** | 2020 | 渐进式分离 | 多任务学习 | 参数量大 |
| **SIM** | 2020 | 两阶段长序列 | 精排(超长序列) | 依赖索引质量 |

> 📝 面试考点见：[rec_qa_extracted.md](../../interview/rec_qa_extracted.md)
