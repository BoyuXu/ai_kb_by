# 推荐系统算法岗 - PyTorch编程实战题库

> 作者：MelonEggLearn  
> 定位：面试手撕代码必备，从Scratch实现核心推荐模型

---

## 目录

1. [FM（因子分解机）](#1-fm因子分解机)
2. [DeepFM](#2-deepfm)
3. [DIN（深度兴趣网络）](#3-din深度兴趣网络)
4. [双塔召回模型](#4-双塔召回模型)
5. [MMOE多任务学习](#5-mmoe多任务学习)
6. [Transformer序列推荐](#6-transformer序列推荐bert4rec风格)

---

## 1. FM（因子分解机）

### 1.1 核心代码

```python
"""
FM: Factorization Machines
- 一阶线性部分 + 二阶特征交叉
- 时间复杂度从 O(n^2) 优化到 O(nk)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FM(nn.Module):
    def __init__(self, num_features, embed_dim=8):
        """
        Args:
            num_features: 特征维度（one-hot后的总维度）
            embed_dim: 隐向量维度
        """
        super(FM, self).__init__()
        
        # 一阶权重: 每个特征对应一个权重
        self.first_order_w = nn.Embedding(num_features, 1)
        self.first_order_bias = nn.Parameter(torch.zeros(1))
        
        # 二阶隐向量: 每个特征对应一个embed_dim维向量
        self.second_order_v = nn.Embedding(num_features, embed_dim)
        
        # 初始化
        nn.init.xavier_uniform_(self.first_order_w.weight)
        nn.init.xavier_uniform_(self.second_order_v.weight)
        
    def forward(self, features, feature_values):
        """
        Args:
            features: [batch_size, num_fields] 特征索引
            feature_values: [batch_size, num_fields] 特征值（通常为1）
        Returns:
            output: [batch_size] 预测分数
        """
        # 一阶部分: sum(w_i * x_i)
        # [batch, num_fields, 1] -> [batch, num_fields]
        first_order = self.first_order_w(features).squeeze(-1)
        first_order = torch.sum(first_order * feature_values, dim=1, keepdim=True)
        
        # 二阶部分: 0.5 * sum_f((sum_i(v_if * x_i))^2 - sum_i(v_if^2 * x_i^2))
        # [batch, num_fields, embed_dim]
        embeddings = self.second_order_v(features)
        feature_values_expanded = feature_values.unsqueeze(-1)  # [batch, num_fields, 1]
        v = embeddings * feature_values_expanded  # [batch, num_fields, embed_dim]
        
        # 平方和: (sum_i(v_if * x_i))^2
        square_of_sum = torch.pow(torch.sum(v, dim=1), 2)  # [batch, embed_dim]
        
        # 和平方: sum_i(v_if^2 * x_i^2)
        sum_of_square = torch.sum(torch.pow(v, 2), dim=1)  # [batch, embed_dim]
        
        # 交叉项
        second_order = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        
        # 最终输出
        output = self.first_order_bias + first_order.squeeze(-1) + second_order.squeeze(-1)
        return torch.sigmoid(output)

# 使用示例
if __name__ == "__main__":
    batch_size = 4
    num_fields = 5  # 特征域数量（如user_id, item_id, category等）
    num_features = 1000  # 总特征维度
    embed_dim = 8
    
    model = FM(num_features, embed_dim)
    
    # 模拟输入：特征索引和特征值
    features = torch.randint(0, num_features, (batch_size, num_fields))
    feature_values = torch.ones(batch_size, num_fields)
    
    output = model(features, feature_values)
    print(f"FM output shape: {output.shape}")  # [4]
```

### 1.2 关键设计决策

| 决策点 | 说明 |
|--------|------|
| **Embedding选择** | 使用`nn.Embedding`而非线性层，支持稀疏特征查找 |
| **二阶优化公式** | 利用 $(a+b+c)^2 - (a^2+b^2+c^2)$ 将 $O(n^2)$ 降到 $O(nk)$ |
| **特征值处理** | 支持加权特征（如连续特征值），通过 `feature_values` 相乘实现 |
| **输出激活** | CTR预估用Sigmoid，评分预测可去掉 |

### 1.3 常见Bug与注意事项

```python
# ❌ Bug 1: 忘记处理特征值（连续特征）
embeddings = self.second_order_v(features)  # 错误！没乘feature_values

# ✅ 正确做法
embeddings = self.second_order_v(features) * feature_values.unsqueeze(-1)

# ❌ Bug 2: 二阶计算维度错误
square_of_sum = torch.pow(torch.sum(v, dim=-1), 2)  # 在embed_dim上求和？错！

# ✅ 正确做法: 在num_fields维度求和，保留embed_dim
square_of_sum = torch.pow(torch.sum(v, dim=1), 2)  # [batch, embed_dim]

# ❌ Bug 3: 初始化问题（梯度消失/爆炸）
self.second_order_v = nn.Embedding(num_features, embed_dim)  # 默认初始化可能不佳

# ✅ 正确做法: Xavier初始化
nn.init.xavier_uniform_(self.second_order_v.weight)

# ⚠️ 注意事项: 特征哈希冲突
# FM假设每个特征有独立的隐向量，如果特征维度太大导致哈希冲突，效果会下降
```

---

## 2. DeepFM

### 2.1 核心代码

```python
"""
DeepFM: Deep Factorization Machines
- FM: 自动学习二阶特征交叉
- DNN: 学习高阶非线性特征交叉
- 共享Embedding，Wide & Deep的改进版
"""

import torch
import torch.nn as nn

class DeepFM(nn.Module):
    def __init__(self, field_dims, embed_dim=8, mlp_dims=[256, 128, 64], dropout=0.2):
        """
        Args:
            field_dims: 每个特征域的维度列表，如 [100, 50, 20] 表示3个field
            embed_dim: Embedding维度
            mlp_dims: DNN隐藏层维度
            dropout: Dropout比率
        """
        super(DeepFM, self).__init__()
        
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim
        
        # Embedding层: FM和DNN共享
        self.embedding = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in field_dims
        ])
        
        # FM一阶权重
        self.fm_first = nn.ModuleList([
            nn.Embedding(dim, 1) for dim in field_dims
        ])
        self.fm_bias = nn.Parameter(torch.zeros(1))
        
        # DNN部分
        input_dim = self.num_fields * embed_dim
        layers = []
        for dim in mlp_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.BatchNorm1d(dim))
            input_dim = dim
        layers.append(nn.Linear(input_dim, 1))
        self.dnn = nn.Sequential(*layers)
        
        # 初始化
        for emb in self.embedding:
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.fm_first:
            nn.init.xavier_uniform_(emb.weight)
            
    def fm_layer(self, embeddings, first_order_vals):
        """FM层: 一阶 + 二阶交叉"""
        # 一阶部分
        first_order = torch.stack(first_order_vals, dim=1).squeeze(-1)  # [batch, num_fields]
        first_order = torch.sum(first_order, dim=1, keepdim=True)
        
        # 二阶部分: 内积和
        # embeddings: list of [batch, embed_dim]
        stacked = torch.stack(embeddings, dim=1)  # [batch, num_fields, embed_dim]
        
        square_of_sum = torch.pow(torch.sum(stacked, dim=1), 2)
        sum_of_square = torch.sum(torch.pow(stacked, 2), dim=1)
        second_order = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        
        return first_order + second_order + self.fm_bias
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_fields] 每个field一个特征索引
        Returns:
            output: [batch_size] 预测CTR
        """
        # Embedding查找: 共享给FM和DNN
        embeddings = [self.embedding[i](x[:, i]) for i in range(self.num_fields)]
        first_order_vals = [self.fm_first[i](x[:, i]) for i in range(self.num_fields)]
        
        # FM输出
        fm_out = self.fm_layer(embeddings, first_order_vals)  # [batch, 1]
        
        # DNN输入: 拼接所有embedding
        dnn_input = torch.cat(embeddings, dim=1)  # [batch, num_fields * embed_dim]
        dnn_out = self.dnn(dnn_input)  # [batch, 1]
        
        # 最终输出: FM + DNN
        output = torch.sigmoid(fm_out.squeeze(-1) + dnn_out.squeeze(-1))
        return output

# 使用示例
if __name__ == "__main__":
    # 假设有3个特征域: user_id(100), item_id(50), category(20)
    field_dims = [100, 50, 20]
    batch_size = 4
    
    model = DeepFM(field_dims, embed_dim=8, mlp_dims=[64, 32])
    
    # 输入: 每个field一个特征索引
    x = torch.stack([
        torch.randint(0, field_dims[0], (batch_size,)),
        torch.randint(0, field_dims[1], (batch_size,)),
        torch.randint(0, field_dims[2], (batch_size,))
    ], dim=1)
    
    output = model(x)
    print(f"DeepFM output shape: {output.shape}")  # [4]
```

### 2.2 关键设计决策

| 决策点 | 说明 |
|--------|------|
| **共享Embedding** | FM和DNN共享embedding层，减少参数量，互相正则化 |
| **DNN结构** | 建议使用BN+Dropout，防止过拟合 |
| **FM二阶实现** | 仍使用优化公式，而非显式两两计算 |
| **输出融合** | FM和DNN输出直接相加，再sigmoid |

### 2.3 常见Bug与注意事项

```python
# ❌ Bug 1: DNN输入维度错误
dnn_input = torch.stack(embeddings, dim=1)  # [batch, num_fields, embed_dim]
# 直接输入会报错，需要展平

# ✅ 正确做法
dnn_input = torch.cat(embeddings, dim=1)  # [batch, num_fields * embed_dim]
# 或
dnn_input = torch.stack(embeddings, dim=1).view(batch_size, -1)

# ❌ Bug 2: 忘记处理多值特征
# 如果item有多个tag，需要pooling后再输入

# ✅ 正确做法
def multi_hot_pooling(self, multi_indices, embedding_layer):
    """多值特征: 取平均embedding"""
    embs = embedding_layer(multi_indices)  # [batch, max_tags, embed_dim]
    mask = (multi_indices != 0).float().unsqueeze(-1)  # 假设0是padding
    summed = torch.sum(embs * mask, dim=1)  # [batch, embed_dim]
    avg = summed / torch.sum(mask, dim=1).clamp(min=1)
    return avg

# ❌ Bug 3: FM一阶和二阶维度不匹配
first_order = torch.sum(...)  # [batch] 没keepdim
second_order = torch.sum(..., dim=1, keepdim=True)  # [batch, 1]
result = first_order + second_order  # 广播错误！

# ✅ 正确做法: 统一维度
first_order = torch.sum(..., dim=1, keepdim=True)  # [batch, 1]

# ⚠️ 注意事项: 类别特征基数大时
# 对高基数特征（如user_id百万级），考虑使用Hash Trick或预训练Embedding
```

---

## 3. DIN（深度兴趣网络）

### 3.1 核心代码

```python
"""
DIN: Deep Interest Network
- Target Attention: 用户历史行为对候选物品的注意力权重
- 局部激活单元: 自适应学习用户兴趣表示
- 支持多样化兴趣表达
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ActivationUnit(nn.Module):
    """
    局部激活单元: 计算候选物品与用户历史行为的注意力权重
    输入: [候选物品embedding, 历史行为embedding, 差值, 点积]
    """
    def __init__(self, embed_dim, hidden_dims=[64, 16]):
        super(ActivationUnit, self).__init__()
        
        # 输入: embed_dim * 4 (候选, 历史, 差值, 点积扩展)
        layers = []
        input_dim = embed_dim * 4
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = dim
        layers.append(nn.Linear(input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, candidate, history):
        """
        Args:
            candidate: [batch, embed_dim] 候选物品
            history: [batch, seq_len, embed_dim] 用户历史行为
        Returns:
            weights: [batch, seq_len] 注意力权重
        """
        seq_len = history.size(1)
        
        # 扩展候选物品到序列长度
        candidate_expanded = candidate.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 元素差值
        diff = candidate_expanded - history
        
        # 元素点积（外积的迹）
        prod = candidate_expanded * history
        
        # 拼接特征: [候选, 历史, 差值, 点积]
        concat = torch.cat([candidate_expanded, history, diff, prod], dim=-1)
        
        # MLP输出注意力分数
        weights = self.mlp(concat).squeeze(-1)  # [batch, seq_len]
        
        return weights

class DIN(nn.Module):
    def __init__(self, feature_dims, embed_dim=8, seq_max_len=50, 
                 hidden_dims=[200, 80]):
        """
        Args:
            feature_dims: dict with 'user', 'item', 'category'等
            embed_dim: embedding维度
            seq_max_len: 用户历史行为最大长度
        """
        super(DIN, self).__init__()
        
        self.seq_max_len = seq_max_len
        self.embed_dim = embed_dim
        
        # Embedding层
        self.user_emb = nn.Embedding(feature_dims['user'], embed_dim)
        self.item_emb = nn.Embedding(feature_dims['item'], embed_dim)
        self.cat_emb = nn.Embedding(feature_dims['category'], embed_dim)
        
        # 激活单元
        self.attention_unit = ActivationUnit(embed_dim * 2)  # item + category
        
        # 其他特征（如上下文）的embedding
        self.context_emb = nn.ModuleDict({
            k: nn.Embedding(v, embed_dim) 
            for k, v in feature_dims.items() 
            if k not in ['user', 'item', 'category']
        })
        
        # 全连接层
        # 输入: 候选物品(2*emb) + 用户画像(emb) + 兴趣表示(2*emb) + 上下文
        fc_input_dim = embed_dim * 5  # 根据实际特征调整
        if 'context' in feature_dims:
            fc_input_dim += embed_dim
            
        layers = []
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(fc_input_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(0.2)
            ])
            fc_input_dim = dim
        layers.append(nn.Linear(fc_input_dim, 1))
        
        self.fc = nn.Sequential(*layers)
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
                
    def get_interest(self, candidate, history_items, history_cats, history_len):
        """
        使用Target Attention获取用户兴趣表示
        Args:
            candidate: [batch, item_emb + cat_emb]
            history_items: [batch, seq_len]
            history_cats: [batch, seq_len]
            history_len: [batch] 实际历史长度
        Returns:
            interest: [batch, embed_dim * 2] 用户兴趣表示
        """
        # 历史行为embedding
        h_item_emb = self.item_emb(history_items)  # [batch, seq_len, emb]
        h_cat_emb = self.cat_emb(history_cats)     # [batch, seq_len, emb]
        history_emb = torch.cat([h_item_emb, h_cat_emb], dim=-1)  # [batch, seq_len, emb*2]
        
        # 候选物品embedding（同样需要item+cat）
        # candidate shape: [batch, emb*2]
        
        # 计算注意力权重
        weights = self.attention_unit(candidate, history_emb)  # [batch, seq_len]
        
        # Mask处理: 对padding位置赋予极小值
        mask = torch.arange(self.seq_max_len, device=history_items.device)
        mask = mask.unsqueeze(0) < history_len.unsqueeze(1)  # [batch, seq_len]
        weights = weights.masked_fill(~mask, -1e9)
        
        # Softmax归一化
        weights = F.softmax(weights, dim=1)  # [batch, seq_len]
        
        # 加权求和
        interest = torch.bmm(weights.unsqueeze(1), history_emb).squeeze(1)
        
        return interest
        
    def forward(self, user_id, target_item, target_cat, 
                history_items, history_cats, history_len, **context_feats):
        """
        Args:
            user_id: [batch]
            target_item: [batch]
            target_cat: [batch]
            history_items: [batch, seq_len]
            history_cats: [batch, seq_len]
            history_len: [batch]
        """
        # 用户画像
        user_emb = self.user_emb(user_id)
        
        # 候选物品
        target_item_emb = self.item_emb(target_item)
        target_cat_emb = self.cat_emb(target_cat)
        target_emb = torch.cat([target_item_emb, target_cat_emb], dim=1)
        
        # 用户兴趣（核心：Target Attention）
        interest = self.get_interest(target_emb, history_items, 
                                      history_cats, history_len)
        
        # 上下文特征
        context_embs = []
        for key, emb_layer in self.context_emb.items():
            if key in context_feats:
                context_embs.append(emb_layer(context_feats[key]))
        
        # 拼接所有特征
        concat = [user_emb, target_emb, interest] + context_embs
        fc_input = torch.cat(concat, dim=1)
        
        # 全连接
        logit = self.fc(fc_input).squeeze(-1)
        return torch.sigmoid(logit)

# 使用示例
if __name__ == "__main__":
    batch_size = 4
    seq_len = 10
    
    feature_dims = {
        'user': 1000,
        'item': 5000,
        'category': 100,
        'hour': 24
    }
    
    model = DIN(feature_dims, embed_dim=8, seq_max_len=seq_len)
    
    # 模拟输入
    user_id = torch.randint(0, 1000, (batch_size,))
    target_item = torch.randint(0, 5000, (batch_size,))
    target_cat = torch.randint(0, 100, (batch_size,))
    history_items = torch.randint(0, 5000, (batch_size, seq_len))
    history_cats = torch.randint(0, 100, (batch_size, seq_len))
    history_len = torch.randint(1, seq_len + 1, (batch_size,))
    hour = torch.randint(0, 24, (batch_size,))
    
    output = model(user_id, target_item, target_cat, 
                   history_items, history_cats, history_len, hour=hour)
    print(f"DIN output shape: {output.shape}")  # [4]
```

### 3.2 关键设计决策

| 决策点 | 说明 |
|--------|------|
| **激活单元输入** | 拼接`[候选, 历史, 差值, 点积]`，充分表达两者关系 |
| **Mask处理** | 对padding位置mask，避免影响softmax |
| **Softmax vs Sum** | DIN论文使用加权sum，无需softmax，保留用户兴趣强度 |
| **特征拼接** | 候选物品的category也要参与attention计算 |

### 3.3 常见Bug与注意事项

```python
# ❌ Bug 1: Mask在Softmax后应用（错误！）
weights = F.softmax(weights, dim=1)
weights = weights * mask.float()  # 概率和不等于1，错误！

# ✅ 正确做法: Mask在Softmax前应用
weights = weights.masked_fill(~mask, -1e9)
weights = F.softmax(weights, dim=1)

# ❌ Bug 2: 注意力权重没有归一化导致数值不稳定
interest = torch.bmm(weights.unsqueeze(1), history_emb)  # 权重可能极大

# ✅ 正确做法: 使用Softmax或使用自适应池化
weights = F.softmax(weights, dim=1)  # 或使用 DIN的原始做法: sum pooling

# ❌ Bug 3: 历史行为embedding未考虑padding
history_emb = self.item_emb(history_items)  # padding位置也会出embedding
interest = torch.bmm(weights.unsqueeze(1), history_emb)  # 错误包含padding

# ✅ 正确做法: 确保mask生效
weights = weights.masked_fill(history_items == 0, -1e9)  # 假设0是padding

# ❌ Bug 4: 候选物品和历史物品使用不同空间
# 错误：候选用item_emb，历史用另一个emb层

# ✅ 正确做法: 共享embedding层，确保同一空间

# ⚠️ 注意事项: 序列长度差异大
# 不同用户历史长度差异大，建议对长序列进行截断/采样，短序列padding

# ⚠️ 注意事项: 实时性要求
# DIN需要实时获取用户历史行为，线上服务需设计高效的特征存储
```

---

## 4. 双塔召回模型

### 4.1 核心代码

```python
"""
双塔召回模型 (DSSM-based Two-Tower)
- User Tower: 编码用户画像和历史行为
- Item Tower: 编码物品特征
- 内积/余弦相似度作为分数
- 支持负采样训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class TwoTowerRecall(nn.Module):
    def __init__(self, user_features, item_features, embed_dim=64, 
                 hidden_dims=[256, 128]):
        """
        Args:
            user_features: dict {feature_name: vocab_size}
            item_features: dict {feature_name: vocab_size}
        """
        super(TwoTowerRecall, self).__init__()
        
        self.embed_dim = embed_dim
        
        # User塔Embedding层
        self.user_embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab, embed_dim)
            for name, vocab in user_features.items()
        })
        
        # Item塔Embedding层
        self.item_embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab, embed_dim)
            for name, vocab in item_features.items()
        })
        
        # User塔MLP
        user_input_dim = len(user_features) * embed_dim
        self.user_mlp = self._build_mlp(user_input_dim, hidden_dims, embed_dim)
        
        # Item塔MLP
        item_input_dim = len(item_features) * embed_dim
        self.item_mlp = self._build_mlp(item_input_dim, hidden_dims, embed_dim)
        
        # 温度参数（可选）
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
                
    def _build_mlp(self, input_dim, hidden_dims, output_dim):
        layers = []
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(dim)
            ])
            input_dim = dim
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)
        
    def user_tower(self, user_feats):
        """
        Args:
            user_feats: dict {feature_name: [batch] indices}
        Returns:
            user_vec: [batch, embed_dim] L2归一化后的向量
        """
        embs = [self.user_embeddings[name](indices) 
                for name, indices in user_feats.items()]
        concat = torch.cat(embs, dim=1)
        user_vec = self.user_mlp(concat)
        # L2归一化，使内积等价于余弦相似度
        user_vec = F.normalize(user_vec, p=2, dim=1)
        return user_vec
        
    def item_tower(self, item_feats):
        """
        Args:
            item_feats: dict {feature_name: indices}
        Returns:
            item_vec: [batch, embed_dim] L2归一化后的向量
        """
        embs = [self.item_embeddings[name](indices) 
                for name, indices in item_feats.items()]
        concat = torch.cat(embs, dim=1)
        item_vec = self.item_mlp(concat)
        item_vec = F.normalize(item_vec, p=2, dim=1)
        return item_vec
        
    def forward(self, user_feats, pos_item_feats, neg_item_feats=None):
        """
        Args:
            user_feats: dict
            pos_item_feats: dict 正样本
            neg_item_feats: dict or None 负样本（训练时）
        Returns:
            loss or similarity scores
        """
        user_vec = self.user_tower(user_feats)
        pos_item_vec = self.item_tower(pos_item_feats)
        
        # 正样本相似度
        pos_sim = torch.sum(user_vec * pos_item_vec, dim=1) / self.temperature
        
        if self.training and neg_item_feats is not None:
            # 负样本相似度
            neg_item_vec = self.item_tower(neg_item_feats)
            neg_sim = torch.sum(user_vec * neg_item_vec, dim=1) / self.temperature
            
            # 采样softmax loss (InfoNCE)
            logits = torch.stack([pos_sim, neg_sim], dim=1)  # [batch, 2]
            labels = torch.zeros(user_vec.size(0), dtype=torch.long, 
                                device=user_vec.device)
            loss = F.cross_entropy(logits, labels)
            return loss
        else:
            # 推理时返回相似度
            return pos_sim

class NegativeSampler:
    """
    负采样策略
    """
    def __init__(self, num_items, method='uniform', popularity=None):
        """
        Args:
            num_items: 物品总数
            method: 'uniform'|'popularity'|'hard'
            popularity: 物品流行度分布（用于popularity采样）
        """
        self.num_items = num_items
        self.method = method
        self.popularity = popularity
        
    def sample(self, batch_size, exclude_pos=None):
        """
        采样负样本
        Args:
            exclude_pos: [batch] 每个样本的正样本ID，用于过滤
        """
        if self.method == 'uniform':
            # 均匀采样
            neg_samples = torch.randint(0, self.num_items, (batch_size,))
        elif self.method == 'popularity':
            # 按流行度采样（更容易采样到热门物品）
            weights = self.popularity / self.popularity.sum()
            neg_samples = torch.multinomial(weights, batch_size, replacement=True)
        elif self.method == 'hard':
            # 困难负采样（需配合ANN索引，这里简化）
            neg_samples = torch.randint(0, self.num_items, (batch_size,))
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        # 确保不采样到正样本
        if exclude_pos is not None:
            mask = (neg_samples == exclude_pos)
            while mask.any():
                neg_samples[mask] = torch.randint(0, self.num_items, (mask.sum(),))
                mask = (neg_samples == exclude_pos)
                
        return neg_samples

# 使用示例
if __name__ == "__main__":
    batch_size = 32
    num_items = 10000
    
    # 特征定义
    user_features = {'user_id': 10000, 'age': 10, 'gender': 2}
    item_features = {'item_id': num_items, 'category': 100, 'brand': 1000}
    
    model = TwoTowerRecall(user_features, item_features, embed_dim=64)
    
    # 模拟数据
    user_feats = {
        'user_id': torch.randint(0, 10000, (batch_size,)),
        'age': torch.randint(0, 10, (batch_size,)),
        'gender': torch.randint(0, 2, (batch_size,))
    }
    pos_item_feats = {
        'item_id': torch.randint(0, num_items, (batch_size,)),
        'category': torch.randint(0, 100, (batch_size,)),
        'brand': torch.randint(0, 1000, (batch_size,))
    }
    
    # 训练模式
    model.train()
    sampler = NegativeSampler(num_items, method='uniform')
    neg_ids = sampler.sample(batch_size, pos_item_feats['item_id'])
    neg_item_feats = {
        'item_id': neg_ids,
        'category': torch.randint(0, 100, (batch_size,)),
        'brand': torch.randint(0, 1000, (batch_size,))
    }
    
    loss = model(user_feats, pos_item_feats, neg_item_feats)
    print(f"Training loss: {loss.item():.4f}")
    
    # 推理模式: 批量生成Item向量，建立ANN索引
    model.eval()
    with torch.no_grad():
        # 生成所有物品向量
        all_item_ids = torch.arange(num_items)
        item_vecs = []
        for i in range(0, num_items, 1000):
            batch_ids = all_item_ids[i:i+1000]
            batch_feats = {
                'item_id': batch_ids,
                'category': torch.randint(0, 100, (len(batch_ids),)),
                'brand': torch.randint(0, 1000, (len(batch_ids),))
            }
            vecs = model.item_tower(batch_feats)
            item_vecs.append(vecs)
        item_vecs = torch.cat(item_vecs, dim=0)  # [num_items, embed_dim]
        
        # 生成User向量
        user_vec = model.user_tower(user_feats)
        
        # 计算相似度，取top-k
        scores = torch.matmul(user_vec, item_vecs.t())  # [batch, num_items]
        top_k = torch.topk(scores, k=100, dim=1)
        print(f"Top-k indices: {top_k.indices.shape}")  # [batch, 100]
```

### 4.2 关键设计决策

| 决策点 | 说明 |
|--------|------|
| **L2归一化** | 输出向量L2归一化，使内积=余弦相似度，便于ANN检索 |
| **温度参数** | 可学习的temperature，调节softmax的尖锐程度 |
| **负采样策略** | 均匀采样简单但效果一般；流行度采样更符合分布；困难负采样提升效果 |
| **塔结构对称性** | User/Item塔可不对称（不同特征复杂度），但输出维度必须相同 |

### 4.3 常见Bug与注意事项

```python
# ❌ Bug 1: 推理时没有L2归一化
user_vec = self.user_mlp(user_emb)  # 忘记归一化
scores = torch.matmul(user_vec, item_vecs.t())  # 数值范围不一致

# ✅ 正确做法: 训练和推理都需归一化
user_vec = F.normalize(user_vec, p=2, dim=1)

# ❌ Bug 2: 负样本采样到正样本
neg_samples = torch.randint(0, num_items, (batch_size,))  # 可能包含正样本

# ✅ 正确做法: 过滤正样本
while (neg_samples == pos_items).any():
    mask = (neg_samples == pos_items)
    neg_samples[mask] = torch.randint(0, num_items, (mask.sum(),))

# ❌ Bug 3: Batch内负采样问题（In-batch Negative）
# 错误：只使用当前batch内的其他样本作为负样本，忽视了热门物品偏差

# ✅ 正确做法: 温度校正或采样校正
scores = scores / temperature  # 降低热门物品的分数

# ❌ Bug 4: Item向量未离线计算缓存
# 错误：每次推理都重新计算所有item向量

# ✅ 正确做法: 定期离线生成item向量，构建Faiss/Milvus索引

# ⚠️ 注意事项: 特征对齐
# User和Item塔如果对同一特征（如category）编码，需共享embedding或对齐分布

# ⚠️ 注意事项: 冷启动物品
# 新物品没有ID embedding，需依赖side feature（category/brand）塔
```

---

## 5. MMOE多任务学习

### 5.1 核心代码

```python
"""
MMOE: Multi-gate Mixture-of-Experts
- 多个Expert网络学习不同模式
- 每个任务有独立的Gate进行动态加权
- 相比Shared-Bottom，缓解任务冲突
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """单个Expert网络"""
    def __init__(self, input_dim, hidden_dim):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.net(x)

class MMOE(nn.Module):
    def __init__(self, num_features, num_experts=4, num_tasks=2, 
                 expert_dim=128, tower_dims=[64, 32]):
        """
        Args:
            num_features: 输入特征维度
            num_experts: Expert数量
            num_tasks: 任务数量（如CTR, CVR）
            expert_dim: Expert输出维度
            tower_dims: 每个任务的Tower结构
        """
        super(MMOE, self).__init__()
        
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        
        # Experts: 所有任务共享
        self.experts = nn.ModuleList([
            Expert(num_features, expert_dim) for _ in range(num_experts)
        ])
        
        # Gates: 每个任务一个Gate，输出Expert加权权重
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_features, num_experts),
                nn.Softmax(dim=1)
            ) for _ in range(num_tasks)
        ])
        
        # Task Towers: 每个任务独立的Tower
        self.towers = nn.ModuleList([
            self._build_tower(expert_dim, tower_dims) 
            for _ in range(num_tasks)
        ])
        
    def _build_tower(self, input_dim, hidden_dims):
        layers = []
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = dim
        layers.append(nn.Linear(input_dim, 1))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: [batch, num_features] 输入特征
        Returns:
            outputs: list of [batch] 每个任务的预测值
        """
        # 所有Expert的输出
        expert_outputs = [expert(x) for expert in self.experts]  # list of [batch, expert_dim]
        expert_output_stack = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, expert_dim]
        
        outputs = []
        for i in range(self.num_tasks):
            # Gate生成权重: [batch, num_experts]
            gate_weights = self.gates[i](x)  # [batch, num_experts]
            
            # 加权融合Expert输出
            # [batch, 1, num_experts] @ [batch, num_experts, expert_dim]
            fused = torch.bmm(gate_weights.unsqueeze(1), expert_output_stack)
            fused = fused.squeeze(1)  # [batch, expert_dim]
            
            # 任务特定的Tower
            output = self.towers[i](fused).squeeze(-1)  # [batch]
            outputs.append(torch.sigmoid(output))
            
        return outputs  # list of [batch] with length num_tasks

class MMOEWithAuxLoss(MMOE):
    """
    MMOE + 辅助损失（可选，用于正则化Expert学习）
    """
    def __init__(self, *args, aux_loss_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_loss_weight = aux_loss_weight
        
    def compute_aux_loss(self, x):
        """
        计算辅助损失：鼓励Expert多样性和负载均衡
        """
        # 收集所有Gate的权重
        gate_weights = torch.stack([gate(x) for gate in self.gates], dim=1)  
        # [batch, num_tasks, num_experts]
        
        # 计算每个Expert的平均权重
        mean_weights = gate_weights.mean(dim=[0, 1])  # [num_experts]
        
        # 负载均衡损失: 希望各Expert权重接近均匀分布
        target = torch.ones_like(mean_weights) / self.num_experts
        balance_loss = F.mse_loss(mean_weights, target)
        
        return balance_loss

# 使用示例: CTR和CVR联合训练
if __name__ == "__main__":
    batch_size = 32
    num_features = 100
    
    model = MMOE(
        num_features=num_features,
        num_experts=4,
        num_tasks=2,  # CTR和CVR
        expert_dim=128,
        tower_dims=[64, 32]
    )
    
    # 模拟输入
    x = torch.randn(batch_size, num_features)
    
    # 标签
    ctr_label = torch.randint(0, 2, (batch_size,)).float()
    cvr_label = torch.randint(0, 2, (batch_size,)).float()
    
    # 前向传播
    ctr_pred, cvr_pred = model(x)
    
    # 多任务损失
    ctr_loss = F.binary_cross_entropy(ctr_pred, ctr_label)
    cvr_loss = F.binary_cross_entropy(cvr_pred, cvr_label)
    
    # 动态加权或固定加权
    total_loss = ctr_loss + cvr_loss
    
    print(f"CTR Loss: {ctr_loss.item():.4f}, CVR Loss: {cvr_loss.item():.4f}")
    
    # 查看Gate权重分布
    with torch.no_grad():
        gate_weights = [gate(x).mean(dim=0) for gate in model.gates]
        print(f"\nGate weights for CTR: {gate_weights[0].numpy().round(3)}")
        print(f"Gate weights for CVR: {gate_weights[1].numpy().round(3)}")
```

### 5.2 关键设计决策

| 决策点 | 说明 |
|--------|------|
| **Expert数量** | 通常4-8个，过多增加参数量，过少表达能力受限 |
| **Gate设计** | Softmax保证权重和为1，每个任务动态选择Expert组合 |
| **共享Experts** | 所有任务共享同一组Experts，通过Gate差异化组合 |
| **Tower独立** | 每个任务独立的Tower网络，学习任务特定表示 |

### 5.3 常见Bug与注意事项

```python
# ❌ Bug 1: Gate权重维度错误
weights = self.gates[i](x)  # [batch, num_experts]
fused = weights @ expert_outputs  # 矩阵乘法错误，维度不匹配

# ✅ 正确做法: 使用bmm进行batch-wise加权
gate_weights = weights.unsqueeze(1)  # [batch, 1, num_experts]
expert_stack = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, dim]
fused = torch.bmm(gate_weights, expert_stack).squeeze(1)

# ❌ Bug 2: 忘记Softmax导致权重不归一化
self.gates = nn.Linear(num_features, num_experts)  # 缺少Softmax

# ✅ 正确做法
gate = nn.Sequential(nn.Linear(num_features, num_experts), nn.Softmax(dim=1))

# ❌ Bug 3: 所有任务梯度冲突未处理
# 简单相加损失可能导致某个任务主导训练

# ✅ 正确做法: 使用不确定性加权或GradNorm
def uncertainty_weighted_loss(losses):
    """自动学习任务权重（Kendall et al.）"""
    precisions = [torch.exp(-log_var) for log_var in log_vars]
    weighted_losses = [prec * loss + log_var 
                       for prec, loss, log_var in zip(precisions, losses, log_vars)]
    return sum(weighted_losses)

# ❌ Bug 4: Expert崩溃（所有Gate选择相同Expert）
# 解决方案1：添加负载均衡辅助损失
# 解决方案2：每个Expert只连接部分任务

# ⚠️ 注意事项: 任务相关性
# MMOE适合任务有一定相关性的场景，任务完全不相关时不如独立模型

# ⚠️ 注意事项: Gate初始化
# 建议用较小的值初始化Gate输出，使初始状态接近均匀分布
```

---

## 6. Transformer序列推荐（BERT4Rec风格）

### 6.1 核心代码

```python
"""
BERT4Rec风格的Transformer序列推荐
- 双向Transformer编码器
- 遮罩语言建模（MLM）训练
- 适用于下一物品预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]

class TransformerRec(nn.Module):
    def __init__(self, num_items, hidden_size=256, num_layers=2, 
                 num_heads=4, dropout=0.2, max_seq_len=100):
        """
        Args:
            num_items: 物品总数（含mask、padding等特殊token）
            hidden_size: 隐藏层维度
            num_layers: Transformer层数
            num_heads: 注意力头数
        """
        super(TransformerRec, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        
        # 特殊token
        self.mask_token = num_items - 1
        self.pad_token = num_items - 2
        
        # 物品Embedding
        self.item_embedding = nn.Embedding(num_items, hidden_size, 
                                           padding_idx=self.pad_token)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_size, max_seq_len)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True  # PyTorch 1.9+支持
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_size, num_items)
        
        # LayerNorm和Dropout
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, item_seq, mask_positions=None):
        """
        Args:
            item_seq: [batch, seq_len] 物品ID序列
            mask_positions: [batch, num_masks] 需要预测的位置（训练时）
        Returns:
            logits: [batch, seq_len, num_items] 或 [batch, num_masks, num_items]
        """
        batch_size, seq_len = item_seq.shape
        
        # Embedding + 位置编码
        item_emb = self.item_embedding(item_seq)  # [batch, seq_len, hidden]
        item_emb = self.pos_encoding(item_emb)
        item_emb = self.layer_norm(item_emb)
        item_emb = self.dropout(item_emb)
        
        # 生成attention mask（防止关注padding位置）
        padding_mask = (item_seq == self.pad_token)  # [batch, seq_len]
        
        # Transformer编码（双向）
        hidden = self.transformer(item_emb, src_key_padding_mask=padding_mask)
        # hidden: [batch, seq_len, hidden]
        
        # 输出预测
        if self.training and mask_positions is not None:
            # 只返回mask位置的预测
            # mask_positions: [batch, num_masks]
            hidden_masked = torch.stack([
                hidden[i, mask_positions[i]] 
                for i in range(batch_size)
            ], dim=0)  # [batch, num_masks, hidden]
            logits = self.output_layer(hidden_masked)  # [batch, num_masks, num_items]
        else:
            # 推理时返回所有位置
            logits = self.output_layer(hidden)  # [batch, seq_len, num_items]
            
        return logits
        
    def predict_next(self, item_seq):
        """
        预测序列下一个物品（自回归风格）
        Args:
            item_seq: [batch, seq_len]
        Returns:
            scores: [batch, num_items] 下一个物品的分数
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(item_seq)  # [batch, seq_len, num_items]
            next_logits = logits[:, -1, :]  # 取最后一个位置
            scores = torch.softmax(next_logits, dim=-1)
        return scores

def create_masked_inputs(item_seq, mask_prob=0.2, num_items=None, 
                         pad_token=None, mask_token=None):
    """
    BERT风格的Masked Item Prediction数据增强
    Args:
        item_seq: [batch, seq_len]
        mask_prob: 遮罩概率
    Returns:
        masked_seq: [batch, seq_len]
        mask_positions: [batch, num_masks]
        labels: [batch, num_masks] 被遮罩位置的原始物品
    """
    batch_size, seq_len = item_seq.shape
    device = item_seq.device
    
    masked_seq = item_seq.clone()
    mask_positions_list = []
    labels_list = []
    
    for i in range(batch_size):
        # 找出有效位置（非padding）
        valid_mask = (item_seq[i] != pad_token)
        valid_positions = torch.where(valid_mask)[0]
        
        num_valid = len(valid_positions)
        num_masks = max(1, int(num_valid * mask_prob))
        
        # 随机选择mask位置
        if len(valid_positions) > 0:
            mask_idx = torch.randperm(len(valid_positions))[:num_masks]
            positions = valid_positions[mask_idx]
            
            for pos in positions:
                prob = random.random()
                if prob < 0.8:
                    masked_seq[i, pos] = mask_token  # 80% mask
                elif prob < 0.9:
                    masked_seq[i, pos] = torch.randint(0, num_items - 2, (1,)).item()
                # 10% 保持不变
                
            mask_positions_list.append(positions)
            labels_list.append(item_seq[i, positions])
    
    # Padding到相同长度
    max_masks = max(len(m) for m in mask_positions_list) if mask_positions_list else 1
    
    mask_positions = torch.full((batch_size, max_masks), -1, 
                                dtype=torch.long, device=device)
    labels = torch.full((batch_size, max_masks), pad_token, 
                       dtype=torch.long, device=device)
    
    for i in range(batch_size):
        if i < len(mask_positions_list):
            m_len = len(mask_positions_list[i])
            mask_positions[i, :m_len] = mask_positions_list[i]
            labels[i, :m_len] = labels_list[i]
            
    return masked_seq, mask_positions, labels

# 使用示例
if __name__ == "__main__":
    import random
    
    batch_size = 8
    seq_len = 20
    num_items = 10000
    
    model = TransformerRec(
        num_items=num_items + 2,  # +2 for mask and pad tokens
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=seq_len
    )
    
    # 模拟数据
    item_seq = torch.randint(0, num_items, (batch_size, seq_len))
    
    # 训练模式：Masked Prediction
    model.train()
    masked_seq, mask_positions, labels = create_masked_inputs(
        item_seq, mask_prob=0.2, num_items=num_items,
        pad_token=model.pad_token, mask_token=model.mask_token
    )
    
    # 前向传播
    logits = model(masked_seq, mask_positions)  # [batch, num_masks, num_items]
    
    # 计算损失（只计算有效mask位置）
    mask = (labels != model.pad_token)
    loss = F.cross_entropy(
        logits.view(-1, model.num_items),
        labels.view(-1),
        reduction='none'
    )
    loss = (loss * mask.view(-1).float()).sum() / mask.sum()
    
    print(f"Training loss: {loss.item():.4f}")
    
    # 推理模式：预测下一个物品
    model.eval()
    next_scores = model.predict_next(item_seq)
    top_k = torch.topk(next_scores, k=10, dim=1)
    print(f"\nTop-10 next item recommendations shape: {top_k.indices.shape}")
```

### 6.2 关键设计决策

| 决策点 | 说明 |
|--------|------|
| **双向编码** | 与GPT的自回归不同，BERT4Rec使用双向Transformer，利用全序列信息 |
| **MLM训练** | Masked Language Modeling，随机遮罩部分物品进行预测 |
| **位置编码** | 使用可学习或正弦位置编码，捕捉序列顺序信息 |
| **Padding处理** | `src_key_padding_mask`确保不关注padding位置 |

### 6.3 常见Bug与注意事项

```python
# ❌ Bug 1: 忘记设置batch_first=True（PyTorch版本问题）
encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4)
# 输入需要是 [seq_len, batch, hidden]

# ✅ 正确做法: PyTorch 1.9+ 支持batch_first
encoder_layer = nn.TransformerEncoderLayer(..., batch_first=True)
# 输入 [batch, seq_len, hidden]

# ❌ Bug 2: Attention mask混淆
# src_mask: 防止关注未来位置（因果mask，用于解码器）
# src_key_padding_mask: 防止关注padding位置

# ✅ 正确做法: 双向编码器只用padding mask
padding_mask = (item_seq == pad_token)  # [batch, seq_len]
hidden = self.transformer(item_emb, src_key_padding_mask=padding_mask)

# ❌ Bug 3: 推理时泄露未来信息
logits = model(item_seq)  # 没有mask，模型能看到答案本身

# ✅ 正确做法: 训练时只预测mask位置，推理时从左到右
# 或改用自回归版本（SASRec）用于顺序推荐

# ❌ Bug 4: 位置编码在Embedding之前加
item_emb = self.pos_encoding(item_seq)  # 错误！item_seq是索引

# ✅ 正确做法
item_emb = self.item_embedding(item_seq)  # 先转embedding
item_emb = self.pos_encoding(item_emb)    # 再加位置编码

# ⚠️ 注意事项: 序列长度限制
# Transformer复杂度O(n^2)，长序列需截断或使用Linear Attention变体

# ⚠️ 注意事项: 物品ID基数大
# 输出层num_items过大时，考虑使用采样softmax或分层softmax

# ⚠️ 注意事项: 与自回归模型对比
# BERT4Rec适合有明确时间间隔的序列（如会话推荐）
# SASRec适合严格的顺序推荐（用户历史行为）
```

---

## 附录：面试速查表

### 模型对比

| 模型 | 核心创新 | 适用场景 | 复杂度 |
|------|----------|----------|--------|
| **FM** | 二阶特征交叉降维 | 稀疏特征CTR预估 | 低 |
| **DeepFM** | FM+DNN共享Embedding | 兼顾低阶高阶特征 | 中 |
| **DIN** | Target Attention | 用户兴趣多样变化 | 中 |
| **双塔** | User/Item分离编码 | 大规模召回 | 低（推理） |
| **MMOE** | 多任务门控机制 | 相关多任务联合学习 | 中 |
| **BERT4Rec** | 双向Transformer | 序列推荐 | 高 |

### 面试常考代码片段

```python
# 1. 二阶交叉优化公式（FM核心）
second_order = 0.5 * (square_of_sum - sum_of_square)

# 2. Attention Mask处理
scores.masked_fill_(mask == 0, -1e9)
weights = F.softmax(scores, dim=-1)

# 3. L2归一化（双塔）
F.normalize(vectors, p=2, dim=1)

# 4. 负采样
torch.multinomial(weights, num_samples, replacement=True)

# 5. 序列Padding
nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)

# 6. 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

*生成时间: 2026-03-12*  
*作者: MelonEggLearn*
