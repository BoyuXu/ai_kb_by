# CTCVR Heterogeneous Hierarchical Decoder for WeChat Video Ads
> 来源：工业论文 (腾讯/微信) | 领域：广告系统 | 学习日期：20260327

## 问题定义
微信视频号广告的CTCVR（点击×转化）联合建模面临：
1. **路径稀疏**：转化率远低于点击率，CVR正样本极少
2. **异质特征**：CTR特征（内容相关）与CVR特征（购买意图相关）来源不同，维度不同
3. **解耦难度**：简单乘积分解 P(CTCVR) = P(CTR) × P(CVR|click) 忽略了两者的深层关联
**目标**：设计层次化解码器，建模CTR到CVR的转化路径，同时处理异质特征。

## 核心方法与创新点

### 1. 层次化解码架构
```
用户特征 + 广告特征
      ↓
  共享Encoder
      ↓
┌──────────────┐
│  CTR Decoder │  → P(CTR)
└──────┬───────┘
       ↓ (CTR hidden state作为条件)
┌──────────────┐
│  CVR Decoder │  → P(CVR|CTR)
└──────────────┘
      ↓
P(CTCVR) = P(CTR) × P(CVR|CTR)
```

CVR Decoder接收CTR Decoder的隐藏状态作为额外输入，显式建模CTR→CVR的条件依赖。

### 2. 异质特征融合
- **CTR特征**：内容特征（视频标签、时长）+ 用户内容偏好
- **CVR特征**：商业特征（价格、类目、用户购买历史）

使用Cross-Attention将异质特征动态融合：

$$
\text{CVR}}_{\text{{\text{Input}}} = \text{CrossAttn}(\text{CTR}}_{\text{{\text{hidden}}}, \text{Commercial}}_{\text{{\text{features}}})
$$

### 3. ESMM框架改进
在ESMM基础上引入层次化解码，避免直接用CTR预测作为CVR的hard特征，而是用soft的hidden state传递更丰富的信息。

## 实验结论
- **CTCVR AUC**：+1.5%（相比独立CTR×CVR模型）
- **GMV**：+3.2%（在线A/B测试）
- **CVR AUC**：+2.1%（层次化条件建模效果显著）
- 异质特征融合比简单拼接提升+0.8% AUC

## 工程落地要点
1. **正样本扩充**：CVR正样本稀少，可以用"加购物车"作为弱正样本辅助训练
2. **样本空间一致性**：CTR和CVR必须在同一样本空间（ESMM框架），否则选择偏差严重
3. **特征时间对齐**：转化可能在点击后数天发生，特征快照时间点需要统一
4. **分布偏移**：视频号广告的用户行为分布随时间变化快，需要频繁更新模型

## 面试考点
Q1: 为什么不直接将P(CTR)的预测值作为CVR的输入特征？
A: P(CTR)预测值是一个标量，信息损失严重。CTR Decoder的hidden state是高维向量，包含了更丰富的用户-广告匹配信息（哪些维度匹配、哪些不匹配）。用hidden state作为CVR Decoder的输入，相当于将CTR的"理解"完整传递给CVR建模，而不是只传递最终的分数。

Q2: 微信视频号广告与搜索广告的CVR建模有何不同？
A: 搜索广告：用户有明确购买意图（搜索"买手机"），CVR相对高，正样本较充足。视频号广告：用户被动浏览，购买意图不明确，CVR极低（<1%），正样本极度稀少。因此视频号广告需要更多的辅助任务（加购、收藏、分享等弱信号）和更强的正则化来避免过拟合。

Q3: 层次化Decoder与并行多任务头有什么区别？
A: 并行多任务头：CTR和CVR的decoder独立，只共享encoder，不存在显式的CTR→CVR信息流。层次化Decoder：CVR decoder显式依赖CTR decoder的输出，建模了CTR→CVR的因果路径。层次化方案在路径依赖性强的场景（广告转化链路）中效果更好。
