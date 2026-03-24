# 广告系统技术演进脉络

> 整理时间：2026-03-16 | MelonEggLearn
> 参考来源：wzhe06/Ad-papers、阿里妈妈/字节跳动/Google公开论文、内部技术分享

---

## 演进时间线总览

```
2000        2004        2008        2011        2015        2019        2022        2025
 │           │           │           │           │           │           │           │
 ●           ●           ●           ●           ●           ●           ●           ●
 │           │           │           │           │           │           │           │
搜索广告    关键词竞价  Quality     RTB诞生    DSP/SSP     Wide&Deep   Auto        LLM+
诞生        AdWords    Score引入   OpenRTB    生态成熟    DNN大爆发   Bidding     隐私计算
Yahoo!      GSP拍卖    eCPM排序    Cookie     PMP/PDB     多任务学习   RL出价      无Cookie
                       点击质量    追踪       程序化      ESMM        预算优化    联邦学习
```

```
时代划分（5大阶段）:

┌──────────────────────────────────────────────────────────────────────────────────┐
│  阶段1         阶段2           阶段3            阶段4          阶段5               │
│  关键词广告     RTB时代         深度学习CTR      Auto Bidding   LLM+隐私           │
│  2000-2010     2011-2015       2016-2019        2019-2022      2023-至今          │
│                                                                                  │
│  搜索广告       程序化购买       端到端预估        智能出价        生成式广告          │
│  CPC竞价       DSP/SSP/ADX     Wide&Deep        oCPX→RL        隐私计算           │
│  Quality Score Cookie追踪      ESMM多任务       ROAS优化        联邦学习           │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 阶段1：关键词广告时代（2000-2010）

### 背景与驱动力

互联网泡沫破灭后，门户网站和搜索引擎急需找到可持续的商业模式。搜索引擎天然具备"用户主动表达意图"的特点——用户输入关键词时，其购买意向极为明确，广告与需求的匹配精准度远超传统展示广告。

**驱动力：**
- 1998年 GoTo.com（后更名为 Overture）首创**按点击付费（Pay Per Click, PPC）**
- 2000年 Google 推出 AdWords，彻底改变广告产业格局
- 2003年 Yahoo! 收购 Overture，搜索广告战争全面开打
- 传统印刷/电视广告无法量化效果，数字广告可精确计算 ROI

**市场规模：**
- 2000年：互联网广告约 80 亿美元（主要是 Banner 展示广告）
- 2010年：全球搜索广告突破 260 亿美元，Google 占据 65%+ 市场份额

---

### 核心技术突破

#### 1.1 广义第二价格拍卖（GSP）

早期 GoTo.com 使用**第一价格拍卖（First Price Auction）**：出价最高者赢得广告位，支付自己的出价。这导致广告主频繁调价，系统不稳定。

Google 借鉴 Vickrey 拍卖理论，引入**广义第二价格拍卖（Generalized Second Price Auction, GSP）**：

```
排名规则：   rank_i = bid_i × CTR_i        （实际上是 eCPM 排序）
实际支付：   price_i = bid_{i+1} × CTR_{i+1} / CTR_i + ε
```

**GSP 的核心性质：**
- 赢家支付**恰好击败下一名所需的最低价格**，而非自己的出价
- 激励相容性（incentive compatible）：广告主倾向于真实出价
- 媒体收益相对稳定，拍卖结果可预期

**与 VCG（Vickrey-Clarke-Groves）的比较：**

| 特性 | GSP | VCG |
|------|-----|-----|
| 激励相容性 | 近似（纳什均衡） | 严格（占优策略） |
| 实现复杂度 | 低 | 高 |
| 媒体收益 | 略低于 VCG | 理论最优 |
| 工业应用 | Google/百度/Yahoo | 部分场景 |

---

#### 1.2 Quality Score：从出价到质量的跨越

2005年，Google 引入 **Quality Score（质量得分）** 机制，这是搜索广告最重要的技术创新之一。

**背景问题：** 纯出价竞争导致"有钱就能上位"，用户体验极差——高出价低质量广告占据头部，用户点击率低，Google 收益反而下降。

**Quality Score 计算：**

```
Quality Score = f(预测CTR, 广告相关性, 落地页体验)

eCPM = bid × Quality Score
     = bid × CTR_predicted × Relevance × LandingPage_Quality
```

**三大核心维度：**

```
┌─────────────────────────────────────────────────────────┐
│                    Quality Score                         │
├───────────────────┬─────────────────┬───────────────────┤
│   预测CTR         │   广告相关性     │   落地页体验        │
│   (权重最高~50%)  │   (~30%)        │   (~20%)           │
├───────────────────┼─────────────────┼───────────────────┤
│ • 历史CTR         │ • 关键词匹配     │ • 页面加载速度      │
│ • 账户历史        │ • 广告文案相关性 │ • 内容相关性        │
│ • 关键词匹配类型  │ • 广告组质量    │ • 跳出率            │
└───────────────────┴─────────────────┴───────────────────┘
```

**Quality Score 的革命性意义：**
- 广告主必须同时关注**出价**和**质量**，不能只堆金钱
- Google 平衡了**用户体验**、**广告主ROI**和**媒体收益**三方利益
- 开创了"相关性优先"的广告生态范式

---

#### 1.3 关键词匹配与 Broad Match

```
匹配类型层级：

精确匹配 [keyword]  ← 最精准，流量最少
短语匹配 "keyword"  ← 中等精准
广泛匹配  keyword   ← 最宽泛，流量最多
否定匹配 -keyword   ← 排除不相关流量
```

**TF-IDF 相关性计算（早期）：**

```
相关性(query, ad) = Σ TF(t, ad) × IDF(t, corpus)

其中：
TF(t, ad) = 词t在广告中出现的频率
IDF(t)    = log(N / df(t))，N为总广告数，df(t)为包含词t的广告数
```

---

#### 1.4 CTR 预估：从统计到 LR

早期 CTR 预估使用逻辑回归（Logistic Regression）：

```
p(click) = σ(w^T × x + b) = 1 / (1 + exp(-(w^T × x + b)))

特征向量 x 包含：
- 查询词 one-hot
- 广告ID one-hot  
- 位置特征
- 时间特征
- 历史CTR统计

损失函数：L = -Σ [y_i log(p_i) + (1-y_i) log(1-p_i)]
```

**限制：** LR 无法自动捕捉特征交叉，需要大量人工特征工程。

---

### 代表公司/产品

| 公司 | 产品 | 关键贡献 |
|------|------|----------|
| Google | AdWords (2000) | GSP拍卖、Quality Score、AdSense |
| Yahoo! | Panama (2007) | 广告排名改革，引入质量因素 |
| Microsoft | adCenter (2006) | 后演变为 Bing Ads |
| 百度 | 凤巢系统 (2007) | 中国版 Quality Score，质量度体系 |
| Overture | GoTo.com (1998) | PPC模式鼻祖 |

---

### 解决的问题 + 遗留的问题

**✅ 解决的问题：**
- 广告效果可量化：首次实现精准 ROI 计算
- 意图匹配：搜索词直接反映用户意图，CTR 大幅提升
- 拍卖机制：GSP 使竞价生态相对稳定
- 质量门槛：Quality Score 淘汰低质广告，提升用户体验

**❌ 遗留的问题：**
- **展示广告缺乏精准定向**：只能按页面内容（内容相关），无法按用户属性
- **跨网站追踪缺失**：无法追踪用户在多个网站的行为轨迹
- **程序化购买效率低**：广告位买卖仍以合同+人工谈判为主
- **实时性差**：无法根据用户实时行为动态调整投放
- **中小媒体变现困难**：无法接入大平台的精准广告

---

### 面试必考点

1. **GSP vs VCG 区别是什么？为什么工业界用 GSP？**
   - GSP：赢家支付击败下一名所需最低价；实现简单，但非严格激励相容
   - VCG：支付社会福利损失；严格激励相容，但计算复杂，收益不如 GSP
   - 工业界选 GSP：实现简单、媒体收益更高、历史惯性

2. **Quality Score 的三个维度及权重如何影响 eCPM？**
   - eCPM = bid × QS，QS 越高出价越低也能排前
   - 核心：CTR 预测占最大权重

---

## 阶段2：RTB 时代（2011-2015）

### 背景与驱动力

搜索广告解决了"用户主动意图"的精准匹配，但展示广告（Display Ads）仍停留在"内容相关"层面——你在汽车网站看到汽车广告，但你可能刚买了车，完全不需要。

**驱动力：**
- **第三方 Cookie 技术成熟**：跨网站用户行为追踪成为可能
- **流量爆炸**：Web 2.0 催生海量媒体流量，人工谈判根本买不完
- **计算能力提升**：云计算使 100ms 内完成实时竞价成为现实
- **广告主需求升级**：从"买版位"转向"买人群"
- **2009年 OpenRTB 协议诞生**：IAB 推动标准化，行业互通

**关键事件时间轴：**
```
2007  DoubleClick 被 Google 以31亿美元收购
2009  OpenRTB 协议 v1.0 发布
2010  Google DoubleClick Ad Exchange 正式上线
2011  RTB 市场规模突破 10 亿美元
2013  程序化广告占展示广告比重超过 20%
2014  Header Bidding 技术出现，颠覆瀑布流
2015  程序化广告占比突破 50%
```

---

### 核心技术突破

#### 2.1 RTB 系统架构与竞价流程

**完整 RTB 交互时序（100ms 内完成）：**

```
用户请求页面
      │
      ▼ (1) 发起广告请求
   Publisher
   (媒体/SSP)
      │
      ▼ (2) 发送 bid request（用户信息+广告位信息，JSON格式）
   Ad Exchange (ADX)
      │
      ├──────────────────────────────────┐
      ▼ (3) 并行发送竞价请求            │
   DSP-1  DSP-2  DSP-3 ... DSP-N       │
      │                                 │
      ▼ (4) 各DSP在 ~50ms 内返回 bid   │
   bid_1  bid_2  bid_3 ... bid_N        │
      │                                 │
      ▼ (5) ADX 汇总竞价，选最高赢家    │
   Winner = argmax(eCPM_i)              │
      │                                 │
      ▼ (6) 赢家支付价格（Second Price）│
   clear_price = second_highest_eCPM   │
      │                                 │
      ▼ (7) 返回胜出广告创意            │
   广告展示给用户 ──────────────────────┘

整体延迟要求：< 100ms（通常 RTB 请求 ~30-50ms）
```

**Bid Request 核心字段（OpenRTB v2.x）：**

```json
{
  "id": "请求唯一ID",
  "imp": [{
    "id": "广告位ID",
    "banner": {"w": 300, "h": 250},
    "bidfloor": 0.5,       // 底价（CPM）
    "bidfloorcur": "USD"
  }],
  "site": {"domain": "example.com", "cat": ["IAB2"]},
  "user": {
    "id": "用户ID（Cookie/设备ID）",
    "buyeruid": "DSP侧用户ID（Cookie Sync后）"
  },
  "device": {"ua": "...", "ip": "...", "geo": {...}},
  "at": 2   // 拍卖类型：1=一价，2=二价
}
```

---

#### 2.2 DSP/SSP/ADX/DMP 生态

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           程序化广告生态                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  广告主层                                                                    │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                                     │
│  │  品牌A  │  │  电商B  │  │  游戏C  │  ← 有预算、有目标（品牌/效果）         │
│  └────┬────┘  └────┬────┘  └────┬────┘                                     │
│       └────────────┴────────────┘                                           │
│                    │                                                         │
│                    ▼                                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  DSP (Demand Side Platform) - 需求方平台                              │   │
│  │  功能：竞价决策、创意管理、预算控制、效果报告、人群定向               │   │
│  │  代表：Google DV360、The Trade Desk、MediaMath、腾讯广点通            │   │
│  └───────────────────────────┬──────────────────────────────────────────┘   │
│                               │                                              │
│                    ┌──────────┴──────────┐                                   │
│                    ▼                     ▼                                   │
│  ┌─────────────────────────┐  ┌─────────────────────────┐                   │
│  │ DMP (Data Mgmt Platform)│  │ ADX (Ad Exchange)       │                   │
│  │ 数据管理平台             │  │ 广告交易平台             │                   │
│  │ 用户画像、人群包、       │  │ 流量聚合、竞价撮合、     │                   │
│  │ 第三方数据接入           │  │ 反作弊、结算            │                   │
│  │ 代表：Oracle BlueKai    │  │ 代表：Google ADX,       │                   │
│  │       Salesforce DMP    │  │       AppNexus          │                   │
│  └─────────────────────────┘  └────────────┬────────────┘                   │
│                                             │                                │
│                    ┌────────────────────────┘                                │
│                    ▼                                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  SSP (Supply Side Platform) - 供给方平台                              │   │
│  │  功能：流量变现、底价管理、多ADX接入、收益最大化                       │   │
│  │  代表：Magnite（Rubicon+Telaria）、PubMatic、Index Exchange            │   │
│  └───────────────────────────┬──────────────────────────────────────────┘   │
│                               │                                              │
│  媒体/出版商层                 │                                              │
│  ┌─────────┐  ┌─────────┐  ┌──┴──────┐                                     │
│  │ 新闻网站│  │ 视频App │  │ 社交媒体│  ← 有流量、需要变现                  │
│  └─────────┘  └─────────┘  └─────────┘                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### 2.3 Cookie 追踪与 Cookie Sync

**Cookie 追踪原理：**

```
用户访问网站A：
  → 网站A设置第一方Cookie：user_id=abc123
  → 第三方广告SDK（如DoubleClick）设置第三方Cookie：dc_uid=xyz789

用户访问网站B：
  → 浏览器自动携带 dc_uid=xyz789
  → DoubleClick 识别：这是同一个用户！
  → 跨站行为拼接：在A买了鞋 + 在B看了跑步文章 = 运动爱好者
```

**Cookie Sync（Cookie 同步/匹配）：**

```
问题：ADX 有用户ID，DSP 有另一套用户ID，互不认识
解决：Cookie Sync（像"互认身份证"）

流程：
1. 用户访问媒体页面
2. SSP 触发 ADX 的像素请求：GET https://adx.com/sync?ssp_uid=111
3. ADX 重定向到 DSP：GET https://dsp.com/sync?adx_uid=AAA&ssp_uid=111
4. DSP 存储映射：{adx_uid: AAA ↔ dsp_uid: BBB}
5. 下次竞价：ADX 传 adx_uid=AAA，DSP 查到对应用户是 dsp_uid=BBB
```

---

#### 2.4 Lookalike 人群扩量

基于种子人群（Seed Audience）找到相似用户：

```
1. 定义种子：已转化用户集合 S = {u1, u2, ..., uk}
2. 提取特征：每个用户向量化 x_i = [年龄,性别,兴趣,行为...]
3. 计算种子中心：μ_seed = mean(x_i for i in S)
4. 相似度计算：sim(u, S) = cosine(x_u, μ_seed)
5. 扩量：选取 sim > threshold 的用户群
```

**Facebook Lookalike（工业级）：**
- 使用深度学习将用户映射到高维 embedding 空间
- 在 embedding 空间中做 ANN（近似最近邻）检索
- 扩量比例可控：1%~10%（越小越精准，越大越宽泛）

---

#### 2.5 Header Bidding：打破瀑布流垄断

**瀑布流（Waterfall）的问题：**

```
优先级瀑布（媒体侧设置）：
ADX_1 (底价$5) → ADX_2 (底价$3) → ADX_3 (底价$1)

问题：ADX_1 有最高优先级，即使 ADX_2 能出$8，也看不到机会
→ 媒体收益严重损失
```

**Header Bidding（竞价头部）解决方案（2014年）：**

```html
<!-- 媒体页面 <head> 中插入 JS -->
<script>
  // 同时向多个 ADX 发起竞价请求
  pbjs.requestBids({
    adUnits: [{...}],
    bidsBackHandler: function(bids) {
      // 汇总所有 ADX 出价，选最高者
      var winner = bids.getBidResponses();
      // 设置关键词，传给 Google Ad Manager
      googletag.cmd.push(function() {
        pbjs.setTargetingForGPTAsync();
        googletag.pubads().refresh();
      });
    }
  });
</script>
```

```
Header Bidding 效果：
媒体收益提升 20-40%（真正的价格竞争）
ADX 地位下降（Google ADX 失去部分垄断优势）
延迟增加（并行请求需等待最慢的ADX响应）
```

---

#### 2.6 Bid Landscape 与赢率模型

DSP 需要预测"以某个价格出价，赢得竞价的概率"：

```
赢率模型（Win Rate Model）：
P(win | bid=b) = P(b > max_competitor_bid)

常用分布假设：
- 对手出价服从对数正态分布：log(bid) ~ N(μ, σ²)
- Gamma 分布：更好拟合长尾

最优出价（最大化期望收益）：
E[profit | bid=b] = (value - b) × P(win | bid=b)
b* = argmax_b { (v - b) × F(b) }

其中 F(b) 为赢率（CDF），解方程：
(v - b*) × f(b*) = F(b*)  → v = b* + F(b*)/f(b*)
```

---

### 代表公司/产品

| 公司 | 产品/贡献 | 时间 |
|------|-----------|------|
| Google | DoubleClick Ad Exchange, DFP | 2008-2010 |
| Yahoo!/MS | Right Media, AppNexus | 2007-2011 |
| The Trade Desk | DSP 独立平台 | 2009 |
| Rubicon Project | SSP 代表 | 2007 |
| LiveRamp | 数据连接/Onboarding | 2011 |
| IAB | OpenRTB 2.0 标准 | 2012 |
| 腾讯 | 广点通（移动RTB） | 2012 |
| 阿里 | 阿里妈妈 TANX | 2011 |

---

### 解决的问题 + 遗留的问题

**✅ 解决的问题：**
- **程序化效率**：广告买卖自动化，媒体库存利用率大幅提升
- **精准定向**：Cookie 追踪使跨站行为定向成为可能
- **市场透明度**：标准化协议（OpenRTB）统一了行业
- **中小媒体变现**：SSP 让小流量网站也能接入高价广告

**❌ 遗留的问题：**
- **隐私问题**：Cookie 追踪涉及用户隐私，监管压力初现（GDPR 前奏）
- **广告欺诈**：机器人流量（Bot Traffic）泛滥，虚假点击损耗巨大
- **延迟问题**：RTB 链路长，100ms 内竞价压力大
- **预测精度不足**：LR/手工特征无法捕捉用户深层兴趣
- **CTR→CVR 跳跃**：点击率高≠转化率高，效果广告主不满意

---

### 面试必考点

1. **RTB 完整链路（100ms内）是什么？各角色作用？**
   - User→Publisher→SSP→ADX→DSP（并行）→ADX汇总→展示
   - ADX 负责竞价汇总，SSP 负责媒体侧管理，DSP 负责广告主侧出价

2. **Cookie Sync 为什么需要？流程是什么？**
   - 各平台用户ID体系不同，需要建立映射关系
   - 通过重定向/像素触发，双方互相存储 ID 映射表

---

## 阶段3：深度学习 CTR 预估时代（2016-2019）

### 背景与驱动力

RTB 体系确立后，广告系统的核心竞争力从"流量采购"转向"预估精度"——谁能更准确地预测用户点击/转化，谁就能在相同竞价下获得更高 ROI。

**驱动力：**
- **数据量爆炸**：移动互联网普及，用户行为数据呈指数级增长
- **GPU 算力普及**：深度学习训练成本下降，大规模 DNN 成为可能
- **特征工程瓶颈**：LR+GBDT 对海量稀疏特征的交叉挖掘接近上限
- **Google、Facebook 开源**：Wide&Deep、FNN 等论文引爆学界+工业界
- **移动广告崛起**：App 内广告比 Web 广告有更丰富的上下文特征

**关键论文里程碑：**
```
2016  Wide & Deep (Google Play, DLRS)
2016  DeepFM (华为诺亚方舟实验室)  
2017  DCN (Google, Deep & Cross Network)
2018  DIN (阿里巴巴, Deep Interest Network)
2018  ESMM (阿里巴巴, Entire Space Multi-task Model)
2019  DIEN (阿里巴巴, Deep Interest Evolution Network)
2019  FiBiNET (微博)
```

---

### 核心技术突破

#### 3.1 Wide & Deep：记忆与泛化的统一

**Google Play 应用推荐（2016），同样适用于广告 CTR 预估**

**核心思想：**

```
Wide 部分（线性模型）= 记忆（Memorization）
Deep 部分（DNN）    = 泛化（Generalization）

Wide: y_wide = w^T [x, φ(x)] + b
      - x：原始特征（稀疏高维）
      - φ(x)：人工交叉特征，如 user_installed_app × impression_app

Deep: y_deep = σ(W_L × ... σ(W_1 × a_0 + b_1) + b_L)
      - a_0：dense embedding of sparse features
      - 3层全连接，ReLU激活

联合训练（Joint Training）：
P(Y=1|x) = σ(y_wide + y_deep)
Loss = -Σ [y log p + (1-y) log(1-p)]
```

**Wide & Deep 架构图：**

```
输入特征
├── 连续特征 ──────────────────────────────────┐
├── 类别特征（Embedding）───────────────────── │
│   ├── user_id  → e_user  ∈ R^32             │
│   ├── item_id  → e_item  ∈ R^32             │   Deep 部分
│   ├── category → e_cat   ∈ R^16     ────────┤
│   └── ...                                   │   [512] → [256] → [128]
│                                             │          ↓
└── 人工交叉特征 ─────────── Wide 部分         │
    user_app × impression_app                 │
    (稀疏高维 one-hot)                         │
              ↓                               │
          w^T x                              ├──→ Sigmoid → P(click)
              └──────────────────────────────┘
```

---

#### 3.2 DeepFM：自动学习特征交叉

**痛点：** Wide&Deep 的 Wide 部分仍需人工设计交叉特征。

**DeepFM = FM + Deep，全自动特征交叉：**

```
FM（因子分解机）部分：
- 一阶项：Σ w_i × x_i
- 二阶交叉：Σ_{i<j} <v_i, v_j> × x_i × x_j
  其中 v_i 是第 i 个特征的 embedding，内积代表交叉权重

计算技巧（O(kn) vs O(n²)）：
Σ_{i<j} <v_i, v_j> x_i x_j = 
  1/2 [ ||Σ v_i x_i||² - Σ ||v_i||² x_i² ]

Deep 部分：
- 将所有特征 embedding concat → 多层 MLP

联合输出：
ŷ = σ(y_FM + y_Deep)
```

**DeepFM vs Wide&Deep：**

| 对比项 | Wide & Deep | DeepFM |
|--------|-------------|--------|
| 低阶交叉 | 需人工特征 | FM自动学习 |
| 高阶交叉 | Deep部分 | Deep部分 |
| 特征工程 | 需要 | 无需 |
| 参数量 | 较少 | 略多 |

---

#### 3.3 DIN：动态兴趣激活

**阿里巴巴（2018），Deep Interest Network**

**核心洞察：** 用户历史行为序列中，并非每个行为对当前 candidate item 都同等重要——需要**注意力机制**动态聚合用户兴趣。

```
传统方法（Pooling）：
user_history = [item1, item2, ..., itemN]
user_rep = mean(e_item1, e_item2, ..., e_itemN)  ← 信息损失严重

DIN 注意力机制：
attention_score_i = a(e_item_i, e_candidate)
                  = MLP([e_item_i, e_candidate, e_item_i ⊙ e_candidate])

user_rep = Σ_i attention_score_i × e_item_i   ← 动态兴趣表达

完整前向传播：
1. 候选商品 embedding: e_c ∈ R^d
2. 历史行为 embedding: e_i ∈ R^d (i=1,...,N)
3. 注意力得分: α_i = softmax(a(e_i, e_c))
4. 用户兴趣向量: v_u = Σ α_i × e_i
5. 拼接 MLP: [v_u, e_c, context] → FC → ŷ
```

**DIN 训练技巧：**
- **Dice 激活函数**（代替 ReLU）：自适应 rectified-linear unit
- **Mini-batch Aware Regularization**：只正则化出现在当前 batch 的参数
- **局部归一化 softmax**（不做全局归一化）

---

#### 3.4 ESMM：解决样本选择偏差的多任务学习

**阿里巴巴（2018），Entire Space Multi-task Model**

**核心问题（CVR 预估的两大挑战）：**

```
问题1：样本选择偏差（Sample Selection Bias, SSB）
  传统CVR训练：只用点击样本训练 P(conversion | click)
  但预测时：在全量展示空间预测
  → 训练空间 ≠ 推断空间，分布不一致！

问题2：数据稀疏（Data Sparsity）
  点击率约 1%，转化率约 0.1%
  CVR 模型可用的正样本极少，难以训练
```

**ESMM 解决方案：**

```
核心恒等式：
pCTCVR = pCTR × pCVR

其中：
pCTR   = P(click | impression)        ← 在展示空间定义
pCVR   = P(conversion | click)        ← 在点击空间定义  
pCTCVR = P(click ∧ conversion | impression)  ← 在展示空间定义

训练目标（两个任务联合）：
L = L_CTR + L_CTCVR
  = CrossEntropy(CTR_pred, click_label)
  + CrossEntropy(pCTR × pCVR, conversion_label)

关键：pCVR = pCTCVR / pCTR（隐式消除SSB）
     CVR 任务利用 CTR 的全量展示样本，解决数据稀疏
```

**ESMM 网络结构：**

```
展示空间（全量）
    │
    ├──── CTR 塔 ──── P(click|impression) ──────────────────┐
    │     (shared embedding)                                 │
    └──── CVR 塔 ──── P(conversion|click)                   × ──→ P(CTCVR)
          (shared embedding)                                 │
                                                            └──→ Loss(CTCVR)
```

---

#### 3.5 多任务学习架构演进

**从单塔到多塔（2016-2019）：**

```
MMoE（Multi-gate Mixture-of-Experts，Google，2018）：

输入
 │
 ├── Expert 1 (MLP)
 ├── Expert 2 (MLP)  
 ├── Expert 3 (MLP)  ← 共享专家网络
 ├── Expert 4 (MLP)
 │
Task A Gate: softmax([g_1, g_2, g_3, g_4]) ──→ 加权组合专家输出 ──→ Tower A ──→ ŷ_A
Task B Gate: softmax([g_1, g_2, g_3, g_4]) ──→ 加权组合专家输出 ──→ Tower B ──→ ŷ_B

优势：不同任务可以选择性利用不同专家，减少任务间干扰
```

**PLE（Progressive Layered Extraction，腾讯，2020）：**

```
共享专家 + 任务专属专家，多层递进式提取
解决"跷跷板问题"（seesaw effect）：一个任务提升，另一个下降
```

---

#### 3.6 特征工程革命：Embedding 化

```
稀疏特征 Embedding 化过程：

用户ID（10亿级）→ Embedding Table(1B × 32) → e_user ∈ R^32
商品ID（1亿级） → Embedding Table(100M × 32) → e_item ∈ R^32
类目ID（1万级） → Embedding Table(10K × 16) → e_cat ∈ R^16

Embedding 参数量：
user_emb = 1B × 32 × 4bytes ≈ 128GB  ← 工业级挑战！
→ 需要 PS（Parameter Server）分布式存储
→ 热点 embedding 缓存到 GPU 显存
→ Hash trick 压缩：对 ID 取模降低参数量
```

---

### 代表公司/产品

| 公司 | 技术/产品 | 代表论文 |
|------|-----------|----------|
| Google | Wide & Deep, DCN | Cheng et al. 2016 |
| Facebook | DLRM | Naumov et al. 2019 |
| 阿里巴巴 | DIN/DIEN/ESMM | Zhou et al. 2018/2019 |
| 华为 | DeepFM | Guo et al. 2017 |
| 腾讯 | PLE/多任务 | Tang et al. 2020 |
| 百度 | 搜索广告DNN | 凤巢算法团队 |

---

### 解决的问题 + 遗留的问题

**✅ 解决的问题：**
- **特征交叉自动化**：FM/DeepFM 无需人工设计交叉特征
- **序列兴趣建模**：DIN/DIEN 捕捉用户动态兴趣
- **多任务协同**：ESMM/MMoE 解决 CVR 偏差和多目标冲突
- **表达能力**：DNN 的非线性显著优于 LR，AUC 大幅提升

**❌ 遗留的问题：**
- **行为序列长度受限**：DIN 通常只能处理 50-200 条行为，长序列推理慢
- **训练-推理不一致**：离线 AUC 高，但线上效果提升有限
- **实时特征缺失**：模型难以利用用户当前实时行为（分钟级）
- **出价策略仍粗糙**：CTR 预估精了，但出价决策仍然是固定公式
- **多目标帕累托难题**：如何同时最优化 CTR、CVR、GMV 仍无优雅解

---

### 面试必考点

1. **Wide & Deep 为什么要联合训练？分开训练有什么问题？**
   - 联合训练使 Wide 部分学到"Deep 的不足"，互补效果更好
   - 分开训练：Deep 可能已经足够好，Wide 学到的是噪声

2. **ESMM 如何解决样本选择偏差？**
   - 利用 pCTCVR = pCTR × pCVR，在展示空间定义 CVR
   - CVR 塔共享 CTR 任务的展示空间训练信号

3. **DIN 的注意力机制和 Transformer 注意力有什么区别？**
   - DIN：MLP 计算注意力分数（考虑交叉特征），不做 scaled dot-product
   - Transformer：Q/K/V，scaled dot-product attention，序列自注意力

---

## 阶段4：Auto Bidding 时代（2019-2022）

### 背景与驱动力

深度学习提升了 CTR/CVR 预估精度，但出价策略（Bidding Strategy）仍然落后：
- 广告主手动设置出价，调整周期慢（天级）
- 固定出价无法响应实时竞争环境变化
- 广告主真正关注的是 oCPC/oCPA（目标转化成本），而非 CPC

**驱动力：**
- **广告主需求升级**：从"CPM/CPC 出价"到"oCPX 目标出价"（Target CPA/ROAS）
- **预算约束优化**：有限预算内最大化目标，需要全局最优分配
- **强化学习成熟**：DQN/PPO 等 RL 算法在序列决策问题上成熟
- **平台竞争**：各大平台（Google tROAS、Facebook CBO）率先落地，行业跟进

**出价模式演进：**
```
CPM → CPC → oCPC/oCPA → tROAS → 全自动出价（AutoBidding）
手动  手动   半自动       全自动     RL智能体
```

---

### 核心技术突破

#### 4.1 oCPX 出价体系

**oCPC（Optimized CPC）= 以 CPC 形式计费，但优化 CVR**

```
核心公式：

eCPM = pCTR × pCVR × target_CPA × 调节系数k

广告主设置：target_CPA = 目标每次转化成本（如100元）
系统自动出价：bid = pCTR × pCVR × target_CPA

在 k=1 时，期望 CPA ≈ target_CPA：
E[CPA] = E[成本/转化] 
       = bid/pCVR 
       = pCTR × pCVR × target_CPA / pCVR 
       = pCTR × target_CPA
       ≈ target_CPA（当广告充分竞争时）
```

**oCPA（Optimized CPA）扩展：**
```
多级转化目标：
oCPC（目标点击成本） → oCPL（目标线索成本） → oCPS（目标销售成本）

转化路径：展示 → 点击 → 表单提交 → 购买 → 复购
每个转化事件可单独设置优化目标
```

---

#### 4.2 PID 预算平滑控制

**问题：** 广告在一天内流量分布不均，如果前期花费过猛，后期无预算错失优质流量。

**PID 控制器（比例-积分-微分）：**

```
目标：按日预算均匀消耗（理想消耗曲线）

定义偏差：
e(t) = 理想消耗进度(t) - 实际消耗进度(t)
     = t/T × Budget - actual_spend(t)

PID 出价调节系数 λ(t)：
λ(t) = K_p × e(t) + K_i × Σe(τ) + K_d × Δe(t)

实际出价：bid(t) = base_bid × (1 + λ(t))

效果：
- 超花预算 → e(t)<0 → λ(t)<0 → bid 降低
- 欠花预算 → e(t)>0 → λ(t)>0 → bid 升高
```

**PID 的局限：**
- 只考虑预算消耗速率，不考虑流量质量
- 参数 K_p, K_i, K_d 需要人工调节（后来用 RL 自动学习）

---

#### 4.3 基于对偶优化的智能出价（LP Bidding）

**预算约束下的出价优化问题（原始问题）：**

```
Primal Problem（广告主视角，最大化转化数）：

max_{b_i}  Σ_i x_i × pCVR_i × w_i(b_i)
s.t.       Σ_i x_i × cost_i(b_i) ≤ Budget
           b_i ≥ 0

其中：
- x_i：第 i 次竞价机会
- pCVR_i：预测转化率
- w_i(b_i)：以出价 b_i 的赢率
- cost_i(b_i)：赢得后的支付价格

对偶问题（引入拉格朗日乘子 λ）：
L(b, λ) = Σ_i x_i[pCVR_i × w_i(b_i) - λ × cost_i(b_i)] + λ × Budget

最优出价（KKT条件）：
b_i* = pCVR_i / λ*

其中 λ* 为影子价格（shadow price）= 预算的边际价值
```

**直觉理解：**
- `λ*`：多一块钱预算能多带来多少转化（边际价值）
- `b_i* = pCVR_i / λ*`：转化概率越高，出价越高；预算越紧，λ* 越大，出价越低
- **核心算法**：二分搜索找到满足预算约束的最优 λ*

---

#### 4.4 强化学习自动出价

**RL 建模框架（Markov Decision Process）：**

```
状态 s_t：
  - 当前时刻 t（一天中的时间位置）
  - 已消耗预算比例
  - 已获得转化数
  - 最近 N 次竞价的赢率、CTR、CVR 统计
  - 当前流量质量指标

动作 a_t：
  - 出价调节系数 λ_t ∈ [0.5, 2.0]（离散化或连续）
  - 实际出价 = base_bid × λ_t

奖励 r_t：
  - 简单版：r_t = 转化数 - α × 超预算惩罚
  - 精细版：r_t = CVR_t × (1 - cost_t/ROI_target)

目标：max E[Σ_t γ^t × r_t]  s.t. Σ_t cost_t ≤ Budget
```

**工业落地算法：**

```
阿里 RL Bidding（双层架构）：
  上层：RL 智能体（DDPG）学习预算分配策略（小时级）
  下层：LP 优化出价（秒级，实时响应竞价）

字节跳动 Auto Bidding：
  使用 PPO 算法，状态包含竞争者出价分布估计
  离线仿真环境（Bidding Simulator）验证策略

美团 Auto Bidding：
  多目标 RL：同时优化 CVR、GMV、ROI
  考虑多广告主竞争（Multi-Agent 视角）
```

**在线仿真（Offline Training → Online Deployment）：**
```
1. 离线数据构建 Simulator：重放历史竞价数据
2. RL 在 Simulator 中探索训练（避免线上 Exploration 风险）
3. 策略网络部署上线（只做 Exploitation）
4. 定期用新数据更新 Simulator，重新训练策略
```

---

#### 4.5 多目标优化与帕累托前沿

**广告系统的多目标冲突：**

```
目标间关系：
CTR↑ ↔ CVR↓（点击多的用户不一定转化）
GMV↑ ↔ DAU↑（商业化 vs 用户体验）
短期收入↑ ↔ 长期留存↓

帕累托前沿（Pareto Frontier）：
在不损害其他目标的前提下，无法再提升某一目标

工业解法：
1. 加权和（Scalarization）：L = Σ w_i × L_i
   问题：权重难确定，非凸区域帕累托点无法获取
   
2. 约束优化：
   max  CTR_pred
   s.t. CVR_pred ≥ threshold
        Diversity ≥ threshold
        
3. EPO（Exact Pareto Optimal）：
   动态调整梯度方向，确保更新方向指向帕累托前沿
   
4. 线上 A/B + 帕累托优化：
   离线搜索帕累托前沿，线上实验选最优操作点
```

---

### 代表公司/产品

| 公司 | 产品 | 技术特点 |
|------|------|----------|
| Google | Smart Bidding (tCPA/tROAS) | 基于 Shapley 值的归因+ML出价 |
| Facebook | Campaign Budget Optimization (CBO) | 跨广告组预算自动分配 |
| 阿里妈妈 | oCPX + RL Bidding | 对偶优化+强化学习双层架构 |
| 字节跳动 | 巨量引擎 Auto Bidding | PPO+竞争感知 |
| 腾讯 | 广点通智能出价 | 多目标 Pareto 优化 |
| 百度 | 凤巢智能出价 | PID+预算预测 |

---

### 解决的问题 + 遗留的问题

**✅ 解决的问题：**
- **广告主操作简化**：从手动调价→设置目标 CPA，系统全自动
- **预算效率提升**：预算平滑消耗，避免早花完或剩余大量
- **转化目标优化**：真正优化广告主关心的目标（CVR/ROI），而非代理指标（CTR）
- **实时响应竞争**：RL 策略能感知竞争环境变化，动态调整

**❌ 遗留的问题：**
- **冷启动难**：新广告主/新广告组缺少历史数据，RL 模型探索风险大
- **多广告主博弈**：单个广告主 RL 优化，但多个智能体同时博弈可能导致系统不稳定
- **归因黑盒**：跨渠道/跨设备的转化归因越来越困难
- **隐私监管压力**：GDPR（2018）、CCPA（2020）开始限制 Cookie 追踪
- **数据孤岛**：广告主数据无法与平台数据安全融合

---

### 面试必考点

1. **oCPC 出价公式推导：为什么 bid = pCTR × pCVR × target_CPA？**
   - 期望支付 = bid × P(win) × P(click)
   - 期望转化数 = P(win) × pCTR × pCVR
   - 期望 CPA = bid/pCVR → 令其等于 target_CPA → bid = pCVR × target_CPA

2. **对偶出价中 λ（影子价格）的含义和计算方法？**
   - λ = 多一块钱预算多得到的转化数（边际价值）
   - 计算：二分搜索满足预算约束的 λ*，然后 b_i* = pCVR_i / λ*

3. **PID 控制预算平滑的参数含义？K_p K_i K_d 各管什么？**
   - K_p：比例项，立即响应偏差（当前消耗多/少）
   - K_i：积分项，消除长期偏差（历史累积偏差）
   - K_d：微分项，预测未来趋势（偏差变化速率）

---

## 阶段5：LLM + 隐私计算时代（2023-至今）

### 背景与驱动力

**双重冲击：**

1. **隐私监管革命：**
   - 2018年 GDPR（欧盟）：用户有权拒绝 Cookie 追踪
   - 2020年 CCPA（加州）：同类规定扩展到美国
   - 2022年 Apple ITP（Safari 完全禁用第三方 Cookie）
   - 2023年 Chrome 开始淘汰第三方 Cookie（最终落地 2024Q3）

2. **LLM 技术爆炸：**
   - 2022年 ChatGPT 引发 LLM 应用浪潮
   - 多模态大模型（GPT-4V、Gemini）可理解图像+文本
   - LLM 开始渗透广告创意生成、人群理解、出价策略

**Cookie 消亡时间线：**
```
2017  Apple ITP 1.0（Safari 限制第三方 Cookie 7天）
2019  Apple ITP 2.2（1天过期）
2020  Firefox 全面默认禁用第三方 Cookie
2022  Apple完全禁用
2023  Chrome 开始计划淘汰（1%流量开始）
2024  Chrome 大规模淘汰第三方 Cookie
```

---

### 核心技术突破

#### 5.1 隐私沙盒（Privacy Sandbox）

**Google Privacy Sandbox 核心 API：**

```
1. Topics API（兴趣分组）：
   - 浏览器本地分析用户兴趣（不上传原始数据）
   - 仅返回当前周最多3个兴趣分类（共350个主题）
   - 广告商只知道用户属于"体育/汽车/美妆"大类
   
   技术实现：
   用户浏览 URLs → 本地模型分类 → 存储最近3周 top-5 话题
   广告请求时：随机返回1个当周话题（概率噪声保护隐私）

2. FLEDGE/Protected Audience API（受众定向）：
   - 广告主在用户设备上"注册"兴趣组（Interest Group）
   - 竞价在浏览器本地完成，服务器只知道胜出广告，不知道用户信息
   
   流程：
   用户访问电商A → 浏览器本地存储 ig={advertiser: A, bid_url: ..., ads: [...]}
   用户访问媒体B → 浏览器本地运行 runAdAuction() → 联系广告主出价服务
                 → 本地竞价 → 展示胜出广告 → 上报（汇总，差分隐私保护）

3. Attribution Reporting API（转化归因）：
   - 点击/展示事件和转化事件分别存储在设备
   - 延迟汇总上报（延迟数小时），加噪声（差分隐私）
   - 禁止跨网站关联
```

---

#### 5.2 联邦学习（Federated Learning）在广告中的应用

**核心场景：广告主数据 × 平台数据安全联合建模**

```
问题：
- 广告主有转化数据（CRM：用户购买记录）
- 平台有用户行为数据（点击/浏览/搜索）
- 两者都不愿意/不能直接共享原始数据
- 但联合建模可以大幅提升转化预测效果

联邦学习解决方案（纵向联邦）：

┌─────────────────────────────────────────────────────────┐
│                     联邦训练流程                          │
│                                                         │
│  广告主方（Party A）         平台方（Party B）            │
│  特征：用户CRM属性           特征：用户行为序列            │
│  标签：转化 y                无标签                       │
│                                                         │
│  1. 各方本地计算中间结果（梯度/嵌入）                       │
│  2. 通过加密传输中间结果（不传原始数据）                    │
│  3. 联合更新全局模型参数                                   │
│  4. 循环直到收敛                                          │
│                                                         │
│  安全保证：                                              │
│  - 半诚实模型：各方不主动泄露但可能推断                    │
│  - 安全多方计算（SMPC）：加密中间结果                      │
│  - 差分隐私（DP）：添加噪声防止推断                        │
└─────────────────────────────────────────────────────────┘
```

**差分隐私（DP）机制：**

```
ε-差分隐私定义：
对任意相邻数据集 D 和 D'（仅差1条记录），算法 M 满足：
P[M(D) ∈ S] ≤ e^ε × P[M(D') ∈ S]

Gaussian 机制（给梯度加噪）：
M(D) = f(D) + N(0, σ²I)

其中 σ = Δf × √(2ln(1.25/δ)) / ε
Δf：函数 f 的全局敏感度（L2范数）

隐私预算 ε：越小隐私保护越强，但模型效用越差
工业实践：通常 ε ∈ [1, 10]，δ ∈ [10^-5, 10^-6]
```

---

#### 5.3 隐私计算技术栈

```
┌────────────────────────────────────────────────────────────────────────┐
│                    广告隐私计算技术栈                                    │
├──────────────┬──────────────┬──────────────┬───────────────────────────┤
│  技术        │  核心原理    │  广告场景    │  代表方案                  │
├──────────────┼──────────────┼──────────────┼───────────────────────────┤
│  安全多方计算 │ 秘密分享/    │ 跨平台联合   │ 蚂蚁链摩斯、百度点石、      │
│  (SMPC)      │ 混淆电路     │ 人群定向     │ 腾讯星云                   │
├──────────────┼──────────────┼──────────────┼───────────────────────────┤
│  联邦学习    │ 本地训练+    │ 广告主联合   │ Google FL、字节隐私         │
│  (FL)        │ 梯度聚合     │ 建模         │ 计算、阿里合规计算          │
├──────────────┼──────────────┼──────────────┼───────────────────────────┤
│  可信执行环境 │ 硬件隔离     │ 广告数据     │ Intel SGX、ARM TrustZone   │
│  (TEE)       │ 加密计算     │ 安全处理     │ AMD SEV                    │
├──────────────┼──────────────┼──────────────┼───────────────────────────┤
│  差分隐私    │ 添加噪声     │ 统计报告     │ Apple DP、Google RAPPOR    │
│  (DP)        │ 保护个体     │ 归因上报     │                            │
├──────────────┼──────────────┼──────────────┼───────────────────────────┤
│  同态加密    │ 密文计算     │ 密文出价     │ Microsoft SEAL             │
│  (HE)        │             │ 联合预测     │ IBM HElib                  │
└──────────────┴──────────────┴──────────────┴───────────────────────────┘
```

---

#### 5.4 无 Cookie 追踪替代方案

**1. 第一方数据战略（First-Party Data）：**
```
品牌/媒体自建数据体系：
- 注册登录用户数据（邮件/手机号）
- 购买/浏览行为（网站自有1st-party Cookie）
- CRM 数据（离线购买+客服记录）

关键技术：
- Identity Graph：将 email/phone/device_id 归一到同一用户
- CDP（Customer Data Platform）：统一用户数据平台
- Hashed Email（SHA256）作为跨平台 ID
```

**2. 概率 ID（Probabilistic Identity）：**
```
无 Cookie 环境下，通过多维信号估计用户同一性：

特征组合：
- IP 地址（变化慢）
- User Agent（浏览器/OS/设备型号）
- 屏幕分辨率 + 时区 + 字体列表
- Canvas fingerprint（canvas元素渲染差异）

匹配概率：
P(same_user | feature_match) = P(feature_match | same_user) × P(same_user)
                                / P(feature_match)
（贝叶斯估计，精度远低于 Cookie）
```

**3. Clean Room 技术：**
```
数据净室（Data Clean Room）：
- 广告主上传加密的用户哈希（如 SHA256(email)）
- 平台在安全环境中匹配，输出聚合统计（不暴露个体）
- 结果：广告主知道"在平台上覆盖了X%目标用户"，但不知道具体是谁

代表产品：
- Google Ads Data Hub
- Facebook Advanced Analytics
- LiveRamp Safe Haven
- AWS Clean Rooms
```

---

#### 5.5 LLM 在广告中的应用

**场景1：广告创意生成（Creative Generation）**

```
传统流程：创意团队 → 设计师 → A/B 测试 → 上线
         耗时：1-2周，成本高

LLM+多模态流程：
  输入：品牌指引 + 产品图片 + 目标人群描述
  
  LLM文案生成：
    prompt = "你是一个广告文案专家，为{brand}的{product}
              针对{audience}生成5个不同风格的标题，要求..."
    output = ["标题1（情感诉求）", "标题2（功能诉求）", ...]
  
  多模态图像生成（DALL-E/Stable Diffusion）：
    结合产品图 + 文案 → 生成广告图片
  
  自动 A/B 测试：多版本上线，系统自动选最优
```

**场景2：LLM 增强用户理解**

```
传统人群画像：规则/统计 → 有限标签（年龄/性别/兴趣）

LLM 语义理解：
  用户搜索/浏览文本 → LLM 语义嵌入 → 细粒度兴趣理解
  
例：
  用户搜索 "how to improve running form for marathon"
  传统：体育 > 跑步
  LLM：准备参加马拉松的中级跑步者，关注技术提升，可能消费：
       跑鞋升级、运动补给品、马拉松报名、运动手表

LLM Embedding 用于广告匹配：
  query_emb = LLM.encode(user_query)  ← 语义向量
  ad_emb    = LLM.encode(ad_title + description)
  relevance = cosine_similarity(query_emb, ad_emb)
```

**场景3：LLM 辅助出价策略**

```
实验性方向：
  - LLM 作为策略生成器：根据市场分析生成出价策略
  - In-Context Learning：少量案例 → LLM 自适应策略
  - LLM 解释 RL 策略：提高自动出价的可解释性

挑战：
  - LLM 推理延迟（秒级）远无法满足 RTB 100ms 要求
  - 目前 LLM 更多用于离线策略生成，而非在线出价
```

**场景4：对话式广告管理**

```
传统：广告主需要学习复杂的广告后台操作
LLM界面：
  广告主："帮我针对25-35岁女性，推广我的新款护肤品，
          预算5000元，目标获客成本不超过50元"
  
  LLM 自动：
  1. 解析意图 → 创建广告系列
  2. 生成人群定向标签
  3. 调用文案生成 API 生成3组创意
  4. 设置 oCPA=50元，预算=5000元
  5. 预览 → 确认 → 上线

代表产品：
  - Google Performance Max（自动化程度最高）
  - Meta Advantage+ Shopping
  - 字节 Kling AI 创意生成
```

---

#### 5.6 无 Cookie 归因新范式

**传统归因 vs 新范式：**

```
传统（基于 Cookie 的多触点归因）：
展示A（Cookie追踪）→ 搜索B（Cookie追踪）→ 点击C → 购买
→ 可以重建完整转化路径，分配功劳

新范式（无 Cookie）：

1. 概率归因（Probabilistic Attribution）：
   - 实验组/对照组（Holdout Test）
   - 有广告组 vs 无广告组的转化率差异 = 广告增量效果（Incrementality）
   - 不需要追踪个体，只需群体统计

2. 媒体组合建模（Media Mix Modeling, MMM）：
   - 回归分析：Sales = f(TV, Digital, OOH, ...) + 季节性 + 基线
   - 基于宏观数据，不依赖 Cookie
   - 时效性差（周级），但隐私友好

3. 聚合转化 API（如 Meta CAPI, Google Enhanced Conversions）：
   - 广告主服务器直接上报转化（Server-to-Server）
   - 用 hashed email/phone 做匹配，而非 Cookie
   - 平台在加密环境中归因，输出汇总数据
```

---

### 代表公司/产品

| 领域 | 公司/产品 | 技术特点 |
|------|-----------|----------|
| 隐私沙盒 | Google Privacy Sandbox | Topics API + FLEDGE |
| 联邦学习 | Google/蚂蚁集团 | 纵向联邦 + DP |
| Clean Room | Google Ads Data Hub | SQL on encrypted data |
| LLM创意 | 字节Kling、Meta Gen AI | 多模态广告生成 |
| 无Cookie归因 | Facebook CAPI、Google ECv2 | S2S + 哈希匹配 |
| 隐私计算 | 腾讯星云、阿里合规计算 | TEE + SMPC |
| LLM广告管理 | Google Performance Max | 全自动广告优化 |

---

### 解决的问题 + 遗留的问题

**✅ 解决的问题：**
- **隐私合规**：Privacy Sandbox/联邦学习满足 GDPR/CCPA 要求
- **第一方数据价值提升**：品牌自有数据战略地位凸显
- **创意效率**：LLM 使广告创意生产效率提升 10x+
- **语义理解**：LLM embedding 大幅提升广告-用户语义匹配精度

**❌ 遗留的问题：**
- **精准度下降**：无 Cookie 定向精准度显著下降，中小广告主受影响最大
- **归因黑盒**：概率归因和 MMM 精度远低于确定性 Cookie 归因
- **LLM 延迟**：在线推理延迟与 RTB 实时性要求仍有巨大 gap
- **隐私计算成本**：SMPC/同态加密计算开销大，工业落地仍有挑战
- **生态碎片化**：各平台"数据围墙"（Walled Garden）越来越高，跨平台归因更难
- **LLM 幻觉风险**：LLM 生成广告文案可能包含不准确信息，需要人工审核

---

### 面试必考点

1. **第三方 Cookie 消亡后，如何替代 Retargeting？**
   - 第一方数据 + CRM 匹配
   - Topics API（兴趣分组）
   - Clean Room 跨平台匹配
   - Lookalike 基于第一方种子人群

2. **联邦学习在广告中如何解决数据孤岛？有什么局限？**
   - 纵向联邦：广告主 CRM × 平台行为，联合训练 CVR 模型
   - 局限：通信开销大、收敛慢、安全性仍有推断风险（梯度泄露攻击）

---

## 广告系统核心组件横切关注点

### 拍卖机制对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         广告拍卖机制演进                                      │
├──────────────┬──────────────┬──────────────┬─────────────────────────────────┤
│  机制        │  出现时间    │  特点        │  典型使用                        │
├──────────────┼──────────────┼──────────────┼─────────────────────────────────┤
│  一价拍卖    │ 1997        │ 赢家付出价   │ 早期GoTo.com，部分程序化          │
│  (FPA)       │             │ 容易过度出价 │ 2019后 Google/FB 部分切换FPA      │
├──────────────┼──────────────┼──────────────┼─────────────────────────────────┤
│  二价拍卖    │ 2000        │ 赢家付次高价 │ Google AdWords经典模型            │
│  (SPA/GSP)   │             │ 激励真实出价 │ 大多数 RTB 平台默认              │
├──────────────┼──────────────┼──────────────┼─────────────────────────────────┤
│  VCG 拍卖    │ 理论        │ 严格激励相容 │ Google AdX 部分场景              │
│              │             │ 实现复杂     │                                  │
├──────────────┼──────────────┼──────────────┼─────────────────────────────────┤
│  Bid Shading │ 2019        │ FPA中降低出价 │ DSP对接 FPA 市场时使用           │
│              │             │ 减少赢家诅咒 │ 预测 clearing price，调整出价    │
└──────────────┴──────────────┴──────────────┴─────────────────────────────────┘
```

### CTR 预估模型演进总结

```
时间     模型               核心贡献                    局限
──────────────────────────────────────────────────────────────────────────
2010     Logistic Regression  简单高效，工业基准         无自动特征交叉
2012     GBDT+LR             树模型自动特征交叉+LR       离线，不可端到端
2014     FM                  O(kn)二阶交叉              缺乏高阶交叉
2016     Wide&Deep           记忆+泛化，端到端           Wide需人工特征
2017     DeepFM              FM+Deep，全自动交叉         仍限于二阶
2017     DCN                 Cross Network，高阶交叉     显式多项式交叉
2018     DIN                 注意力序列建模              序列长度受限
2018     ESMM                多任务解决SSB               需要CTR/CVR正相关
2019     DIEN                GRU序列兴趣进化             推理延迟高
2020     DCN V2              矩阵乘法高阶交叉            参数量大
2021     BST                 Transformer序列建模         计算密集
2022     ETA                 哈希加速长序列              精度-效率折中
2023     HSTU(TikTok)        Transformer全序列，SOTA     部署挑战大
```

---

## 面试高频问题

### 1. GSP 拍卖中如何计算每个广告位的实际 CPC？

**答：**

```
假设3个广告位，5个竞价者（按 eCPM 排序）：

位置  点击率  竞价者  bid  pCTR  eCPM=bid×pCTR
1     α1=0.1  A       5    0.1   0.5
2     α2=0.05 B       4    0.08  0.32
3     α3=0.02 C       3    0.06  0.18
           D       2    0.05  0.10
           E       1    0.03  0.03

位置1（A赢）实际CPC：
CPC_A = B.bid × B.pCTR / A.pCTR = 4 × 0.08 / 0.1 = 3.2元
（A支付：恰好击败B需要的出价 × 对应 CPC）

位置2（B赢）实际CPC：
CPC_B = C.bid × C.pCTR / B.pCTR = 3 × 0.06 / 0.08 = 2.25元
```

---

### 2. Wide & Deep 的 Wide 部分为什么要用交叉特征而不是直接用原始特征？

**答：**
- Wide 部分设计目的是**记忆（Memorization）**：记住特定的特征组合与结果的强关联
- 例如：`user_installed=Netflix AND impression=Hulu → 高CTR`，这种精确规则需要交叉特征
- 单独原始特征（user_installed=Netflix）泛化性太强，无法记住"安装Netflix的人对Hulu感兴趣"这种特定规律
- **Deep 做泛化，Wide 做记忆**：两者互补

---

### 3. ESMM 中为什么用 pCTCVR 而不是直接训练 pCVR？

**答：**
- 直接训练 pCVR：只能用点击样本（点击后才有转化标签）
- 训练空间（点击数据）≠ 推断空间（全量展示数据）→ **样本选择偏差**
- pCTCVR = pCTR × pCVR，在**展示空间**有完整标签（是否点击+转化）
- ESMM 将 CVR 任务转化为在展示空间优化 pCTCVR，隐式消除偏差
- 同时，展示样本比点击样本多 100x，解决**数据稀疏**问题

---

### 4. RTB 系统中，DSP 如何在 50ms 内完成出价？

**答：**
```
关键优化手段：
1. 预计算 Embedding：用户特征 embedding 提前计算并缓存（Redis）
2. 模型轻量化：
   - 线上模型：2层小 DNN 或 LR（离线 DNN 特征转换后线上 LR）
   - 知识蒸馏：大模型离线蒸馏成小模型
3. 特征工程前置：特征抽取提前到广告打包阶段
4. 硬件加速：GPU 服务器 batch 推理 / FPGA 定制化推理
5. 请求过滤：
   - Black List：已展示过多次的广告不再竞价
   - 低质流量过滤（Bot检测）：提前终止
6. 预算预检：预算耗尽的广告活动直接跳过
```

---

### 5. 对偶出价中影子价格 λ 如何在线更新？

**答：**
```
离线预算约束优化（L时间窗口）：
  L(λ) = E[pCVR / λ] - Budget  ← 对偶函数

在线更新（次梯度方法）：
  λ_{t+1} = λ_t - η × (Budget_remain / T_remain - cost_t)

PID 变种（更稳定）：
  e(t) = target_spend_rate - actual_spend_rate
  λ(t) = λ_base × exp(K_p × e(t))  ← 乘性更新更稳定

关键：λ 需要在预算消耗过快时升高（降低出价），预算剩余时降低（提高出价）
```

---

### 6. 联邦学习梯度泄露攻击是什么？如何防御？

**答：**
```
攻击原理（DLG - Deep Leakage from Gradients）：
  给定梯度 ∇W，攻击者可以反向推断原始训练数据 x：
  x* = argmin_x ||∇L(W, x) - ∇W||²

防御方案：
1. 差分隐私（DP）：梯度加噪声，精度换隐私
   ∇W_noisy = ∇W + N(0, σ²I)，σ 由隐私预算 ε 决定
   
2. 梯度压缩：只传 Top-K 梯度，降低信息量
   同时减少通信开销

3. 秘密分享（SMPC）：梯度分片传输，单方无法重建
   适合可信多方场景

4. 梯度聚合时序保护：批量聚合，延迟上传，降低单样本可识别性
```

---

### 7. CTR 预估中，正负样本极度不均衡（1:100）如何处理？

**答：**
```
方法1：负样本采样（工业最常用）
  负样本随机采样到 1:N（N=3~10）
  但需要校准：p_calibrated = p_raw / (N × p_raw + 1 - p_raw)
  
方法2：损失函数调整
  Focal Loss: L = -α(1-p)^γ log(p)，降低简单负样本权重
  
方法3：正样本过采样（SMOTE）
  在特征空间插值生成伪正样本（CTR预估中少用）

方法4：分层采样
  按展示位置/时间分层，保持各层样本比例
  
工业实践：
  通常负采样率 = 0.01~0.1（从百分之一到百分之十的负样本）
  + 预测值校准（因为负采样改变了先验概率）
```

---

### 8. RTB 中的 Bid Shading 是什么？为什么需要它？

**答：**
```
背景：2019年左右，主要 ADX 从二价拍卖切换到一价拍卖
  二价拍卖：赢家付次高价 → 鼓励真实出价
  一价拍卖：赢家付自己出价 → 如果按真实价值出价，会付超额（赢家诅咒）

Bid Shading（出价遮蔽）：DSP 在一价拍卖中主动降低出价

核心公式：
  shaded_bid = value × P(win at shaded_bid) optimally
  
最优 Bid Shading：
  max (value - b) × P(win|b)
  → 对手出价分布的 CDF F(b)
  → 最优 b* 满足：F(b*) / f(b*) = value - b*

实现：
  1. 历史竞价数据估计 clearing price 分布（log-normal等）
  2. 对每次竞价机会，预测 clearing price 分位点
  3. 出价 = min(value, predicted_clearing_price × 1.05)
```

---

### 9. 广告系统中如何处理用户长序列行为（>1000条）？

**答：**
```
挑战：DIN/DIEN attention 复杂度 O(N)，N=1000时线上推理超时

解决方案演进：

方案1：截断（Naive）
  只用最近 50-200 条，简单但丢失长期兴趣

方案2：SIM（Search-based Interest Model，阿里2020）
  GSU（General Search Unit）：快速从全量序列检索 Top-K 相关行为
  ESU（Exact Search Unit）：对检索结果精细建模
  
  GSU 检索：
    hard search：类目/品牌精确匹配
    soft search：item embedding ANN 检索 Top-K
  
  ESU 建模：DIN attention on Top-K（K=200），O(K) 而非 O(N)

方案3：ETA（Efficient Transformer for Long Ads，字节）
  对长序列做 locality-sensitive hashing，降低注意力复杂度
  O(N) → O(N√N) 或更低

方案4：HSTU（Hierarchical Sequential Transduction Units，TikTok 2024）
  全序列 Transformer，工程优化（Flash Attention等），直接处理数千条行为
```

---

### 10. 如何设计一个广告 CVR 预估系统（系统设计题）？

**答：**
```
完整 CVR 预估系统设计：

┌─────────────────────────────────────────────────────────────┐
│                    CVR 预估系统架构                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  数据层                                                      │
│  ├── 实时特征流（Kafka）：用户实时点击/浏览                  │
│  ├── 离线特征（Hive）：用户画像、广告历史统计                 │
│  └── 标签回流（归因）：转化事件（延迟1-7天）                 │
│                                                              │
│  特征工程                                                    │
│  ├── 用户特征：人口属性、设备信息、实时行为序列              │
│  ├── 广告特征：文案、创意、历史CVR、出价                     │
│  ├── 上下文特征：时间、页面、位置                            │
│  └── 交叉特征：用户×广告 历史交互统计                        │
│                                                              │
│  模型层                                                      │
│  ├── 基础模型：ESMM（解决SSB）或 Two-Tower CVR              │
│  ├── 序列模型：DIN/SIM（用户历史行为建模）                   │
│  └── 多任务：CTR+CVR联合训练，共享底层表示                   │
│                                                              │
│  训练                                                        │
│  ├── 全量样本 ESMM（展示空间标签）                           │
│  ├── 增量训练（Online Learning，分钟级更新）                 │
│  └── 样本校准（负采样纠偏）                                  │
│                                                              │
│  服务                                                        │
│  ├── 模型服务（TF Serving/TorchServe）                      │
│  ├── 特征缓存（Redis，热点用户/广告特征）                    │
│  ├── 延迟目标：p99 < 20ms                                   │
│  └── 降级策略：特征缺失时使用统计均值                        │
│                                                              │
│  监控                                                        │
│  ├── 在线 AUC（滑动窗口）                                   │
│  ├── 预估值-实际转化率 校准曲线                              │
│  └── 特征分布漂移告警（PSI监控）                             │
└─────────────────────────────────────────────────────────────┘

关键设计决策：
1. ESMM vs 直接CVR：优先选ESMM（解决SSB）
2. 归因窗口：短窗口（1天）用于快速反馈，长窗口（7天）用于训练
3. 冷启动：新广告用相似广告的CVR迁移；新用户用人口属性+设备特征
4. 实时性：User特征实时更新（分钟级），模型参数增量更新（小时级）
```

---

## 附录：广告系统关键公式速查

```
基础竞价公式：
  eCPM = bid × pCTR × Quality_Score
  GSP 实付：price_i = bid_{i+1} × CTR_{i+1} / CTR_i + ε

CTR 预估（LR）：
  p(click) = σ(w^T x) = 1 / (1 + e^{-w^T x})

oCPC 出价：
  bid = pCVR × target_CPA

对偶优化出价：
  bid_i* = pCVR_i / λ*，其中 λ* = argmin{Σ cost ≤ Budget}

ESMM：
  pCTCVR = pCTR × pCVR，Loss = L_CTR + L_CTCVR

DIN 注意力：
  v_u = Σ_i softmax(MLP([e_i; e_c; e_i⊙e_c])) × e_i

差分隐私（Gaussian机制）：
  M(D) = f(D) + N(0, σ²I)，σ = Δf√(2ln(1.25/δ)) / ε

Bid Shading（FPA最优出价）：
  b* = argmax_b {(v-b) × F(b)}
  解：v = b* + F(b*)/f(b*)
```

---

> 📝 整理：MelonEggLearn | 更新：2026-03-16
> 覆盖阶段：关键词广告→RTB→深度学习CTR→Auto Bidding→LLM+隐私计算
> 下一步建议阅读：`ads/ESMM详解.md`、`ads/AutoBidding技术演进_从规则到RL.md`、`ads/广告系统知识库.md`
