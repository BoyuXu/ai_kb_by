# Kronos 论文研究报告 & 特征工程方案

> 研究员：MelonEgg | 日期：2026-03-20 | 论文：arXiv 2508.02739 (AAAI 2026)

---

## Part 1：Kronos 核心机制解析

### 1.1 整体框架

Kronos 是两阶段架构：**离散化（Tokenizer）→ 自回归建模（Autoregressive Transformer）**

```
原始K线序列 (OHLCVA × T根)
       │
       ▼
【阶段一：K线 Tokenizer】（Transformer AutoEncoder + BSQ量化）
  Encoder → 连续隐向量 ξ_t ∈ R^k
       │
  BSQ量化: ξ_t → b_t ∈ {-1,1}^k  （k=20 bits，约100万种状态）
       │
  分解为双子token：
  ├── coarse subtoken b^c_t ∈ {-1,1}^(k/2)  ← 粗粒度，捕捉趋势方向
  └── fine subtoken b^f_t ∈ {-1,1}^(k/2)    ← 细粒度，捕捉内部结构
       │
       ▼
token序列 [b_1, b_2, ..., b_T]

【阶段二：Decoder-only Transformer（自回归预训练）】
  p(b_{T+h} | b_{1:T+h-1})  ← next-token prediction
  训练目标：预测下一根K线的 (coarse token, fine token)
       │
       ▼
下游任务（Zero-shot）：
  预测未来H根K线 → 计算预测涨幅 → 截面RankIC
```

### 1.2 Tokenizer 的量化方法（BSQ）

**Binary Spherical Quantization (BSQ)**，不同于传统VQ-VAE的 lookup-table：

- 将连续隐向量 ξ_t 投影到 k 个可学习超平面上，取符号（+1 或 -1）
- k=20 bits → 词表大小 2^20 ≈ 100万（太大）
- 分解技巧：分成 n=2 个子空间，每个 k/2=10 bits → 词表 2^10 = 1024（可处理）
- 损失函数：`L = L_coarse + L_fine + λ * L_quant`
  - L_coarse：用 coarse subtoken 重建原始K线（捕捉大趋势）
  - L_fine：用 coarse+fine 精确重建（捕捉细节）
  - L_quant：BSQ commitment loss，防止量化崩塌

**关键洞见**：每一根 OHLCVA 六元组被当作一个整体单元量化为一个 token，保留了价量之间的内在关联。

### 1.3 归一化（最重要的 Scale-Invariant 设计）

输入预处理（论文 Appendix C, Input Preprocessing）：
- 用序列起始价格做归一化，使模型只看到相对价格变化，不受绝对价位影响
- 这是 Kronos 的核心设计哲学：**价格模式是 scale-invariant 的**

### 1.4 下游选股任务接入（RankIC）

论文中 Investment Simulation 用的是 **Zero-shot 推理**：
1. 用历史90根日线K线作为 context（lookback）
2. 调用 `KronosPredictor.predict(pred_len=5)` 预测未来5根K线
3. 取预测的第5日收盘价 / 当前收盘价 - 1 = 预测5日涨幅
4. 对全市场股票截面计算此因子的 RankIC
5. **无需 fine-tuning，直接 zero-shot** 即 RankIC +87% 于 non-pre-trained baseline

### 1.5 数据规模与 A 股覆盖

- 预训练数据：45 个全球交易所，120 亿条 K 线记录，7 种时间粒度
- **A股包含**：代码示例为 `XSHG_5min_600977.csv`（上交所5分钟线）
- 支持分钟线到月线多时间粒度，日线 A 股数据在训练集内

---

## Part 2：适配性评估（无 GPU 场景）

### 2.1 Kronos-mini CPU/MPS 推理可行性

| 指标 | 评估 |
|------|------|
| 模型大小 | 4.1M 参数，内存占用 ~16MB（fp32），完全无压力 |
| 推理框架 | PyTorch，原生支持 MPS（Apple Silicon）和 CPU |
| 是否需要 GPU | ❌ 不需要，CPU/MPS 均可推理 |
| M 系列 Mac 适配 | ✅ MPS 加速支持，推荐 device="mps" |

**推理延迟估算（5000只股票 × 90根日线）：**

```
单次推理（1只股票，seq_len=90，pred_len=5）：
  - Tokenizer Encoder 前向: ~2ms (CPU)
  - Transformer 自回归解码 5步: ~10ms (CPU)
  - 总计: ~12ms/只

批量推理（batch_size=64）：
  - CPU: ~64 × 2ms = ~100-150ms/batch（矩阵乘法并行）
  - MPS: ~3-5x 加速 → ~30-50ms/batch

全量 5000 只:
  - CPU: 5000/64 ≈ 79 batches × 150ms ≈ 12秒
  - MPS: 5000/64 ≈ 79 batches × 50ms ≈ 4秒
```

✅ **结论：每日收盘后 MPS 推理 5000 只股票约 4 秒，完全可接受。**

### 2.2 Fine-tune 条件评估

| 需求 | 我们的情况 | 评估 |
|------|----------|------|
| 样本量 | 5年 × 5000只 × 250天 ≈ 625万条日线 | ✅ 充足 |
| 数据格式 | OHLCV（量已有，amount 需补） | ✅ 基本满足 |
| 计算资源 | 无 GPU，M 系列 Mac | ⚠️ Fine-tune 较慢，但 4.1M 参数可行 |
| 前置工作 | 需要 feature_daily 表存好 5 年数据 | ⚠️ 正在建设中 |

**关于 zero-shot vs fine-tune**：
- 论文 zero-shot 已 +87% baseline，直接用 zero-shot 是最快路径
- Fine-tune 对 A 股日线特化可能再提升 10-20%，但需要 GPU 或长时间 CPU 训练
- **建议先 zero-shot 验证因子有效性，再决定是否 fine-tune**

---

## Part 3：两套特征方案（CoderBy 实现指引）

---

### 🟢 方案A：K线离散化特征（零模型成本，纯特征工程）

**核心思路**：借鉴 Kronos tokenizer 的哲学——**用归一化的相对比例 + 多维联合离散化**，将 OHLCV 的结构信息编码为整数特征，供 LightGBM 直接使用。

#### A1. K线体型编码（Candle Body Type）— 1个特征

**原理**：Kronos coarse subtoken 捕捉的正是每根K线的宏观形态。

```python
def candle_body_type(open_, high, low, close):
    """
    返回K线体型类别编码 (int 0-7)
    """
    total_range = high - low + 1e-8
    body = close - open_
    body_ratio = abs(body) / total_range           # 实体占振幅比例
    upper_shadow = (high - max(open_, close)) / total_range  # 上影线比例
    lower_shadow = (min(open_, close) - low) / total_range   # 下影线比例
    
    if body_ratio > 0.65 and body > 0:     return 1  # 大阳线
    elif body_ratio > 0.65 and body < 0:   return 2  # 大阴线
    elif body_ratio > 0.25 and body > 0:   return 3  # 小阳线
    elif body_ratio > 0.25 and body < 0:   return 4  # 小阴线
    elif body_ratio < 0.1:                 return 5  # 十字星（开收几乎相同）
    elif upper_shadow > 0.45:              return 6  # 上影线显著（倒锤、射击之星）
    elif lower_shadow > 0.45:             return 7  # 下影线显著（锤子、上吊线）
    else:                                  return 0  # 其他/纺锤体
```

#### A2. 收盘价位分位（Close Position）— 1个数值特征 + 1个分桶特征

**原理**：Kronos fine subtoken 捕捉价格在振幅区间的精确位置，这是最具预测力的信号之一。

```python
# 连续特征（最重要！）
close_pos = (close - low) / (high - low + 1e-8)
# 值域 [0,1]：0=收在最低点，1=收在最高点
# close_pos > 0.8 通常是强势信号

# 分桶特征 (int)
close_pos_bucket = pd.cut(close_pos, bins=[0, 0.25, 0.5, 0.75, 1.0],
                           labels=[1, 2, 3, 4]).astype(int)
```

#### A3. 量价关系编码（Volume-Price Relation）— 1个分类特征

**原理**：Kronos 将 Volume 和 Amount 作为 OHLCVA 的组成部分，量价关系是 tokenizer 学到的核心模式之一。

```python
def vol_price_type(close, close_prev, volume, vol_ma20):
    """
    返回量价关系类型 (int 0-5)
    """
    vol_ratio = volume / (vol_ma20 + 1e-8)
    pct_chg = (close - close_prev) / (close_prev + 1e-8)
    
    if vol_ratio > 1.5 and pct_chg > 0.02:    return 1  # 放量大涨（突破）
    elif vol_ratio > 1.5 and pct_chg < -0.02: return 2  # 放量大跌（恐慌）
    elif vol_ratio > 1.5 and abs(pct_chg) < 0.01: return 3  # 放量平盘（多空分歧）
    elif vol_ratio < 0.6 and pct_chg > 0.01:  return 4  # 缩量上涨（温和走强）
    elif vol_ratio < 0.6 and pct_chg < -0.01: return 5  # 缩量下跌（阴跌）
    else:                                       return 0  # 正常
```

#### A4. 近5日K线序列模式（Sequential Pattern）— 3个特征

**原理**：Kronos 是序列模型，捕捉 K线间的时序依赖。我们用滑动窗口统计近期序列模式。

```python
# 基于 candle_body_type，对近5日编码组合
codes_5d = [candle_body_type(o,h,l,c) for o,h,l,c in zip(...)]  # 长度5的列表

# 特征1：序列模式哈希（捕捉任意组合）
seq_hash_5d = hash(tuple(codes_5d)) % 512  # int

# 特征2：近5日强势K线数量（body_type in [1,3]）
bullish_count_5d = sum(1 for c in codes_5d if c in [1, 3])  # int [0,5]

# 特征3：近5日阴线数量
bearish_count_5d = sum(1 for c in codes_5d if c in [2, 4])  # int [0,5]
```

#### A5. 归一化价格结构特征（Kronos-style Scale-Invariant）— 4个特征

**原理**：Kronos 最核心的设计是用相对比例代替绝对价格，让模型对价格水位无感。我们复制此思想。

```python
# 参考价格：过去N日均价（或前一日收盘）
ref_price = close_prev_20d_mean  # 20日均价作为参考

# 振幅相对参考价格的比例
amplitude_ratio = (high - low) / ref_price          # 今日振动强度（归一化）

# 收盘相对20日均价的偏离
close_to_ma20_ratio = (close - close_ma20) / ref_price  # 价格偏离度

# 实体相对历史波动率的比例（类似 Bollinger Band 思路）
daily_vol_5d = rolling_std(close_pct_chg, 5)
body_to_vol_ratio = abs(close - open_) / (close_prev * daily_vol_5d + 1e-8)

# 高低点相对历史高低的位置（近20日）
high_rank = (high - rolling_min(low, 20)) / (rolling_max(high, 20) - rolling_min(low, 20) + 1e-8)
```

#### A6. 连续K线模式检测（Pattern Indicators）— 5个布尔特征

```python
# 三连阳（连续3日 close > open，通常是趋势确认）
three_white_soldiers = int(close_t > open_t and close_t1 > open_t1 and close_t2 > open_t2)

# 三连阴
three_black_crows = int(close_t < open_t and close_t1 < open_t1 and close_t2 < open_t2)

# 缩量整理后放量突破（前3日vol小，今日vol大且价涨）
vol_breakout = int(vol_ma3_prev < vol_ma20 * 0.7 and volume > vol_ma20 * 1.5 and pct_chg > 0.02)

# 上影线拒绝（今日高点比昨日高点高，但收低 → 压力位）
upper_rejection = int(high > high_prev and close < close_prev and upper_shadow > 0.4)

# 下影线支撑（今日低点比昨日低点低，但收高 → 支撑位）
lower_support = int(low < low_prev and close > close_prev and lower_shadow > 0.4)
```

#### 方案A 汇总

| 特征组 | 特征数量 | 计算复杂度 | 数据依赖 |
|--------|---------|----------|---------|
| K线体型编码 | 1 | O(1) | OHLC |
| 收盘价位分位 | 2 | O(1) | OHLC |
| 量价关系编码 | 1 | O(1) | OHLCV + vol_ma20 |
| 近5日序列模式 | 3 | O(5) | 历史OHLC |
| 归一化价格结构 | 4 | O(20) | 历史Close |
| 连续K线模式检测 | 5 | O(3) | 历史OHLCV |
| **合计** | **16个** | **低** | **已有字段** |

**实现工时估算**：
- CoderBy 实现：1天（都是 pandas 向量化操作，无模型依赖）
- 历史回填（5年数据）：0.5天（逐日 rolling 计算）

**预期 AUC 提升**：
- 基于 Kronos 论文的启发，价格形态离散化特征比连续特征更鲁棒（抗噪声）
- 预期与现有动量/波动率特征互补（相关性低）
- 保守估计：AUC +0.002 ~ +0.005，IC 绝对值提升 0.005 ~ 0.01
- 若序列模式哈希特征有效（Kronos 证明K线序列信息有价值），可能更高

---

### 🟡 方案B：Kronos Embedding 接入（利用预训练模型能力）

**核心思路**：下载 Kronos-mini，对每只股票历史 90 根日线生成预测涨幅因子（或 Encoder 隐状态 embedding），作为额外特征送入 LightGBM。

#### B1. 最简接入方式：预测涨幅因子（推荐第一步）

```python
from model import Kronos, KronosTokenizer, KronosPredictor

# 一次性加载（每日运行时常驻内存）
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
predictor = KronosPredictor(model, tokenizer, device="mps", max_context=2048)

# 对每只股票生成因子
def compute_kronos_factor(stock_df, lookback=90, pred_len=5):
    """
    stock_df: columns=['open','high','low','close','volume','amount']
    返回：预测5日涨幅（float）
    """
    x_df = stock_df.iloc[-lookback:][['open','high','low','close','volume','amount']]
    x_ts = stock_df.index[-lookback:]
    
    # 生成未来5个交易日的时间戳
    future_dates = pd.bdate_range(start=stock_df.index[-1], periods=pred_len+1)[1:]
    
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_ts,
        y_timestamp=future_dates,
        pred_len=pred_len,
        T=0.8,         # 较低温度，减少随机性
        top_p=0.9,
        sample_count=10  # 采样10次取均值，减少方差
    )
    # 预测5日涨幅因子
    predicted_return = pred_df['close'].iloc[-1] / x_df['close'].iloc[-1] - 1
    return predicted_return
```

**输出特征**：
- `kronos_pred_ret_5d`：预测5日涨幅（float）← 直接作为因子
- 可以扩展：预测5日最大涨幅 `kronos_pred_high_5d`，预测波动率 `kronos_pred_vol_5d`

#### B2. 进阶方式：Encoder Embedding（更丰富的特征）

提取 Encoder 最后一层 hidden state 作为多维 embedding（需修改代码）：

```python
def extract_kronos_embedding(model, tokenizer, stock_df, lookback=90):
    """
    提取 128 维 Kronos Embedding
    """
    # 预处理（同 predict，但只跑 Encoder）
    x_normalized = normalize_kline(stock_df.iloc[-lookback:])
    tokens = tokenizer.encode(x_normalized)  # [lookback] token ids
    
    with torch.no_grad():
        # 取 Encoder 最后一个 token 的 hidden state
        hidden = model.encoder(tokens)  # [lookback, d_model]
        embedding = hidden[-1].cpu().numpy()  # [d_model] ← 最后时刻的隐状态
    
    return embedding  # shape: (d_model,) 约 256 维
    
# 降维处理
from sklearn.decomposition import PCA
pca = PCA(n_components=16)
# 先在历史数据上 fit，然后 transform
embeddings_16d = pca.fit_transform(all_embeddings)  # [n_stocks, 16]
```

#### B3 推理性能与存储评估

| 指标 | 数值 | 说明 |
|------|------|------|
| 模型加载时间 | ~2秒 | 一次性，常驻内存 |
| 单股推理（CPU，sample_count=1） | ~15ms | seq_len=90 |
| 单股推理（MPS，sample_count=1） | ~5ms | Apple Silicon 加速 |
| 5000只全量（MPS，sample_count=10） | ~250秒 ≈ 4分钟 | sample_count=10增加10x延迟 |
| 5000只全量（MPS，sample_count=1） | ~25秒 | 快速验证方案 |
| 每日存储（float32，1个因子） | 5000 × 4B = 20KB | 极小 |
| 每日存储（16维 embedding） | 5000 × 16 × 4B = 320KB | 可接受 |

**注意**：sample_count=10 采样10次取均值可以显著降低预测方差，建议在正式使用时开启。

#### B4 方案B 风险点

| 风险 | 级别 | 应对 |
|------|------|------|
| Kronos 在 A 股日线截面上未做 RankIC 验证 | ⚠️ 中 | 先在历史数据上回测验证 IC |
| 论文 RankIC 用的是分钟线，我们用日线 | ⚠️ 中 | 测试不同 pred_len（5/10/20日） |
| 当天收盘后推理需要约4分钟（MPS）| ✅ 低 | 收盘15:30后有充足时间 |
| sample_count=10 可能随机性较大 | ⚠️ 中 | 固定 random seed，多次取均值 |
| 需要 amount（成交额）字段 | ⚠️ 低 | Tushare 有，已在 pipeline 中 |

#### 方案B 实现工时估算

| 步骤 | 工时 |
|------|------|
| 安装依赖（requirements.txt），下载模型 | 0.5天 |
| 封装 KronosPredictor 成因子计算函数 | 0.5天 |
| 集成进每日 pipeline，历史回填（5年） | 1天 |
| 验证IC，与LightGBM集成 | 0.5天 |
| **合计** | **2.5天** |

---

## Part 4：综合评估与建议

### 对比矩阵

| 维度 | 方案A（离散化特征） | 方案B（Kronos Embedding） |
|------|------------------|------------------------|
| 实现成本 | ✅ 1天 | ⚠️ 2.5天 |
| 基础设施要求 | ✅ 无额外依赖 | ⚠️ 需安装 PyTorch + 下载模型 |
| 推理延迟 | ✅ 毫秒级 | ✅ ~4分钟/天（可接受） |
| 可解释性 | ✅ 高（人类可理解） | ⚠️ 低（黑盒embedding） |
| 预期 AUC 提升 | ⚠️ 温和（+0.002~0.005） | ✅ 较高（+0.005~0.01）（若IC显著） |
| 风险 | ✅ 低 | ⚠️ 中（需验证A股IC有效性） |
| 与现有特征互补性 | ✅ 高（形态特征 vs 量化特征） | ✅ 高（端到端时序 vs 手工特征） |

### ⭐ PM 建议：两步走策略

**第一步（立即执行，方案A）**：
- CoderBy 实现方案A的16个特征，加入当前 LightGBM pipeline
- 预期1天完成，风险极低，可立即开始评估

**第二步（并行验证，方案B简化版）**：
- 只实现 B1（预测涨幅因子），不做 embedding 提取，工时从 2.5天降至 1天
- 使用 `sample_count=1, pred_len=5` 快速出因子，先验证 IC 是否有效
- 若 IC > 0.03（A 股日频），则说明 Kronos zero-shot 在 A 股有效，值得投入

**不建议直接做 Fine-tune**：
- Fine-tune 需要 GPU（M 系列 Mac CPU fine-tune 4.1M 参数模型约需数小时）
- Zero-shot 已经是 +87% 的提升，先验证基础效果

---

## 附：CoderBy 实现 Checklist

### 方案A（优先）
- [ ] 实现 `candle_body_type(open, high, low, close) → int` 函数
- [ ] 实现 `close_pos(open, high, low, close) → float` + 分桶
- [ ] 实现 `vol_price_type(close, close_prev, volume, vol_ma20) → int`
- [ ] 实现近5日序列模式（需要历史5日的 candle_body_type）
- [ ] 实现4个归一化价格结构特征（需要 rolling_std, rolling_mean 20日）
- [ ] 实现5个连续K线模式布尔特征（需要历史3日 OHLCV）
- [ ] 历史回填：对 feature_daily 表 5 年数据批量计算
- [ ] 加入 LightGBM 特征集，训练验证 AUC 变化

### 方案B（并行验证）
- [ ] 安装 Kronos 依赖：`pip install -r requirements.txt`
- [ ] 下载模型：`NeoQuasar/Kronos-mini` + `NeoQuasar/Kronos-Tokenizer-2k`
- [ ] 封装 `compute_kronos_factor(stock_df) → float` 函数
- [ ] 验证单股推理正确性（对比 GitHub 示例）
- [ ] 历史回填：每个交易日对5000只股票计算因子（存入 feature_daily）
- [ ] 计算因子 IC 时间序列，评估是否有效
- [ ] 若有效，作为额外特征加入 LightGBM

---

*文件路径：`~/Documents/ai-kb/quant-research/kronos_feature_proposal.md`*
*CoderBy 可直接依据本文档开始实现。*
