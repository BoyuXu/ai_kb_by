# 高频交易与市场微观结构

> 市场微观结构研究「价格如何形成」。理解订单簿、买卖价差和订单流，是高频策略和执行算法的基础。

## 1. 限价订单簿（LOB）结构

| 层级 | 内容 | A股可用性 | 用途 |
|------|------|-----------|------|
| Level 1 | 最优买卖价(BBO) + 量 | 所有终端 | 基础信号 |
| Level 2 | 多档买卖盘（5档/10档） | 付费行情 | 深度分析 |
| Level 3 | 逐笔委托明细 | 上交所SSEL2 | 订单流建模 |

- **快照模式**：定时推送完整订单簿（A股常用，如3秒一次）
- **增量模式**：只推送变化部分，带宽效率高。实际系统通常增量+定期快照校验

```python
import pandas as pd
from collections import defaultdict

class SimpleOrderBook:
    """简单限价订单簿模拟"""
    def __init__(self):
        self.bids = defaultdict(float)  # price -> quantity
        self.asks = defaultdict(float)

    def add_order(self, side: str, price: float, qty: float):
        book = self.bids if side == 'buy' else self.asks
        book[price] += qty

    def cancel_order(self, side: str, price: float, qty: float):
        book = self.bids if side == 'buy' else self.asks
        book[price] = max(0, book[price] - qty)
        if book[price] == 0: del book[price]

    def best_bid(self): return max(self.bids.keys()) if self.bids else None
    def best_ask(self): return min(self.asks.keys()) if self.asks else None
    def spread(self):
        bb, ba = self.best_bid(), self.best_ask()
        return (ba - bb) if (bb and ba) else None

    def snapshot(self, levels=5):
        bid_px = sorted(self.bids.keys(), reverse=True)[:levels]
        ask_px = sorted(self.asks.keys())[:levels]
        return pd.DataFrame({
            'bid_qty': [self.bids[p] for p in bid_px],
            'bid_price': bid_px,
            'ask_price': ask_px, 'ask_qty': [self.asks[p] for p in ask_px],
        })

ob = SimpleOrderBook()
for p, q in [(10.05,500),(10.04,1200),(10.03,800)]: ob.add_order('buy', p, q)
for p, q in [(10.06,300),(10.07,900),(10.08,600)]: ob.add_order('sell', p, q)
print(f"BBO: {ob.best_bid()} x {ob.best_ask()}, Spread: {ob.spread():.2f}")
```

## 2. 关键微观结构概念

### 2.1 买卖价差（Bid-Ask Spread）
- **绝对价差**：$S = P_{ask} - P_{bid}$
- **相对价差**：$S_{rel} = (P_{ask} - P_{bid}) / M$，$M$为中间价
- **有效价差**：实际成交价与中间价的偏差，反映真实交易成本

价差三要素分解：(1) 订单处理成本；(2) 存货成本——做市商持仓风险补偿；(3) **逆向选择成本**——被知情交易者挑走的亏损补偿。

### 2.2 市场深度（Market Depth）
各价格档位挂单总量。深度大 → 大单冲击小 → 流动性好。注意「幻影流动性」：挂单可随时撤销。

### 2.3 订单流不平衡（OFI）
价格变动的直接驱动力是**主动买与主动卖的不平衡**：
$$OFI_t = \sum_{i} (\Delta BidQty_i - \Delta AskQty_i)$$

> **面试常问**：OFI 与价格变动的关系？OFI 为正表示买方力量增强，与短期价格变动显著正相关。

### 2.4 VWAP / TWAP
- **VWAP** = $\sum P_i V_i / \sum V_i$：机构评估执行质量的基准，按历史日内成交量分布拆单
- **TWAP** = $\frac{1}{N}\sum P_i$：等时间间隔均匀拆单，适用于缺乏成交量预测的场景

> **面试常问**：VWAP策略的原理和局限？见第6节。

### 2.5 Kyle's Lambda
$$\Delta P = \lambda \cdot OFI + \epsilon$$
$\lambda$ 是价格对订单流的敏感度。$\lambda$ 越大 → 流动性越差。通过线性回归估计。

### 2.6 Python：计算 OFI 和 Kyle's Lambda

```python
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

def compute_ofi(quotes: pd.DataFrame) -> pd.Series:
    """quotes需含 bid_price, bid_size, ask_price, ask_size"""
    bp_up = quotes['bid_price'] >= quotes['bid_price'].shift(1)
    bp_dn = ~bp_up
    ap_dn = quotes['ask_price'] <= quotes['ask_price'].shift(1)
    ap_up = ~ap_dn
    delta_bid = np.where(bp_up, quotes['bid_size'],
                np.where(bp_dn, -quotes['bid_size'].shift(1),
                         quotes['bid_size'] - quotes['bid_size'].shift(1)))
    delta_ask = np.where(ap_dn, quotes['ask_size'],
                np.where(ap_up, -quotes['ask_size'].shift(1),
                         quotes['ask_size'] - quotes['ask_size'].shift(1)))
    return pd.Series(delta_bid - delta_ask, index=quotes.index, name='OFI').fillna(0)

def estimate_kyle_lambda(price_chg, ofi):
    model = LinearRegression().fit(ofi.reshape(-1,1), price_chg)
    return model.coef_[0]

# 示例
np.random.seed(42); n = 1000
quotes = pd.DataFrame({
    'bid_price': 10.0 + np.cumsum(np.random.randn(n)*0.01),
    'bid_size': np.random.randint(100,1000,n),
    'ask_price': 10.01 + np.cumsum(np.random.randn(n)*0.01),
    'ask_size': np.random.randint(100,1000,n),
})
quotes['mid'] = (quotes['bid_price'] + quotes['ask_price']) / 2
ofi = compute_ofi(quotes)
lam = estimate_kyle_lambda(quotes['mid'].diff().fillna(0).values[1:], ofi.values[1:])
print(f"Kyle's Lambda: {lam:.6f}")
```

## 3. 做市策略基本原理

**做市商角色**：同时挂买卖单，赚取价差，为市场提供流动性（immediacy）。

**两大风险**：
1. **Inventory Risk**：持仓偏离零点 → 方向性风险。需动态调整报价倾斜
2. **Adverse Selection**：知情交易者掌握价格信息，与其成交几乎必然亏损。做市利润来自不知情交易者

> **面试常问**：做市商如何盈利？什么是逆向选择？
> 赚取买卖价差，前提是与不知情交易者成交的利润覆盖与知情交易者的亏损。逆向选择是知情交易者利用信息优势在有利价格成交，导致做市商系统性亏损。

**Avellaneda-Stoikov 模型（2008）**：
- 保留价格：$r = s - q \gamma \sigma^2 (T-t)$（$s$中间价，$q$持仓，$\gamma$风险厌恶）
- 最优价差：$\delta^* = \gamma \sigma^2(T-t) + \frac{2}{\gamma}\ln(1+\gamma/\kappa)$
- 核心洞察：持仓越偏→报价越倾斜；波动率越高→价差越宽；临近收盘→价差收窄

## 4. Tick 数据处理

### 4.1 清洗要点
| 问题 | 处理 |
|------|------|
| 异常值 | 偏离中位数>N个标准差则剔除 |
| 零价格/零量 | 直接删除 |
| 拆分/除权 | 前复权或后复权 |
| 集合竞价 | 分开处理或剔除（9:15-9:25, 14:57-15:00）|

### 4.2 时钟选择
- **Calendar Time**：等时间间隔采样，最常用但信息利用率低
- **Trade Time**：每N笔成交采样，活跃期密、清淡期疏
- **Volume Time**：每累积V手采样，Lopez de Prado推荐——采样后收益率更接近正态分布
- **Dollar Bar**：每累积D元采样，考虑价格变化影响。参见 [[ml_in_quant]]

### 4.3 特征提取
- **已实现波动率**：$RV = \sqrt{\sum r_i^2}$
- **Trade Imbalance**：$(V_{buy}-V_{sell})/(V_{buy}+V_{sell})$
- **成交速率**：单位时间成交笔数/量
- **订单簿斜率**：各档挂单量随价格的衰减速度

### 4.4 Python：Tick 聚合为 OHLCV

```python
import pandas as pd, numpy as np

def tick_to_ohlcv(ticks, freq='1min'):
    t = ticks.set_index('timestamp')
    ohlcv = t['price'].resample(freq).agg(open='first',high='max',low='min',close='last')
    ohlcv['volume'] = t['volume'].resample(freq).sum()
    ohlcv['vwap'] = (t['price']*t['volume']).resample(freq).sum() / ohlcv['volume']
    return ohlcv.dropna(subset=['open'])

def tick_to_volume_bars(ticks, vol_threshold=10000):
    t = ticks.copy()
    t['bar_id'] = t['volume'].cumsum() // vol_threshold
    bars = t.groupby('bar_id').agg(
        timestamp=('timestamp','first'), open=('price','first'),
        high=('price','max'), low=('price','min'), close=('price','last'),
        volume=('volume','sum'), trades=('price','count'))
    return bars

# 示例
np.random.seed(42); n = 5000
ticks = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-15 09:30', periods=n, freq='200ms'),
    'price': 15.0 + np.cumsum(np.random.randn(n)*0.005),
    'volume': np.random.randint(1,50,n)*100,
})
print(f"1min K线: {len(tick_to_ohlcv(ticks))}条, Volume bars: {len(tick_to_volume_bars(ticks, 50000))}条")
```

## 5. A股高频限制与现实

| 约束 | A股 | 美股 |
|------|-----|------|
| 日内回转 | **T+1** | T+0 |
| 最小变动 | 0.01元 | 0.01美元 |
| 涨跌幅 | 主板10%/创业板20% | 无（有熔断） |
| 撤单限制 | 频繁撤单标记异常 | 相对宽松 |
| 做市商 | 科创板/北交所有 | 成熟制度 |

**A股高频策略方向**：
- **ETF套利**：ETF价格与成分股净值偏差。溢价→买成分股+申购ETF+卖出；折价反之
- **期现套利**：股指期货与现货基差套利。基差过大→做空期货+做多现货
- **统计套利**：配对交易，价差偏离做均值回复，可持仓过夜。参见 [[strategy_development]]
- **底仓T+0**：持有底仓后日内高卖低买做差价；融券T+0受限于额度

## 6. 面试考点汇总

> **面试常问**：做市商如何盈利？

赚取买卖价差。利润=与不知情交易者成交赚的spread - 与知情交易者成交的逆向选择亏损。需控制存货风险，避免持仓偏离。

> **面试常问**：什么是逆向选择？

知情交易者掌握未反映在价格中的信息，倾向于在价格即将涨时买做市商卖单、即将跌时卖给做市商买单。做市商无法区分，必须加宽价差补偿。

> **面试常问**：VWAP策略的原理和局限？

**原理**：预测日内成交量U型分布，按比例拆单，使执行均价贴近全天VWAP。
**局限**：(1) 成交量偏离历史时效果差；(2) 无alpha，被动跟随；(3) 大单自身推动VWAP；(4) 不适合有时效性的信息驱动交易。

---

**相关文档**：[[factor_investing]] | [[ml_in_quant]] | [[strategy_development]]
