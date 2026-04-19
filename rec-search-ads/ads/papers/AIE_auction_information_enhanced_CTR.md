# AIE: Auction Information Enhanced Framework for CTR Prediction in Online Advertising
> 来源：arXiv:2408.07907 | 领域：ads | 学习日期：20260419

## 核心方法
1. **拍卖信息融合**：将竞价环境的后验信息（竞争程度、出价分布）融入CTR预估
2. **去偏设计**：解决拍卖信息引入的数据偏差问题
3. **在线广告场景**：CTR预估不仅依赖用户-广告匹配，还受竞价环境影响

## 面试考点
- Q: 拍卖信息为什么能提升CTR预估？
  - A: 竞争激烈的广告位展示的广告质量更高（选择偏差），拍卖信息帮助模型感知这种环境效应
