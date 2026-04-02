# Uplift Modeling（增量效果建模）

## 概述
Uplift Modeling 解决的核心问题：在广告/推荐场景中，如何识别"因为干预而转化"的用户，而非"本来就会转化"的用户。从因果推断视角估计个体处理效应（ITE），指导预算分配和人群定向。

## 目录结构

```
uplift/
├── INDEX.md                 # 本文件
├── papers/                  # 相关论文笔记
└── synthesis/
    └── Uplift建模技术演进与工业实践.md   # 核心综述
```

## 关键词
Uplift Modeling, CATE, ITE, ATE, Meta-Learner, S-Learner, T-Learner, X-Learner, DragonNet, TarNet, CFRNet, DESCN, EUEN, AUUC, Qini Curve, 因果推断, 反事实学习

## 推荐阅读顺序
1. 先读 synthesis 综述，建立全景认知
2. 重点关注 CATE 估计的数学框架
3. 对比 meta-learner 系列与端到端建模的优劣
4. 理解工业落地（双塔 uplift、DESCN）的工程权衡
