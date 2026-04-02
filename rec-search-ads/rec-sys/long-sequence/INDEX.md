# 长序列用户行为建模

## 概述
用户行为序列是推荐系统最重要的信号源之一。随着数据积累，序列长度从几十（DIN）扩展到万级（SIM）甚至百万级（HSTU），如何在延迟约束下高效建模超长序列成为工业核心挑战。

## 目录结构

```
long-sequence/
├── INDEX.md                 # 本文件
├── papers/                  # 相关论文笔记
└── synthesis/
    └── 长序列用户行为建模技术演进.md   # 核心综述
```

## 关键词
DIN, DIEN, SIM, ETA, HSTU, Mamba4Rec, SSM, Target Attention, GRU, AUGRU, GSU, ESU, RoPE, YaRN, 长序列建模, 用户行为序列

## 推荐阅读顺序
1. DIN → DIEN：理解 target attention 和序列兴趣演化
2. SIM → ETA：理解万级序列的工程解法
3. HSTU → Mamba4Rec：理解 Transformer/SSM 在序列推荐的前沿
4. 长上下文扩展：理解位置编码外推技术
