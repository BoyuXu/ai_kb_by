# X Recommendation Algorithm (Twitter the-algorithm)

- **Date**: 2023 (open-sourced), updated 2025-09
- **Type**: Open-Source Repo
- **URL**: https://github.com/twitter/the-algorithm

## 核心内容

X开源"For You"推荐Feed源码。近1000个源文件，~65000行Scala代码。覆盖For You Timeline和推荐通知两个产品面。

## 关键架构

候选生成 → 排序 → 重排序的经典三阶段pipeline。包含实时特征、社交图谱信号、用户行为信号的多维度特征体系。

## 工业实践价值

真实大规模社交媒体推荐系统的完整代码参考，对理解工业级推荐系统架构极有价值。
