# Beyond Quality: Unlocking Diversity in Ad Headline Generation with Large Language Models

**ArXiv:** 2508.18739 | **Date:** 2025-08

## 核心问题
广告标题生成中，现有方法主要优化质量/CTR，忽视**多样性**，导致输出同质化，无法覆盖不同受众细分。

## 方案：DIVER（Diversity-aware headline generation）

### 两大组件

**1. 语义与风格感知数据生成流水线**
- 自动生成高质量训练对
- 确保广告内容 + 多样化标题的配对数据
- 涵盖语义多样性和风格多样性

**2. 多阶段多目标优化框架**
- 阶段1：监督微调（SFT）奠定基础
- 阶段2：强化学习（RL）联合优化多样性和质量目标

## 线上 A/B 测试（大型内容分享平台，数亿用户）
- 广告主价值（ADVV）：**+4.0%**
- CTR：**+1.4%**

## 技术洞见
多样性和质量不是对立的，联合优化可以同时提升两者。

## 面试考点
- 广告标题多样性的业务价值？
- 如何量化文本多样性？
- RL 如何多目标优化 diversity + quality？

**Tags:** #ads #creative #headline-generation #diversity #llm #rl
