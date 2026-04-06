# GenFacet: End-to-End Generative Faceted Search via Multi-Task Preference Alignment

**ArXiv:** 2603.19665 | **Date:** 2026-03 | **Org:** JD.com | **Venue:** E-Commerce

## 核心问题
电商 faceted search（分面导航）的两个痛点：
1. Facet 生成不够动态，无法响应实时趋势
2. 用户交互到精准搜索查询的转化不够智能

## 核心方案
工业级端到端生成框架，基于统一 LLM 将 faceted search 重构为两个耦合生成任务：

### Task 1: Context-Aware Facet Generation
动态合成趋势响应型导航选项（facets）。

### Task 2: Intent-Driven Query Rewriting
将用户的 facet 交互转化为精准搜索查询，闭合检索循环。

## 训练方法
多任务训练流水线：**Teacher-Student Distillation + GRPO**，直接优化下游搜索满意度。

## 线上效果（JD.com 部署）
- Facet CTR：相对提升 **42.0%**
- 用户转化率（UCVR）：相对提升 **2.0%**

## 面试考点
- Faceted search 的业务价值？
- GRPO 如何用于多任务偏好对齐？
- 端到端 vs 流水线式系统的权衡？

**Tags:** #search #faceted-search #generative #multi-task #e-commerce #jd
