# Fast and Faithful: Real-Time Verification for Long-Document RAG Systems

**ArXiv:** 2603.23508 | **Date:** 2026-03

## 核心问题
生产环境 RAG 流水线中，长文档溯源验证面临速度与覆盖率的权衡：
- 大型 LLM 可检查长上下文，但对交互式服务太慢
- 轻量分类器速度快但上下文窗口受限，遗漏截断段落外的证据

## 核心方案
实时验证组件集成到生产 RAG 流水线，支持全文档溯源，满足延迟约束。

## 关键技术
- **处理文档长度**：最长 32K tokens
- **自适应推理策略**：根据工作负载动态平衡响应时间和验证覆盖率
- 在延迟约束下实现全文档定位（full-document grounding）

## 工业意义
在生产系统中保证 RAG 输出的真实性（faithfulness）是企业部署的关键需求，本文提供了可实用的实时解决方案。

## 面试考点
- RAG faithfulness 的定义和常见度量方式？
- 如何在延迟和验证覆盖率之间做权衡？
- Long-document RAG 有哪些特殊挑战？

**Tags:** #llm-infra #rag #faithfulness #verification #production
