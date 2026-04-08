# Hydra: Unifying Document Retrieval and Generation in a Single VLM

> 来源：arXiv 2026 | 领域：search | 学习日期：20260408

## 问题定义

文档检索和文档生成（摘要、QA）通常使用独立模型，资源浪费。

**核心问题**：能否用单一视觉语言模型同时完成检索和生成？

## 核心方法与创新点

1. **Dual-Head Architecture**：
   - 检索头 + 生成头，共享 4B 参数基座
   - 参数共享率高

2. **Adapter Toggle**：
   - 关键发现：生成训练是不必要的
   - 切换适配器即可恢复生成能力

3. **内存效率**：
   - GPU 内存减少 41%
   - 性能接近基线 (nDCG@5: 0.8842 vs 0.8892)

## 工程启示

- 多任务文档处理系统的参数共享策略
- 减少部署成本的实用方案
