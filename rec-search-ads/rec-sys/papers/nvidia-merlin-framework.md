# NVIDIA Merlin: End-to-End GPU-Accelerated Recommender System Framework

- **Type**: Open-Source Framework
- **URL**: https://github.com/NVIDIA-Merlin/Merlin

## 核心组件

- **NVTabular**: 表格数据特征工程，比CPU快10x
- **HugeCTR**: GPU加速训练框架，支持大规模embedding表分布式训练
- **Transformers4Rec**: 序列/会话推荐
- **Triton**: 推理服务

## 关键技术

端到端GPU加速pipeline：数据预处理→模型训练→推理部署。支持TensorFlow/PyTorch/HugeCTR多框架。

## 面试考点

推荐系统GPU加速的端到端方案，embedding表的分布式切分策略。
