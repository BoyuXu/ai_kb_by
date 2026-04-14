# RTP-LLM: Alibaba High-Performance LLM Inference Engine

- **Type**: Open-Source Engine
- **URL**: https://github.com/alibaba/rtp-llm

## 核心技术

- **高性能CUDA内核**: PagedAttention, FlashAttention, FlashDecoding
- **WeightOnly量化**: INT8/INT4 (GPTQ/AWQ)
- **多LoRA部署**: 单模型实例服务多LoRA
- **多模态**: 图片+文本输入
- **多机多卡**: Tensor Parallelism

## 工业部署

淘宝、天猫、饿了么等阿里业务线全面部署，服务数十亿用户。

## 面试考点

LLM推理加速技术栈，量化方法对比，多LoRA serving架构。
