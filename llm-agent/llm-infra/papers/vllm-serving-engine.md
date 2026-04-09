# vLLM: Easy, Fast, and Cheap LLM Serving

- **Type**: Open-Source Engine
- **URL**: https://github.com/vllm-project/vllm

## 核心技术

- **PagedAttention**: 类OS虚拟内存管理KV cache，减少80%GPU内存浪费
- **V1架构**: 多进程（scheduler/engine core/GPU workers），ZeroMQ通信，1.7x吞吐提升
- **最新特性**: Gemma 4支持、零气泡异步调度+推测解码、ViT CUDA Graphs

## 硬件支持

NVIDIA, AMD ROCm, Intel XPU/Gaudi, Google TPU, AWS Trainium, ARM CPUs

## 面试考点

PagedAttention原理，KV cache管理策略，推测解码加速。
