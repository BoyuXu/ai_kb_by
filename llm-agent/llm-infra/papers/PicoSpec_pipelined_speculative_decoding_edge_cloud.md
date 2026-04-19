# PicoSpec: Pipelined Collaborative Speculative Decoding for Edge-Cloud LLM Inference
> 来源：arXiv:2603.19133 | 领域：llm-infra | 学习日期：20260419

## 问题定义
LLM推理面临端云协同难题：纯云端成本高，纯端侧受限于设备性能。投机解码（Speculative Decoding）可加速，但在端云场景中存在互等问题——端侧draft等云端verify，云端verify等端侧draft。

## 核心方法与创新点
1. **异步流水线（Asynchronous Pipeline）**：解耦端侧drafting和云端verification，消除互等
2. **Training-Free**：无需重训练，通用框架适配标准模型
3. **独立Rejection Sampling**：针对通信开销优化的拒绝采样算法
4. **隐藏网络延迟**：端侧持续投机而不等待云端反馈

## 实验结论
- 最高 **2.9x** 加速
- 首个training-free的异步分布式投机推理框架

## 面试考点
- Q: 投机解码的核心思想？
  - A: 小模型快速生成draft tokens → 大模型并行verify → 接受正确token/拒绝错误token，保证输出分布不变
- Q: 端云协同推理的挑战？
  - A: ①网络延迟不确定；②带宽有限（传输KV cache开销大）；③端侧计算能力受限
