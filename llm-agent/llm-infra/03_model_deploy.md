# 模型部署基础面试考点

> 来源：AIGC-Interview-Book 模型部署基础章节
> 更新：2026-03-12

---

## 一、部署框架体系

### 1.1 主流推理框架

| 框架 | 适用场景 | 核心特点 |
|------|---------|---------|
| ONNX Runtime | 通用，跨平台 | 支持多种后端（CPU/CUDA/TensorRT） |
| TensorRT | NVIDIA GPU推理 | 算子融合、量化、极致性能 |
| Triton Inference Server | 服务化部署 | 多模型、多框架统一服务 |
| vLLM | LLM推理服务 | PagedAttention，高吞吐量 |
| SGLang | LLM推理 | RadixAttention，复杂提示高效处理 |
| LMDeploy | LLM推理 | TurboMind引擎，国内常用 |
| Ollama | 本地LLM部署 | 一键部署，面向个人用户 |
| TensorRT-LLM | NVIDIA LLM推理 | 融合TensorRT+LLM优化 |

### 1.2 ONNX（Open Neural Network Exchange）

**作用：** 模型格式中间件，解耦训练框架和推理框架

**流程：**
```
PyTorch/TensorFlow训练 → 导出ONNX → ONNX Runtime / TensorRT推理
```

**导出示例：**
```python
torch.onnx.export(
    model, dummy_input, "model.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}}
)
```

**优化：**
- onnxsim：图化简（常量折叠、冗余节点消除）
- ONNXRuntime图优化级别：Basic/Extended/All

### 1.3 TensorRT

**核心优化手段：**

1. **算子融合（Layer Fusion）**：Conv + BN + ReLU → 单一算子，减少Memory带宽
2. **精度校准（Quantization）**：FP32 → INT8/FP16
3. **Kernel Auto-tuning**：针对具体GPU架构选最优CUDA Kernel
4. **动态形状（Dynamic Shape）**：支持变长输入

**部署流程：**
```
PyTorch模型 → ONNX → TensorRT Builder → Engine（.plan文件）→ 推理
```

**命令行工具：**
```bash
trtexec --onnx=model.onnx --saveEngine=model.plan --fp16 --int8
```

### 1.4 Triton Inference Server

**特点：**
- 支持多框架（TensorRT/ONNX/PyTorch/TensorFlow）
- 动态Batching：自动聚合请求提升GPU利用率
- 多模型并发服务
- REST/gRPC API

**config.pbtxt核心配置：**
```
name: "my_model"
platform: "onnxruntime_onnx"  # 或 tensorrt_plan
max_batch_size: 8
instance_group [{ count: 1, kind: KIND_GPU, gpus: [0] }]
```

---

## 二、LLM 推理优化

### 2.1 vLLM 核心技术

**PagedAttention：**
- 问题：KV Cache需要连续显存，导致碎片化和提前OOM
- 解决：将KV Cache分成固定大小的Page（类比OS虚拟内存），按需分配
- 效果：显存利用率从≈40%提升到≈96%，吞吐量大幅提升

**Continuous Batching：**
- 传统：静态batch，等所有序列完成再处理下批
- Continuous：逐token动态调度，完成的序列立即释放，新请求填入

### 2.2 SGLang

**RadixAttention：**
- 对有共同前缀的请求共享KV Cache（Radix Tree结构）
- 适合系统提示（System Prompt）相同的场景

**前端（SGL Language）：**
- 结构化生成语言，支持分支/并发/循环

### 2.3 LMDeploy / TurboMind

- 量化：W4A16（权重INT4，激活FP16）
- 连续批处理 + 动态split-fuse
- 适合国内部署场景

---

## 三、模型压缩

### 3.1 量化（Quantization）

**两种方式：**

| 方式 | 时机 | 代表方法 |
|------|------|---------|
| PTQ（Post-Training Quantization）| 训练后 | GPTQ, AWQ, LLM.int8() |
| QAT（Quantization-Aware Training）| 训练中 | 模拟量化，更高精度 |

**主流量化：**
- **GPTQ**：逐层量化，补偿误差，适合W4/W8
- **AWQ**：保留重要权重通道，精度损失小
- **LLM.int8()**：混合精度，异常值FP16，其余INT8

**量化等式：**
$$
W_{int} = \text{clamp}\left(\text{round}\left(\frac{W}{s}\right), -128, 127\right)
$$

### 3.2 剪枝（Pruning）

| 类型 | 说明 | 适用 |
|------|------|------|
| 非结构化剪枝 | 置零单个权重 | 压缩率高但不规则，需稀疏库 |
| 结构化剪枝 | 移除整个channel/head | 直接降维，推理加速明显 |
| 迭代剪枝 | 剪枝→微调→迭代 | 精度损失可控 |

**重要性度量：**
- 权重大小（L1/L2范数）
- 激活值方差
- 泰勒展开的梯度信息

### 3.3 知识蒸馏（Knowledge Distillation）

**基本框架：**
```
Teacher Model（大）→ Soft Labels（软标签）
                        ↓
Student Model（小）← 监督学习（Hard Labels + Soft Labels）
```

**损失函数：**
$$
L = (1-\alpha)\cdot L_{CE}(y, \hat{y}) + \alpha \cdot T^2 \cdot L_{KL}(p_T, p_S)
$$

- $T$：温度参数，T越大软标签越平滑
- $\alpha$：软标签权重

**中间层蒸馏：** 对齐Teacher和Student的中间层表示（FitNets）

**LLM蒸馏场景：**
- 大模型 → 小模型（如GPT-4 → GPT-3.5）
- Token概率/Logits对齐
- 链式推理蒸馏（CoT蒸馏）

---

## 四、推理延迟优化方案

### 4.1 算法层

| 优化方法 | 效果 |
|---------|------|
| 量化（INT8/INT4）| 推理速度2-4x |
| 模型剪枝 | 减少计算量 |
| 投机采样 | 吞吐量提升2-3x |
| FlashAttention | 减少IO，速度2x+ |
| 算子融合 | 减少Kernel Launch开销 |

### 4.2 系统层

| 优化方法 | 说明 |
|---------|------|
| 动态Batching | 聚合请求，提升GPU利用率 |
| 连续Batching | 逐token调度 |
| 张量并行 | 多GPU分担计算 |
| 预填充/解码分离 | Prefill和Decode不同优化策略 |
| 缓存优化 | PagedAttention、RadixAttention |

### 4.3 硬件层

- CUDA Kernel优化（Triton语言）
- 利用Tensor Core（需要矩阵维度对齐）
- NVLink高速互联（多GPU通信）
- 异步流水线（计算/IO重叠）

---

## 五、分布式训练并行方式

| 并行类型 | 说明 | 典型框架 |
|---------|------|---------|
| 数据并行（DP）| 每卡完整模型，不同数据 | DDP（PyTorch）|
| 张量并行（TP）| 矩阵按列/行分片 | Megatron-LM |
| 流水线并行（PP）| 层级分卡 | GPipe, PipeDream |
| 专家并行（EP）| MoE专家分卡 | DeepSpeed-MoE |
| ZeRO | 优化器/梯度/参数分片 | DeepSpeed ZeRO |

**ZeRO三个阶段：**
- Stage 1：分片优化器状态
- Stage 2：分片梯度
- Stage 3：分片模型参数（显存最省，通信最多）

---

## 六、高频面试题

1. **ONNX的作用是什么？**
   → 模型中间格式，解耦训练框架和推理框架，支持跨平台部署

2. **TensorRT如何加速推理？**
   → 算子融合、精度量化（INT8）、Kernel自动调优、减少显存访问

3. **vLLM的PagedAttention解决了什么问题？**
   → KV Cache显存碎片化问题；通过分页管理显存利用率从40%→96%

4. **知识蒸馏的温度参数T有什么作用？**
   → T越大，软标签越平滑（不同类别概率差异缩小），Student能学到更多"暗知识"

5. **PTQ和QAT的区别？**
   → PTQ训练后量化，简单快速；QAT训练中模拟量化，精度更高但需重训

6. **结构化剪枝和非结构化剪枝的区别？**
   → 结构化：移除整行/列，可直接降维加速；非结构化：零散置零，需稀疏硬件支持

7. **Triton推理服务器的动态Batching是什么？**
   → 将多个独立请求在服务端聚合成一个Batch推理，提升GPU利用率

8. **ZeRO-3和普通数据并行的区别？**
   → ZeRO-3将参数、梯度、优化器状态全部分片，每卡只存1/N；代价是通信量增加
