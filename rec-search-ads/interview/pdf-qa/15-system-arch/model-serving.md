# 模型部署与推理优化：Serving 框架、量化、Batching、系统优化

## 1. 模型服务框架

### 1.1 主流框架对比
```
框架           支持格式            核心优势                    适用场景
TF Serving    SavedModel         TF 原生/版本管理/gRPC       TF 生态团队
Triton        ONNX/TRT/多格式    多框架通吃/动态Batching     混合框架/GPU密集
TorchServe    .mar (PyTorch)     自定义handler/灵活          PyTorch 生态
ONNX Runtime  .onnx              跨框架标准/CPU推理优化      CPU部署/轻量场景

选型决策树：
  TF 模型 + 成熟生态 → TF Serving
  PyTorch 模型 + GPU 推理 → Triton (ONNX→TRT)
  PyTorch 模型 + 灵活部署 → TorchServe
  CPU 推理 + 跨框架 → ONNX Runtime
  多框架混用 → Triton（统一管理）
```

### 1.2 模型格式转换链路
```
PyTorch → ONNX → TensorRT (GPU 推理最优路径)
  torch.onnx.export(model, dummy_input, "model.onnx",
      dynamic_axes={"input": {0: "batch_size"}})
  trtexec --onnx=model.onnx --saveEngine=model.trt --fp16

PyTorch → TorchScript → TorchServe (PyTorch 生态)
  scripted = torch.jit.script(model)  # 或 torch.jit.trace
  scripted.save("model.pt")

TF → SavedModel → TF Serving (TF 生态)
  tf.saved_model.save(model, "saved_model/1/")

转换注意事项：
  1. 动态维度声明：batch_size 和 序列长度必须标记为动态
  2. 算子兼容性：自定义算子可能不被 ONNX/TRT 支持
     → 注册 custom op 或用支持的算子改写
  3. 精度验证：转换后必须比对输出
     max_diff = max(abs(output_original - output_converted))
     要求：max_diff < 1e-5 (FP32) 或 < 1e-3 (FP16)
  4. 性能基准测试：转换后 benchmark 延迟和吞吐量
```

### 1.3 模型版本管理与热更新
```
版本管理：
  目录结构：
    model_repo/
    ├── model_a/
    │   ├── 1/   (v1)
    │   ├── 2/   (v2, 当前服务)
    │   └── 3/   (v3, 灰度中)
    └── config.pbtxt

部署策略：
  蓝绿部署：v2 全量 → v3 全量（一次性切换）
  金丝雀：v3 先 5% 流量 → 观察 1h → 逐步扩到 100%
  影子模式：v3 与 v2 并行推理，只用 v2 结果，对比两者差异

模型热更新机制：
  双 Buffer 方案：
    1. 当前模型在 Buffer A 服务
    2. 后台加载新模型到 Buffer B
    3. 加载完成后原子切换（指针交换）
    4. 旧 Buffer A 延迟回收（等待在途请求完成）

  优势：零停机时间，请求无感知
  TF Serving / Triton 原生支持
```

---

## 2. 模型量化

### 2.1 训练后量化（PTQ, Post-Training Quantization）
```
原理：
  FP32 权重/激活值 → INT8/FP16
  量化公式：q = round(x / scale) + zero_point
  反量化：x_approx = (q - zero_point) * scale
  scale = (x_max - x_min) / (2^bits - 1)

类型：
  动态量化：推理时动态计算激活值的 scale
    优点：不需要校准数据
    缺点：每次推理额外计算 scale 的开销

  静态量化：用校准数据集预先确定 scale
    步骤：跑一轮推理收集激活值分布 → 确定 scale/zero_point
    优点：推理时无额外开销
    缺点：需要有代表性的校准数据

效果：
  FP32 → FP16：精度损失 < 0.1%，速度 2x
  FP32 → INT8：精度损失 < 1%，速度 2-4x
  模型体积：直接缩小到 1/2 (FP16) 或 1/4 (INT8)
```

### 2.2 量化感知训练（QAT, Quantization-Aware Training）
```
原理：在训练过程中模拟量化误差

训练流程：
  前向传播：权重/激活值做模拟量化（加入量化噪声）
  反向传播：使用直通估计器（STE）传递梯度
  效果：模型学会适应量化带来的精度损失

代码框架（PyTorch）：
  model = torch.quantization.prepare_qat(model)
  for batch in dataloader:
      output = model(batch)       # 前向带模拟量化
      loss.backward()             # STE 反向传播
  model = torch.quantization.convert(model)

对比 PTQ：
  PTQ：简单快速，精度损失略大
  QAT：训练成本高，精度损失更小
  推荐：先试 PTQ，精度不够再用 QAT
```

### 2.3 混合精度策略
```
不同层使用不同精度：
  Embedding 层：保持 FP32（查表操作，量化收益小但风险大）
  全连接层：FP16（计算密集，量化收益大）
  注意力层：FP16（矩阵乘法加速明显）
  输出层：FP32（保证最终预测精度）

实现（TensorRT）：
  parser.set_dtype(layer_embedding, trt.float32)
  parser.set_dtype(layer_fc, trt.float16)
```

---

## 3. 推理加速

### 3.1 动态 Batching
```
原理：
  单个推理请求 → 不充分利用 GPU 并行能力
  聚合多个请求 → 一次前向传播 → 提升吞吐量

Triton 动态 Batching 配置：
  dynamic_batching {
    preferred_batch_size: [8, 16, 32]
    max_queue_delay_microseconds: 1000  # 最大等待 1ms
  }

权衡：
  batch 越大 → 吞吐量越高，但延迟增加
  max_queue_delay 越大 → batch 越满，但等待时间越长
  推荐配置：max_delay = P99_budget * 0.1

实际效果：
  batch=1  → GPU 利用率 10-20%，延迟 5ms
  batch=32 → GPU 利用率 70-90%，延迟 8ms（吞吐量 6x）
```

### 3.2 TensorRT 优化
```
优化手段：

算子融合（Layer Fusion）：
  Conv + BN + ReLU → 一个融合 kernel
  减少 GPU kernel launch 次数和中间结果显存占用
  自动完成，无需手动

Kernel 自动调优（Auto-Tuning）：
  对每个算子尝试多种实现
  选择在当前 GPU 上最快的版本
  首次构建 engine 时耗时（几分钟~几小时），之后复用

FP16/INT8 推理：
  TensorRT 自动选择精度策略
  INT8 需要提供校准数据集

构建命令：
  trtexec --onnx=model.onnx \
          --saveEngine=model.trt \
          --fp16 \
          --workspace=4096 \
          --optShapes=input:16x128 \
          --minShapes=input:1x128 \
          --maxShapes=input:64x128
```

### 3.3 模型压缩组合策略
```
压缩手段组合顺序（按 ROI 排序）：

第一步：知识蒸馏
  Teacher（大模型）→ Student（小模型）
  蒸馏损失 = alpha * KL(soft_teacher, soft_student)
              + (1-alpha) * CE(hard_label, student)
  温度 T 控制 softmax 平滑度
  效果：模型参数量减少 5-10x，精度损失 1-3%

第二步：结构化剪枝
  剪掉整行/整列/整个注意力头
  训练 → 剪枝 → 微调 → 再剪枝（迭代进行）
  效果：参数量进一步减少 2-4x

第三步：量化
  FP32 → FP16/INT8
  效果：推理速度再提升 2-4x

三步叠加效果：
  原始模型 100ms → 蒸馏后 30ms → 剪枝后 15ms → 量化后 5ms
```

---

## 4. 微服务部署架构

### 4.1 推荐系统微服务拆分
```
服务拆分（按推荐阶段）：

请求入口 → API Gateway（鉴权/限流/路由）
    |
    v
推荐调度服务（编排全链路）
    |
    ├──→ 召回服务（多路并行 RPC）
    |      ├─ 协同过滤召回
    |      ├─ 向量召回（ANN）
    |      ├─ 热门召回
    |      └─ 实时召回
    |
    ├──→ 特征服务（批量特征查询）
    |
    ├──→ 排序服务（模型推理）
    |      ├─ 粗排
    |      └─ 精排
    |
    └──→ 重排服务（业务规则 + 多样性）

延迟预算分配：
  召回 10ms + 特征查询 5ms + 粗排 5ms + 精排 15ms + 重排 5ms = 40ms
  各阶段独立超时控制
```

### 4.2 服务治理
```
熔断（Circuit Breaker）：
  监控下游错误率 → 超阈值自动断开 → 走降级逻辑
  半开状态：定时放少量请求试探恢复
  实现：Hystrix / Sentinel / Istio

限流（Rate Limiting）：
  令牌桶：固定速率发放令牌，请求消耗令牌
  漏桶：固定速率处理请求，超出排队或丢弃
  分级限流：VIP 用户更高配额

超时控制：
  每个 RPC 调用设置独立超时
  总超时 = 各阶段超时之和 + buffer
  超时后走降级策略而非等待

降级策略优先级：
  1. 缓存兜底（返回上次推荐结果）
  2. 简化模型（跳过精排，用粗排直接返回）
  3. 热门推荐（全局热门，无个性化）
  4. 静态列表（运营预配置的固定推荐）
```

### 4.3 负载均衡
```
接入层：
  Nginx/HAProxy 四层/七层负载均衡
  算法：加权轮询 / 最少连接 / 一致性哈希

服务间：
  gRPC 负载均衡（客户端/代理/服务端）
  服务发现：Consul / Eureka / K8s Service
  健康检查：定时心跳 + 主动探活

亲和性：
  同一用户的请求路由到同一实例（利用本地缓存）
  一致性哈希：user_id → 固定实例
  优势：提高本地缓存命中率
```

---

## 5. K8s 弹性伸缩

### 5.1 HPA（Horizontal Pod Autoscaler）
```
基于指标自动调整 Pod 副本数：

内置指标：
  CPU 利用率 > 70% → 扩容
  内存利用率 > 80% → 扩容

自定义指标（推荐系统更实用）：
  QPS > 阈值 → 扩容
  推理队列长度 > N → 扩容
  GPU 利用率 > 80% → 扩容

配置示例：
  apiVersion: autoscaling/v2
  spec:
    minReplicas: 3
    maxReplicas: 20
    metrics:
    - type: Pods
      pods:
        metric:
          name: requests_per_second
        target:
          type: AverageValue
          averageValue: "1000"

扩容策略：
  快速扩容：每 15s 检查一次，最多扩 4 个 Pod
  慢速缩容：每 5min 检查一次，最多缩 1 个 Pod
  防止频繁抖动
```

### 5.2 混合伸缩策略
```
时间维度：
  白天流量峰值：HPA 按 QPS 水平扩容
  夜间流量低谷：VPA 缩小单 Pod 资源 + HPA 缩副本

事件维度：
  大促/活动：预热扩容（提前 30min 扩到预估峰值）
  突发热点：HPA 快速响应 + 预热缓存

成本优化：
  GPU Pod：按推理 QPS 精确计算所需 GPU 数
  CPU Pod：混合使用 Spot Instance 降低成本
  自动化：根据历史数据预测次日流量模式
```

---

## 6. 全链路延迟优化

### 6.1 优化策略（按 ROI 排序）
```
策略                  收益          成本     优先级
特征预计算+缓存       -10ms         低       P0
模型量化 FP16         -15ms         低       P0
动态 Batching         -5ms          低       P1
多路召回并行           -20ms        中       P0
模型蒸馏              -30ms         高       P1
TensorRT 算子融合     -10ms         中       P1
架构拆分微服务         -5ms         高       P2
```

### 6.2 实战案例：P99 从 80ms 优化到 40ms
```
Step 1：分布式追踪定位瓶颈
  Jaeger 追踪显示：特征查询 30ms（最大瓶颈）

Step 2：特征查询优化（30ms → 8ms）
  Redis Pipeline 批量查询（减少网络往返）
  本地缓存热特征（命中率 92%）

Step 3：召回并行化（25ms → 10ms）
  4 路召回从串行改为并行 RPC
  设置每路 8ms 超时，超时返回空

Step 4：精排模型加速（20ms → 12ms）
  ONNX → TensorRT 转换
  FP16 量化 + 算子融合

Step 5：重排轻量化（5ms → 3ms）
  复杂 DPP 重排 → 规则 + 小模型

总计：80ms → 33ms（超额完成）
```

---

## 7. 面试高频问题

```
Q: 为什么不直接用 Flask/FastAPI 部署模型？
A: 1) Python GIL 限制并发 2) 无动态 Batching（GPU利用率低）
   3) 无模型版本管理 4) 无 GPU 调度优化
   生产环境必须用 TF Serving/Triton 等专业框架。

Q: 量化推理 INT8 精度损失如何控制？
A: 1) 先试 PTQ，精度不够再用 QAT
   2) 混合精度：敏感层保持 FP32，计算密集层用 INT8
   3) 用有代表性的校准数据集确定 scale
   4) 量化后必须做精度验证和 A/B 测试。

Q: 动态 Batching 的 max_delay 怎么设？
A: max_delay = P99_budget * 0.1 或更小。例如 P99 目标 50ms，
   max_delay 设 5ms。batch_size 和 delay 需要在吞吐量和延迟
   之间找平衡，建议用 benchmark 工具实测不同配置。

Q: 模型上线后效果下降怎么排查？
A: 按顺序排查：1) 最近部署变更（回滚验证）
   2) 数据漂移（PSI > 0.25 告警）3) 特征缺失/延迟
   4) 代码 bug（特征计算逻辑变更）5) 流量分布变化
   6) 外部因素（竞品/季节）。

Q: 蒸馏、剪枝、量化三者的最佳组合顺序？
A: 先蒸馏（减知识冗余，精度损失最小）→ 再剪枝（减结构冗余）
   → 最后量化（减精度冗余）。三者可叠加，总加速可达 10-20x。
```
