# 出价系统工程架构：延迟 / 分布式 / 特征实时化 / 日志闭环

> 整理日期：2026-03-18 | 作者：MelonEggLearn
> 难度：⭐⭐⭐⭐⭐ | 面试高频：✅（大厂必考）

---

## ASCII 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          实时出价系统完整架构                                     │
│                                                                                 │
│  ┌──────────────┐    ┌──────────────────────────────────────────────────────┐   │
│  │  广告交易平台  │    │                   出价服务集群（Bidding Cluster）        │   │
│  │  (Ad Exchange)│    │                                                      │   │
│  │              │    │  ┌────────────┐  ┌────────────┐  ┌────────────────┐  │   │
│  │  Bid Request ├───►│  │ 特征服务    │  │ 模型服务    │  │ 出价决策服务   │  │   │
│  │  (100ms TTL) │    │  │(Feature Svc)│  │(Model Svc) │  │(Bidding DSP)  │  │   │
│  │              │◄───┤  │            │  │            │  │               │  │   │
│  │  Bid Response│    │  │  1-2ms     │  │  3-5ms     │  │   1ms         │  │   │
│  └──────────────┘    │  └──────┬─────┘  └──────┬─────┘  └───────────────┘  │   │
│                      │         │               │                             │   │
│  ┌──────────────┐    └─────────┼───────────────┼─────────────────────────────┘  │
│  │  Win Notice  │              │               │                                 │
│  │  (赢价通知)  │              ▼               ▼                                 │
│  └──────┬───────┘    ┌─────────────────────────────┐                            │
│         │            │       实时特征存储层           │                            │
│         │            │  Redis / Memory KV Store     │                            │
│         │            │  ┌──────────┐ ┌──────────┐  │                            │
│         │            │  │用户实时   │ │广告实时   │  │                            │
│         │            │  │行为特征   │ │统计特征   │  │                            │
│         │            │  └──────────┘ └──────────┘  │                            │
│         │            └──────────────────────────────┘                            │
│         │                        ▲                                               │
│         │            ┌───────────┴──────────────────┐                            │
│         │            │       实时流处理层              │                            │
│         │            │  Kafka → Flink → Redis        │                            │
│         │            └──────────────────────────────┘                            │
│         │                                                                         │
│         │            ┌──────────────────────────────────────────────────────┐   │
│         └───────────►│               日志系统 & 训练闭环                      │   │
│                       │  bid_request → bid_response → win_notice → conversion│   │
│                       │  ↓ 日志ETL → 特征工程 → 模型训练 → 蓝绿部署 → 新模型  │   │
│                       └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 一、延迟预算（<10ms 总目标）

### 1.1 延迟分配框架

在实时竞价（RTB）系统中，从 Ad Exchange 发出 Bid Request 到收到 Bid Response，通常有 **80-150ms** 的窗口（不同平台不同）。其中网络传输往返约占 20-50ms，留给出价系统处理的时间往往只有 **<10ms**。

```
完整时间线：
[Ad Exchange 发出请求]
    │
    ├─ 网络传输（往程）: 2-3ms
    │
    ▼
[出价服务收到请求] ← 计时开始
    │
    ├─ 特征获取（Feature Fetch）: 1-2ms
    │   ├─ 用户实时特征（Redis/内存KV）: 0.5-1ms
    │   ├─ 广告基础特征（本地缓存）: 0.1ms
    │   └─ 上下文特征（请求自带）: 0ms
    │
    ├─ 模型推理（CTR/CVR预估）: 3-5ms
    │   ├─ 特征拼接/预处理: 0.5ms
    │   ├─ 模型前向计算: 2-4ms
    │   └─ 结果后处理: 0.1ms
    │
    ├─ 出价计算（Bid Calculation）: 1ms
    │   ├─ 出价公式应用（eCPM计算）: 0.3ms
    │   ├─ 预算检查: 0.2ms
    │   └─ 出价系数叠加: 0.3ms
    │
    └─ 出价服务响应: 0.2ms
[计时结束] → 总处理: ~6-8ms

    ├─ 网络传输（回程）: 2-3ms
    │
    ▼
[Ad Exchange 收到应答] 总端到端: ~10-14ms
```

### 1.2 各层延迟分配

| 处理阶段 | 目标延迟 | 关键依赖 | 超时处理 |
|---------|---------|---------|---------|
| 特征获取 | 1-2ms | Redis/KV Store（本地/同IDC） | 使用默认特征值继续 |
| 模型推理 | 3-5ms | GPU 推理 / 量化模型 | 使用历史均值出价 |
| 出价计算 | 1ms | 纯内存计算 | 无超时风险 |
| 序列化/网络 | 1-2ms | Protobuf 序列化 | 无法优化，仅压缩 |
| **总计** | **<10ms** | | |

### 1.3 延迟优化的工程手段

#### A. 特征获取优化

```
策略1：本地内存缓存（L1）
- 广告基础特征（ID、标题、落地页URL）缓存到进程内存
- 命中率通常 >90%，延迟 < 0.1ms
- 代价：内存消耗，需定期刷新

策略2：同机房 Redis（L2）
- 用户实时特征：同IDC Redis，RTT < 0.5ms
- 相比跨IDC（2-5ms）大幅降低

策略3：特征预拉取（Pre-fetch）
- 在竞价请求到达前，提前获取高频用户特征到本地缓存
- 基于预测模型预判哪些用户/广告组合会被请求

策略4：批量合并请求
- 同一请求中多个广告槽位，批量请求特征而非逐个查询
```

#### B. 模型推理优化

```
量化加速：
- FP32 → FP16: 速度提升 1.5-2x，精度损失 <0.1%
- FP32 → INT8: 速度提升 2-4x，精度损失 ~0.5%（需要校准）
- FP32 → INT4: 速度提升 4-8x，精度损失更大（大模型专用）

硬件加速：
- GPU 批处理：将多个并发请求合并为 batch 推理
  （batch_size=32 vs batch_size=1，吞吐量提升 10-50x，但增加排队延迟）
- 针对延迟敏感场景：batch_size=1，使用 TensorRT 或 ONNX Runtime

模型架构优化：
- 知识蒸馏：大模型（Teacher）→ 小模型（Student），推理速度提升 3-10x
- 剪枝（Pruning）：去除权重接近0的神经元，稀疏化模型
- 早退出（Early Exit）：置信度高时提前结束前向传播
```

#### C. 出价计算优化

```python
# 核心出价公式（内存计算，<1ms）
def calculate_bid(pctr: float, pcvr: float, bid_price: float,
                  adjustments: dict) -> float:
    """
    pctr: 预估点击率
    pcvr: 预估转化率
    bid_price: 目标CPA或出价参数
    """
    # eCPM = pCTR × pCVR × 目标CPA × 1000
    ecpm = pctr * pcvr * bid_price * 1000
    
    # 叠加调整系数（乘法）
    multiplier = 1.0
    for adj in adjustments.values():
        multiplier *= (1.0 + adj)
    
    return ecpm * multiplier
```

### 1.4 超时熔断设计

```
熔断触发条件：
- 单次请求超时（>8ms）：使用默认出价，记录超时日志
- 连续N次失败（如N=10）：触发熔断，切换降级模式
- 错误率 > 5%（1分钟窗口）：触发降级

熔断器状态机：
CLOSED（正常）→ [错误率达阈值] → OPEN（熔断）
OPEN（熔断）→ [等待恢复时间，如30秒] → HALF_OPEN（探测）
HALF_OPEN（探测）→ [成功率恢复] → CLOSED
HALF_OPEN（探测）→ [仍有错误] → OPEN

降级出价策略：
- 优先级1：使用广告历史均值出价（近7天平均）
- 优先级2：使用全局默认出价（配置文件中固定值）
- 优先级3：不参与竞价（极端情况，避免乱出价）
```

---

## 二、分布式出价服务

### 2.1 无状态设计

出价服务的核心设计原则是**完全无状态**：每次 Bid Request 独立处理，服务实例之间不共享状态。

```
✅ 正确做法（无状态）：
┌────────────────────────────────────────┐
│  Bid Request 到达                       │
│  → 从请求中获取 user_id, ad_id          │
│  → 从外部存储（Redis）获取特征           │
│  → 本地模型推理                          │
│  → 返回出价                             │
│  → 结束（无任何本地状态保留）            │
└────────────────────────────────────────┘

❌ 错误做法（有状态）：
- 在服务内存中维护用户会话（Session）
- 在服务内存中缓存广告预算状态
- 依赖前一次请求的结果

无状态的好处：
1. 任意实例可处理任意请求（可以无限水平扩展）
2. 实例宕机不影响其他实例（高可用）
3. 部署简单，金丝雀发布方便
4. 无状态 = 天然幂等，重试安全
```

### 2.2 水平扩展

```
┌──────────────────────────────────────────────────────────┐
│                   负载均衡层                               │
│                                                          │
│   ┌─────────────────────────────────────────────────┐   │
│   │  L4/L7 负载均衡（Nginx / HAProxy / Envoy）       │   │
│   │  策略：轮询 / 最少连接 / 一致性哈希              │   │
│   └──────────┬──────────────┬───────────────────────┘   │
│              │              │                            │
│    ┌─────────▼───┐  ┌───────▼───┐                       │
│    │ Bidding-01  │  │Bidding-02 │  ... Bidding-N        │
│    │（无状态）    │  │（无状态） │                       │
│    └─────────────┘  └───────────┘                       │
│                                                          │
│   服务发现：Consul / etcd / Kubernetes Service           │
│   - 新实例启动后自动注册                                  │
│   - 实例宕机后自动摘除（心跳检测）                        │
│   - 支持按权重分配（灰度发布用）                          │
└──────────────────────────────────────────────────────────┘
```

**服务发现实现：**

```yaml
# Consul 服务注册示例
service:
  name: "bidding-service"
  id: "bidding-01"
  port: 8080
  tags: ["bidding", "v2.1.0"]
  check:
    http: "http://localhost:8080/health"
    interval: "5s"
    timeout: "2s"
    deregister_critical_service_after: "30s"
```

### 2.3 服务拆分

出价系统通常拆分为三个独立微服务，各自独立扩容和部署：

```
┌───────────────────────────────────────────────────────────┐
│                   服务拆分架构                             │
│                                                           │
│  ┌──────────────────────────────────────────────────┐    │
│  │  1. 特征服务（Feature Service）                   │    │
│  │                                                   │    │
│  │  职责：整合用户特征、广告特征、上下文特征          │    │
│  │  输入：user_id, ad_id, context                   │    │
│  │  输出：特征向量（稠密/稀疏）                      │    │
│  │  存储：Redis, HBase, 本地缓存                    │    │
│  │  延迟目标：< 2ms                                 │    │
│  └──────────────────────────────────────────────────┘    │
│                                                           │
│  ┌──────────────────────────────────────────────────┐    │
│  │  2. 模型服务（Model Service）                     │    │
│  │                                                   │    │
│  │  职责：CTR/CVR 预估，输出概率分                   │    │
│  │  输入：特征向量                                   │    │
│  │  输出：pCTR, pCVR, pROI                         │    │
│  │  框架：TensorFlow Serving / TorchServe / TRT     │    │
│  │  延迟目标：< 5ms                                 │    │
│  └──────────────────────────────────────────────────┘    │
│                                                           │
│  ┌──────────────────────────────────────────────────┐    │
│  │  3. 出价决策服务（Bidding Decision Service）       │    │
│  │                                                   │    │
│  │  职责：将预估值转为出价，应用策略和约束            │    │
│  │  输入：pCTR, pCVR + 业务参数（目标CPA、预算状态） │    │
│  │  输出：final_bid（出价金额）                      │    │
│  │  逻辑：出价公式 + 出价调整 + 预算检查             │    │
│  │  延迟目标：< 1ms                                 │    │
│  └──────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────┘
```

**拆分的收益：**
- 特征服务可以独立扩容（数据密集型）
- 模型服务可以独立升级（GPU 资源密集型）
- 出价决策服务轻量，可以高并发部署
- 各服务故障域隔离，互不影响

---

## 三、特征工程实时化

### 3.1 实时特征流水线

```
用户行为事件流：
点击/曝光/购买/加购/搜索

    │ (Kafka Topic: user-behavior-events)
    ▼
┌──────────────────────────────────────┐
│         Kafka（消息队列）             │
│  - 分区：按 user_id 哈希分区          │
│  - 保留：7天（用于回溯和重放）         │
│  - 吞吐：100万 msg/s                 │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│         Flink（实时流计算）           │
│                                      │
│  实时计算任务示例：                   │
│  - 用户过去1小时点击次数              │
│  - 用户过去24小时消费金额             │
│  - 用户对某品类的实时兴趣分           │
│  - 广告过去1小时的CTR统计            │
│                                      │
│  窗口类型：                          │
│  - 滚动窗口（Tumbling）: 5min统计    │
│  - 滑动窗口（Sliding）: 1h/5min步   │
│  - 会话窗口（Session）: 30min超时    │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│    Redis / 内存KV存储（特征存储）      │
│                                      │
│  Key结构：                           │
│  user:{user_id}:realtime → hash      │
│  {                                   │
│    "click_1h": 5,                   │
│    "ctr_preference": 0.023,         │
│    "last_category": "electronics",  │
│    "session_depth": 3               │
│  }                                  │
│                                      │
│  TTL：通常1小时（避免过期数据）        │
└──────────────────────────────────────┘
```

### 3.2 特征新鲜度 vs 计算成本

| 特征类型 | 更新频率 | 存储位置 | 计算成本 | 典型特征 |
|---------|---------|---------|---------|---------|
| 实时特征 | 秒级 | Redis | 高（Flink持续计算） | 过去1h点击数、实时搜索词 |
| 近实时特征 | 分钟级 | Redis | 中 | 过去24h行为统计 |
| 离线特征 | 小时/天级 | HBase/KV | 低（批处理） | 历史LTV、兴趣标签、人口属性 |
| 广告特征 | 分钟/小时级 | 本地缓存 | 低 | 广告标题、历史CTR、预算状态 |

**权衡原则：**
```
特征新鲜度收益 > 实时计算成本 → 做实时特征
特征新鲜度收益 < 离线计算成本 → 用离线特征

实践建议：
- 1小时内行为：实时（Flink实时聚合）
- 1天内行为：近实时（Flink微批，5分钟更新）
- 7天以上：离线（Spark每小时批处理）
```

### 3.3 特征穿越（Feature Leakage）的工程防护

特征穿越是指在训练时使用了"未来信息"——即在预测时刻实际不可知的特征，导致模型在线上表现远差于离线评估。

```
典型特征穿越场景：
┌─────────────────────────────────────────────────────────────────┐
│  错误示例：                                                      │
│  训练样本的标签：用户是否在24小时内点击                            │
│  训练特征：用户过去24小时的总点击次数（包含了本次点击事件！）         │
│                                                                  │
│  原因：打标签时用了 t+24h 的数据，但该特征也在 t 时刻更新         │
│  后果：离线 AUC 虚高 0.1~0.2，线上完全失效                       │
└─────────────────────────────────────────────────────────────────┘
```

**工程防护方案：**

```python
# 防穿越核心原则：特征时间戳 < 标签事件时间戳

def build_training_sample(bid_request_time: datetime,
                          conversion_time: datetime,
                          features_snapshot: dict) -> dict:
    """
    所有特征必须是 bid_request_time 时刻的快照
    不得使用 bid_request_time 之后产生的数据
    """
    # ✅ 正确：使用请求时刻的特征快照
    valid_features = {
        k: v for k, v in features_snapshot.items()
        if v['timestamp'] <= bid_request_time
    }
    
    # ❌ 错误示例（应避免）：
    # wrong_feature = query_current_value(user_id, 'click_24h')
    # 因为训练时 current 已经包含了转化后的行为
    
    label = 1 if conversion_time is not None else 0
    return {"features": valid_features, "label": label}
```

**日志系统防穿越：** 在竞价日志中，同时记录特征值的**生效时间戳**，训练时严格过滤晚于请求时间的特征值。

---

## 四、模型部署与加速

### 4.1 模型压缩

#### 知识蒸馏（Knowledge Distillation）

```
Teacher Model（大模型，准确但慢）
    │
    │ 软标签（Soft Labels / Logits）
    ▼
Student Model（小模型，快但稍不准）
    │
    │ 蒸馏损失 = α × KL散度(Teacher, Student) + (1-α) × 交叉熵(Hard Label)
    ▼
部署 Student Model（速度提升 3-10x，精度损失 <2%）
```

#### INT8 量化

```python
# PyTorch INT8 动态量化示例
import torch
from torch.quantization import quantize_dynamic

# 训练好的 FP32 模型
model_fp32 = load_trained_model()

# 动态量化：权重 FP32 → INT8
model_int8 = quantize_dynamic(
    model_fp32,
    {torch.nn.Linear, torch.nn.Embedding},  # 量化这些层
    dtype=torch.qint8
)

# 效果：推理速度提升 2-4x，模型大小减少 ~75%
# 精度损失：AUC 通常降低 < 0.001
```

#### 剪枝（Pruning）

```
结构化剪枝：去除整个神经元/注意力头
非结构化剪枝：将权重矩阵中小于阈值的权重置0（稀疏化）

实际效果（以 CTR 模型为例）：
- 30% 剪枝率：速度提升 20%，AUC 损失 <0.0005
- 50% 剪枝率：速度提升 40%，AUC 损失 ~0.001
- 70% 剪枝率：速度提升 60%，AUC 损失明显，需要重新微调
```

### 4.2 推理框架对比

| 框架 | 适用模型 | 延迟 | GPU支持 | 特点 |
|-----|---------|-----|--------|-----|
| TensorFlow Serving | TF 模型 | 中 | ✅ | 成熟稳定，gRPC接口 |
| TorchServe | PyTorch 模型 | 中 | ✅ | 灵活，支持自定义Handler |
| TensorRT | 任意（转换后） | 最低 | ✅（NVIDIA专属） | 极致优化，部署复杂 |
| ONNX Runtime | 跨框架 | 低 | ✅ | 通用性强，CPU/GPU均可 |
| Triton Inference Server | 多框架 | 低 | ✅ | 多模型管理，动态batching |

**推荐选型：**
```
延迟最敏感（<5ms）: TensorRT（NVIDIA GPU）或 ONNX Runtime（量化+线程优化）
多模型管理: NVIDIA Triton
在线快速迭代: TorchServe（Python自定义逻辑灵活）
```

### 4.3 模型热更新：蓝绿部署

```
蓝绿部署流程：

[当前生产环境]          [新模型版本]
     │                      │
  Blue（100%流量）         Green（0%流量）
     │                      │
     │   新模型训练完成       │
     │                      │
     ▼   离线评估 AUC 通过   ▼
  Blue（100% → 0%）     Green（0% → 100%）
     │                      │
     │   灰度验证（见下）     │
     │                      │
     └──────────────────────┘

灰度发布步骤：
Step 1: 5% 流量 → Green（观察24小时，看线上CTR/CPC/ROI）
Step 2: 20% 流量 → Green（观察12小时）
Step 3: 50% 流量 → Green（观察6小时）
Step 4: 100% 流量 → Green（Blue 进入备用）

回滚条件（任一满足立即回滚）：
- 线上 CTR 下降 > 5%（相对变化）
- 平均 CPC 上升 > 10%
- 服务错误率 > 1%
- P99 延迟 > 15ms（超过阈值）
```

---

## 五、竞价日志系统（训练闭环）

### 5.1 日志类型与采集

```
竞价完整日志链路：

[1] bid_request（竞价请求）
    {
      "request_id": "req_abc123",
      "timestamp": 1710000000.123,
      "user_id": "u_12345",
      "device": "mobile_ios",
      "ad_slot": "feed_banner_01",
      "floor_price": 5.0,
      "publisher": "pub_456"
    }

[2] bid_response（出价响应）
    {
      "request_id": "req_abc123",
      "bid_price": 12.5,
      "ad_id": "ad_789",
      "campaign_id": "camp_321",
      "pctr": 0.023,
      "pcvr": 0.008,
      "ecpm": 12.5,
      "features_hash": "f7a8b9...",  # 特征快照哈希，用于防穿越
      "latency_ms": 6.8
    }

[3] win_notice（赢价通知）
    {
      "request_id": "req_abc123",
      "win_price": 10.2,  # 实际支付价格（二价/统一价）
      "cleared_price": 10.2,
      "timestamp": 1710000000.200
    }

[4] impression（曝光通知）
    {
      "request_id": "req_abc123",
      "impression_id": "imp_001",
      "timestamp": 1710000002.500  # 广告渲染后记录
    }

[5] click（点击日志）
    {
      "impression_id": "imp_001",
      "click_id": "clk_999",
      "timestamp": 1710000015.800
    }

[6] conversion（转化日志）
    {
      "click_id": "clk_999",
      "conversion_id": "conv_555",
      "value": 299.0,  # 转化金额
      "conversion_type": "purchase",
      "timestamp": 1710003600.000
    }
```

### 5.2 日志关联与训练样本构建

```python
# 训练样本构建逻辑（简化版）
def build_training_dataset(bid_logs, win_logs, impression_logs,
                           click_logs, conversion_logs, lookback_window=24*3600):
    """
    将各类日志关联成训练样本
    标签：24小时窗口内是否有转化
    """
    # 建立 request_id → 各类日志的映射
    samples = []
    
    for bid_req in bid_logs:
        req_id = bid_req['request_id']
        bid_time = bid_req['timestamp']
        
        # 检查是否赢得竞价
        win = win_logs.get(req_id)
        if not win:
            continue  # 未赢得，无曝光
        
        # 获取曝光
        impression = impression_logs.get(req_id)
        if not impression:
            continue
        
        # 检查是否有点击（通过曝光ID关联）
        imp_id = impression['impression_id']
        click = click_logs.get(imp_id)
        
        # 检查是否有转化（通过点击ID关联）
        label_ctr = 1 if click else 0
        label_cvr = 0
        if click:
            clk_id = click['click_id']
            conv = conversion_logs.get(clk_id)
            if conv and (conv['timestamp'] - bid_time) <= lookback_window:
                label_cvr = 1
        
        samples.append({
            'features': bid_req['features_snapshot'],
            'label_ctr': label_ctr,
            'label_cvr': label_cvr,
            'win_price': win['win_price'],
            'bid_price': bid_req['bid_price']
        })
    
    return samples
```

### 5.3 训练闭环

```
完整训练闭环（每天执行）：

[竞价日志采集]（实时写入 Kafka → HDFS/对象存储）
    │
    ▼
[日志ETL清洗]（Spark离线 / Flink近实时）
    │ 过滤无效日志、关联转化、防穿越处理
    ▼
[特征工程]（Spark 批处理）
    │ 原始日志 → 特征向量 + 标签
    ▼
[模型训练]（分布式训练，GPU集群）
    │ 增量训练（在上版模型基础上）或全量重训
    ▼
[离线评估]（AUC / GAUC / 校准度 / NDCG）
    │ 通过阈值才允许上线
    ▼
[蓝绿部署]（灰度发布，5%→20%→100%）
    │
    ▼
[线上监控]（实时看板：CTR/CPC/ROI/P99延迟）
    │ 异常自动触发回滚
    ▼
[新日志产生] → 回到顶部，形成闭环
```

---

## 六、容错与降级

### 6.1 模型超时降级

```python
import asyncio
from typing import Optional

async def get_bid_with_fallback(
    request: BidRequest,
    model_service: ModelService,
    timeout_ms: float = 5.0,
    fallback_bid: Optional[float] = None
) -> float:
    """
    带超时保护的出价函数
    """
    try:
        # 设置超时
        pctr, pcvr = await asyncio.wait_for(
            model_service.predict(request.features),
            timeout=timeout_ms / 1000
        )
        bid = calculate_bid(pctr, pcvr, request.target_cpa)
        return bid
        
    except asyncio.TimeoutError:
        # 超时降级：使用历史均值
        logger.warning(f"Model timeout for req {request.id}, using fallback")
        metrics.increment("bidding.model_timeout")
        
        # 降级策略（优先级从高到低）
        if request.ad_id in historical_avg_bids:
            return historical_avg_bids[request.ad_id]  # 广告历史均值
        elif request.campaign_id in campaign_avg_bids:
            return campaign_avg_bids[request.campaign_id]  # 活动均值
        elif fallback_bid:
            return fallback_bid  # 固定值
        else:
            return DEFAULT_GLOBAL_BID  # 全局默认值
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        metrics.increment("bidding.model_error")
        return DEFAULT_GLOBAL_BID
```

### 6.2 熔断器模式（Circuit Breaker）

```python
class CircuitBreaker:
    """
    熔断器实现
    """
    def __init__(self, failure_threshold=10, recovery_timeout=30,
                 success_threshold=5):
        self.failure_threshold = failure_threshold   # 触发熔断的连续失败次数
        self.recovery_timeout = recovery_timeout      # 熔断恢复等待时间（秒）
        self.success_threshold = success_threshold    # HALF_OPEN → CLOSED 需要的连续成功次数
        
        self.state = "CLOSED"    # CLOSED / OPEN / HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
    
    def can_attempt(self) -> bool:
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            # 检查是否到了恢复探测时间
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.success_count = 0
                return True  # 允许一次探测请求
            return False
        else:  # HALF_OPEN
            return True
    
    def on_success(self):
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "CLOSED"
                self.failure_count = 0
        elif self.state == "CLOSED":
            self.failure_count = 0
    
    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.critical("Circuit breaker OPEN! Switching to degraded mode")
```

### 6.3 灰度发布：流量分配策略

```
新模型灰度发布（Traffic Splitting）：

方案1：随机分流（最简单）
    - 生成随机数 r ∈ [0, 1)
    - r < 0.05 → 新模型（5%）
    - r >= 0.05 → 旧模型（95%）
    - 缺点：同一用户可能分配到不同模型，体验不一致

方案2：用户哈希分流（推荐）
    - bucket = hash(user_id) % 100
    - bucket < 5 → 新模型（5%）
    - bucket >= 5 → 旧模型（95%）
    - 优点：同一用户始终使用同一模型，可以做用户维度的效果统计

方案3：广告活动分流
    - 选定部分广告活动（campaign）用新模型
    - 适合精细化的A/B实验
    - 缺点：不同活动的广告主特征不同，可能引入噪声
```

**灰度进度表：**

```python
CANARY_STAGES = [
    {"name": "1% 探测", "traffic_pct": 1,   "min_duration_hours": 2,   "auto_advance": False},
    {"name": "5% 金丝雀", "traffic_pct": 5,  "min_duration_hours": 24,  "auto_advance": True},
    {"name": "20% 扩量", "traffic_pct": 20,  "min_duration_hours": 12,  "auto_advance": True},
    {"name": "50% 对比", "traffic_pct": 50,  "min_duration_hours": 6,   "auto_advance": False},
    {"name": "100% 全量", "traffic_pct": 100, "min_duration_hours": 0,   "auto_advance": False},
]

# 自动晋级条件（每个阶段必须满足）
def check_advance_conditions(metrics_window):
    return (
        metrics_window['ctr_change_pct'] > -0.02  # CTR 不降超过2%
        and metrics_window['cpc_change_pct'] < 0.05  # CPC 不涨超过5%
        and metrics_window['error_rate'] < 0.001    # 错误率<0.1%
        and metrics_window['p99_latency_ms'] < 12   # P99延迟<12ms
    )
```

---

## 七、常见考点5条

### 考点1：出价系统的延迟如何控制在10ms以内？

**参考答案：**

出价系统的延迟预算分配：特征获取（1-2ms）+ 模型推理（3-5ms）+ 出价计算（1ms）+ 序列化（1-2ms）= 总计约8ms。

**关键优化手段：**
1. **特征获取**：同IDC Redis（<1ms）+ 进程内缓存（<0.1ms），避免跨机房访问
2. **模型推理**：INT8量化（速度提升2-4x）+ TensorRT/ONNX Runtime + GPU batching
3. **出价计算**：纯内存计算，预先加载预算状态，不做任何IO
4. **超时保护**：单次请求超时（>8ms）立即使用降级出价，不等待

---

### 考点2：出价系统为什么要无状态设计？

**参考答案：**

无状态（Stateless）设计意味着每次竞价请求完全独立，服务实例不维护任何跨请求的内存状态。

**原因：**
1. **水平扩展**：任意实例可处理任意请求，流量增加时直接加实例，无需迁移状态
2. **高可用**：实例宕机直接摘除，其他实例无感知（无状态丢失问题）
3. **简化部署**：蓝绿部署、滚动更新不需要状态迁移
4. **调试简单**：请求日志包含完整上下文，可复现任意历史请求

**状态存放在哪？** 所有状态（用户特征、广告预算、活动配置）存放在外部存储（Redis/数据库），服务实例按需读取。

---

### 考点3：特征穿越（Feature Leakage）如何产生，如何防护？

**参考答案：**

**产生原因：** 训练时使用了预测时刻之后才产生的数据作为特征。典型场景：以"用户24小时总点击次数"作为特征，而标签也是24小时内转化，导致特征包含了标签期间产生的点击。

**后果：** 离线 AUC 虚高（0.1-0.2），线上效果大幅下降，严重时出价策略完全失效。

**工程防护：**
1. **特征时间戳记录**：在竞价日志中记录每个特征的"数据生效时间"
2. **严格过滤**：训练时，只使用时间戳 ≤ 请求时刻的特征值
3. **特征快照**：竞价时将当前特征值快照保存到日志，训练直接用快照，不重新查询
4. **离线 vs 线上一致性测试**：定期对比离线 pCTR 分布和线上 CTR 分布，偏差过大需排查穿越问题

---

### 考点4：如何设计竞价日志的训练闭环？

**参考答案：**

训练闭环的核心链路：`实时竞价日志 → 日志ETL → 特征工程 → 模型训练 → 部署 → 产生新日志`。

**关键设计要点：**

1. **日志完整性**：需要采集 bid_request（出价请求）、bid_response（出价响应）、win_notice（赢价）、impression（曝光）、click（点击）、conversion（转化）六类日志，通过 request_id 串联
2. **延迟归因**：转化事件（如付款）可能发生在点击后数小时，需要设置归因窗口（如24-72小时）
3. **训练频率**：通常每日训练一次（全量或增量），实时性要求高的场景用 Flink 实现在线学习
4. **防穿越**：使用日志中记录的特征快照，而非重新查询当前值

---

### 考点5：如何安全地将新出价模型上线到生产？

**参考答案：**

**蓝绿部署 + 灰度发布**的标准流程：

1. **离线评估**：新模型的 AUC/GAUC、校准度（Calibration）、NDCG 均优于当前模型，且通过显著性检验
2. **影子模式**（可选）：新模型接收全量请求，但不实际出价，仅记录出价结果用于离线对比
3. **灰度发布**：5% → 20% → 50% → 100%，每阶段观察线上CTR/CPC/ROI/延迟
4. **监控告警**：
   - CTR 下降 >3%：立即人工介入
   - CTR 下降 >5%：自动回滚
   - P99延迟 >12ms：自动回滚
5. **快速回滚**：Blue 环境（旧模型）保持热备，回滚只需切换流量分配配置，<1分钟完成

---

## 附录：核心技术栈

```
实时流处理:  Kafka + Flink (Apache Kafka 3.x + Flink 1.18)
特征存储:    Redis Cluster（实时）+ HBase（历史）
模型推理:    TensorRT（GPU）/ ONNX Runtime（CPU）
服务发现:    Consul / Kubernetes Service
负载均衡:    Envoy / Nginx
分布式追踪:  Jaeger / Zipkin
监控:        Prometheus + Grafana
日志:        ELK（Elasticsearch + Logstash + Kibana）
训练平台:    PyTorch + DDP（分布式数据并行）
```

---
> 笔记编号：ads/20260318_bidding_system_engineering | 状态：✅ 完成
