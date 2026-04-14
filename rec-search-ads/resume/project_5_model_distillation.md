# 项目5：CTR 模型蒸馏与在线服务优化

## 项目概览

这是项目1（CTR 优化）的深化和工程化扩展。当 CTR 模型精度优化到 AUC 0.78 后，新的瓶颈出现了：**推理延迟和计算成本**。我通过知识蒸馏、INT8 量化、结构化剪枝等技术，将推理延迟从 20ms 降低到 6ms（70% 降低），同时降低 GPU 成本 60%，精度衰减控制在 <1%。项目周期 8 个月，涉及模型优化、TensorRT 编译、线上部署三个环节。

---

## 一、瓶颈分析

### 1.1 现状：CTR 模型的两个痛点

经过项目1的优化，CTR 预估模型达到 AUC 0.78。但线上遇到了新问题：

```
问题 1：推理延迟太高
  ├─ 模型：DeepFM + Stacking（三个 Base Model）
  ├─ 推理延迟：p50 = 18ms，p99 = 25ms ← 太高！
  ├─ RTB 竞价 SLA：<50ms 响应时间（包括网络往返）
  │   ├─ 网络延迟：5-10ms（往返）
  │   ├─ 特征获取：5-10ms
  │   ├─ 模型推理：18ms（当前）← 用掉 36% 的预算
  │   └─ 后处理 + 网络返回：5ms
  └─ 结果：竞价逻辑（竞价、出价计算等）没有时间执行！

问题 2：GPU 成本高
  ├─ 模型参数量：50M（DeepFM）+ 30M（LightGBM）+ 25M（XGBoost 编码）
  ├─ GPU 内存占用：200MB（单个模型）
  ├─ 推理 QPS：2000 req/s（单机，V100 GPU）
  ├─ 月成本：$2000（GPU 租赁）← 虽然不是主要成本，但有优化空间
  └─ 问题：随着流量增长，GPU 成本线性增加
```

### 1.2 目标与约束

```
目标：
  ├─ 推理延迟：20ms → <8ms（p99）
  ├─ 精度衰减：<1%（AUC 可以从 0.78 → 0.772）
  ├─ 模型大小：450MB → <100MB（便于边缘部署）
  ├─ GPU 成本：月省 $1200
  └─ 线上稳定性：零故障

约束：
  ├─ 模型更新频率不变（仍需每周重训）
  ├─ 特征工程不能简化（否则精度掉太多）
  └─ 需要支持灰度发布和快速回滚
```

---

## 二、知识蒸馏（Knowledge Distillation）

### 2.1 核心原理

**问题**：为什么直接训练小模型不行？

```
直接训练小模型（FM）：
  ├─ 训练数据：只有 hard label（0 或 1）
  ├─ 小模型能力有限，容易欠拟合
  ├─ 结果：AUC = 0.772（衰减 0.8%）
  └─ 问题：丧失了大模型蕴含的信息

知识蒸馏：
  ├─ 训练数据：大模型的软标签（如 0.04 表示预估 CTR 4%）
  ├─ 软标签包含"大模型的想法"（哪些样本相似、哪些容易出错）
  ├─ 小模型通过学习软标签，继承大模型的知识
  ├─ 结果：AUC = 0.775（衰减仅 0.3%）
  └─ 收益：比直接训练好 0.5%
```

### 2.2 温度缩放（Temperature Scaling）

关键是如何生成"好的"软标签。

硬标签：

$$
\text{hard label} = \begin{cases} 1 & \text{if } y_i = 1 \\ 0 & \text{otherwise} \end{cases}
$$

软标签（未缩放）：

$$
p_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}
$$

问题：如果模型对某个样本很有信心（z=10，p=0.99999），软标签就接近 hard label，信息少。

解决：加入温度参数 $T$，"平滑"概率分布

$$
p_i^T = \frac{1}{1 + e^{-z_i / T}}
$$

当 $T$ 增大时，分布变平滑（soft）：

```
z = 2（模型中等有信心，预估 CTR 88%）

T = 0.5：p = sigmoid(4) = 0.982 ← 很尖锐（接近 hard label）
T = 1.0：p = sigmoid(2) = 0.881 ← 正常
T = 5.0：p = sigmoid(0.4) = 0.599 ← 平滑（保留更多信息）
T = 10.0：p = sigmoid(0.2) = 0.550 ← 非常平滑（接近均匀）
```

**选择温度**：通常在训练过程中搜索最优 T，通常 T ∈ [2, 10]。

### 2.3 蒸馏损失函数

```
总损失 = α × Hard Loss + (1 - α) × Soft Loss

L_total = α × L_CE(y, ŷ_student) + (1-α) × L_KL(p_teacher^T, p_student^T)
```

其中：
- $L_CE$：学生与真实标签的交叉熵（hard target）
- $L_KL$：学生与教师的 KL 散度（soft target）
- $\alpha$：权重（通常 0.3-0.5）

代码：

```python
import tensorflow as tf

class KnowledgeDistillation:
    def __init__(self, teacher_model, student_model, temperature=5.0, alpha=0.3):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    def distillation_loss(self, y_true, y_teacher, y_student):
        """计算蒸馏损失"""
        # Hard loss：学生与真实标签
        hard_loss = tf.keras.losses.binary_crossentropy(y_true, y_student)
        
        # Soft loss：学生与教师的 KL 散度（使用温度缩放）
        # 对数似然
        teacher_soft = y_teacher / self.temperature
        student_soft = y_student / self.temperature
        
        # KL(teacher || student)
        kl_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.nn.softmax(teacher_soft),
            logits=tf.nn.softmax(student_soft)
        )
        
        # 加权组合
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * kl_loss
        return tf.reduce_mean(total_loss)
    
    def train_step(self, X_batch, y_batch):
        """单个训练步"""
        with tf.GradientTape() as tape:
            # 教师预测（冻结，不更新）
            y_teacher = self.teacher(X_batch, training=False)
            
            # 学生预测（更新）
            y_student = self.student(X_batch, training=True)
            
            # 计算蒸馏损失
            loss = self.distillation_loss(y_batch, y_teacher, y_student)
        
        # 反向传播（只更新学生）
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        
        return loss
    
    def train(self, X_train, y_train, epochs=100, batch_size=256):
        """完整训练"""
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.shuffle(len(X_train)).batch(batch_size)
        
        for epoch in range(epochs):
            losses = []
            for X_batch, y_batch in dataset:
                loss = self.train_step(X_batch, y_batch)
                losses.append(loss.numpy())
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {np.mean(losses):.4f}")
```

### 2.4 Student 模型的选择

需要在"模型大小/速度"和"精度"之间权衡：

| Student 架构 | 参数数 | 推理延迟 | AUC | 衰减 | 选择 |
|------------|--------|---------|-----|------|------|
| FM（因子分解机） | 1M | 5ms | 0.762 | 2.3% | ✗（衰减太多） |
| 2 层 MLP | 5M | 8ms | 0.771 | 1.0% | ? |
| **3 层 MLP** | 8M | 12ms | 0.775 | 0.3% | ✓ |
| 4 层 MLP | 12M | 18ms | 0.778 | 0.0% | ✗（没优化的价值） |
| DeepFM（完整）| 50M | 20ms | 0.783 | - | 基线 |

选择：**3 层 MLP**
- 精度衰减仅 0.3%（几乎无损）
- 推理延迟 12ms（从 20ms 快 40%）
- 参数紧凑（8M），易于部署

### 2.5 蒸馏效果

```
实验对比：

基线（DeepFM）：
  AUC = 0.783（精度最高）
  延迟 p99 = 20ms（太慢）
  参数 = 50M

方案 A：直接训练 3 层 MLP（无蒸馏）
  AUC = 0.772（衰减 1.1%）
  延迟 p99 = 12ms ✓
  参数 = 8M ✓

方案 B：蒸馏 3 层 MLP（T=5, α=0.3）
  AUC = 0.775（衰减仅 0.3%）✓
  延迟 p99 = 12ms ✓
  参数 = 8M ✓

收益：相比方案 A，蒸馏多获得 0.8% 的 AUC，同时延迟和参数都一样。
```

---

## 三、INT8 量化

### 3.1 原理

浮点数（float32）使用 32 位表示。整数（int8）使用 8 位。量化就是把 float32 转换为 int8，压缩 4x。

```
Float32：
  指数（8位）| 尾数（23位）
  范围：[-10^38, 10^38]
  精度：小数点后 6-7 位
  
Int8：
  符号（1位）| 数值（7位）
  范围：[-128, 127]
  精度：整数
```

**线性量化**：

$$
x_{\text{int8}} = \text{round}\left( \frac{x_{\text{float32}} - x_{\min}}{x_{\max} - x_{\min}} \times 127 \right)
$$

反量化：

$$
x_{\text{float32}} = \frac{x_{\text{int8}}}{127} \times (x_{\max} - x_{\min}) + x_{\min}
$$

### 3.2 量化感知训练（QAT）

直接量化一个训练好的模型，精度会损失很多（5-10%）。更好的方法是在**训练时**就模拟量化。

```python
class QuantizationAwareTraining:
    def __init__(self, model):
        self.model = model
        self.scale_factors = {}  # 记录每层的量化参数
    
    def fake_quantize(self, x, layer_name):
        """
        模拟量化：
        1. 计算 x 的范围
        2. 线性映射到 int8
        3. 反映射回 float32（用于梯度计算）
        """
        # 计算量化参数
        x_min = tf.reduce_min(x)
        x_max = tf.reduce_max(x)
        scale = (x_max - x_min) / 255.0  # int8 范围是 [0, 255]
        
        # 量化
        x_quantized = tf.cast(
            tf.round((x - x_min) / scale),
            tf.int8
        )
        
        # 反量化（保留梯度流动）
        x_dequantized = tf.cast(x_quantized, tf.float32) * scale + x_min
        
        # 记录参数（用于推理时）
        self.scale_factors[layer_name] = (x_min, scale)
        
        return x_dequantized
    
    def train_step(self, X_batch, y_batch):
        """训练步：在前向传播中模拟量化"""
        with tf.GradientTape() as tape:
            # 经过量化的前向传播
            x = X_batch
            for i, layer in enumerate(self.model.layers):
                x = layer(x)
                # 模拟量化
                x = self.fake_quantize(x, layer_name=f"layer_{i}")
            
            logits = x
            loss = tf.keras.losses.binary_crossentropy(y_batch, logits)
        
        # 正常的反向传播
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss
```

### 3.3 量化校准

量化后精度的关键是**选择合适的量化范围**（$x_{\min}, x_{\max}$）。

错误做法：直接用训练数据的最小值和最大值。

```
问题：
  训练数据的异常值（outliers）会拉长量化范围
  导致中间的值量化精度低
  
例子：
  权重范围：[-0.5, 0.5]，但有一个 outlier = 10
  
  用全局范围：[-10, 10]（太宽）
  → 0.5 量化为 6（精度低）
  
  用百分位数范围：[-0.6, 0.6]（排除 outliers）
  → 0.5 量化为 127（精度高）
```

**解决**：用百分位数（如 99.9%）而不是全局最值

```python
def calibrate_quantization_params(validation_data, percentile=99.9):
    """用验证数据校准量化参数"""
    scale_factors = {}
    
    for layer_name, layer in enumerate(model.layers):
        # 获取这层的权重
        weights = layer.get_weights()[0]  # 假设第一个是权重
        
        # 计算百分位数
        lower = np.percentile(weights, 100 - percentile)
        upper = np.percentile(weights, percentile)
        
        # 保存参数
        scale_factors[layer_name] = (lower, upper)
    
    return scale_factors
```

### 3.4 量化效果

```
基线（Float32）：
  模型大小：8M（3 层 MLP）
  推理延迟：12ms
  AUC：0.775

INT8 量化（直接）：
  模型大小：2M（4x 压缩）✓
  推理延迟：5ms（60% 降低）✓
  AUC：0.768（衰减 0.7%）✗

INT8 量化 + QAT（量化感知训练）：
  模型大小：2M ✓
  推理延迟：5ms ✓
  AUC：0.773（衰减仅 0.2%）✓✓

结论：QAT 虽然多费一点训练时间，但精度衰减可以控制到可接受水平。
```

---

## 四、在线服务优化

### 4.1 TensorRT 编译

TensorRT 是 NVIDIA 的推理优化库，可以进一步加速模型：

- **图优化**：融合相邻的操作（如 Batch Norm 和 ReLU 合并）
- **层融合**：多个小的 kernel 合并为一个大 kernel，减少内存访问
- **自动调优**：尝试不同的实现方式，选择最快的

```python
import tensorrt as trt

def compile_model_to_tensorrt(onnx_model_path, output_path):
    """
    将 ONNX 模型编译为 TensorRT
    """
    # TensorRT logger
    logger = trt.Logger(trt.Logger.WARNING)
    
    # 创建 builder
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    
    # 设置最大 batch size
    config.max_workspace_size = 1 << 30  # 1GB
    
    # 读取 ONNX 模型
    parser = trt.OnnxParser(builder, logger)
    with open(onnx_model_path, 'rb') as f:
        parser.parse(f.read())
    
    # 构建 TensorRT 引擎
    engine = builder.build_engine(network, config)
    
    # 序列化并保存
    with open(output_path, 'wb') as f:
        f.write(engine.serialize())
    
    return engine

# 使用
engine = compile_model_to_tensorrt('model.onnx', 'model.trt')

# 推理
context = engine.create_execution_context()
```

**效果**：TensorRT 编译后，推理延迟再快 20-30%（12ms → 8-9ms）

### 4.2 批处理（Batching）

单个请求的延迟可能很低（1ms），但如果每个请求都单独调用模型，GPU 利用率很低。

**批处理思想**：累积多个请求，一次性推理。

```python
class BatchingService:
    def __init__(self, max_batch_size=256, max_wait_time_ms=10):
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.request_queue = []
        self.event = threading.Event()
    
    def infer_async(self, features):
        """
        异步推理：请求被加入队列，等待批处理
        """
        future = threading.Future()
        self.request_queue.append((features, future))
        
        # 如果队列满或等待时间超过阈值，立即推理
        if len(self.request_queue) >= self.max_batch_size:
            self._process_batch()
        
        return future
    
    def _process_batch(self):
        """处理一个 batch"""
        if not self.request_queue:
            return
        
        # 取出请求
        batch = self.request_queue[:self.max_batch_size]
        self.request_queue = self.request_queue[self.max_batch_size:]
        
        # 提取特征
        features_list = [features for features, _ in batch]
        X_batch = np.array(features_list)
        
        # 批量推理
        predictions = model.predict(X_batch, batch_size=len(batch))
        
        # 返回结果
        for i, (_, future) in enumerate(batch):
            future.set_result(predictions[i])
```

**效果**：
- 单个请求延迟：1ms（由于等待）
- 批处理延迟：1ms + 8ms batch 推理 / batch_size
- 例如 batch_size=100：(1 + 8/100) = 1.08ms ← 几乎没增加延迟，但 GPU 利用率从 10% → 90%！

### 4.3 缓存热点特征

特征计算往往比模型推理还慢（获取数据库数据、RPC 调用等）。

```python
class FeatureCache:
    def __init__(self, cache_size=10000):
        self.cache = {}
        self.cache_size = cache_size
        self.access_count = {}  # 记录访问频率
    
    def get_features(self, feature_key):
        """获取特征，优先从缓存"""
        if feature_key in self.cache:
            # 命中
            self.access_count[feature_key] += 1
            return self.cache[feature_key]
        
        # 未命中，计算
        features = compute_features(feature_key)  # 可能很慢
        
        # 加入缓存
        if len(self.cache) < self.cache_size:
            self.cache[feature_key] = features
        else:
            # 缓存满，移除最少访问的
            least_accessed = min(self.access_count, key=self.access_count.get)
            del self.cache[least_accessed]
            del self.access_count[least_accessed]
            
            self.cache[feature_key] = features
        
        return features
```

**效果**：
- 热点特征的获取延迟：从 5ms（计算）→ 0.1ms（缓存）
- 命中率通常在 70-80%
- 整体特征获取延迟：5ms × 20% + 0.1ms × 80% = 1.08ms

---

## 五、完整流程与效果

### 5.1 优化流程

```
原始 DeepFM：
  特征：3000维
  模型：50M 参数
  延迟：20ms ┐
  成本：$2000│
         p99 │
         
     ↓ 蒸馏 → 3 层 MLP（8M）
     
蒸馏后：
  延迟：12ms ┐
  AUC：0.775│
  参数：8M ├── 满足延迟要求，精度可接受
          │
     ↓ INT8 量化
     
量化后：
  延迟：5ms（数据加载延迟消失）
  模型大小：2M
  AUC：0.773 ┐
           ├── 继续快速
     ↓ TensorRT 编译
     
编译后：
  延迟：4ms ┐
  吞吐：8000 QPS│ 推理部分已经达到硬件极限
              
加上其他（特征获取、网络）：
  总延迟：4ms（推理）+ 3ms（特征）+ 2ms（网络） = 9ms p99
         
最终：
  延迟：9ms ✓（< 8ms target，满足）
  成本：$800/月 ✓（省 $1200）
  AUC：0.773 ✓（仅衰减 0.5%）
```

### 5.2 线上 AB 测试

虽然新模型是蒸馏后的，需要验证线上效果不会因为精度微降而被用户察觉。

```
对照组（原 DeepFM）：50% 流量
实验组（蒸馏 MLP）：50% 流量

时长：1 周
```

结果：

| 指标 | 对照 | 实验 | 差异 | 显著性 |
|------|------|------|------|--------|
| CTR | 4.12% | 4.10% | -0.5% | 不显著 |
| CVR | 3.50% | 3.49% | -0.3% | 不显著 |
| RPM | 0.85 元 | 0.845 元 | -0.6% | 不显著 |

**结论**：虽然 AUC 衰减 0.5%，但线上用户察觉不到。这验证了蒸馏的有效性——AUC 的小幅衰减不一定导致线上指标衰减。

### 5.3 性能指标总结

| 指标 | 优化前 | 优化后 | 改进 |
|------|--------|--------|------|
| 推理延迟（p99） | 20ms | 6ms | -70% ✓✓✓ |
| 模型大小 | 450MB | 50MB | -89% ✓✓ |
| AUC | 0.783 | 0.773 | -0.5% ⚠️ |
| 单机 QPS | 2000 | 8000 | +4x ✓✓✓ |
| GPU 成本 | $2000/月 | $800/月 | -$1200 ✓✓ |
| GPU 内存占用 | 200MB | 50MB | -75% ✓ |

---

## 六、关键洞察

### 6.1 知识蒸馏不是"老生常谈"

很多人一开始对蒸馏有误解：

```
误解 1："蒸馏就是压缩，肯定会损失精度"
  真相：用对了蒸馏，精度衰减可以控制到 <1%
        比直接训练小模型好 10 倍

误解 2："温度参数随意选就行"
  真相：温度选择直接决定蒸馏效果
        T=1 几乎等同于没有蒸馏
        T=10 虽然软化了，但可能过度平滑
        最优 T 通常在 3-7 之间
```

### 6.2 量化是个系统性的问题

INT8 量化不是"一键压缩"，有很多细节：

```
直接量化（Post-Training Quantization）：
  ├─ 优点：简单（只需训练好的模型）
  └─ 缺点：精度损失 5-10%

量化感知训练（QAT）：
  ├─ 优点：精度损失控制 <1%
  └─ 缺点：需要重新训练（2-3 小时）

最关键的是校准数据的选择：
  ├─ 用训练数据校准：可能过拟合
  ├─ 用验证数据校准：最安全
  └─ 用百分位数（99.9%）而不是全局最值
```

### 6.3 推理优化的关键是整体

不能只优化单个环节：

```
优化层级：

Level 1：模型架构优化（蒸馏、剪枝）
  ├─ 收益：30-50%
  └─ 难度：高（需要重新训练）

Level 2：量化与编译（INT8、TensorRT）
  ├─ 收益：20-40%
  └─ 难度：中（需要理解硬件）

Level 3：系统级优化（批处理、缓存）
  ├─ 收益：50-200%（特别是吞吐）
  └─ 难度：低（不需要重新训练）

组合效果 > 单个的加和：
  30% × 30% × 100% = 9% ✗ (错）
  
  实际：通过协同优化
  蒸馏 30% + 量化 25% + 编译 15% + 批处理 50% 
  = 最多能减少推理延迟 70% + 增加吞吐 4x
```

### 6.4 不是所有优化都值得做

```
蒸馏 → 剪枝：值得（延迟快 40%，参数少 80%）
蒸馏 → 再蒸馏：不值得（精度基本没改，复杂度增加）
蒸馏 → 量化：值得（再快 60%，精度衰减 <1%）
量化 → 再量化（如 int4）：不值得（精度衰减 5-10%，复杂度很高）
```

---

## 七、讲故事要点

### 7.1 30 秒电梯演讲

> "我优化了 CTR 预估模型的推理性能。原来的 DeepFM 模型虽然精度很高（AUC 0.78），但推理延迟 20ms，太慢了。我用三个技术：（1）知识蒸馏把模型从 50M 压到 8M，（2）INT8 量化进一步压缩和加速，（3）TensorRT 编译优化。最终推理延迟从 20ms 降到 6ms（70% 降低），模型大小从 450MB 降到 50MB，精度衰减控制在 <1%。这让我们的 GPU 成本月省 $1200，同时满足了 RTB 竞价的延迟需求。"

### 7.2 完整讲述

**问题**：CTR 模型经过优化（项目1），精度达到 AUC 0.78。但出现了新的瓶颈：推理延迟 20ms，太高了。RTB 竞价需要在 50ms 内返回出价，推理占用了 40% 的时间预算，竞价逻辑没时间执行。同时，GPU 成本也在增加（$2000/月）。

**解决方案**：我用递进式的优化策略：
1. **知识蒸馏**：用大模型（DeepFM）的软标签来训练小模型（3 层 MLP）。虽然参数少了 6x，但通过学习大模型的"想法"，精度衰减控制在 0.3%。
2. **INT8 量化**：将浮点权重转换为整数，进一步压缩 4x，加速 60%。关键是用"量化感知训练"在训练时就模拟量化，避免推理时精度损失。
3. **TensorRT 编译**：NVIDIA 的推理引擎，通过图优化和层融合，再快 20-30%。
4. **系统级优化**：批处理累积请求、缓存热点特征，在不影响单个请求延迟的情况下，吞吐增加 4x。

**结果**：
- 推理延迟：20ms → 6ms（70% 降低）
- 模型大小：450MB → 50MB（89% 压缩，可以部署到边缘设备）
- 精度衰减：AUC 从 0.783 → 0.773（仅 0.5%，线上用户察觉不到）
- GPU 成本：月省 $1200
- 吞吐：QPS 从 2000 → 8000（4x 提升）

**学到的**：最大的收获是理解了优化是**系统性的工作**，不能只看单个环节。蒸馏、量化、编译、系统优化的组合效果远大于单个的加和。另一个洞察是：**微小的精度衰减（如 0.5% AUC）不一定导致线上指标衰减**。有时候，为了换取 70% 的延迟收益，损失 0.5% 的 AUC 是完全值得的。

---

## 八、总结

这个项目让我学到：

1. **推理优化需要硬件和算法的共同理解**：不能只会写模型，还要懂 GPU、内存、编译这些东西。

2. **知识蒸馏是个很强大的工具**：在线上系统中，蒸馏不仅用于压缩模型，还能提升推理稳定性（因为小模型的方差小）。

3. **整体优化的思维**：从模型架构、到量化方案、到编译、到系统设计，每一层都有优化空间。要有"木桶原理"的意识——瓶颈在哪里。

4. **trade-off 的取舍**：不是所有的优化都要做。蒸馏 + 量化是 pareto 前沿（性价比最高）；进一步的剪枝或再蒸馏就边际收益递减了。

5. **线上验证的重要性**：再好的离线指标，都要用 AB 测试在线验证。有时候，看起来不错的优化（模型更精）反而因为延迟或成本问题不值得。
