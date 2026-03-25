# 用户意图语义增强 + 广告召回优化：基于 LLM Profile 的语义召回系统

> 一句话定位：用 LLM 将用户行为序列转化为语义 Profile，结合向量化广告库实现语义召回，提升冷启动用户和长尾广告的匹配质量

**核心技术栈：** LLM User Profiling | BGE-M3 Embedding | Milvus HNSW | ANN Recall | 双塔模型 | 多路召回融合 | INT8 量化部署

---

## 1. 项目背景与痛点

### 业务规模

| 指标 | 规模 |
|------|------|
| 用户 DAU | 100万~500万 |
| 广告库规模 | 100万+素材 |
| 每日新增广告 | 8000~10000条 |
| 峰值 QPS | 2万~5万 |
| 冷启动用户占比 | 约 20~30%（新注册 + 低活跃） |

### 核心痛点

**痛点一：传统标签系统的局限**
- 用户标签体系依赖平台内行为积累（年龄/性别/兴趣类别，共 500~1000个）
- 标签粒度粗（「喜欢运动」无法区分「跑步」vs「篮球」vs「瑜伽」）
- 标签更新延迟：通常 T+1 天更新，无法捕捉用户实时意图变化

**痛点二：冷启动用户体验差**
- 新注册用户：无历史行为，标签为空
- 低活跃用户：标签稀疏，召回广告相关性低
- 结果：冷启动用户 CTR 比活跃用户低 40~60%，广告主对新用户流量报怨

**痛点三：长尾广告曝光不足**
- 传统协同过滤召回偏向热门广告（马太效应）
- 长尾广告（曝光 < 1000次/天）因历史数据少，难以被召回
- 对广告主不公平，对用户也可能错过真正相关的广告

**解决思路：** 用 LLM 将用户行为序列提炼成语义化 Profile，广告文案也做语义向量化，实现「用户意图 ↔ 广告含义」的语义空间匹配

---

## 2. 系统架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                  用户意图语义增强召回系统                          │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    离线 Profile 生成                        │ │
│  │                                                            │ │
│  │  用户行为日志（T-N天）                                       │ │
│  │  ┌─────────────────────────────────────────────────────┐  │ │
│  │  │ 搜索词序列 | 点击标题序列 | 购买商品名 | 浏览类目     │  │ │
│  │  └──────────────────────┬──────────────────────────────┘  │ │
│  │                         ▼                                   │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │           LLM 语义推断（批量离线）                     │  │ │
│  │  │  输出：「该用户近期准备健身，对蛋白粉/运动装备感兴趣，  │  │ │
│  │  │         消费能力中等，倾向于性价比产品」               │  │ │
│  │  └──────────────────────┬──────────────────────────────┘  │ │
│  │                         ▼                                   │ │
│  │          Embedding 模型（BGE-M3）→ 用户语义向量              │ │
│  │          存储到 Redis（在线召回使用）                         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    广告语义向量库                           │ │
│  │                                                            │ │
│  │  广告文案 + 落地页关键词 → BGE-M3 → 广告向量               │ │
│  │  Milvus（HNSW 索引，100万广告，6GB）                       │ │
│  │  每日增量更新（新增 1万条广告）                              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    在线召回（< 20ms）                       │ │
│  │                                                            │ │
│  │  用户请求                                                   │ │
│  │      │                                                     │ │
│  │      ├── 语义召回：用户语义向量 → ANN 检索 → 200个候选     │ │
│  │      ├── 标签召回：传统定向标签 → 倒排索引 → 500个候选     │ │
│  │      └── 协同过滤：历史 CF 模型 → 200个候选               │ │
│  │                          │                                 │ │
│  │              多路融合（加权合并去重）                        │ │
│  │                          │                                 │ │
│  │              排序层（CTR 预估模型打分）                      │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 数据流时序

```
离线流（T-1 天执行）：
  用户行为日志 → Spark 批处理聚合序列 → LLM Profile 生成
  → BGE-M3 向量化 → 写入 Redis + Milvus 用户向量库
  
  广告新增 → 广告文案向量化 → 插入 Milvus 广告向量库
  耗时：4~6小时（批量处理）

在线流（每次广告请求，< 20ms）：
  用户 ID → 查询 Redis 获取语义向量（< 1ms）
  → Milvus ANN 检索 Top-200 候选（< 5ms）
  → 多路合并 + 粗排（< 5ms）
  → 精排 CTR 模型（< 10ms）
  → 返回 Top-N 广告
```

---

## 3. 核心算法

### 3.1 用户行为序列的语义化

**输入格式设计：**

```python
USER_PROFILE_PROMPT = """
你是用户行为分析专家。请根据用户最近的行为序列，推断其当前意图和兴趣。

## 用户行为序列（最近14天，时间从远到近）
### 搜索记录
{search_queries}  # 示例："蛋白粉什么牌子好", "健身房推荐", "跑步鞋"

### 点击的内容标题
{clicked_titles}  # 示例："如何3个月增肌10斤", "初学者健身计划"

### 购买或加购的商品
{purchase_items}  # 示例："运动水壶（已购）", "哑铃（加购未买）"

### 浏览的类目路径
{browse_categories}  # 示例：运动健康 > 健身器材 > 哑铃

## 输出要求
请输出以下 JSON 结构：
{
  "current_intent": "用户当前最强烈的需求（一句话，不超过30字）",
  "interests": ["兴趣标签1", "兴趣标签2", ...],  # 细粒度标签，最多8个
  "consumption_stage": "浏览期/考虑期/决策期",
  "price_sensitivity": "价格敏感/中等/不敏感",
  "user_description": "一段自然语言描述，用于向量化（50~100字）"
}
"""
```

**实际输出示例：**

```json
{
  "current_intent": "正在准备开始健身，需要基础器材和营养补剂",
  "interests": ["健身入门", "增肌", "蛋白粉", "哑铃", "运动服装", "跑步", "健康饮食"],
  "consumption_stage": "决策期",
  "price_sensitivity": "中等",
  "user_description": "该用户近期持续搜索健身相关内容，已购买运动水壶，
  正在考虑购买哑铃，对蛋白粉品牌有明确兴趣，是健身新手，注重性价比"
}
```

**向量化：**
```python
# 将 user_description 向量化作为语义召回的 Query
user_vector = bge_m3_model.encode(profile["user_description"],
                                   normalize_embeddings=True)
# 存储到 Redis（TTL 24小时，每天更新）
redis.set(f"user_semantic:{user_id}", user_vector.tobytes(), ex=86400)
```

### 3.2 广告语义向量化

```python
def encode_ad(ad: dict) -> np.ndarray:
    """
    将广告多维信息融合为单一语义向量
    """
    # 拼接广告的多个文本字段
    ad_text = f"""
    广告标题：{ad['title']}
    广告描述：{ad['description']}
    商品名称：{ad['product_name']}
    类目：{ad['category']}
    核心卖点：{ad['selling_points']}
    适用人群：{ad['target_audience']}
    """
    
    # BGE-M3 支持最长 8192 token，足够覆盖广告所有字段
    vector = bge_m3_model.encode(ad_text.strip(), normalize_embeddings=True)
    return vector  # shape: (1024,)

# 批量向量化（利用 GPU batch 推理）
def batch_encode_ads(ads: list, batch_size=256) -> np.ndarray:
    texts = [format_ad_text(ad) for ad in ads]
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        vectors = bge_m3_model.encode(batch, normalize_embeddings=True,
                                      show_progress_bar=False)
        all_vectors.append(vectors)
    return np.vstack(all_vectors)
```

### 3.3 HNSW 索引参数详解

```
HNSW（Hierarchical Navigable Small World）索引参数：

M = 32：
  - 含义：每个节点在图中的最大连接数（出度）
  - 太小（M=8）：精度下降，recall@100 可能 < 85%
  - 太大（M=64）：构建慢，内存占用大（索引大小 ∝ M）
  - M=32 在精度和资源消耗间取得平衡

efConstruction = 200：
  - 含义：构建索引时，每步搜索的候选集大小（越大构建越准但越慢）
  - 建议：至少是 2*M，200 对百万量级是合理选择

ef（在线检索参数）= 64：
  - 含义：查询时维护的候选集大小（越大精度越高但延迟增加）
  - ef=64 在 recall@100 ≈ 97% 和延迟 < 5ms 间平衡
  - 若需要更高精度，可提高到 ef=128（延迟约增加 1倍）

精度-速度权衡：
  ef=32：  recall@100 ≈ 93%，延迟 ~2ms
  ef=64：  recall@100 ≈ 97%，延迟 ~4ms  ← 推荐
  ef=128： recall@100 ≈ 99%，延迟 ~8ms
```

**Milvus 索引构建代码：**

```python
from pymilvus import Collection, utility

index_params = {
    "metric_type": "IP",          # 内积（向量已 L2 归一化，等价余弦相似度）
    "index_type": "HNSW",
    "params": {
        "M": 32,
        "efConstruction": 200
    }
}

collection.create_index(
    field_name="embedding",
    index_params=index_params
)

# 在线检索
search_params = {"ef": 64}

results = collection.search(
    data=[user_vector],           # 查询向量
    anns_field="embedding",
    param=search_params,
    limit=200,                    # 召回 200 个候选
    output_fields=["ad_id", "category", "ctr_score"]
)
```

### 3.4 增量索引更新策略

```python
class AdVectorIndexManager:
    """管理广告向量库的增量更新"""
    
    def daily_update(self, new_ads: list):
        """每日新增广告的增量插入"""
        # 1. 向量化新广告
        vectors = batch_encode_ads(new_ads)
        
        # 2. 插入 Milvus（支持在线插入，不需重建索引）
        entities = [
            [ad['id'] for ad in new_ads],           # id
            vectors.tolist(),                         # embedding
            [ad['ctr_history'] for ad in new_ads],   # 历史 CTR（若有）
            [ad['category_id'] for ad in new_ads],   # 类目
        ]
        collection.insert(entities)
        
        # 3. 强制 flush（确保数据持久化）
        collection.flush()
        
        print(f"增量插入 {len(new_ads)} 条广告，"
              f"Milvus Growing Segment 延迟 < 5秒可搜索")
    
    def weekly_optimize(self):
        """每周执行索引优化（离线，不影响在线服务）"""
        # compaction: 合并小 segment，优化存储效率
        utility.compact(collection.name)
        
        # 在备用节点重建索引后切流量（零停机）
        # 此操作在 K8s 环境下通过 rolling update 实现
```

---

## 4. 与 CTR 模型的集成

### 4.1 双塔结构中的语义塔设计

```
传统双塔（仅标签特征）：
  用户塔：[年龄 | 性别 | 兴趣标签 ID | 历史点击广告 ID]
  广告塔：[广告 ID | 类目 ID | 历史 CTR]

增强双塔（加入语义特征）：
  用户塔：[传统特征] + [语义 Embedding 256维]
  广告塔：[传统特征] + [语义 Embedding 256维]
  
  匹配分数 = 传统双塔分数 × α + 语义相似度 × (1-α)
  其中 α 是可学习参数，根据 A/B 实验调整
```

### 4.2 Embedding 特征对齐

```python
class SemanticTower(nn.Module):
    """将 1024维 BGE-M3 向量压缩到 CTR 模型使用的维度"""
    
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.projection(embedding)
```

### 4.3 Fine-tuning vs Frozen Embedding

```
策略对比：

Frozen Embedding（推荐，初期使用）：
  BGE-M3 参数冻结，只训练 Projection 层和 CTR 模型
  优点：训练稳定，GPU 显存需求低，不需要大量标注数据
  缺点：BGE-M3 的语义空间未针对广告 CTR 任务优化
  
Fine-tuning（数据量足够时使用）：
  用广告点击数据对 BGE-M3 的最后几层 Fine-tune（LoRA）
  优点：语义空间向 CTR 相关性对齐，效果更好
  缺点：需要大量配对数据（点击 vs 未点击），训练成本高
  
实践建议：
  第一阶段：Frozen + 新增 Projection 层（快速上线）
  第二阶段：积累 3个月线上数据后，用 LoRA Fine-tune
  指标：Recall@100 提升 2~5pp（相比 Frozen 方案）
```

---

## 5. 系统规模估算

### 用户 Profile 生成成本

```
规模：100万 DAU，每天更新活跃用户 Profile
活跃用户定义：过去 24小时有行为的用户

实际需要更新的用户数：
  - 新注册用户：约 1万/天
  - 行为更新超过阈值的活跃用户：约 10万/天
  - 合计：约 11万用户/天需要更新 Profile

LLM 调用成本（GPT-4o-mini，批量模式）：
  - 每用户 Profile 生成：约 800 tokens 输入 + 200 tokens 输出
  - 10个用户合并一次 API 调用：约 9000 tokens/次
  - 总调用次数：11万 / 10 = 1.1万次/天
  - 费用：输入 $0.15/1M × 9K × 1.1万 = $14.85/天
          输出 $0.6/1M × 2K × 1.1万 = $13.20/天
  - 合计：约 $28/天（约 200元/天，可接受）

使用本地模型（Qwen-7B）替代：
  - 4张 A100（每张 80GB），INT8 量化后单张可跑 7B 模型
  - 批量推理吞吐：约 1000 tokens/秒/卡 × 4卡 = 4000 tokens/秒
  - 11万用户 × 1000 tokens/用户 ÷ 4000 = 约 27.5秒（极快）
  - 成本：接近 0（只有电费）
```

### 广告向量库规模

```
存储计算：
  广告总量：100万条
  Embedding 维度：1024（BGE-M3）
  数据类型：float32（4字节）
  原始向量存储：1M × 1024 × 4 = 4GB
  HNSW 索引额外开销（M=32）：约 1.5倍 = 6GB
  元数据（广告 ID、类目、CTR 等）：约 200MB
  总计：约 6.2GB（可放入单台 64GB 内存服务器）

检索性能：
  100万向量，HNSW（ef=64）：
  - 延迟：< 5ms（单线程），< 2ms（多线程并发）
  - QPS：单机可支撑 2000+ QPS（足够覆盖峰值 2~5万广告请求，
    因为多个广告请求可复用同一用户的检索结果）
```

### 在线延迟分解

```
用户广告请求端到端延迟：

1. 获取用户语义向量（Redis 查询）：< 1ms
2. Milvus 语义召回 Top-200：< 5ms
3. 传统标签召回（倒排索引）：< 2ms
4. 多路合并去重：< 1ms
5. CTR 模型精排（GPU 推理）：< 10ms
6. 过滤 + 排序 + 返回：< 1ms
─────────────────────────────────
合计：< 20ms（满足在线广告系统要求）
```

---

## 6. 本地小模型部署方案（适合小公司）

### 4 张 A100 的资源分配

```
GPU 资源规划（4张 A100 80GB）：

卡 0-1：Embedding 推理服务（BGE-M3）
  - 用于广告向量化（每日新增 1万条）
  - 用于用户 Profile 向量化（每日 11万条）
  - INT8 量化后显存占用：约 6GB/卡
  - 批量吞吐：约 2000 tokens/秒/卡

卡 2：CTR 精排模型
  - DeepFM/DCN-V2 模型（参数量 1亿左右）
  - 显存：约 4GB
  - 在线 QPS：> 5000（远超业务需求）

卡 3：LLM 用户 Profile 生成（可选，替代 API）
  - Qwen-7B INT8 量化，显存：约 8GB
  - 批量离线处理，不占用在线请求延迟

CPU 服务器（2台，32核/256GB）：
  - Milvus 向量检索服务（内存密集型）
  - Redis 缓存服务
  - ClickHouse 数据查询服务
```

### INT8 量化部署

```python
from transformers import AutoModel
from auto_gptq import AutoGPTQForCausalLM  # 或使用 llama.cpp / vLLM

# BGE-M3 INT8 量化（Embedding 推理）
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

# 转换为 ONNX + INT8 量化
model = ORTModelForFeatureExtraction.from_pretrained(
    "BAAI/bge-m3",
    export=True,
    provider="CUDAExecutionProvider"
)
# INT8 量化后：显存从 ~12GB 降到 ~6GB，速度提升 1.5~2倍

# 量化对精度的影响：
# FP32 vs INT8 BGE-M3 的 Recall@100 差异通常 < 0.5%，可以接受
```

---

## 7. 面试高频考点

**Q：LLM 生成的用户 Profile 准确吗？如何评估？**

A：
- **挑战**：没有 Profile 的 Ground Truth 标签，无法直接评估准确性
- **间接评估方法**：
  1. **召回相关性**：用 Profile 做 ANN 检索，人工评估召回广告是否与用户行为相关（样本标注）
  2. **在线指标**：使用语义召回的广告 CTR，对比传统标签召回的 CTR（A/B 测试）
  3. **Profile 一致性**：同一用户多次生成的 Profile 应高度相似（余弦相似度 > 0.9）
  4. **Case Study**：抽取 100个用户的行为序列和生成 Profile，人工评估合理性
- **典型问题**：行为稀疏时 LLM 容易过度推断（看了1篇健身文章就说"热衷健身"），需要在 Prompt 中加入置信度约束

---

**Q：语义召回 vs 协同过滤召回，各自适用什么场景？**

A：

| 维度 | 语义召回 | 协同过滤召回 |
|------|---------|------------|
| 适用场景 | 冷启动用户、长尾广告 | 数据充足的活跃用户 |
| 数据需求 | 用户行为序列（少量即可）| 大量历史点击数据 |
| 发现长尾 | 好（语义空间无偏） | 差（热门偏见严重） |
| 准确性 | 受 LLM 生成质量影响 | 受协同滤波矩阵稀疏性影响 |
| 可解释性 | 强（可看 Profile 解释）| 弱（黑盒相似性） |
| 计算成本 | ANN 检索 < 5ms | 矩阵计算/查表 < 1ms |

---

**Q：冷启动用户（新用户）的 Profile 怎么初始化？**

A：
1. **注册信息推断**：用注册时填写的年龄/性别/城市做基础 Profile
2. **设备信息推断**：手机型号、系统版本 → 价格段推断（iPhone 14 vs 低端机）
3. **安装来源推断**：从哪个 App 跳转而来（如从健身 App 跳来 → 加入「健康」兴趣）
4. **实时兴趣更新**：用户在 App 内首次点击/搜索后，立即触发 Profile 更新（不等到次日离线批处理）
5. **通用 Profile 兜底**：新用户前 5分钟使用「通用新用户 Profile」（基于新用户群体的平均行为）

---

**Q：如何证明语义增强对最终 CTR 有正向效果？**

A：
严格的 A/B 测试设计：
1. **实验分组**：随机将用户 ID 分为对照组（纯传统召回）和实验组（传统 + 语义召回）
2. **观察指标**：
   - 主指标：CTR（点击率）、CVR（转化率）
   - 辅助指标：召回多样性（覆盖的广告类目数）、新广告曝光比例
3. **显著性检验**：Z-test 或 T-test，P-value < 0.05 才算显著
4. **分层分析**：
   - 冷启动用户（< 30天）的 CTR 提升应更明显
   - 活跃用户（> 180天）提升可能较小（传统方法已够好）
5. **实验周期**：至少 2周（覆盖完整的周期性因素）
6. **Guardrail 指标**：确保用户体验指标（广告点踩率、APP 使用时长）没有下降

---

**Q：向量维度怎么选？1536维 vs 768维 vs 256维的权衡？**

A：

| 维度 | 模型示例 | 精度 | 存储（100万向量）| 检索延迟 | 推荐场景 |
|------|---------|------|---------|---------|---------|
| 1536维 | OpenAI text-embedding-3-large | 最高 | 6GB | 较慢 | 高精度要求，资源充足 |
| 1024维 | BGE-M3 | 高 | 4GB | 中等 | 推荐，均衡 |
| 768维 | BGE-large | 较高 | 3GB | 快 | 资源受限 |
| 256维 | 压缩/量化 | 中等 | 1GB | 极快 | 超大规模（10亿级别）|

对于 100万广告规模，推荐 1024维（BGE-M3），性能和成本的最优平衡点。

---

**Q：在线 ANN 检索的 recall 不够高怎么办？有哪些提升手段？**

A：
1. **提高 ef 值**：ef=64→128，精度从 97% 提升到 99%（延迟约翻倍）
2. **多向量检索**：用多个视角的 Embedding 联合检索（如用户的多个兴趣 Profile 分别检索，取并集）
3. **Cascade 检索**：先用粗粒度索引快速过滤，再用精细索引精确检索
4. **重排模型（Re-ranking）**：ANN 召回 500个候选后，用 Cross-Encoder 精确重排（更准但更慢）
5. **维度压缩检验**：若是降维导致精度损失，改用 Matryoshka Embedding（支持变长维度的模型）
6. **数据质量**：检查广告文案质量（空文案、重复文案会污染向量空间）

---

**Q：多路召回融合时，语义召回和传统召回如何打分统一？**

A：
不同召回路的分数量纲不同，需要归一化后合并：

```python
def merge_recall_results(semantic_results, cf_results, tag_results):
    """
    多路召回结果融合
    semantic_results: [(ad_id, cosine_sim), ...]  # 分数范围 [-1, 1]
    cf_results: [(ad_id, cf_score), ...]           # 分数范围 [0, ∞)
    tag_results: [(ad_id, tag_match_count), ...]   # 分数范围 [0, N]
    """
    # 方法一：各路分数 Min-Max 归一化后加权合并
    merged = {}
    
    for ad_id, score in semantic_results:
        merged[ad_id] = merged.get(ad_id, 0) + 0.4 * normalize(score, 'semantic')
    
    for ad_id, score in cf_results:
        merged[ad_id] = merged.get(ad_id, 0) + 0.4 * normalize(score, 'cf')
    
    for ad_id, score in tag_results:
        merged[ad_id] = merged.get(ad_id, 0) + 0.2 * normalize(score, 'tag')
    
    # 按融合分数排序
    sorted_results = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    return [ad_id for ad_id, _ in sorted_results[:1000]]  # 取 Top-1000 进精排
```

---

## 8. 项目效果（量化指标）

| 指标 | 优化前 | 优化后 | 提升幅度 |
|------|------|------|---------|
| 冷启动用户（<30天）CTR | 基线 | +20%~35% | 核心改进点 |
| 长尾广告曝光占比（<1000次/天）| ~5% | ~12% | +7pp |
| 整体召回多样性（Intra-list Diversity）| 基线 | +15% | 减少重复广告 |
| 整体 CTR（含活跃用户）| 基线 | +3%~8% | 语义召回整体贡献 |
| 广告主满意度（新广告 eCPM）| 基线 | +10%~15% | 更多长尾广告被展示 |

**注：** 效果数据参考同类系统（百度文心广告语义召回、腾讯广告语义增强等）的公开报告范围。实际效果取决于数据质量和调优程度。

---

## 9. 技术亮点总结（面试用）

1. **LLM Profile 作为中间表示**：将离散的行为序列转化为连续的语义向量，弥合了「用户行为」和「广告语义」之间的语言 Gap
2. **冷启动专项优化**：语义召回不依赖历史交互数据，天然解决冷启动问题
3. **资源优化方案**：INT8 量化 + Batch 推理，在 4张 A100 上完整跑通全套流程
4. **工程可落地性**：离线生成 Profile（非实时），Milvus 增量更新，在线延迟 < 20ms，满足生产环境要求
5. **可验证效果**：通过 A/B 测试框架，分层分析冷启动用户 vs 活跃用户的不同改善幅度

---

*文档版本：v1.0 | 适用场景：搜广推算法工程师面试 | 业务规模：中小型广告平台*
