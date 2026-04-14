# 广告创意生成 + CTR 评估闭环：基于 LLM+RAG 的自动化创意优化系统

> 一句话定位：用 LLM 自动生成高质量广告创意，结合 CTR 预估模型实现「生成-评估-反馈」闭环，大幅提升创意迭代效率

**核心技术栈：** LLM Creative Generation | RAG | Milvus/FAISS | BGE Embedding | CTR Prediction | MMR | Constrained Decoding | A/B Testing

---

## 1. 项目背景与痛点

### 业务规模

| 指标 | 规模 |
|------|------|
| 广告库总量 | 100万+ 素材 |
| 每日新增广告 | 8000~10000条 |
| 人工创意制作成本 | 平均 30~60分钟/条（含文案、设计迭代） |
| A/B 测试出结论周期 | 1~2周（需足够的曝光量） |

### 核心痛点

**痛点一：创意质量参差不齐**
- 百万级广告库中，头部 10% 的创意贡献了 60%+ 的转化
- 底部 40% 的创意几乎无效，但仍在消耗预算
- 优质创意的经验难以快速复制到其他广告组

**痛点二：人工审核成本高**
- 每天新增 1万条广告，人工审核创意质量不现实
- 传统关键词过滤只能处理合规问题，无法评估质量
- 优化师需要凭经验判断哪个文案版本更好

**痛点三：迭代周期长**
- 一条新文案上线到 A/B 出结论：1~2周
- 每条广告只测试 2~3个版本（成本限制）
- 错过了快速探索高 CTR 文案空间的机会

**解决思路：** 用 LLM 批量生成多版本候选 → CTR 预估模型快速筛选 → 优质版本优先上线，缩短迭代周期

---

## 2. 系统架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                  广告创意生成与评估系统                            │
│                                                                 │
│  输入层                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐    │
│  │  商品信息    │  │  用户画像    │  │  投放约束条件         │    │
│  │  标题/描述  │  │  年龄/兴趣  │  │  字数/违禁词/风格     │    │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘    │
│         └────────────────┼──────────────────────┘              │
│                          ▼                                      │
│  RAG 层                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  历史高 CTR 创意库（向量库）                               │   │
│  │  商品信息向量化 → ANN 检索 → Top-K 参考创意               │   │
│  │  检索策略：语义相似 + 行业相关 + CTR 加权                  │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             ▼                                   │
│  生成层                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  LLM（GPT-4/Claude）                                    │   │
│  │  输入：商品 + 用户画像 + 参考创意 + 约束                   │   │
│  │  输出：20个候选文案版本                                    │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             ▼                                   │
│  评估层                                                         │
│  ┌──────────────────┐   ┌──────────────────────────────────┐   │
│  │  CTR 预估模型     │   │  合规检查                         │   │
│  │  对 20个版本打分  │   │  违禁词 / 格式 / 字数限制          │   │
│  └────────┬─────────┘   └───────────────┬──────────────────┘   │
│           └──────────────────┬───────────┘                      │
│                              ▼                                   │
│  多样性过滤（MMR 算法）                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  在 CTR 分数高 + 语义多样性大 的版本中选出 Top-3           │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             ▼                                   │
│  输出层：Top-3 候选创意 → A/B 测试 → 线上效果反馈               │
│                             │                                   │
│              ┌──────────────┘                                   │
│              ▼                                                   │
│  闭环更新：高 CTR 创意 → 更新向量库 → 提升 RAG 质量              │
└─────────────────────────────────────────────────────────────────┘
```

### 数据流说明

```
离线流（T-1 天预生成）：
  新广告入库 → 提取商品特征 → LLM 批量生成候选创意
  → CTR 预估打分 → 排序存储 → 次日可用

在线流（实时生成，可选）：
  广告主手动触发 → 简化 Prompt（去掉复杂 RAG）
  → 快速生成 5~10个版本 → 立即可用
  延迟：5~15 秒（可接受，非 RTB 链路）

闭环更新（每日批量）：
  统计昨日 CTR 数据 → CTR 高于阈值的创意
  → 向量化 → 插入 Milvus 创意库
  → 更新 CTR 加权分数
```

---

## 3. 核心技术细节

### 3.1 Prompt 设计

```python
CREATIVE_GENERATION_PROMPT = """
你是资深广告文案创作专家。请根据以下信息，生成 {n_versions} 个不同风格的广告文案。

## 商品信息
- 商品名称：{product_name}
- 核心卖点：{selling_points}
- 价格：{price}（原价 {original_price}）
- 商品类目：{category}

## 目标用户画像
- 年龄段：{age_range}
- 性别偏好：{gender}
- 兴趣标签：{interests}
- 消费能力：{consumption_level}

## 参考优质创意（来自同类目高 CTR 广告，仅供参考风格，不要直接复制）
{reference_creatives}

## 创作约束
- 标题字数：{min_len}~{max_len} 字
- 禁用词汇：{forbidden_words}
- 禁止夸大宣传：不使用「最」「第一」「绝对」等极限词
- 风格多样：每个版本应有明显不同的切入角度

## 输出格式（严格遵守 JSON 格式）
返回 JSON 数组，每个元素包含：
- "title": 广告标题
- "style": 文案风格（情感共鸣/功能突出/价格优惠/场景化/痛点解决）
- "highlight": 核心亮点词（3个以内）

请直接输出 JSON，不要有其他文字。
"""
```

### 3.2 RAG 实现细节

**向量化策略：**

```python
from sentence_transformers import SentenceTransformer

# 使用 BGE-M3 做中文创意向量化（本地部署，无 API 费用）
model = SentenceTransformer("BAAI/bge-m3")

def encode_creative(creative: dict) -> np.ndarray:
    # 将广告文案、类目、核心卖点拼接后向量化
    text = f"{creative['title']} | {creative['category']} | {creative['selling_point']}"
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding  # 维度：1024
```

**Milvus 索引构建：**

```python
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection

# Schema 设计
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="ctr", dtype=DataType.FLOAT),      # 历史 CTR
    FieldSchema(name="category_id", dtype=DataType.INT32),
    FieldSchema(name="creative_text", dtype=DataType.VARCHAR, max_length=512),
]

# 创建 HNSW 索引（平衡精度与速度）
index_params = {
    "metric_type": "IP",          # 内积（向量已归一化，等价余弦相似度）
    "index_type": "HNSW",
    "params": {
        "M": 32,                  # 每个节点的最大连接数（越大精度越高，构建越慢）
        "efConstruction": 200     # 构建时的搜索深度
    }
}
```

**CTR 加权检索策略：**

$$
\text{score}}_{\text{{final}} = \alpha \cdot \text{sim}}_{\text{{semantic}} + (1-\alpha) \cdot \text{ctr}}_{\text{{norm}}
$$

其中：
- $\text{sim}}_{\text{{semantic}}$：当前商品与历史创意的语义相似度（余弦相似度）
- $\text{ctr}}_{\text{{norm}}$：历史创意的 CTR 归一化值（按类目内排名归一化到 0~1）
- $\alpha = 0.6$：语义相似度权重更高，避免过度依赖历史偏见

```python
def retrieve_reference_creatives(query_embedding, category_id, top_k=10):
    # 向量检索（召回候选集）
    search_params = {"ef": 64}  # 在线检索的搜索深度（越大越准但越慢）
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k * 3,  # 多召回一些，后续重排
        expr=f"category_id == {category_id}"  # 同类目过滤
    )
    
    # CTR 加权重排
    candidates = results[0]
    for hit in candidates:
        semantic_score = hit.score
        ctr_normalized = normalize_ctr(hit.entity.ctr, category_id)
        hit.final_score = 0.6 * semantic_score + 0.4 * ctr_normalized
    
    candidates.sort(key=lambda x: x.final_score, reverse=True)
    return candidates[:top_k]
```

### 3.3 CTR 预估模型对生成文案的评估

**挑战：** CTR 模型通常是基于历史展示数据训练的，对于从未曝光的新创意，预测存在偏差（分布外问题）

**文本特征提取方案：**

```
方案对比：
┌──────────────────┬──────────────────────────┬──────────────────────┐
│ 方案              │ 优点                      │ 缺点                 │
├──────────────────┼──────────────────────────┼──────────────────────┤
│ TF-IDF           │ 计算快，无需 GPU            │ 无语义，OOV 问题严重  │
│ BERT Embedding   │ 语义理解强，泛化好           │ 需要 GPU，延迟高     │
│ BGE-M3 Embedding │ 中文效果好，可本地部署        │ 维度高，存储占用大    │
│ SimCSE Fine-tune  │ 针对广告场景优化，效果最好  │ 需要标注数据 fine-tune│
└──────────────────┴──────────────────────────┴──────────────────────┘
推荐方案：BGE-M3 Embedding（1024维）+ 降维到 256维（PCA）作为 CTR 模型的文本特征
```

**CTR 预估用于创意评分：**

$$
\text{CTR}}_{\text{{pred}} = \text{Model}(\mathbf{e}_{user}, \mathbf{e}_{creative}, \mathbf{x}_{context})
$$

其中 $\mathbf{e}_{creative}$ 是生成创意的 embedding，通过 CTR 模型的文本塔提取

### 3.4 多样性过滤：MMR 算法

最大边际相关性（Maximal Marginal Relevance）：

$$
\text{MMR} = \arg\max_{d_i \in R \setminus S} \left[ \lambda \cdot \text{CTR}(d_i) - (1-\lambda) \cdot \max_{d_j \in S} \text{Sim}(d_i, d_j) \right]
$$

其中：
- $R$：所有候选创意集合（20个候选）
- $S$：已选中的创意集合（初始为空）
- $\text{CTR}(d_i)$：候选创意的预测 CTR 分数
- $\text{Sim}(d_i, d_j)$：两条创意间的语义相似度
- $\lambda = 0.7$：质量和多样性的权衡参数

```python
def mmr_select(candidates, embeddings, ctr_scores, n_select=3, lambda_=0.7):
    selected = []
    selected_embeddings = []
    remaining = list(range(len(candidates)))
    
    # 贪心选择：每次选 MMR 分数最高的
    for _ in range(n_select):
        best_idx = None
        best_score = -float('inf')
        
        for idx in remaining:
            ctr_score = ctr_scores[idx]
            
            # 与已选创意的最大相似度
            if selected_embeddings:
                max_sim = max(cosine_similarity(embeddings[idx], emb)
                             for emb in selected_embeddings)
            else:
                max_sim = 0
            
            mmr_score = lambda_ * ctr_score - (1 - lambda_) * max_sim
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        selected.append(candidates[best_idx])
        selected_embeddings.append(embeddings[best_idx])
        remaining.remove(best_idx)
    
    return selected
```

---

## 4. 系统规模设计

### 向量库规模

```
存储估算：
  - 广告量：100万条
  - Embedding 维度：1024（BGE-M3）
  - 每个向量大小：1024 × 4字节（float32）= 4KB
  - 100万向量原始存储：1M × 4KB = 4GB
  - HNSW 索引额外开销：约原始大小的 1.5倍
  - 元数据（CTR、类目等）：约 100MB
  - 总计：约 6~8GB（可放入单台服务器内存）

检索性能：
  - HNSW 索引参数：M=32, efConstruction=200
  - 在线检索 ef=64，召回精度 recall@10 ≈ 98%
  - 检索延迟（100万量级）：< 5ms（单机内存检索）
  - 并发支持：100 QPS（单台 CPU 服务器）

增量更新策略：
  - 每日新增 1万条，不触发全量重建
  - Milvus 支持在线插入，自动合并到 HNSW 索引
  - 每周进行一次索引优化（compaction），离线执行不影响在线服务
```

### LLM 生成成本

```
离线批量生成（推荐方案）：
  - 每日新增 1万条广告，为每条生成 20个候选
  - 每次 LLM 调用：约 1500 tokens 输入 + 1000 tokens 输出
  - 批量模式（10条广告合并为 1次调用）：每次 ~8000 tokens
  - 总调用次数：1万 / 10 = 1000次/天
  - 成本（GPT-4 Turbo）：
    输入 $0.01/1K × 8 × 1000 = $80/天
    输出 $0.03/1K × 1 × 1000 = $30/天
    合计：约 $110/天

  使用本地 Qwen-72B 或 GPT-4o-mini 可降至 $20~30/天
```

### 在线服务架构

```
在线 CTR 预估服务：
  - 模型：轻量级 DCN-V2 或 DeepFM
  - 部署：单卡 A100（实测 1000 QPS 以上）
  - 文本特征：预先计算并缓存，在线查表即可
  - 延迟：< 5ms（批量 50条创意打分）

Milvus 集群：
  - 规模：2~4台 CPU 服务器（16核/64GB内存）
  - 读写分离：在线检索 + 离线索引构建分开部署
  - 高可用：主备复制，故障切换 < 30秒
```

---

## 5. 算法优化

### 5.1 Few-shot 动态选择

```python
def select_few_shot_examples(query_embedding, example_pool, n=3):
    """
    从 few-shot 示例库中动态选择最相关的 N 个示例
    避免使用固定示例（固定示例可能对某些类目效果差）
    """
    similarities = cosine_similarity(query_embedding, example_pool['embeddings'])
    top_n_indices = np.argsort(similarities)[-n:][::-1]
    
    # 额外过滤：确保示例的历史 CTR 高于类目平均值
    filtered = [i for i in top_n_indices
                if example_pool['ctr'][i] > example_pool['category_avg_ctr'][i]]
    
    return [example_pool['examples'][i] for i in filtered[:n]]
```

### 5.2 约束解码（Constrained Decoding）

确保生成的创意满足格式约束：

```python
import guidance

# 使用 guidance 库实现约束生成
@guidance.gen_block
def generate_ad_title(lm, max_len=20, min_len=10):
    lm += guidance.select(
        options=generate_options(max_len),
        list_append=True
    )
    # 保证：字数在 [min_len, max_len] 之间
    # 保证：不包含违禁词（词表过滤）
    # 保证：输出合法 JSON 格式
    return lm
```

简化方案（不依赖 guidance）：
```python
def validate_and_fix(generated_text, constraints):
    try:
        titles = json.loads(generated_text)
    except json.JSONDecodeError:
        # 提取 JSON 块，修复常见错误
        titles = repair_json(generated_text)
    
    # 过滤不合规创意
    valid_titles = []
    for item in titles:
        title = item.get("title", "")
        if (constraints.min_len <= len(title) <= constraints.max_len
                and not any(w in title for w in constraints.forbidden_words)):
            valid_titles.append(item)
    
    return valid_titles
```

### 5.3 多样性采样参数设置

```python
GENERATION_PARAMS = {
    "temperature": 0.9,    # 高温度增加多样性（不用 1.0 以防乱码）
    "top_p": 0.95,         # Nucleus sampling，过滤低概率 token
    "presence_penalty": 0.6,  # 惩罚重复词汇，增加新颖性
    "frequency_penalty": 0.3,  # 降低高频词使用，避免千篇一律
}

# 生成多版本时，对每个版本使用略不同的随机种子
# 确保同一 Prompt 的 N 次调用产生不同结果
```

---

## 6. 面试高频考点

**Q：如何评估生成创意的质量？离线指标 vs 在线指标？**

A：
- **离线指标**：
  - CTR 预估分数（模型预测，低成本快速评估）
  - 语言质量分：困惑度（Perplexity，越低越流畅）、BLEU/ROUGE（对比参考创意）
  - 多样性指标：N 个候选的平均成对语义距离（越大越多样）
  - 合规通过率：违禁词命中率、格式合法率
- **在线指标**（真正重要的指标）：
  - 实际 CTR：生成创意上线后的真实点击率
  - CVR：转化率（更难提升，但更有价值）
  - CTR 提升率：对比人工创意作为基线
  - AB 测试显著性：确保提升不是随机波动（P-value < 0.05）

---

**Q：RAG 检索的 top-k 怎么选？太多太少各有什么问题？**

A：
- **too small（top-k=1~3）**：参考信息不足，生成多样性差；容易过拟合单个高 CTR 创意的风格
- **too large（top-k=20+）**：Prompt 超长，增加 token 成本；噪声增多，LLM 注意力被稀释；召回的低相关创意干扰生成质量
- **推荐 top-k=5~8**：实验上通常在此区间效果最好
- **动态 top-k**：数据量少的新类目用较大的 k（借助跨类目知识），数据量大的成熟类目用较小的 k（精准匹配）

---

**Q：CTR 预估模型能准确评估未曝光的创意吗？如何处理分布外问题？**

A：
这是核心挑战。CTR 模型对训练分布内的样本预测准确，但对全新风格的创意存在偏差：
1. **问题根源**：训练数据是历史曝光的创意，新生成的创意在 embedding 空间中可能离训练样本较远
2. **缓解方案**：
   - 保守策略：CTR 模型只用于初筛（淘汰明显差的），不过度依赖分数排名
   - Uncertainty Estimation：使用 MC Dropout 或 Ensemble 估计预测不确定性，对不确定性高的创意降低置信度
   - 强制上线探索：一定比例（如 10%）的创意强制上线，不经 CTR 模型过滤，用于收集真实数据
3. **闭环修正**：收集未曝光→曝光后的真实 CTR，用于持续 fine-tune CTR 模型

---

**Q：如何处理违禁词和合规性检查？**

A：
- **关键词匹配**：维护违禁词表（极限词+行业特定词），生成后过滤
- **语义相似违禁**：用 embedding 相似度检测语义相近的违规表达（如「最优」→「最佳」是规避行为）
- **正则表达式**：检测价格误导（如虚假折扣）、联系方式泄露等
- **LLM 二次审查**：对于边界案例，用小模型（GPT-3.5）做一次合规评分
- **人工审核兜底**：评分在阈值附近的创意，发送人工审核队列

---

**Q：生成创意和人工创意的 CTR 对比结果如何？**

A：
- 行业参考数据（来自字节跳动 AutoCreative、百度 AIGC 广告等公开案例）：
  - 生成创意 vs 人工创意：CTR 相近（差距 -5%~+10%）
  - **关键优势不是单条质量，而是效率和覆盖**：人工 1条/小时 vs 系统 1000条/秒
  - 结合人工筛选：人工从 20个候选中选最好的，CTR 通常比单独人工创作高 15%~25%
  - A/B 测试结论：系统辅助（人工+AI）> 纯人工 > 纯 AI（无人工干预）

---

**Q：向量库百万级别的实时更新怎么做？增量索引 vs 重建？**

A：
- **Milvus 增量插入**：支持在线 insert，新向量插入到 growing segment，异步 flush 到磁盘
  - 优点：实时性好，不影响在线检索
  - 缺点：insert 后到可检索有短暂延迟（通常 1~5秒）
- **索引重建**：每周执行一次 compaction，合并小 segment，重建 HNSW 索引
  - 操作在备用节点上执行，完成后切流量，不影响线上服务
- **实践策略**：增量插入满足日常需求（每日 1万条新创意），定期重建保证检索性能不衰退
- **删除处理**：标记删除（soft delete）而非物理删除，避免频繁重建索引

---

## 7. 项目效果（量化指标）

| 指标 | 优化前 | 优化后 | 提升幅度 |
|------|------|------|---------|
| 创意制作时间 | 30~60分钟/条 | 5分钟/条（人工挑选）| 减少 85%+ |
| 每条广告测试版本数 | 2~3个版本 | 10~20个候选 | 提升 5~8倍 |
| A/B 测试出结论周期 | 1~2周 | 3~5天（更多版本并行测试）| 缩短 60% |
| 最优创意 CTR 提升 | 基线 | +10%~20%（行业参考）| 来自更大候选集 |
| 人工审核通过率 | ~80% | 初筛后 ~95% | 减少人工负担 |

---

## 8. 技术亮点总结（面试用）

1. **RAG + CTR 双重过滤**：不只是生成，而是「生成 → 评估 → 筛选」的完整流程
2. **MMR 多样性**：解决了纯分数排名导致的同质化问题，真正给出多样化的方案
3. **闭环机制**：线上效果反馈 → 更新创意库 → 提升 RAG 质量，系统越用越好
4. **成本控制**：通过批量生成 + 本地 Embedding 模型，将 AI 创作成本控制在可接受范围
5. **工程化思考**：离线预生成 vs 在线实时生成的权衡，贴合实际业务场景

---

*文档版本：v1.0 | 适用场景：搜广推算法工程师面试 | 业务规模：中小型广告平台*
