# 知识库文档分类规则（v2 — 2026-03-24 更新）

## 新文档入库流程

每次生成新文档时，严格按以下规则分类和命名。

### 1. 判断领域
- CTR/CVR/竞价/bidding/creative/广告/auction → `ads/`
- recall/ranking/cold-start/推荐/sequence/embedding/user behavior → `rec-sys/`
- retrieval/reranking/query/search/检索/sparse/dense → `search/`
- LLM推理/KV Cache/MoE/显存/serving/attention/fine-tuning/alignment → `llm-infra/`
- 跨多领域/多任务/通用ML → `cross-domain/`
- 面试技巧/讲述方法 → `interview/`

### 2. 判断类型
- 单篇论文笔记（有 arXiv 编号/论文标题） → `papers/`
- 大厂工业案例（有公司名/产品名） → `practices/`
- 多篇论文对比/演进脉络/框架总结 → `synthesis/`

### 3. ⚠️ 命名规范（重要！）

#### Papers 命名：用论文名，不要日期前缀
```
✅ CADET_context_conditioned_ads_ctr.md
✅ DIN_deep_interest_network.md
✅ ColBERT_v2_late_interaction.md
✅ ESMM_entire_space_multitask.md

❌ 20260325_cadet_context_ctr.md      ← 禁止日期前缀
❌ 20260325_din.md                     ← 禁止日期前缀
```
- 保留关键缩写（DIN、ESMM、ColBERT、SPLADE 等）
- 文件名用英文 snake_case
- 不超过 60 字符

#### Synthesis 命名：用中文标题
```
✅ 广告CTR_CVR预估与校准.md
✅ 推荐排序模型演进.md
✅ 混合检索技术演进.md
✅ LLM集成框架_广告系统.md

❌ std_ads_ctr_cvr_calibration.md      ← 禁止英文代号
❌ 20260325_rec_sys_synthesis.md        ← 禁止日期前缀
```

#### Practices 命名
```
✅ google_ads_llm_application.md
✅ netflix_recommendation_system.md
```

### 4. 引用规范
- Synthesis 文件头部必须有 `> 📚 参考文献` 区域
- 引用格式：`[简称](../papers/filename.md) — 一句话描述`
- 每次更新 synthesis 时检查引用完整性

### 5. Synthesis 内容标准
每个 synthesis 文件必须包含：
- **📚 参考文献**：可跳转链接到对应 papers
- **📐 核心公式与原理**：3-5 个关键公式，含解释
- **🎯 核心洞察**：5 条以上
- **📈 技术演进脉络**：时间线或流程图
- **🎓 面试考点**：≥10 个 Q&A（30秒答案格式）

### 6. 索引更新
- 新增文件后更新对应领域 INDEX.md
- 新增 synthesis 时检查是否应合并到已有文件（避免碎片化）

### 7. 判断"新建 vs 合并"
- 如果新学内容与已有 synthesis 主题高度重叠（>70%）→ **合并到已有文件**，补充新内容
- 如果是全新主题或已有文件已超过 1500 行 → **新建文件**
- 合并时同步更新引用链接

### 8. 完整入库流程 checklist
```
□ 判断领域（ads/rec-sys/search/llm-infra/cross-domain）
□ 判断类型（papers/practices/synthesis）
□ 按规范命名（papers=论文名英文，synthesis=中文标题）
□ 放入正确目录：{domain}/{type}/filename.md
□ 如果是 synthesis：添加参考文献+公式+≥10个Q&A
□ 更新 INDEX.md
□ git add + commit
```
