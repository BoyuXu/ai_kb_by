# 知识库文档分类规则

## 新文档入库流程

每次生成新文档时，按以下规则分类：

### 1. 判断领域
- 关键词含 CTR/CVR/竞价/bidding/creative/广告 → `ads/`
- 关键词含 recall/ranking/cold-start/推荐/sequence → `rec-sys/`
- 关键词含 retrieval/reranking/query/search/检索 → `search/`
- 关键词含 LLM推理/KV Cache/MoE/显存/serving → `llm-infra/`
- 跨多领域 → `cross-domain/`

### 2. 判断类型
- 单篇论文笔记（有arxiv编号/论文标题） → `papers/`
- 大厂工业案例（有公司名/产品名） → `practices/`
- 多篇论文对比/演进脉络/框架总结 → `synthesis/`

### 3. 命名规范
- Papers: `YYYYMMDD_snake_case_short_name.md`
- Practices: `company_product_topic.md`
- Synthesis: `topic_keyword.md` 或 `std_domain_topic.md`

### 4. 引用规范
- Synthesis 文件头部必须有 `> 📚 参考文献` 区域
- 引用格式：`[简称](../papers/filename.md) — 一句话描述`
- 每次更新 synthesis 时检查引用完整性

### 5. 索引更新
- 新增文件后更新对应领域 INDEX.md
