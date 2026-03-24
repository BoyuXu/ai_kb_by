# AI 知识库整体结构（搜广推 + LLM）

## 🎯 使命

这是 MelonEggLearn 的学习型知识库，用于系统梳理和积累搜索、广告、推荐等领域的算法知识和工程经验。

**目标**：
- 📚 沉淀学术论文理解
- 🏢 记录工业最佳实践
- 🧠 总结演进规律和框架
- 🚀 支撑面试准备和工程应用

---

## 📂 目录树

```
~/Documents/ai-kb/
│
├── KNOWLEDGE_BASE_STRUCTURE.md  ← 你在这里
├── CLASSIFICATION_RULES.md      ← 文档分类规则
├── LEARNING_CONFIG.md           ← 学习配置
├── LLM_INTEGRATION_ROADMAP.md   ← LLM 集成全景
│
├── 🔍 search/                   ← 搜索系统领域
│   ├── INDEX.md                 ← 搜索知识库导航 📌 从这里开始
│   ├── papers/                  ← ~75 篇学术论文笔记
│   ├── practices/               ← 工业实践案例
│   └── synthesis/               ← 12 篇提炼总结
│       ├── std_search_*.md (6)  (标准化文档)
│       ├── 20260320~23_*.md (4) (专题综合)
│       ├── 01_search_ranking.md
│       └── llm_integration_framework.md
│
├── 🎯 ads/                      ← 广告系统领域
│   ├── INDEX.md                 ← 广告知识库导航 📌 从这里开始
│   ├── papers/                  ← ~85 篇学术论文笔记
│   ├── practices/               ← 2 篇工业实践案例
│   └── synthesis/               ← 15 篇提炼总结
│       ├── std_ads_*.md (7)     (标准化文档)
│       ├── 20260320~23_*.md (4) (专题综合)
│       ├── ads_ranking_evolution.md
│       ├── auto_bidding_evolution.md
│       ├── llm_integration_framework.md
│       └── mixing_ranking_evolution.md
│
├── 🎬 rec-sys/                  ← 推荐系统领域
│   ├── INDEX.md                 ← 推荐知识库导航 📌 从这里开始
│   ├── papers/                  ← ~80 篇学术论文笔记
│   ├── practices/               ← 工业实践案例
│   └── synthesis/               ← 26 篇提炼总结
│       ├── std_rec_*.md (9)     (标准化文档)
│       ├── 20260321~23_*.md (6) (专题综合)
│       ├── 00~09_*.md (10)      (基础总结)
│       └── llm_integration_framework.md
│
├── 🧠 llm-infra/                ← LLM 基础设施领域
│   ├── INDEX.md                 ← LLM 基础设施导航 📌 从这里开始
│   ├── *.md (根级)              ← ~80 篇论文笔记（含 KV Cache/MoE/RAG/Speculative Decoding）
│   ├── papers/                  ← 新论文归档
│   ├── practices/               ← 工业实践案例
│   └── synthesis/               ← 14 篇提炼总结
│       ├── std_llm_*.md (7)     (标准化文档)
│       └── 20260320~23_*.md (7) (专题综合)
│
├── 🔗 cross-domain/             ← 跨域综合
│   └── synthesis/               ← 11 篇提炼总结
│       ├── std_cross_*.md (6)   (标准化文档)
│       └── 20260320~24_*.md (5) (专题综合)
│
├── 🎤 interview/                ← 面试准备
│   ├── cards/                   ← 面试卡片
│   ├── sessions/                ← 模拟面试记录
│   ├── synthesis/               ← 6 篇面试综合
│   │   ├── card_*.md (5)        (知识卡片)
│   │   └── 20260324_interview_storytelling.md
│   ├── qa-bank.md               ← 问题库
│   ├── system-design.md         ← 系统设计
│   └── scenario-questions.md    ← 场景题
│
├── 📄 papers/                   ← 根级通用
│   └── reading-list.md          ← 阅读清单
│
├── 📝 synthesis/                ← 根级模板
│   └── TEMPLATE.md              ← 综合文档模板
│
├── 📦 其他辅助目录
│   ├── llm4rec/                 ← LLM for Rec 专题
│   ├── quant-ml/                ← 量化 ML
│   ├── quant-research/          ← 量化研究
│   ├── tech-evolution/          ← 技术演进
│   ├── weekly/                  ← 周报
│   ├── resume/                  ← 简历
│   ├── repo/ & repos/           ← 代码仓库笔记
│   ├── agent-context-management/← Agent 上下文管理
│   └── scripts/                 ← 工具脚本
```

---

## 📊 文档统计

| 领域 | Papers | Practices | Synthesis | 总计 |
|------|--------|-----------|-----------|------|
| **search** | ~75 | 0 | 12 | ~87 |
| **ads** | ~85 | 2 | 15 | ~102 |
| **rec-sys** | ~80 | 0 | 26 | ~106 |
| **llm-infra** | ~80 | 0 | 14 | ~94 |
| **cross-domain** | - | - | 11 | 11 |
| **interview** | - | - | 6 | 6+ |
| **总计** | ~320 | 2 | 84 | ~406 |

---

## 🏗️ 三层架构

每个领域目录遵循统一三层结构：

1. **papers/** — 单篇论文笔记（有 arXiv 编号/论文标题）
2. **practices/** — 大厂工业案例（有公司名/产品名）
3. **synthesis/** — 多篇论文对比/演进脉络/框架总结

### 命名规范
- Papers: `YYYYMMDD_snake_case_short_name.md`
- Practices: `company_product_topic.md`
- Synthesis: `topic_keyword.md` 或 `std_domain_topic.md`

### 引用规范
- Synthesis 文件头部有 `> 📚 参考文献` 区域
- 引用格式：`[简称](../papers/filename.md) — 一句话描述`

详细分类规则参见 [CLASSIFICATION_RULES.md](./CLASSIFICATION_RULES.md)

---

## 📝 最后更新
- **最后更新**: 2026-03-24
- **更新内容**: 二次重组 — 根级文件归位 + 新建 llm-infra/synthesis、cross-domain、interview/synthesis
