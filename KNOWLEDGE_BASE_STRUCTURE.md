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
├── LEARNING_CONFIG.md           ← 学习配置
├── LLM_INTEGRATION_ROADMAP.md   ← LLM 集成全景
│
├── 🔍 search/                   ← 搜索系统领域
│   ├── INDEX.md                 ← 搜索知识库导航 📌 从这里开始
│   ├── papers/                  ← 72 篇学术论文笔记
│   │   ├── 20260313_colbert_v2.md
│   │   ├── 20260316_dpr-dense-retrieval.md
│   │   ├── 20260316_llm-for-ir-survey.md
│   │   └── ... (其他论文笔记)
│   ├── practices/               ← 0 篇工业实践案例 (待补充)
│   │   └── (补充大厂案例)
│   └── synthesis/               ← 2 篇提炼总结
│       ├── 01_search_ranking.md          (搜索系统演进)
│       └── llm_integration_framework.md  (LLM 集成框架)
│
├── 🎯 ads/                      ← 广告系统领域
│   ├── INDEX.md                 ← 广告知识库导航 📌 从这里开始
│   ├── papers/                  ← 82 篇学术论文笔记
│   │   ├── 20260313_cadet_context_ctr.md
│   │   ├── 20260316_gsp-vcg-auction.md
│   │   └── ... (其他论文笔记)
│   ├── practices/               ← 2 篇工业实践案例
│   │   ├── p3_ltr_to_rl_ranking.md
│   │   └── p5_industrial_practice.md
│   └── synthesis/               ← 4 篇提炼总结
│       ├── llm_integration_framework.md  (LLM 集成框架)
│       ├── auto_bidding_evolution.md     (自动出价演进)
│       ├── ads_ranking_evolution.md      (广告排序演进)
│       └── mixing_ranking_evolution.md   (混合排序演进)
│
├── 🎬 rec-sys/                  ← 推荐系统领域
│   ├── INDEX.md                 ← 推荐知识库导航 📌 从这里开始
│   ├── papers/                  ← 69 篇学术论文笔记
│   │   ├── 00_overview.md
│   │   ├── 01_ctr_models_deep_dive.md
│   │   └── ... (其他论文笔记)
│   ├── practices/               ← 0 篇工业实践案例 (待补充)
│   │   └── (补充Netflix/YouTube/TikTok等大厂案例)
│   └── synthesis/               ← 11 篇提炼总结
│       ├── llm_integration_framework.md  (LLM 集成框架)
│       ├── recommendation_evolution.md   (推荐系统演进)
│       └── ... (其他总结)
│
├── 📝 interview/                ← 面试准备
│   ├── qa-bank.md              ← 常见面试题库
│   └── cards/                  ← 面试知识卡片
│
├── 💼 resume/                   ← 简历与项目库
│   ├── 30_projects_portfolio.md ← 30 个项目投资组合
│   └── ...
│
├── 🚀 tech-evolution/           ← 技术演进路线
│   ├── 01_recall_evolution.md   ← 召回系统演进
│   ├── 02_ranking_evolution.md  ← 排序系统演进
│   └── ...
│
└── 📊 data/                     ← 日志和元数据
    ├── papers_queue.jsonl       ← 待处理论文队列
    ├── processed_log.jsonl      ← 已处理文档日志
    └── knowledge_graph.md       ← 知识图谱可视化
```

---

## 🎯 快速开始

### 我是新手，怎么学？

1. **选择你感兴趣的领域**
   - 🔍 搜索系统 → 文本检索、排序、RAG
   - 🎯 广告系统 → CTR 预估、竞价、出价
   - 🎬 推荐系统 → 召回、排序、重排

2. **从 INDEX.md 开始**
   - 每个领域都有 `INDEX.md`，包含全景导航
   - 按学习阶段查找（入门 → 深度 → 工程）
   - 按研究方向查找（特定算法/技术）

3. **按这个顺序学习**
   ```
   synthesis/ (总结概览)
      ↓
   papers/ (论文细节)
      ↓
   practices/ (工业案例)
      ↓
   interview/ (面试题)
   ```

### 我是工程师，怎么用？

1. **查看架构与框架**
   - `synthesis/llm_integration_framework.md` - LLM 集成框架
   - `synthesis/XXX_evolution.md` - 系统演进历程

2. **学习工业最佳实践**
   - `practices/` 下的大厂案例
   - `papers/` 中的工程优化论文

3. **快速查找**
   - 用 `INDEX.md` 的快速导航
   - 用 Cmd+F 搜索特定技术

### 我在准备面试，怎么复习？

1. **阅读 synthesis 文档**
   - 快速掌握全景和关键概念
   - 理解各部分如何协作

2. **深入 papers 高频方向**
   - CTR 预估、召回、排序等核心算法
   - 最新 LLM 集成方向

3. **查看 interview/ 题库**
   - 常见面试题及答案框架
   - 知识卡片速记

---

## 📚 分类规则与定义

### Papers (学术论文笔记)

**定义**：原始论文的学习笔记  
**命名规则**：`20260313_colbert_v2.md`（日期 + 论文简称）  
**内容**：
- 论文核心思想总结
- 关键公式推导
- 优缺点分析
- 与相关工作对比

**何时记录**：
- 阅读 arXiv 论文后
- 跟进学术会议新论文
- 学习算法基础理论

**使用场景**：
- 🎓 学生/研究者：深入理论理解
- 📖 工程师：学习算法细节
- 💡 面试准备：理论考察

---

### Practices (工业实践案例)

**定义**：大厂真实系统设计、工程经验、技术选型总结  
**命名规则**：`google_search_system.md`、`meta_ads_llm.md`  
**内容**：
- 系统架构设计
- 核心技术选型
- 工程优化经验
- 性能指标与权衡

**何时记录**：
- 学习大厂技术博客
- 参加技术分享会
- 工业论文（NSDI、USENIX 等）
- 自身工程经验总结

**使用场景**：
- 💼 工程师：系统设计参考
- 🎓 学生：理解生产环境
- 💡 面试准备：工程题目
- 🚀 创业：MVP 设计参考

---

### Synthesis (提炼总结)

**定义**：基于 papers + practices 的演进总结、框架性文档、知识精华提炼  
**命名规则**：
- `XXX_evolution.md` - 某方向的演进历程
- `XXX_framework.md` - 某方向的架构框架
- `01_XXX.md` - 某方向的总体概览

**内容特征**：
- **必须有明确的引用**（指向 papers/ 和 practices/）
- 按时间/复杂度层次组织
- 强调最新前沿（2024-2026）
- 工程应用指南

**引用格式**：
```markdown
## 传统排序时代
参考：see [papers/20260316_bm25.md](../papers/20260316_bm25.md)

## LLM 时代新范式
参考：
- 学术基础：see [papers/20260316_llm-for-ir-survey.md](../papers/20260316_llm-for-ir-survey.md)
- 工业实践：see [practices/google_search_system.md](../practices/google_search_system.md)
```

**何时记录**：
- papers + practices 足够丰富后
- 发现明确的演进规律或框架
- 需要高度概括总结

**使用场景**：
- 📚 快速了解一个方向
- 🎓 系统学习路径
- 💡 面试准备：整体思路
- 🚀 项目设计：架构参考

---

## 🔗 三层引用关系

```
Papers (原始学习笔记)
  │
  ├─ 知识沉淀（理论理解）
  │
  ▼
Practices (工业应用案例)
  │
  ├─ 工程智慧（系统设计）
  │
  ▼
Synthesis (演进总结与框架)
  │
  └─ 知识精华（全景理解）
```

**设计哲学**：
- **Papers** 记录学到了什么
- **Practices** 记录怎么用了什么
- **Synthesis** 记录为什么这样用

---

## 📊 当前统计

### 按领域分布

| 领域 | Papers | Practices | Synthesis | 总计 |
|------|--------|-----------|-----------|------|
| **Search** 搜索 | 72 | 0 | 2 | 74 |
| **Ads** 广告 | 82 | 2 | 4 | 88 |
| **Rec-Sys** 推荐 | 69 | 0 | 11 | 80 |
| **总计** | **223** | **2** | **17** | **242** |

### 发展路线

- ✅ Papers 文档积累（223 篇）
- 📌 Practices 补充中（仅 2 篇）
  - 需要补充：Google、Meta、Netflix、Alibaba、Bytedance 等大厂案例
- 📌 Synthesis 持续完善（17 篇）
  - 需要补充：各领域的演进总结、框架文档

---

## 🚀 后续维护

### 日常使用

每次新增文档时遵循：

1. **判断分类**
   - 是学术论文笔记？→ `papers/`
   - 是工业实践案例？→ `practices/`
   - 是架构框架总结？→ `synthesis/`

2. **文件命名**
   - Papers: `20260313_论文简称.md`
   - Practices: `公司_场景.md`
   - Synthesis: `01_方向.md` 或 `方向_evolution.md`

3. **添加引用**（synthesis 文件必须）
   ```markdown
   ## 📚 参考资料与引用
   
   - [papers/XXX.md](../papers/XXX.md)
   - [practices/YYY.md](../practices/YYY.md)
   ```

4. **更新 INDEX.md**
   - 新增文件数统计
   - 在相应分类下添加链接
   - 更新 "最后更新时间"

5. **记录日志**
   - 运行脚本更新 `processed_log.jsonl`
   - 或手动记录新增

### 月度回顾

- 整理新增论文，补充到领域
- 识别可以总结为 synthesis 的 papers 组
- 补充 practices 中的工业案例
- 更新 INDEX.md 和导航

### 季度梳理

- 重构 synthesis 文档，优化架构
- 补充缺失的方向
- 更新 LLM_INTEGRATION_ROADMAP.md
- 准备新的学习计划

---

## 🎯 核心导航

### 🔍 搜索系统

快速进入：[search/INDEX.md](./search/INDEX.md)

**关键路径**：
- 入门：[01_search_ranking.md](./search/synthesis/01_search_ranking.md)
- 框架：[llm_integration_framework.md](./search/synthesis/llm_integration_framework.md)
- 最新：papers/ 中 20260323 开头的文件

---

### 🎯 广告系统

快速进入：[ads/INDEX.md](./ads/INDEX.md)

**关键路径**：
- CTR 预估：[20260319_ctr-prediction-comprehensive-survey.md](./ads/papers/20260319_ctr-prediction-comprehensive-survey.md)
- 自动竞价：[auto_bidding_evolution.md](./ads/synthesis/auto_bidding_evolution.md)
- LLM 集成：[llm_integration_framework.md](./ads/synthesis/llm_integration_framework.md)

---

### 🎬 推荐系统

快速进入：[rec-sys/INDEX.md](./rec-sys/INDEX.md)

**关键路径**：
- CTR 建模：[01_ctr_models_deep_dive.md](./rec-sys/synthesis/01_ctr_models_deep_dive.md)
- 系统演进：[recommendation_evolution.md](./rec-sys/synthesis/recommendation_evolution.md)
- LLM 融合：[llm_integration_framework.md](./rec-sys/synthesis/llm_integration_framework.md)

---

## 💡 使用建议

### 对于不同角色

**🎓 学生/初级工程师**
- 从 synthesis 了解全景
- 深入 papers 学习理论
- 对标 practices 理解工程

**👨‍💼 中高级工程师**
- 重点看 synthesis 和 practices
- 按需查阅 papers 细节
- 用于系统设计决策

**🔬 研究者**
- 系统浏览 papers
- 参考 synthesis 的论文组织
- 补充新研究方向

**💼 面试/求职**
- synthesis 作为答题框架
- papers 作为深度理解
- practices 作为工程例子

---

## 📝 元数据

- **创建时间**: 2024-03-10
- **最后更新**: 2026-03-24
- **维护者**: MelonEggLearn
- **协议**: Knowledge Base for Personal Learning
- **访问**: 本地文件系统 `~/Documents/ai-kb/`

---

## 🔗 相关资源

### 开源项目
- [fun-rec](https://github.com/datawhalechina/fun-rec) - 推荐系统入门
- [AlgoNotes](https://github.com/shenweichen/AlgoNotes) - 算法工程笔记

### 公众号
- "机器学习与推荐算法"

### 学习社区
- arXiv 和顶会（RecSys、SIGIR、WWW、KDD）
- GitHub 开源项目

---

> 💡 **核心理念**: Papers 是学习，Practices 是实战，Synthesis 是智慧。  
> 三层合一，才能从知识走向能力。
