# AI 知识库（ai-kb）

> 搜广推（搜索/广告/推荐）+ LLM/Agent 算法工程师成长知识库
> 定位：**面试备战 + 工业实践参考**，不是论文摘抄，是有思考的提炼

## 📁 目录结构

```
ai-kb/
├── rec-search-ads/          # 搜广推核心（按域+阶段组织）
│   ├── rec-sys/             # 推荐系统
│   │   ├── 01_recall/       # 召回（双塔/SemanticID/生成式）
│   │   ├── 02_rank/         # 精排（CTR/DIN/序列建模）
│   │   ├── 03_rerank/       # 重排（多样性/LLM重排）
│   │   └── 04_multi-task/   # 多任务/因果/冷启动
│   ├── ads/                 # 广告系统
│   │   ├── 01_recall/       # 广告召回
│   │   ├── 02_rank/         # CTR预估/ESMM/LTV
│   │   ├── 03_rerank/       # 混排/广告重排
│   │   ├── 04_bidding/      # 出价/AutoBidding/RTB
│   │   └── 05_creative/     # 创意生成/多模态
│   ├── search/              # 搜索系统
│   │   ├── 01_recall/       # 检索（稠密/稀疏/混合）
│   │   ├── 02_rank/         # LTR
│   │   ├── 03_rerank/       # LLM Rerank
│   │   └── 04_query/        # Query理解
│   ├── cross-domain/        # 跨域通用（多目标/偏差/工程/冷启动）
│   ├── interview/           # 面试题库 + AB实验 + 面试模板
│   └── industry-challenges-2026.md  # ⭐ 行业痛点调研报告
├── llm-agent/               # LLM + Agent
│   ├── llm-infra/           # LLM 推理/训练/RAG/Agent
│   │   ├── papers/          # 论文笔记（按主题分类）
│   │   └── synthesis/       # 综述（⭐ 从这里开始读）
│   └── agent-context-management/  # Agent 上下文工程
├── fundamentals/            # 基础算法深度文档（12篇）
│   # Attention/LoRA/DIN/MMOE/Uplift/BM25/CTR校准...
├── tech-evolution/          # 技术演进史（6个方向）
│   # 召回/精排/重排/广告/搜索/LLM 的完整演进时间线
└── _meta/                   # 报告/索引（不是知识内容）
```

## 📊 知识库统计

| 领域 | Papers | Synthesis | 合计 |
|------|--------|-----------|------|
| 🔍 搜索系统 | 63 | 12 | 75 |
| 📢 广告系统 | 76 | 15 | 93 |
| 🎯 推荐系统 | 64 | 26 | 90 |
| 🤖 LLM 基础设施 | — | 17 | 17 |
| 🔗 跨域综合 | — | 12 | 12 |
| **合计** | **203+** | **82+** | **287+** |

## 🚀 推荐学习路径

### 路径1：推荐系统面试速成（1周）

```
Day1: fundamentals/attention_transformer.md + din_sequence_modeling.md
Day2: rec-sys/02_rank/synthesis/CTR模型深度解析.md
Day3: rec-sys/01_recall/synthesis/召回系统工业界最佳实践.md
Day4: rec-sys/03_rerank/synthesis/推荐系统重排与多样性.md
Day5: interview/面试官最想听到什么.md + qa-bank.md (面试题)
Day6-7: tech-evolution/02_ranking_evolution.md (完整演进)
```

### 路径2：广告系统面试（1周）

```
Day1: fundamentals/auction_theory.md + ctr_calibration.md
Day2: ads/02_rank/synthesis/ESMM系列CVR估计演进.md + LTV预测.md
Day3: ads/04_bidding/synthesis/AutoBidding技术演进.md
Day4: ads/04_bidding/synthesis/广告出价体系全景.md
Day5: industry-challenges-2026.md（广告部分）
Day6-7: tech-evolution/04_ads_evolution.md
```

### 路径3：LLM/Agent 方向（1周）

```
Day1: llm-infra/synthesis/KVCache与LLM推理优化全景.md
Day2: llm-infra/synthesis/LLM微调技术.md + LoRA与PEFT.md
Day3: llm-infra/synthesis/RAG系统全景.md + RAG_vs_Finetune决策框架.md
Day4: llm-infra/synthesis/RLVR_vs_RLHF后训练路线.md + GRPO.md
Day5: llm-infra/synthesis/Agent失败模式与解法.md
Day6-7: tech-evolution/06_llm_evolution.md
```

## ⭐ 最值得反复看的文件

| 文件 | 为什么值得反复看 |
|------|---------------|
| `interview/面试官最想听到什么.md` | 面试前必看，纠正表达方式 |
| `industry-challenges-2026.md` | 理解业界真实痛点，面试有深度 |
| `fundamentals/auction_theory.md` | 广告出价的根基 |
| `llm-infra/synthesis/LLM常见认知误区.md` | 避免在面试中说错话 |
| `cross-domain/冷启动问题统一视角.md` | 三域对比，展示系统思维 |

---
*最后更新：2026-04-03 | MelonEggLearn*
