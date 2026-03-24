# Agent Memory & Context Management Papers — Index

## 📚 核心论文库（Curated Collection）

### Tier-1: Must-Read Foundation Papers

#### 1. Memori: A Persistent Memory Layer for Efficient, Context-Aware LLM Agents
- **ArXiv:** https://arxiv.org/abs/[待补]
- **Date:** 2026-03-20
- **Authors:** Luiz C. Borro, Luiz A. B. Macarini, Gordon Tindall, Michael Montero, Adam B. Struck
- **Status:** 📌 To Read
- **Why:** 核心持久化内存架构，直接解决 context window 问题
- **Key Ideas:**
  - Persistent Memory Layer（PML）
  - Context-aware retrieval
  - Efficiency optimization for long-horizon tasks

#### 2. All-Mem: Agentic Lifelong Memory via Dynamic Topology Evolution
- **ArXiv:** https://arxiv.org/abs/[待补]
- **Date:** 2026-03-19
- **Authors:** Can Lv, Heng Chang, Yuchen Guo, Shengyu Tao, Shiji Zhou
- **Status:** 📌 To Read
- **Why:** 解决终身 Agent 的内存问题，months/years 尺度
- **Key Ideas:**
  - Dynamic Topology（非静态设计）
  - Continual Writing & Retrieval
  - Token efficiency

#### 3. D-Mem: A Dual-Process Memory System for LLM Agents
- **ArXiv:** https://arxiv.org/abs/[待补]
- **Date:** 2026-03-19
- **Authors:** Zhixing You, Jiachen Yuan, Jason Cai
- **Status:** 📌 To Read
- **Why:** 认知科学启发的双流程内存架构
- **Key Ideas:**
  - Dual-process (System 1 & System 2 thinking in memory)
  - High-fidelity persistent storage
  - Self-adapting mechanism

#### 4. Graph-Native Cognitive Memory for AI Agents
- **ArXiv:** https://arxiv.org/abs/[待补]
- **Date:** 2026-03-17
- **Authors:** Young Bin Park
- **Status:** 📌 To Read
- **Why:** 形式化的知识表示与信念修正
- **Key Ideas:**
  - Graph-based representation
  - Belief Revision Semantics
  - Versioned Memory Architectures

### Tier-2: Application & Optimization Papers

#### 5. Chronos: Temporal-Aware Conversational Agents with Structured Event Retrieval
- **ArXiv:** https://arxiv.org/abs/[待补]
- **Date:** 2026-03-17
- **Authors:** Sahil Sen, Elias Lumer, Anmol Gulati, Vamse Kumar Subbiah
- **Status:** 📌 To Read
- **Why:** 时间感知的记忆检索，对多轮对话很关键
- **Key Ideas:**
  - Temporal indexing
  - Structured event representation
  - Long-term conversational memory

#### 6. AdaMem: Adaptive User-Centric Memory for Long-Horizon Dialogue Agents
- **ArXiv:** https://arxiv.org/abs/[待补]
- **Date:** 2026-03-17
- **Authors:** Shannan Yan, Jingchen Ni, Leqi Zheng, Jiajun Zhang, Peixi Wu, Dacheng Yin, Jing Lyu, Chun Yuan, Fengyun Rao
- **Status:** 📌 To Read
- **Why:** 个性化内存的自适应更新机制
- **Key Ideas:**
  - User-centric design
  - Adaptive summarization
  - Long-horizon dialogue retention

#### 7. CraniMem: Cranial Inspired Gated and Bounded Memory for Agentic Systems
- **ArXiv:** https://arxiv.org/abs/[待补]
- **Date:** 2026-03-03
- **Authors:** Pearl Mody, Mihir Panchal, Rishit Kar, Kiran Bhowmick, Ruhina Karani
- **Status:** 📌 To Read
- **Why:** 生物启发的门控内存，解决内存爆炸问题
- **Key Ideas:**
  - Gating mechanism (biological inspiration)
  - Bounded memory (容量限制)
  - Selective retention

#### 8. NextMem: Towards Latent Factual Memory for LLM-based Agents
- **ArXiv:** https://arxiv.org/abs/[待补]
- **Date:** 2026-02-26
- **Authors:** Zeyu Zhang, Rui Li, Xiaoyan Zhao, Yang Zhang, Wenjie Wang, Xu Chen, Tat-Seng Chua
- **Status:** 📌 To Read
- **Why:** 事实一致性与潜在知识表示
- **Key Ideas:**
  - Latent factual representation
  - Consistency maintenance
  - Knowledge retention over long interactions

### Tier-3: Emerging & Specialized Topics

#### 9. MemMA: Coordinating the Memory Cycle
- **Date:** 2026-03-19
- **Key Contribution:** Memory cycle coordination in multi-agent settings
- **Status:** 📌 To Read

#### 10. D-MEM: Dopamine-Gated Agentic Memory
- **Date:** 2026-03-15
- **Key Contribution:** RL-driven selective memory via RPE routing
- **Status:** 📌 To Read

#### 11. Structured Distillation for Personalized Agent Memory
- **Date:** 2026-03-13
- **Key Contribution:** 11x token reduction with retrieval preservation
- **Status:** 📌 To Read

#### 12. MemArchitect: A Policy Driven Memory Governance Layer
- **Date:** 2026-03-18
- **Key Contribution:** Governance framework for persistent LLM memory
- **Status:** 📌 To Read

---

## 🎯 Reading Schedule (Proposed)

### Week 1 (Mar 24-30)
- Mon-Tue: Memori + D-Mem (2 papers)
- Wed: All-Mem + CraniMem (2 papers)
- Thu-Fri: Chronos + AdaMem (2 papers)
- Weekend: Summary + notes

### Week 2 (Mar 31 - Apr 6)
- Mon-Tue: Graph-Native Cognitive Memory + NextMem (2 papers)
- Wed-Thu: MemMA + D-MEM + MemArchitect (3 papers)
- Fri: Structured Distillation + others
- Weekend: Cross-paper analysis

### Week 3-4
- Deep dive into selected papers
- Implement toy examples
- Prepare interview materials

---

## 📊 Tracking

### Paper Categories
- **Memory Architecture:** Memori, D-Mem, All-Mem, CraniMem, MemArchitect
- **Retrieval & Indexing:** Chronos, Graph-Native
- **Personalization:** AdaMem, NextMem
- **Multi-Agent:** MemMA
- **Optimization:** Structured Distillation, D-MEM (RL-driven)

### Interview Prep
- [ ] Summarize 5-min pitch for each Tier-1 paper
- [ ] Identify trade-offs (accuracy vs efficiency)
- [ ] Compare with recommendation systems memory (CTR modeling)
- [ ] Design question: "如何为搜索系统设计 Agent Memory?"

---

**Last Updated:** 2026-03-24  
**Total Papers:** 12+ (expanding)  
**Status:** 🔄 Active Collection
