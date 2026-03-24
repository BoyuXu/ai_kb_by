#!/bin/bash
# MelonEggLearn 持续自动学习 v2
# 覆盖全部剩余内容源，有内容就一直跑
# 运行: nohup bash auto_learn_v2.sh >> auto_learn_v2.log 2>&1 &

KIMI="/usr/local/bin/kimi"
AI_KB="$HOME/Documents/ai-kb"
LOG="$AI_KB/auto_learn_v2.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M')] $1" | tee -a "$LOG"; }

run() {
  local id="$1"; local prompt="$2"
  log "▶ START: $id"
  $KIMI --work-dir "$AI_KB" --print -p "$prompt" >> "$LOG" 2>&1
  local code=$?
  [ $code -eq 0 ] && log "✅ DONE: $id" || log "❌ FAIL: $id (exit $code)"
  return $code
}

log "====== MelonEggLearn v2 启动 ======"

# ─────────────────────────────────────────
# BLOCK 1: AIGC-Interview-Book 剩余目录
# ─────────────────────────────────────────

run "ml_basics" "
你是MelonEggLearn。读取 $AI_KB/repos/AIGC-Interview-Book/机器学习基础/ 所有md文件。

整理机器学习核心面试考点：

### 1. 损失函数
- BCE/CE/Focal Loss/MSE/Huber——推导+适用场景
- 正负样本不均衡处理（Focal/上采样/下采样）

### 2. 优化方法
- SGD/Adam/AdamW/LAMB原理推导
- 学习率调度（Warmup/Cosine/OneCycle）
- 梯度裁剪、权重衰减

### 3. 机器学习基础
- 偏差-方差权衡
- L1/L2正则化的几何解释
- 交叉验证、AUC/GAUC/NDCG指标
- 树模型：XGBoost/LightGBM原理对比

### 4. 模型评估
- 混淆矩阵/Precision/Recall/F1
- 推荐系统离线指标 vs 在线指标的gap

每节给出1-2道高频面试题+答案。
输出：$AI_KB/rec-sys/04_ml_fundamentals.md
"

run "data_structure" "
你是MelonEggLearn。读取 $AI_KB/repos/AIGC-Interview-Book/数据结构基础/数据结构基础.md

整理算法岗数据结构考点（重点：推荐/广告工程实际用到的）：

1. 哈希表：碰撞处理、一致性哈希（分布式特征存储）
2. 堆/优先队列：TopK问题（召回合并、实时榜单）
3. 树结构：B+树（特征索引）、Trie（前缀匹配）
4. 图：BFS/DFS、最短路径（GNN基础）
5. 链表/双端队列：LRU Cache实现（特征缓存）

每个结构给出Python实现 + 推荐系统实战场景。
追加到：$AI_KB/interview/coding-prep.md
"

run "mock_interview_ads" "
你是MelonEggLearn。模拟一场广告算法岗面试（字节/腾讯/阿里妈妈）。

## 面试题目（含参考答案）：

### 基础概念（5题）
1. CTR/CVR预估的差异及样本空间问题
2. ESMM模型如何解决CVR样本稀疏？
3. 广告竞价：GSP vs VCG机制
4. 探索与利用：ε-greedy/UCB/Thompson Sampling
5. 预算分配与Pacing算法

### 系统设计（详细）
题：设计一个实时广告CTR预估系统，支持10w QPS
- 特征工程（实时+离线）
- 模型架构选型
- 在线学习更新频率
- A/B实验体系
- 核心指标监控

### 代码题
实现ESMM的PyTorch代码（含CTR塔+CVR塔+联合训练）

### 场景题
广告系统中如何处理位置偏差（Position Bias）？

输出：$AI_KB/interview/mock-interview-ads.md
"

# ─────────────────────────────────────────
# BLOCK 2: AlgoNotes 精华文章（分批抓取+摘要）
# ─────────────────────────────────────────

run "algo_notes_p1_ranking" "
你是MelonEggLearn。从以下精排领域顶级工业界文章中提炼核心知识（基于你的知识库+文章标题/内容推断）：

文章列表（按优先级）：
1. 京东推荐算法精排技术实践
2. 贝壳CVR转化率预估模型实践
3. 张俊林：推荐系统排序Embedding建模
4. 快手KDD2022 D2Q观看时长预估
5. 阿里新一代Rank技术
6. 万字长文梳理CTR预估模型发展
7. 网易云音乐用户行为序列深度建模
8. Life-long兴趣建模：SIM模型
9. TWIN双塔序列模型
10. COLD级联精排

对每篇文章：
- 核心贡献（1-3句话）
- 技术亮点（关键创新点）
- 工程落地难点
- 面试时如何引用

输出：$AI_KB/rec-sys/05_industry_ranking_papers.md
（标题：# 精排工业界精华 · AlgoNotes精读）
"

run "algo_notes_p2_recall" "
你是MelonEggLearn。整理召回系统工业界最佳实践：

1. 微信看一看内容推荐：万亿参数双塔模型
2. 快手双塔召回：负样本采样策略
3. Pinterest PinSage图召回
4. YouTube DNN召回经典架构
5. 小红书MHSA多头自注意力召回
6. 阿里EGES Graph Embedding
7. 字节TDM树形深度检索
8. 多路召回分数校准（序关系保持）
9. 向量数据库选型：Faiss/Milvus/Proxima对比
10. ANN近似最近邻：HNSW原理

对每个主题：核心原理 + 工程细节 + 优劣对比
输出：$AI_KB/rec-sys/06_industry_recall_papers.md
"

run "algo_notes_p3_multimodal_rec" "
你是MelonEggLearn。整理多模态+LLM推荐系统最新进展：

1. 内容理解在推荐中的应用（图文视频特征）
2. CLIP/BLIP在推荐中的特征提取
3. LLM作为推荐系统的几种范式：
   - 直接生成（P5/LLMRec）
   - 特征增强（LLM Embedding）  
   - 意图理解（Query改写/冷启动）
4. 知识图谱增强推荐（KGCN/KGAT）
5. 因果推断在推荐去偏中的应用

输出：$AI_KB/llm-infra/04_multimodal_llm_rec.md
"

# ─────────────────────────────────────────
# BLOCK 3: 个人学习计划深度整理
# ─────────────────────────────────────────

run "personal_learning_plan" "
你是MelonEggLearn。读取文件：/Users/boyu/Downloads/learning-plan (2).docx

这是Boyu的个人学习笔记，包含Attention/Transformer/RNN/LSTM/推荐系统全链路内容。

请：
1. 提炼Boyu已掌握的知识点（从笔记深度判断）
2. 识别知识盲区（笔记中模糊/缺失的部分）
3. 针对盲区生成补充学习内容
4. 结合已有知识库，制作个性化复习计划

输出：$AI_KB/rec-sys/00_personal_notes.md
格式：
# Boyu 个人学习档案
## 已掌握（强项）
## 待加强（弱项）
## 个性化补充笔记
## 推荐复习顺序
"

# ─────────────────────────────────────────
# BLOCK 4: 高级专题
# ─────────────────────────────────────────

run "causal_inference_rec" "
你是MelonEggLearn。整理推荐系统中的因果推断专题：

1. 为什么需要因果推断？（混淆偏差、曝光偏差、流行度偏差）
2. 倾向分（Propensity Score）去偏
3. IPS/SNIPS加权校正
4. Counterfactual Learning
5. DML（双重机器学习）
6. 工业界案例：快手D2Q、抖音去偏实践
7. 面试如何回答「如何处理推荐系统的偏差问题」

输出：$AI_KB/rec-sys/07_causal_inference.md
"

run "rerank_diversity" "
你是MelonEggLearn。整理重排&多样性专题：

1. 重排的目标：从精排到展示的差距
2. 经典重排模型：PRM/DLCM/SetRank
3. 多样性算法：MMR/DPP（行列式点过程）
4. 新颖性 vs 多样性 vs 准确性的权衡
5. 生成式重排（LLM排序）
6. Context-aware重排
7. 工业界：淘宝序列检索重排/快手重排演进

输出：$AI_KB/rec-sys/08_rerank_diversity.md
"

run "interview_scenario_questions" "
你是MelonEggLearn。整理推荐/广告算法岗「场景题」（最难、最考经验的题型）：

### 场景1：新用户冷启动
如何为一个完全新用户做个性化推荐？（从注册那一刻开始）

### 场景2：物品冷启动
一件刚上架的新商品，如何快速获得曝光和评估？

### 场景3：黑产/刷量检测
推荐系统中如何识别和处理机器刷量/刷曝光行为？

### 场景4：负反馈处理
用户点了「不感兴趣」，如何更新用户画像和推荐策略？

### 场景5：推荐结果变差定位
线上推荐效果突然下降，排查思路是什么？

### 场景6：双塔模型线上线下不一致
离线AUC很好，线上CTR没提升，可能原因和排查方向？

每个场景：背景分析 + 完整解题思路 + 面试加分点
输出：$AI_KB/interview/scenario-questions.md
"

run "weekly_arxiv_2" "
你是MelonEggLearn，第2期技术雷达（2026-03-11补充）。

搜索并整理：
1. 2026年1-3月推荐系统最新论文（RecSys/WWW/SIGIR）
2. DeepSeek-V3/R1对推荐系统领域的影响
3. 最新开源推荐系统框架（RecBole/FEA/OpenMatch）
4. 工业界最新技术博客（2026年1月之后）
5. GitHub: Star增长最快的推荐/搜索相关项目

输出：$AI_KB/weekly/2026-03-11_tech_radar_v2.md
"

# ─────────────────────────────────────────
# BLOCK 5: 编程实战强化
# ─────────────────────────────────────────

run "coding_practice_recsys" "
你是MelonEggLearn。生成推荐系统算法岗编程实战题库。

## 必会实现（含完整PyTorch代码）

### 1. FM（因子分解机）
from scratch实现，含二阶特征交叉

### 2. DeepFM
FM + DNN混合架构

### 3. DIN（深度兴趣网络）
Target Attention实现

### 4. 双塔召回模型
含负样本采样策略

### 5. MMOE多任务
门控机制实现

### 6. Transformer编码器
推荐序列建模版本（BERT4Rec风格）

每个模型：
- 核心代码（完整可运行）
- 关键设计决策说明
- 常见bug和注意事项

输出：$AI_KB/interview/coding-pytorch.md
"

log "====== v2 全部批次完成 ======"
# 统计产出
echo '=== 产出统计 ===' >> "$LOG"
find "$AI_KB" -name '*.md' | wc -l >> "$LOG"
openclaw system event --text 'MelonEggLearn v2 全部完成！知识库大幅更新' --mode now 2>/dev/null || true
