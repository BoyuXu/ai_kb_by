# Large Language Models for Information Retrieval: A Survey

> 来源：arxiv | 日期：20260316 | 领域：search

## 问题定义

信息检索（IR）经历了从词频统计（BM25）→ 神经稠密检索（DPR）→ 大语言模型（LLM）的演化。LLM 以其强大的语言理解和生成能力，正在重塑 IR 的各个环节：查询理解、文档检索、相关性判断、答案生成。

本综述系统梳理 LLM 在 IR 中的应用，涵盖检索、排序、生成三个阶段。

## 核心方法与创新点

### 1. LLM 用于查询理解（Query Understanding）

- **查询扩展**：HyDE（Hypothetical Document Embedding）——让 LLM 先生成假设性答案，用答案 embedding 检索（避免 query 太短、信息不足）。
- **查询改写**：用 LLM 将口语化/模糊查询改写为更精准的检索词。
- **意图识别**：分类用户意图（导航/信息/事务），指导检索策略选择。

### 2. LLM 用于稠密检索

- **LLM as Bi-encoder**：用 LLM（BERT/E5/GTE）的 [CLS] token 或均值池化作为 embedding，构建双塔检索。
- **E5 系列**：通过 Text Embeddings 大规模预训练，跨语言、跨领域检索能力强。
- **BGE-M3**：支持 dense + sparse + multi-vector 的统一检索，单模型搞定多种场景。

### 3. LLM 用于重排序（Reranking）

- **Cross-encoder 重排**：LLM 同时编码 (query, doc) 对，判断相关性。精度高但推理慢（每对都要完整前向传播）。
- **Listwise Reranking**：RankGPT——将多个文档同时送入 LLM，直接输出排序列表。比 pointwise 效率高，能捕捉文档间相对顺序。
- **Instructable Reranking**：用 Instruction Tuning 让 LLM 理解特定排序准则（"按新颖性排序"vs"按相关性排序"）。

### 4. LLM 用于生成式检索（Generative IR）

- **DSI（Differentiable Search Index）**：将整个文档库"记忆"进 LLM 参数，直接根据 query 生成 doc ID。
- **GENRE**：生成式实体检索，直接生成实体名称（适合 KB 检索）。

### 5. RAG（Retrieval-Augmented Generation）

检索增强生成：先检索相关文档，再将文档作为 context 输入 LLM 生成答案。解决 LLM 知识截止和幻觉问题。

## 实验结论

- LLM Reranker 在 BEIR 基准上相比 BM25：nDCG@10 +15-25%（跨领域泛化强）。
- RAG vs 纯生成 LLM：在 NQ（Natural Questions）上准确率 +20-40%，幻觉率显著降低。
- HyDE：在零样本检索场景（无标注数据）上相比 BM25 +10-15%。

## 工程落地要点

- **Bi-encoder vs Cross-encoder**：bi-encoder 快（可以预先计算 doc embedding），cross-encoder 准（实时计算 query-doc 交互）。实际系统：bi-encoder 召回 Top-K，cross-encoder 精排 Top-N（N≪K）。
- **向量索引**：Elasticsearch 8.x 内置 dense_vector，支持 HNSW 近似检索；生产环境也常用 Qdrant/Weaviate/Milvus。
- **Embedding 更新**：文档库更新时需要增量更新 embedding（不能全量重算），用 FAISS 的 IVF 索引支持动态添加。

## 面试考点

- Q: BM25 的公式是什么？有哪些参数？
  A: BM25(q,d) = Σ IDF(t) × f(t,d)×(k1+1) / (f(t,d) + k1×(1-b+b×|d|/avgdl))。k1（词频饱和，通常1.2-2.0）控制词频的边际效益；b（长度归一化，通常0.75）控制文档长度的影响。IDF = log((N-df+0.5)/(df+0.5))。

- Q: 稠密检索 vs 稀疏检索的优缺点？
  A: 稀疏（BM25）：精确词匹配，可解释，推理快，无需训练；缺点是不能处理同义词/语义相似。稠密（DPR/BERT）：捕捉语义相似性，泛化好；缺点是需要训练、延迟较高、不可解释。现代系统通常 hybrid（BM25 + 稠密检索融合）。

- Q: 什么是 HyDE？它解决什么问题？
  A: HyDE（Hypothetical Document Embedding）：先让 LLM 根据 query 生成一个假想的完整答案文档，然后用这个生成文档的 embedding 去检索真实文档库。解决的问题：用户 query 往往很短（几个词），信息量不足；生成的假想答案更接近目标文档的语言风格，检索效果更好。
