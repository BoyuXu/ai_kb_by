# Evidence Units: Ontology-Grounded Document Organization for Parser-Independent Retrieval

> 来源：arXiv:2604.00500 | 领域：search/document_parsing | 学习日期：2026-04-11
> 作者：Yeonjee Han

## 问题定义

当前文档索引系统将结构化文档的每个解析元素（段落、表格、图片）独立处理，导致**语义连贯的单元被拆散**——一个表格和解释它的段落被分成不同的检索候选，丢失了上下文关系。不同解析器（MinerU, Docling）的输出格式差异大，进一步加剧碎片化。

## 核心方法

提出 **Evidence Unit (EU)**：语义完整的文档块作为统一检索单元。Parser-independent pipeline 四步：

1. **本体角色归一化（Ontology-grounded Role Normalization）**：扩展 DoCO 本体，将不同解析器的输出映射到统一语义 schema
2. **语义全局分配算法（Semantic Global Assignment）**：用完整相似度矩阵最优匹配段落和视觉资产（表格/图/公式），组装成 EU
3. **图验证层（Graph-based Validation）**：Neo4j 图数据库形式化 EU 构建规则，通过两个不变量验证完整性
4. **跨解析器验证（Cross-parser Validation）**：确认不同解析器产出的 EU 空间覆盖收敛

## 关键创新

- **从元素级到单元级检索**：不再是段落/表格独立检索，而是将语义关联的元素组装成完整 Evidence Unit
- **解析器无关**：同一框架适配 MinerU、Docling 等多种解析器
- **本体驱动**：基于文档本体（DoCO）的角色映射，比启发式规则更鲁棒

## 实验亮点

在 OmniDocBench v1.0（1,340 页，1,551 QA 对）上：
- **Recall@1 提升 3.4 倍**：0.15 → 0.51
- **检索 LCS +0.31**
- 文本类查询提升最大：Recall@1 从 0.08 → 0.47
- 跨解析器性能一致

## 工业价值

对 RAG 系统中的文档切分（chunking）有直接指导意义：好的 chunk 不是固定长度切分，而是**语义完整的证据单元**。本体驱动的方法比 naive chunking 在检索质量上有数量级提升。

[[推理增强检索技术综述]] | [[LLM增强信息检索与RAG技术进展]]
