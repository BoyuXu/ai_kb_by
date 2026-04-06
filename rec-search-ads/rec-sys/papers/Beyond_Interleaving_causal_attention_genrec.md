# Beyond Interleaving: Causal Attention Reformulations for Generative Recommender Systems

**Status:** Note - Exact ArXiv ID not found in search | **Domain:** rec-sys | **in_progress**

## 核心主题
生成式推荐系统中，标准 interleaving（item-action 交错）方法的注意力机制重构。

## 背景
生成式推荐通常将用户历史表示为 item 和 action 的交错序列，传统 Transformer causal attention 在这种格式下存在局限：
- 不能有效区分 item token 和 action token 的因果关系
- 位置编码对于结构化序列不够灵活

## 可能的核心贡献
- 针对生成式推荐的 causal attention 重构方案
- 超越简单交错的序列建模方法
- 更有效捕获 item-action 的因果依赖关系

## 相关背景知识
生成式推荐（TIGER、GenRec 等）将推荐建模为序列生成任务，核心挑战是如何在 Transformer 中有效编码用户行为序列的时间和因果结构。

## 待补充
需进一步确认论文 ArXiv ID 和具体技术细节。

**Tags:** #rec-sys #generative-recommendation #causal-attention #transformer
