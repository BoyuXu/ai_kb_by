# Beyond the Flat Sequence: Hierarchical and Preference-Aware Generative Recommendations (HPGR)
> 来源：arXiv:2603.00980 | 领域：rec-sys | 学习日期：20260419

## 问题定义
生成式推荐（GR），如HSTU，将用户交互序列视为"扁平序列"进行建模。这种假设忽略了用户行为的内在结构，导致两个关键问题：①无法捕捉session级别的时序层次结构；②密集注意力在语义稀疏的长序列中引入大量噪声。

## 核心方法与创新点
1. **两阶段范式（Two-Stage Paradigm）**：
   - Stage 1: Session分割 + Session级编码，提取层次化行为结构
   - Stage 2: 偏好感知的跨Session注意力，过滤噪声交互
2. **Session-based Temporal Hierarchy**：将长序列按session切分，session内密集attention，session间稀疏attention
3. **Preference-Aware Attention**：根据用户显式/隐式偏好信号，动态调整注意力权重，抑制无关session的噪声
4. **超越HSTU和MTGR**：在ACM Web Conference 2026上发表，达到SOTA

## 实验结论
- 在多个数据集上显著超越HSTU和MTGR baseline
- 长序列场景提升更为明显（1000+ interactions）
- 计算效率优于full attention（稀疏attention减少O(n²)到O(n·k)）

## 工程落地要点
- Session分割策略需根据业务定义（时间间隔 > 30min / 上下文切换）
- 层次化编码天然适合工业级长序列用户建模（如抖音/YouTube）
- 可与现有HSTU架构渐进式融合

## 面试考点
- Q: HSTU的核心局限性？
  - A: 扁平序列假设→注意力噪声 + 无法捕捉session结构 + O(n²)计算
- Q: 如何定义session？
  - A: 时间间隔阈值 / 上下文变化（搜索query切换） / 行为类型转换（浏览→购买）
