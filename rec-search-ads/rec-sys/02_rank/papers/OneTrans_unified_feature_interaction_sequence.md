# OneTrans: One Transformer for Unified Feature Interaction and Sequence Modeling
> 来源：工业论文 | 领域：推荐系统 | 学习日期：20260327

## 问题定义
现代推荐系统通常使用两类独立模块：
1. **特征交叉模块**（DCN、DeepFM、xDeepFM等）：处理结构化特征的高阶交叉
2. **序列建模模块**（GRU、BERT4Rec、SASRec等）：处理用户行为序列
两个模块独立设计、独立参数，导致：
- 参数冗余（两套参数）
- 特征交叉和序列信息无法互相增强
- 系统复杂度高，维护成本大
**目标**：用单一Transformer统一完成特征交叉和序列建模。

## 核心方法与创新点

### 1. Field Tokenization
将每个特征field（用户年龄、商品类目、价格等）转化为token：

$$
\text{token}}_{\text{i = \text{Embedding}}(\text{field}}_{\text{i) + \text{FieldTypeEmbedding}}(i)
$$

序列行为同样token化，与结构化特征拼接成统一序列。

### 2. 统一注意力
同一个Transformer处理所有token（结构化特征token + 序列行为token）：
- 特征交叉 = 结构化token之间的注意力
- 序列建模 = 行为token之间的注意力（加causal mask）
- 跨类型交互 = 结构化token与行为token的互注意力

### 3. 位置编码设计
- 序列token：使用相对位置编码（保留时序信息）
- 结构化token：使用学习的field-type embedding（无顺序概念）
- 混合场景下两类token的位置编码不共享

### 4. 参数效率

$$
\text{参数量} = \text{OneTrans} \approx 0.7 \times (\text{DCN-v2} + \text{SASRec})
$$

参数量减少约30%，因为共享了底层特征提取层。

## 实验结论
- **AUC提升**：+0.3%~+0.5%（多个数据集）
- **参数量减少**：-30%（相比DCN-v2 + SASRec双模块）
- **推理延迟**：与原始方案相当（Transformer高度优化）
- 消融实验：跨类型交互（结构化↔序列）贡献了约40%的提升

## 工程落地要点
1. **Field顺序**：结构化特征的token顺序对结果有影响，建议按特征重要性排序
2. **序列长度控制**：混合序列（结构化+行为）总长度需控制在512以内
3. **Mask设计**：行为序列需要causal mask，结构化特征之间不需要，实现时要分区域处理
4. **特征稀疏性**：缺失字段用[MASK] token填充，参考BERT的处理方式
5. **增量更新**：序列部分可以增量计算KV Cache，结构化特征部分需要全量重算

## 常见考点
Q1: OneTrans与传统"双塔+序列模型"架构的核心区别？
A: 传统双塔将特征交叉和序列建模视为独立子问题，信息流不互通。OneTrans将所有特征（结构化+序列）统一token化后，通过单一Transformer的跨token注意力机制，让特征交叉和序列建模相互感知和增强，同时减少了30%参数量。

Q2: Field Tokenization的设计原则是什么？
A: (1)每个field一个token，field embedding包含field类型信息；(2)数值型特征需要分桶离散化或加入数值编码（如FeatEmb=categorical_embed + numeric_proj）；(3)多值field（如标签列表）可以先pool后作为一个token，也可以展开为多个token。

Q3: 为什么混合注意力比独立模块更有效？
A: 用户年龄、性别等统计特征可以帮助解释行为序列的偏好模式（比如年轻用户的点击偏好与中年用户不同），而行为序列反过来可以修正统计特征的粗粒度表达。注意力机制让两类信息动态加权融合，比简单拼接特征后输入MLP更能捕获细粒度的交互关系。
