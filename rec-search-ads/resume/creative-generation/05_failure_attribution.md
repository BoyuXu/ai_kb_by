# 05 失败案例归因分析

> 核心论文：SHAP (Lundberg & Lee, NeurIPS 2017)、LIME (Ribeiro et al., KDD 2016)、Integrated Gradients (Sundararajan et al., ICML 2017)

---

## 一、广告创意失败的定义

### 1.1 失败的量化标准

```
失败判定规则（需结合业务经验调参）：

Level 1 - 轻度失败（关注）：
  CTR < 历史该品类均值 × 0.7 (30%以下)
  曝光 > 1000次（保证统计意义）

Level 2 - 中度失败（归因分析）：
  CTR < 历史该品类均值 × 0.5 (50%以下)
  曝光 > 500次

Level 3 - 严重失败（立即下线+根因分析）：
  CTR < 历史该品类均值 × 0.3
  或 广告审核不通过
  或 用户举报率 > 0.1%
```

### 1.2 失败的几种来源

```
广告创意系统失败分类：

┌─────────────────────────────────────────────┐
│ 1. 检索层失败                                │
│    - 检索到的参考创意与目标商品不匹配          │
│    - 新品类无参考，使用跨类参考质量差          │
│                                             │
│ 2. 生成层失败                                │
│    - CoT推理中间步骤出错（错误的用户画像推断）  │
│    - LLM幻觉（虚构产品卖点）                  │
│    - Prompt约束失效（生成了违禁词）            │
│                                             │
│ 3. 用户画像失配                              │
│    - 定向人群与创意语气不符                   │
│    - 用户兴趣标签错误（标签穿越/过期）         │
│                                             │
│ 4. 竞争环境因素                              │
│    - 出价不足（位置差）                       │
│    - 时段不对（深夜投放需求类广告）            │
└─────────────────────────────────────────────┘
```

**归因目标：精确定位失败来自哪一层，避免把检索问题误诊为LLM问题**

---

## 二、SHAP值详解

### 2.1 Shapley值的博弈论起源

**背景：** N个人合作完成一个项目，项目总收益如何公平分配？每个人的"边际贡献"就是Shapley值。

**Shapley值公式：**

$$
\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [v(S \cup \{i\}) - v(S)]
$$

**直观理解：**
- $v(S)$：特征子集S对应的模型预测值
- $v(S \cup \{i\}) - v(S)$：加入特征i后的边际贡献
- 对所有可能的子集S取加权平均 → 特征i的公平贡献

**为什么是加权平均？**
因为特征加入顺序不同，边际贡献也不同（交互效应），必须对所有排列取平均才公平

### 2.2 SHAP在广告CTR预测中的应用

```python
import shap
import numpy as np
import pandas as pd
from typing import List, Dict

class AdCreativeAttributor:
    def __init__(self, ctr_model, feature_names: List[str]):
        """
        ctr_model: 已训练的CTR预测模型（LightGBM/XGBoost等树模型）
        feature_names: 特征列表（如文本向量+用户特征+商品特征）
        """
        self.model = ctr_model
        self.feature_names = feature_names
        # TreeSHAP（对树模型精确高效）
        self.explainer = shap.TreeExplainer(ctr_model)
    
    def attribute_failure(
        self,
        creative_features: np.ndarray,  # 单条创意的特征向量
        creative_id: str
    ) -> Dict:
        """
        对单条失败创意进行SHAP归因
        """
        # 计算SHAP值
        shap_values = self.explainer.shap_values(
            creative_features.reshape(1, -1)
        )[0]  # shape: [n_features]
        
        # 基准值（无任何特征时的预测，通常等于训练集均值CTR）
        base_value = self.explainer.expected_value
        
        # 预测值分解：base_value + sum(shap_values) = predicted_ctr
        predicted_ctr = base_value + shap_values.sum()
        
        # 找出负贡献最大的特征
        feature_contributions = sorted(
            zip(self.feature_names, shap_values),
            key=lambda x: x[1]  # 按贡献值排序
        )
        
        top_negative = feature_contributions[:5]   # 负贡献最大的5个特征
        top_positive = feature_contributions[-5:]  # 正贡献最大的5个特征
        
        return {
            "creative_id": creative_id,
            "base_ctr": base_value,
            "predicted_ctr": predicted_ctr,
            "top_negative_features": top_negative,
            "top_positive_features": top_positive,
            "failure_pattern": self._classify_failure(top_negative)
        }
    
    def _classify_failure(self, top_negative_features) -> str:
        """根据负贡献特征分类失败模式"""
        feature_names = [f[0] for f in top_negative_features]
        
        if any("user_age" in f or "user_interest" in f for f in feature_names):
            return "user_mismatch"
        elif any("text_embedding" in f for f in feature_names):
            return "creative_quality"
        elif any("bid" in f or "position" in f for f in feature_names):
            return "bid_insufficient"
        elif any("category_ctr" in f for f in feature_names):
            return "cold_start"
        else:
            return "unknown"
    
    def batch_attribute(
        self,
        failed_creatives: pd.DataFrame,
        threshold: float = 0.5  # CTR低于均值多少比例算失败
    ) -> pd.DataFrame:
        """批量归因，每天对低CTR创意运行"""
        results = []
        for _, row in failed_creatives.iterrows():
            result = self.attribute_failure(
                row[self.feature_names].values,
                row["creative_id"]
            )
            results.append(result)
        
        return pd.DataFrame(results)
```

### 2.3 SHAP值可视化解读

```
单条失败创意的SHAP解读示例：

创意ID: ad_12345
基准CTR（全局均值）: 0.050
预测CTR: 0.018（远低于均值）

SHAP分解：
  基准CTR          =  +0.050
  user_age_mismatch = -0.018  ← 最大负贡献：用户年龄不匹配
  text_style_formal = -0.012  ← 文案风格太正式
  low_category_ctr  = -0.008  ← 品类历史CTR偏低
  seasonal_factor   = +0.006  ← 季节加成
  image_quality     = +0.002  ← 图片质量好
  ─────────────────────────────
  预测CTR           =  0.020

结论：主要失败原因是用户定向年龄段（45-55岁）与文案风格（年轻网络用语）不匹配
建议：调整定向到25-35岁，或改写成偏保守风格
```

---

## 三、LIME局部近似解释

### 3.1 原理

**LIME（Local Interpretable Model-Agnostic Explanations）核心思想：**
在预测样本的**邻域内**，用简单的线性模型来近似复杂模型的局部行为

**优化目标：**

$$
\xi(x) = \arg\min_{g \in G} \underbrace{\mathcal{L}(f, g, \pi_x)}_{\text{局部忠实度}} + \underbrace{\Omega(g)}_{\text{模型复杂度}}
$$

- $f$：复杂黑盒模型（CTR预测模型）
- $g$：简单解释模型（线性回归）
- $\pi_x$：以x为中心的局部权重（越近权重越大）
- $\Omega(g)$：正则化项（限制线性模型的参数数量）

**实现：**
```python
import lime
from lime.lime_text import LimeTextExplainer
from lime.lime_tabular import LimeTabularExplainer

class LIMEAttributor:
    def __init__(self, ctr_model, feature_names: List[str]):
        self.model = ctr_model
        self.explainer = LimeTabularExplainer(
            training_data=self.get_training_data(),
            feature_names=feature_names,
            mode="regression"
        )
    
    def explain_creative(
        self,
        creative_features: np.ndarray,
        n_features: int = 10  # 只展示最重要的10个特征
    ):
        """
        LIME解释：为什么这条创意的CTR被预测为这么低？
        """
        explanation = self.explainer.explain_instance(
            creative_features,
            self.model.predict,
            num_features=n_features,
            num_samples=500  # 邻域采样次数
        )
        
        # 提取特征-权重对
        feature_weights = explanation.as_list()
        return {
            "local_explanation": feature_weights,
            "local_model_score": explanation.score,  # 局部线性模型R²
            "predicted_value": explanation.predicted_value
        }
```

### 3.2 SHAP vs LIME 实用对比

| 维度 | SHAP | LIME |
|------|------|------|
| 理论基础 | 博弈论（公理满足） | 局部线性近似 |
| 全局一致性 | ✅（满足公理）| ❌（邻域依赖，不稳定）|
| 计算速度 | TreeSHAP：快；KernelSHAP：慢 | 中等（需要采样） |
| 模型类型 | 树模型最优；也支持其他 | **模型无关**（任何模型）|
| 文本特征 | 需要词嵌入特征 | 原生支持文本 |
| 生产使用 | ✅ 批量离线归因 | ✅ 单条实时解释 |

---

## 四、广告创意失败分类框架

### 4.1 失败模式库

| 失败模式 | 主要SHAP负贡献特征 | 识别特征 | 解决方案 |
|---------|-----------------|---------|---------|
| 目标用户不匹配 | user_age_group, user_interest | 用户特征负贡献 > 总负贡献60% | 重新选择定向人群，或改写文案风格 |
| 创意语言风格偏差 | text_embedding_dist, style_score | 文本embedding与高CTR创意距离 > 0.5 | 使用RAG补充相近品类参考，重新生成 |
| 出价不足 | avg_position, bid_to_win_ratio | 位置特征负贡献 > 50% | 提高出价或选低竞争时段 |
| 品类冷启动 | category_hist_ctr, category_sample_size | 品类特征全部为空/为0 | 使用跨品类迁移，借用相似品类参考 |
| 审核不通过 | policy_violation_prob | 合规特征 < 阈值 | 内容过滤，触发合规审查流水线 |
| 季节性错配 | seasonal_factor, holiday_boost | 时间特征负贡献 | 根据当前时段筛选对应季节的参考创意 |

### 4.2 系统性归因分析（汇总视角）

```python
def systemic_failure_analysis(
    daily_failed_creatives: pd.DataFrame
) -> dict:
    """
    汇总分析：找出哪类特征系统性导致失败
    """
    pattern_counts = daily_failed_creatives["failure_pattern"].value_counts()
    
    # 如果某类失败模式占比 > 40%，说明是系统性问题
    systemic_issues = {
        pattern: count
        for pattern, count in pattern_counts.items()
        if count / len(daily_failed_creatives) > 0.4
    }
    
    recommendations = {}
    if "user_mismatch" in systemic_issues:
        recommendations["user_mismatch"] = (
            "系统性用户画像不匹配，建议审查用户特征工程，"
            "检查定向标签是否过期或穿越"
        )
    
    if "cold_start" in systemic_issues:
        recommendations["cold_start"] = (
            "冷启动品类失败率高，建议优化跨品类迁移策略，"
            "或为新品类建立快速积累机制（降价引流期）"
        )
    
    return {
        "date": pd.Timestamp.now().date(),
        "total_failures": len(daily_failed_creatives),
        "pattern_distribution": pattern_counts.to_dict(),
        "systemic_issues": systemic_issues,
        "recommendations": recommendations
    }
```

---

## 五、系统落地

### 5.1 生产化批量归因流程

```python
# 每天凌晨2点运行的归因任务
def daily_attribution_job():
    # 1. 从ClickHouse拉取低CTR创意
    failed_creatives = clickhouse_client.query("""
        SELECT creative_id, features, actual_ctr, expected_ctr
        FROM ad_performance
        WHERE date = yesterday()
          AND actual_ctr < expected_ctr * 0.5
          AND impressions > 500
        LIMIT 50000
    """)
    
    # 2. 批量SHAP归因（TreeSHAP，高效）
    attributor = AdCreativeAttributor(ctr_model, feature_names)
    attribution_results = attributor.batch_attribute(failed_creatives)
    
    # 3. 结果写回ClickHouse
    clickhouse_client.insert("failure_attribution", attribution_results)
    
    # 4. 生成日报
    daily_report = systemic_failure_analysis(attribution_results)
    send_report_to_ops_team(daily_report)
```

### 5.2 计算复杂度与优化

**TreeSHAP的时间复杂度：**
- 精确计算：$O(TLD^2)$，其中T=树数，L=叶节点数，D=树深度
- 对LightGBM（T=100, L=64, D=6）：约0.5ms/样本
- 5万条失败创意：约25秒（完全可接受）

**KernelSHAP（模型无关版本）：**
- 计算复杂度：$O(2^n \cdot N_{samples})$，样本量敏感
- 实践中：每条样本约1~10秒（深度神经网络的归因用）
- 对于神经网络CTR模型，推荐用IntegratedGradients代替

### 5.3 与ClickHouse集成

```sql
-- 归因结果存储表
CREATE TABLE failure_attribution (
    date Date,
    creative_id String,
    base_ctr Float32,
    predicted_ctr Float32,
    actual_ctr Float32,
    failure_pattern Enum('user_mismatch', 'creative_quality', 
                          'bid_insufficient', 'cold_start', 'unknown'),
    top_negative_features Array(String),
    shap_values Array(Float32),
    insert_time DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY date
ORDER BY (date, failure_pattern, creative_id);

-- 运营查询：哪些广告主的失败率最高？
SELECT 
    advertiser_id,
    failure_pattern,
    count() as failure_count,
    avg(actual_ctr) as avg_ctr
FROM failure_attribution
WHERE date >= today() - 7
GROUP BY advertiser_id, failure_pattern
ORDER BY failure_count DESC
LIMIT 20;
```

---

## 六、面试考点 Q&A

**Q1：SHAP vs LIME vs 积分梯度，各自适用什么场景？**

A：三者的核心区别在于模型假设和适用场景。SHAP（尤其是TreeSHAP）对树模型最高效，有公理化保证（效率性、对称性、虚拟性），适合生产环境批量归因（0.5ms/样本）；LIME是模型无关的局部近似，适合单条样本的实时解释，尤其擅长文本（LimeTextExplainer可以直接高亮关键词），但不稳定（不同运行结果可能不同）；积分梯度（Integrated Gradients）适合深度神经网络，直接从梯度计算attribution，满足完整性公理（$\sum \phi_i = f(x) - f(x')$），是神经网络可解释性的标准方法。广告场景推荐：树模型用TreeSHAP，神经网络CTR用积分梯度，单条创意实时解释用LIME。

**Q2：Shapley值的计算复杂度是多少？如何近似？**

A：精确计算Shapley值需要枚举所有子集（$2^N$个），对N=100个特征几乎不可能。近似方法：(1) TreeSHAP——利用树结构将复杂度降至$O(TLD^2)$，是精确的（不是近似），仅限树模型；(2) KernelSHAP——用蒙特卡洛采样估计，复杂度与采样次数线性相关，1000次采样通常足够；(3) DeepSHAP——针对神经网络，基于DeepLIFT快速估计，比KernelSHAP快10~100倍；(4) 特征分组——将高度相关的特征合并为一组，大幅减少有效特征数。广告场景：若用LightGBM CTR模型，直接用TreeSHAP精确计算；若用DNN，用DeepSHAP近似。

**Q3：如何解释SHAP值的正负方向？**

A：SHAP值的正负是相对于**基准预测值**（训练集平均CTR）的偏差。正SHAP值（如+0.02）：该特征使预测CTR高于平均值，是"有利"因素；负SHAP值（如-0.015）：该特征使预测CTR低于平均值，是"不利"因素。关键理解：绝对值大小 = 影响程度，正负号 = 方向。例如"用户年龄=55岁"对一条年轻人文案的SHAP值为-0.018，意味着这个特征将CTR拉低了1.8个百分点。注意：SHAP值是特征在当前样本的贡献，与全局重要性（平均|SHAP|）不同。

**Q4：归因分析结果如何反馈到模型改进？**

A：建立归因→改进的闭环：(1) 特征层面——SHAP汇总发现某类特征系统性负贡献，说明该特征在CTR模型中权重不准，触发模型重训练（加入该特征的交叉特征或调整权重）；(2) 数据层面——发现冷启动失败率高，增加冷启动场景的训练样本（主动收集数据）或优化迁移学习策略；(3) 系统层面——发现"检索参考质量差"是主因，说明RAG检索模块需要优化，而不是生成模型的问题；(4) 规则层面——归因发现某类违禁词模式频繁导致失败，直接加入规则过滤层（比重训模型快得多）。

**Q5：失败案例归因和A/B测试有什么互补关系？**

A：两者互为补充，解决不同层面的问题。A/B测试告诉你"哪个方案更好"但不告诉你"为什么"；失败归因告诉你"为什么失败"但不能验证改进方案的效果。最佳实践：先用归因找到失败原因（如"用户画像不匹配"），再针对性地设计改进方案，然后用A/B测试验证改进效果。此外，归因还能缩短A/B测试周期：不需要等待显著性，归因分析可以在3天内给出初步诊断，指导快速迭代。反之，A/B测试失败时（两组没有显著差异），归因可以解释为什么改进无效（可能是定向人群问题，而不是文案问题）。

**Q6：对文本特征（广告文案）的SHAP归因如何做？**

A：文本特征的SHAP归因有两种粒度。词袋级（粗）：将文案转为TF-IDF或词频向量，直接用TreeSHAP计算，可以找到"哪些词的出现/不出现影响了CTR预测"；词级精细（推荐）：用LimeTextExplainer，对文案中的每个词进行遮盖（masking），观察预测变化，得到词级别的重要性分数（高亮哪些词对CTR影响最大）。例如：文案「限时优惠，仅剩3件，立即抢购！」中，"限时优惠"SHAP=-0.008（平台认为是促销垃圾信息），"立即抢购"SHAP=-0.012（违规语气词），帮助改写为合规版本。

**Q7：如何处理特征间的共线性对SHAP的影响？**

A：共线性是SHAP最大的挑战之一。问题场景：用户画像特征（年龄、消费水平、兴趣标签）高度相关，SHAP会在这些特征间"分配"贡献，导致每个特征的SHAP值看起来都不大，但实际上是"用户特征整体"的影响。解决方法：(1) 特征分组——将高度相关（r > 0.7）的特征分组，计算组内总SHAP，而非单特征；(2) 使用SHAP Interaction Values——专门量化特征i和j的交互效应；(3) 主成分分析（PCA）——对高相关特征组做PCA，用主成分参与归因，事后映射回原始特征；(4) 警告机制——计算特征相关性矩阵，当某些特征相关性 > 0.8时，在归因报告中标注"共线性警告，结果仅供参考"。

---

*参考文献：*
- *SHAP: Lundberg & Lee, NeurIPS 2017*
- *LIME: Ribeiro et al., KDD 2016*
- *Integrated Gradients: Sundararajan et al., ICML 2017*
- *Interpretable Machine Learning: Christoph Molnar, 2020, 第6-8章*
