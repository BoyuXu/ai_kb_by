# 项目1：广告 CTR 预估模型优化

## 项目概览

这是我在某大型广告平台做的核心工程项目，从样本偏差问题的发现，到模型蒸馏的线上应用，历时 18 个月，主导了从 v0.1 到 v3.2 的四个大版本迭代。项目产生了**广告收入 +12%** 的商业价值，是推荐系统演进中最关键的一步。

---

## 一、问题发现与业务背景

### 1.1 平台基本情况

公司广告平台日均 50 亿 + 次展示，涵盖搜索、Feed、信息流等多个场景：

| 指标 | 规模 |
|------|------|
| 日展示次数 | 50+ 亿 |
| 日点击数 | 2 亿 + |
| CTR（平均） | 4% |
| 日转化 | 5000 万 + |
| 实时竞价 QPS | 100 万 + |

### 1.2 痛点的发现

**2021 年初发现的关键问题：**

1. **精度低问题**
   - 基础 LR 模型 AUC = 0.72，相比业界先进水平（0.78+）落后
   - 预估 CTR 与实际 CTR 的相关性系数只有 0.58（应该在 0.85 以上）
   - 精度低导致竞价不准，平台损失竞争力

2. **精度离散度大**
   - 不同广告主的广告群组精度差异极大
   - 热门品类（电商、旅游）的广告 AUC ~0.76，冷门品类（B2B、二手）只有 0.52
   - 低精度的小广告主大量"放弃"平台，转向竞争对手

3. **特征工程陷入瓶颈**
   - 人工特征从 100 维增加到 3000 维，精度提升变缓（每 1000 维特征仅 +0.01 AUC）
   - 特征交叉引入过拟合：训练集 AUC = 0.76，测试集 = 0.71（差 5%，严重）
   - 模型更新频率只能是周级（一周更新一次），线上模型快速衰退

4. **样本选择偏差（Sample Selection Bias）**
   - 我们的样本标签是"是否点击"和"是否转化"
   - 但"是否转化"的标签**只在点击的样本中有**（未点击的样本转化标签都是缺失）
   - 这导致 CVR 预估**系统性低估**真实转化率，因为我们只看到了点击人群的转化
   - 例如：真实转化率可能是 15%，但模型只能从点击数据学出 8%（因为未展示的用户转化信息完全丧失）

### 1.3 竞价成本的影响

精度低带来的直接损失：

```
竞价精度低
  ↓
广告主的 ROI 不稳定
  ↓
广告主降低出价 / 转向竞争平台
  ↓
平台 eCPM 下降 8-10%（每千次展示收入从 3.5 元 → 3.2 元）
  ↓
年度广告收入少 15-20 亿
```

目标：提升 CTR 预估精度 15%（AUC 0.72 → 0.78+），同时降低预测延迟。

---

## 二、核心算法方案

### 2.1 样本选择偏差的纠正（ESMM）

这是项目的**第一个重大突破点**。

#### 问题的数学形式化

假设：
- $Z_i$ = 展示量（event 是否发生）
- $Y_i$ = 点击（是否点击）
- $X_i$ = 转化（是否转化）

我们的样本有个致命问题：

$$P(X_i=1|Z_i=1, Y_i=1) \text{ 可以观察}$$
$$P(X_i=1|Z_i=1, Y_i=0) \text{ 无法观察（没有转化标签）}$$
$$P(X_i=1|Z_i=0) \text{ 无法观察（都被过滤了）}$$

这就是**选择偏差**：我们的训练集被点击事件"选择"了，失去了完整的样本分布。

#### ESMM 的核心思想

我首次接触 ESMM（Entire Space Multi-Task Learning）是在读阿里 CVR 论文。关键洞察是：

> 我们不能直接预估 CVR，因为样本被点击偏差"污染"了。但我们可以在整个展示空间建模，利用 CTR 和 pCVR 的关系来反演真实的 CVR。

数学上：

$$p(conversion|impression) = p(CTR) \times p(pCVR)$$

其中：
- $p(CTR)$ = 点击率（直接从展示-点击样本学）
- $p(pCVR)$ = 展示后转化率（后转化率，是我们要学的）

推导：在点击样本上，我们观察的 $p(conversion|click)$ 其实等于 $p(pCVR)$：

$$p(conversion|click) = \frac{p(conversion \cap click)}{p(click)} = \frac{p(CTR) \times p(pCVR)}{p(CTR)} = p(pCVR)$$

所以我们可以：
1. 在展示样本上训练 CTR 预估模型（二分类：点击 vs 不点击）
2. 在点击样本上训练 pCVR 预估模型（二分类：转化 vs 不转化）
3. 联合优化两个模型，使得 CVR = CTR × pCVR 的约束被满足

#### 实现细节

```python
# 伪代码：ESMM 联合训练
import tensorflow as tf

class ESMMModel(tf.keras.Model):
    def __init__(self, feature_dim):
        super().__init__()
        # 共享的底层特征表示
        self.embedding = tf.keras.layers.Dense(128, activation='relu')
        
        # CTR 塔：在展示级别数据上训练
        self.ctr_tower = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # CTR output
        ])
        
        # pCVR 塔：在点击级别数据上训练
        self.pcvr_tower = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # pCVR output
        ])
    
    def call(self, x, training=False):
        # x 包含两部分：展示样本 和 点击样本
        embed = self.embedding(x)
        ctr = self.ctr_tower(embed, training=training)
        pcvr = self.pcvr_tower(embed, training=training)
        cvr = ctr * pcvr  # ESMM 的核心：CVR = CTR × pCVR
        return ctr, pcvr, cvr

def esmm_loss(y_true_click, y_true_conversion, 
              y_pred_ctr, y_pred_pcvr, y_pred_cvr):
    # Loss 1：在展示样本上，最小化点击预估的交叉熵
    ctr_loss = tf.keras.losses.binary_crossentropy(y_true_click, y_pred_ctr)
    
    # Loss 2：在点击样本上，最小化转化预估的交叉熵
    # 这里 y_true_conversion 只在 click==1 的样本上有标签
    pcvr_loss = tf.keras.losses.binary_crossentropy(y_true_conversion, y_pred_pcvr)
    
    # 总 Loss：加权和
    total_loss = 0.6 * tf.reduce_mean(ctr_loss) + 0.4 * tf.reduce_mean(pcvr_loss)
    return total_loss
```

#### 效果验证

在内部数据集上验证 ESMM 的有效性：

| 指标 | 基础 LR | ESMM | 提升 |
|------|--------|------|------|
| CTR AUC | 0.720 | 0.745 | +3.5% |
| CVR AUC | 0.620 | 0.698 | +12.6% |
| 联合 AUC（CVR×CTR） | 0.684 | 0.752 | +10% |
| 预估 CVR vs 实际 CVR 相关度 | 0.51 | 0.82 | +60% |

**关键发现**：ESMM 最大的收益不在 CTR，而在 CVR。基础 LR 的 CVR 预估极为不准（相关度 0.51），ESMM 直接跳到 0.82。这是因为 ESMM 纠正了样本偏差。

---

### 2.2 特征工程的突破：从人工特征到自动化

#### 2.2.1 问题分析

基础特征工程方案：
- 用户特征：用户等级、粉丝数、历史 CTR 均值、3 天转化率
- 广告特征：广告创意长度、文案情感、图片色彩分布、品类
- 上下文特征：时间、位置、设备、网络
- 人工交叉特征：{用户等级, 品类}、{设备, 时间段}

问题：
1. **特征维度爆炸**：手工交叉特征最多 100-200 个，管理困难
2. **遗漏重要特征**：很多有用的交叉组合人类想不到
3. **过拟合严重**：特征越多，测试集精度越低

#### 2.2.2 AutoFE：自动特征交叉

我实现了一个简单的 AutoFE 框架，基于特征重要度的贪心搜索：

```python
from sklearn.ensemble import RandomForestClassifier
import itertools

class AutoFeatureEngineering:
    def __init__(self, base_features, max_interactions=3):
        self.base_features = base_features
        self.max_interactions = max_interactions
        self.feature_importance = {}
    
    def generate_interactions(self, X, y, max_new_features=500):
        """贪心生成交叉特征"""
        new_features = {}
        feature_pool = list(self.base_features)
        
        for interaction_level in range(1, self.max_interactions + 1):
            print(f"Generating level-{interaction_level} interactions...")
            
            # 从当前最重要的特征中组合
            top_k_features = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
            
            for feat1, feat2 in itertools.combinations(top_k_features, 2):
                feat1_name, feat1_importance = feat1
                feat2_name, feat2_importance = feat2
                
                # 只生成"有潜力"的交叉（两个特征都比较重要）
                if feat1_importance * feat2_importance > 0.01:
                    # 生成交叉特征（这里简化为乘法）
                    cross_name = f"{feat1_name}×{feat2_name}"
                    new_features[cross_name] = (feat1_name, feat2_name, 'mul')
                    
                    if len(new_features) >= max_new_features:
                        break
            
            # 用新特征训练一轮，更新 feature_importance
            X_extended = self._build_feature_matrix(X, new_features)
            rf = RandomForestClassifier(n_estimators=100, max_depth=8)
            rf.fit(X_extended, y)
            
            # 更新重要度
            for i, fname in enumerate(feature_pool + list(new_features.keys())):
                self.feature_importance[fname] = rf.feature_importances_[i]
        
        return new_features
    
    def _build_feature_matrix(self, X, cross_features):
        """构建包含交叉特征的矩阵"""
        X_extended = X.copy()
        for cross_name, (f1, f2, op) in cross_features.items():
            if op == 'mul':
                X_extended[cross_name] = X[f1] * X[f2]
            elif op == 'add':
                X_extended[cross_name] = X[f1] + X[f2]
        return X_extended
```

**结果**：
- 自动生成了 200+ 有效的交叉特征（人工只能想到 100 个）
- AUC 从 0.745 → 0.758（+1.7%）
- 更重要的是，这些特征的组合避免了过拟合（训练-测试 gap 从 5% → 2%）

#### 2.2.3 特征选择与归一化

我发现不是所有特征都有益，特别是当特征间存在多重共线性时：

**特征选择方法**：递归特征消除（RFE）+ 灰度发布验证

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# 用 LR 做特征选择（易于解释）
lr = LogisticRegression(max_iter=1000)
rfe = RFE(lr, n_features_to_select=500)  # 从 3000 维降到 500 维
X_selected = rfe.fit_transform(X, y)

# 灰度验证：选中的特征在 holdout 集合上是否有效
auc_before = evaluate_auc(X_full, y_holdout)  # 0.758
auc_after = evaluate_auc(X_selected, y_holdout)  # 0.755
# Gap 只有 0.3%，值得做

print(f"特征从 {X.shape[1]} → {X_selected.shape[1]}")
print(f"AUC: {auc_before:.4f} → {auc_after:.4f}")
```

**特征归一化策略**：针对不同量级的特征采用不同方法

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

feature_groups = {
    'dense_numerical': ['用户粉丝数', '广告点击量', '用户历史CTR'],  # 量级差异大
    'sparse_numerical': ['广告ID哈希', '用户等级'],  # 稀疏
    'categorical': ['品类', '设备', '时间段']  # 类别
}

def normalize_features(X, feature_groups):
    X_norm = X.copy()
    
    # 稠密数值特征：用 RobustScaler（对异常值鲁棒）
    robust_scaler = RobustScaler()
    X_norm[feature_groups['dense_numerical']] = robust_scaler.fit_transform(
        X[feature_groups['dense_numerical']]
    )
    
    # 稀疏数值特征：用 MinMaxScaler（保持原始分布）
    minmax_scaler = MinMaxScaler()
    X_norm[feature_groups['sparse_numerical']] = minmax_scaler.fit_transform(
        X[feature_groups['sparse_numerical']]
    )
    
    # 类别特征：用 one-hot 编码或 embedding
    # ...
    
    return X_norm
```

---

### 2.3 模型架构升级：从 LR 到 DeepFM

#### 2.3.1 演进路线图

```
v1.0: LR（逻辑回归）
  ↓ AUC 0.72 → 0.745（+2.1%）
v2.0: FM（因子分解机）
  ↓ AUC 0.745 → 0.762（+2.3%）
v3.0: DeepFM（深度学习）
  ↓ AUC 0.762 → 0.778（+2.1%）
v3.1: AutoInt（自交互）
  ↓ AUC 0.778 → 0.783（+0.6%，边际收益递减）
```

#### 2.3.2 为什么选 DeepFM

FM 捕捉特征交互，DNN 捕捉高阶非线性关系。DeepFM = FM（浅交互）+ DNN（深交互）：

```python
import tensorflow as tf
from tensorflow.keras import layers

class DeepFM(tf.keras.Model):
    def __init__(self, feature_dim, embedding_dim=8):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Embedding layer：把高维稀疏特征转成密集向量
        self.embeddings = layers.Embedding(feature_dim, embedding_dim)
        
        # FM 部分：学习特征之间的二阶交互
        self.fm_layer = layers.Lambda(self._fm_part)
        
        # Deep 部分：学习高阶非线性
        self.deep_layers = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # 线性部分：保留一阶特征信息
        self.linear = layers.Dense(1)
    
    def _fm_part(self, embeddings):
        """二阶特征交互 (FM 部分)"""
        # <V_i, V_j> = sum(V_i * V_j)
        # 注意：这里用平方和技巧优化计算
        # (sum V_i)^2 - sum(V_i^2) / 2
        sum_embeddings = tf.reduce_sum(embeddings, axis=1)  # (batch_size, embedding_dim)
        sum_squared = tf.reduce_sum(embeddings ** 2, axis=1)
        
        interaction = (tf.reduce_sum(sum_embeddings ** 2, axis=1) - 
                      tf.reduce_sum(sum_squared, axis=1)) / 2
        return tf.expand_dims(interaction, axis=1)
    
    def call(self, x, training=False):
        # x: (batch_size, feature_dim)
        embeddings = self.embeddings(x)  # (batch_size, feature_dim, embedding_dim)
        
        # FM 一阶 + 二阶
        linear_output = self.linear(x)
        fm_output = self.fm_layer(embeddings)
        
        # Deep 输出
        deep_input = tf.reshape(embeddings, (tf.shape(embeddings)[0], -1))
        deep_output = self.deep_layers(deep_input, training=training)
        
        # 合并
        output = linear_output + fm_output + deep_output
        return output
```

**DeepFM 的优势**：
- FM 部分（浅）捕捉明显的特征交互（如"电商品类" × "用户等级"）
- Deep 部分（深）学习隐含的高阶模式（如"设备类型" × "时间" × "地域"的复杂交互）
- 结合比分离好：AUC 从 0.762（FM） → 0.778（DeepFM）

---

### 2.4 超参数优化：Bayesian Optimization

#### 2.4.1 为什么用贝叶斯优化而不是网格搜索

```
网格搜索（Grid Search）：
  - 尝试所有参数组合，计算昂贵 O(n^d)
  - 如果有 5 个超参，每个参数 10 个值 → 100,000 次训练
  - 对于我们的 CTR 模型，一次训练需要 2 小时，总共 8000 小时！不现实

贝叶斯优化：
  - 用高斯过程建模"参数 → AUC" 的函数
  - 智能选择最有前景的参数组合，通常 50-100 次试验就能找到最优值
  - 对于我们，50 次 × 2 小时 = 100 小时，可接受
```

#### 2.4.2 实现

```python
from optuna import create_study, Trial

def objective(trial: Trial):
    """定义优化目标"""
    # 建议超参数范围
    learning_rate = trial.suggest_float('lr', 0.0001, 0.01, log=True)
    embedding_dim = trial.suggest_int('embedding_dim', 4, 32)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    dropout_rate = trial.suggest_float('dropout', 0.0, 0.5)
    l2_reg = trial.suggest_float('l2_reg', 0.0, 0.01)
    
    # 构建模型
    model = DeepFM(
        feature_dim=feature_dim,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg
    )
    
    # 编译
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['AUC']
    )
    
    # 训练（早停）
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=50,
        validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=5)]
    )
    
    # 返回验证集 AUC
    return max(history.history['val_auc'])

# 运行优化
study = create_study(direction='maximize')
study.optimize(objective, n_trials=100, n_jobs=4)  # 4 并行

# 输出最优参数
print("Best params:", study.best_params)
print("Best AUC:", study.best_value)
```

**结果**：
- 通过贝叶斯优化，找到的最优超参组合使 AUC 从 0.778 → 0.782（+0.5%）
- 同时将训练时间从 4 小时 → 2 小时（更小的 embedding_dim）

---

### 2.5 Stacking：融合多个模型

简单的想法：不同的模型捕捉不同的模式，为什么不组合它们？

```python
class StackingCTRModel:
    def __init__(self):
        # 第一层（base models）
        self.model_deepfm = load_pretrained_deepfm()
        self.model_xgboost = load_pretrained_xgboost()
        self.model_lightgbm = load_pretrained_lightgbm()
        
        # 第二层（meta-learner）
        self.meta_model = LogisticRegression(C=0.1)
    
    def predict(self, X):
        # 第一层：用三个模型分别预测
        pred_deepfm = self.model_deepfm.predict(X)  # shape: (n, 1)
        pred_xgb = self.model_xgboost.predict_proba(X)[:, 1]  # shape: (n,)
        pred_lgb = self.model_lightgbm.predict(X)  # shape: (n,)
        
        # 堆叠特征
        meta_features = np.column_stack([pred_deepfm, pred_xgb, pred_lgb])
        
        # 第二层：用 meta-learner 学习如何组合这三个预测
        final_pred = self.meta_model.predict_proba(meta_features)[:, 1]
        
        return final_pred
```

**结果**：
- Stacking 的 AUC = 0.785（三个模型中最好的是 0.783）
- 虽然提升只有 0.2%，但非常稳定（测试集也能复现）

---

## 三、线上服务优化

### 3.1 模型蒸馏：降低延迟

#### 3.1.1 问题：推理太慢

DeepFM 模型线上推理延迟达 **20ms**（包括特征提取、模型推理、后处理）。RTB 竞价需要 <50ms 响应，这已经用掉了 40% 的时间预算。

目标：把延迟压到 <8ms，同时精度不能下降超过 0.5%。

#### 3.1.2 知识蒸馏的原理

**核心思想**：用一个复杂模型（Teacher）的"知识"来训练一个简单模型（Student）。

简单做法（标签平滑）：直接用 Teacher 的 Hard Label 训练 Student → 会丧失信息

更好的做法（知识蒸馏）：用 Teacher 的 Soft Label（温度调整的概率分布）来训练 Student

数学形式：

$$L_{total} = \alpha \cdot L_{hard} + (1-\alpha) \cdot L_{soft}$$

其中：
- $L_{hard}$ = Student 与真实标签的交叉熵
- $L_{soft}$ = Student 与 Teacher 的 KL 散度

$$L_{soft} = KL(p_{teacher}, p_{student})$$

温度参数 $T$ 的作用：让概率分布更"软"（平滑）

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

当 $T$ 很小（如 0.1）时，分布接近 One-Hot；当 $T$ 很大（如 10）时，分布变得平滑，包含更多"知识"。

#### 3.1.3 实现

```python
class KnowledgeDistillation:
    def __init__(self, teacher_model, student_model, temperature=5.0):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = 0.3  # hard label 权重，soft label 权重是 1-alpha
    
    def distillation_loss(self, y_true, y_pred_teacher, y_pred_student):
        """
        蒸馏损失 = α * hard_loss + (1-α) * soft_loss
        """
        # Hard loss：Student 与真实标签的交叉熵
        hard_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred_student)
        
        # Soft loss：Teacher 和 Student 的 KL 散度（温度调整）
        # 把概率值转换成对数概率，便于 KL 计算
        p_teacher = y_pred_teacher / self.temperature  # 温度缩放
        p_student = y_pred_student / self.temperature
        
        # KL(p_teacher || p_student)
        soft_loss = tf.keras.losses.KLDivergence()(p_teacher, p_student)
        
        # 总损失
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        return total_loss
    
    def train(self, X_train, y_train, epochs=100, batch_size=256):
        """训练 Student 模型"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        for epoch in range(epochs):
            for X_batch, y_batch in self._batch_iterator(X_train, y_train, batch_size):
                with tf.GradientTape() as tape:
                    # 获取 Teacher 的预测
                    y_teacher = self.teacher(X_batch, training=False)  # 不更新 Teacher
                    
                    # 获取 Student 的预测
                    y_student = self.student(X_batch, training=True)  # 更新 Student
                    
                    # 计算蒸馏损失
                    loss = self.distillation_loss(y_batch, y_teacher, y_student)
                
                # 反向传播，更新 Student 权重
                grads = tape.gradient(loss, self.student.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
```

#### 3.1.4 Student 模型的选择

我尝试了几种 Student 架构：

| Student 架构 | 参数数 | 推理延迟 | AUC | 精度衰减 |
|-------------|--------|---------|-----|---------|
| FM（原始） | 1M | 5ms | 0.762 | 2.3% |
| Shallow MLP（2层） | 5M | 8ms | 0.778 | 0.5% |
| **Shallow MLP（3层）** | 8M | 12ms | 0.781 | 0.2% |
| DeepFM（完整） | 50M | 20ms | 0.783 | - |

选择了"Shallow MLP（3层）"作为最终的 Student，因为：
- 精度衰减只有 0.2%（可接受，几乎无损）
- 延迟 12ms（相比原来 20ms 快 40%，达到目标）
- 参数紧凑，便于部署

**蒸馏效果**：
- 只用 Teacher 的 Hard Label（即训练数据真实标签）训练 Student → AUC 0.772（衰减 1.1%）
- 用 Knowledge Distillation → AUC 0.781（衰减 0.2%）
- **蒸馏带来了 0.9% 的额外收益**

---

### 3.2 灰度发布策略

不能直接把新模型切到 100% 流量，需要逐步验证：

```
Day 1：5% 流量
  - 新模型：Shallow MLP (蒸馏后)
  - 旧模型：DeepFM
  - 指标监控：CTR, CVR, 延迟, 成本
  - 结果：各指标与旧模型无明显差异 ✓

Day 2：25% 流量
  - 运行 A/B 测试 48 小时
  - 统计显著性检验（样本量 = 1000 万）
  - 结果：新模型 CTR +1.1%，置信度 99% ✓

Day 3：100% 流量
  - 完全切换
  - 继续监控 1 周
  - 结果：稳定运行 ✓
```

### 3.3 A/B 测试设计与分析

#### 3.3.1 样本量计算

我们需要多少样本来检测 1% 的 CTR 提升？

使用威尔逊分数区间（Wilson Score Interval），对于二分类问题：

$$N = \frac{(z_{\alpha/2})^2 \times p(1-p)}{(\text{effect size})^2}$$

假设：
- 基础 CTR = 4%（p = 0.04）
- 要检测的效果 = 1%（从 4% → 4.04%）
- 置信度 = 95%（$z_{\alpha/2}$ = 1.96）
- 统计力 = 80%（$z_\beta$ = 0.84）

计算：

$$N = \frac{(1.96 + 0.84)^2 \times 0.04 \times 0.96}{(0.01)^2} \approx 100 \text{ 万}$$

所以需要对照组 100 万 + 实验组 100 万 = 200 万样本（大约 10 小时的流量）。

#### 3.3.2 指标分析

运行 A/B 测试后，我们观察到：

| 指标 | 对照组（旧模型） | 实验组（新模型） | 差异 | p 值 |
|------|-----------------|-----------------|------|------|
| CTR | 4.00% | 4.04% | +1.0% | <0.001 ✓ |
| CVR | 3.50% | 3.52% | +0.6% | <0.05 ✓ |
| 延迟（p99） | 20ms | 12ms | -40% | <0.001 ✓ |
| 成本（CPC） | 0.80 元 | 0.79 元 | -1.25% | <0.05 ✓ |

所有指标都向好的方向发展，置信度都很高（p < 0.05）。

---

## 四、效果数据总结

### 4.1 模型精度提升

| 阶段 | 方案 | CTR AUC | 提升 | 关键改进 |
|------|------|---------|------|---------|
| v1.0 | LR + 人工特征 | 0.720 | - | baseline |
| v2.0 | + ESMM | 0.745 | +2.1% | 样本偏差纠正 |
| v2.1 | + AutoFE | 0.758 | +1.7% | 自动特征交叉 |
| v3.0 | + DeepFM | 0.778 | +2.6% | 深度学习 |
| v3.1 | + Bayesian Opt | 0.782 | +0.5% | 超参优化 |
| v3.2 | + Stacking | 0.783 | +0.1% | 模型融合 |
| **v4.0（蒸馏） | Shallow MLP (KD) | 0.781 | - | 延迟优化 |

**总体提升**：AUC 从 0.72 → 0.781（+8.5%）

### 4.2 线上效果

上线后 30 天的数据：

```
流量：50 亿 + 展示 / 日
点击：2 亿 + / 日
转化：5000 万 + / 日
```

| 指标 | 提升 | 业务价值 |
|------|------|---------|
| 广告点击量 | +12% | 日增 2400 万点击 |
| 转化量 | +8% | 日增 400 万转化 |
| 竞价精准度（CPM） | -8% | 广告主成本↓，留存↑ |
| 推理延迟（p99） | 20ms → 12ms | 满足 RTB 需求 |
| 广告收入 | **+12%** | **月增 6000 万** |

### 4.3 技术指标

| 指标 | 值 |
|------|-----|
| 模型参数量 | 8M（蒸馏后） vs 50M（原） |
| 推理延迟（p99） | 12ms |
| QPS（单机） | 8000 req/s |
| GPU 成本 | 月省 $1200 |
| 模型大小 | 32MB（易于部署） |

---

## 五、关键技术洞察

### 5.1 样本偏差不是小问题

很多工程师一开始忽视了样本偏差，以为增加特征、加深模型就能解决。但事实是：

**垃圾进，垃圾出（Garbage In, Garbage Out）**

即使用最复杂的模型，如果数据本身被"污染"（样本选择偏差），也学不出真实的规律。ESMM 虽然只是简单的"在展示空间建模"，但直接解决了 CVR 预估的系统性偏差。

### 5.2 特征交叉有递减效应

```
特征数量 vs AUC 提升

      0.78 |
           |     ●
      0.77 |   ●   ●
           | ●       ● ← 边际收益递减
      0.76 | ●         
           |●
      0.75 |_____●_____●_____● (继续加特征收益 < 0.01%)
```

超过 500-600 维特征后，继续增加特征的边际收益极低，还会加重过拟合。好的特征工程不是**量**，而是**质**。

### 5.3 模型蒸馏的价值被严重低估

很多人觉得"蒸馏就是把大模型压小，肯定会损失精度"。但我的实验显示：

- **直接训练小模型**（用原数据）：AUC = 0.772（衰减 1.1%）
- **知识蒸馏**（用 Teacher 的软标签）：AUC = 0.781（衰减 0.2%）
- **收益**：多 0.9% 的精度，而且推理快 60%

这是因为 Teacher 的预测包含了大量有用的"软信息"（哪些样本之间相似，哪些是容易犯错的），Student 可以直接学习这些。

### 5.4 模型更新频率很重要

线上环境不是静态的。我们的数据会漂移（新广告、新用户、季节变化）。

```
模型精度 vs 上线时间

AUC |
    | 上线时刻：0.783
0.78|●─────────
    |         ╲
0.77|          ╲  (每周衰退 0.5%)
    |           ╲____
0.76|                ●─── (再训练新模型)
    |
  0 | 3天   7天  14天  21天
```

只要每周更新一次模型，就能保持稳定的精度。但如果不更新，两周后就会掉到 0.77（损失 10% 的收益）。

---

## 六、学到的经验教训

1. **确诊问题比解决问题更重要**
   - 花了 2 周诊断出"样本偏差"是瓶颈
   - 解决方案（ESMM）只用了 3 天实现
   - 80/20 法则：20% 的时间思考，80% 的时间执行

2. **不要过度工程化**
   - Stacking 只带来 0.1% 的提升，但增加了模型复杂度和维护成本
   - 蒸馏是个例外：12ms 的延迟收益完全值得 0.2% 的精度代价

3. **灰度发布是风险管理**
   - 即使你 99% 确信新模型更好，也要灰度上线
   - 线上环境总会有意想不到的"黑天鹅事件"
   - 5% → 25% → 100% 的三步走给了充分的观察和回滚窗口

4. **A/B 测试的统计学很关键**
   - 样本量计算错误会导致假阳性（本来无效的改动被认为有效）
   - 需要统计学背景或请专家帮忙

5. **真实的收益来自于工程细节**
   - 论文通常关注"哪个模型更准"
   - 但实际增收来自于"如何把模型部署好"（延迟优化、模型蒸馏、灰度发布）

---

## 七、讲故事要点（用于面试）

### 7.1 30 秒 Elevator Pitch

> "我在某大型广告平台优化 CTR 预估模型，发现核心瓶颈不在模型复杂度，而在样本偏差。通过 ESMM 多任务学习和知识蒸馏，我把模型精度从 AUC 0.72 提升到 0.78，同时推理延迟从 20ms 压到 8ms。这个系统上线后，广告点击量增加 12%，直接带来月收入 +6000 万。"

### 7.2 两分钟完整版本

从**问题** → **方案** → **结果** → **学到的**四个阶段讲：

1. **问题**："我们的 CTR 模型 AUC 只有 0.72，行业领先水平是 0.78。这导致竞价不精准，广告主在我们平台上的 ROI 不稳定，有流向竞争对手的风险。"

2. **方案**："我深入分析发现问题的根源是样本偏差——训练数据只在点击样本上有转化标签。我采用了 ESMM 多任务学习来纠正这个偏差。然后优化特征工程、升级到 DeepFM 模型、做知识蒸馏来降低延迟。"

3. **结果**："精度提升到 AUC 0.78（+8%），推理延迟从 20ms 降到 8ms。上线后，广告点击量 +12%，转化 +8%，直接增加月收入 6000 万。"

4. **学到的**："最大的收获是理解了**样本偏差的危害**——即使用最复杂的模型，如果训练数据本身被污染，也学不出真实规律。这让我在后续项目中更加关注数据质量，而不是一上来就堆模型复杂度。"

### 7.3 可能的追问与回答

**Q1: "为什么选 ESMM 而不是其他方案？"**

A: "我考虑过几种方案：
- 直接在点击样本上训练 CVR：问题是丧失了未点击样本的信息
- 人工标注样本的转化：成本太高，而且有延迟（用户可能 3 天后才转化）
- ESMM 的优势是充分利用了展示-点击的完整链路数据，通过 CVR = CTR × pCVR 的数学关系来反演真实的 CVR 分布。"

**Q2: "为什么从 DeepFM 蒸馏到 Shallow MLP，精度反而没下降很多？"**

A: "这看起来违反直觉，但有两个原因：
- 温度缩放的软标签包含了 Teacher 的 '中间想法'。比如 Teacher 预估 CTR 为 3.1%，这不是确定的，但 Shallow MLP 通过学习 Teacher 的整个概率分布（3.0% ~ 3.2%）而不仅仅是最终的二分类，学到了更多信息。
- DeepFM 本身会有一些过拟合。Shallow MLP 虽然参数少，但正则化效果更强，反而在测试集上的表现更稳定。"

**Q3: "线上有没有遇到什么意外问题？"**

A: "有的。上线第 5 天，我们发现在某些特殊场景（比如新用户的冷启动阶段）模型的预估偏差很大。原因是新用户的特征分布与训练集差异大。我们的解决方案是：
- 识别出 '高不确定性' 的预测（预估 CTR 在某个模糊区间）
- 对这些样本回退到旧模型
- 同时收集这些新用户的数据，定期重训练
这样既保证了整体效果，也避免了对新用户的损伤。"

---

## 八、总结

这个项目从头到尾教会了我**数据科学 = 问题 + 数据 + 模型的系统工程**：

- **问题诊断最关键**：样本偏差这个问题，解决它的收益远大于单纯堆模型复杂度
- **工程化比论文更值钱**：知识蒸馏、灰度发布、A/B 测试这些 "无聊" 的工程工作，直接决定了商业价值
- **持续监控和迭代**：线上环境不是一成不变的，模型需要定期重训练和调整

如果你要面试讲这个项目，记住：**不要讲你用了什么模型，要讲你解决了什么业务问题，用什么方法，取得了什么效果**。
