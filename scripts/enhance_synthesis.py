#!/usr/bin/env python3
"""
Enhance synthesis files with:
1. Core formulas section (## 📐 核心公式与原理)
2. Expand Q&A to >= 10 questions
"""

import os
import re
import sys

KB = os.path.expanduser("~/Documents/ai-kb")

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def count_questions(content):
    return len(re.findall(r'^### Q\d+:', content, re.MULTILINE))

def has_formula_section(content):
    return '## 📐 核心公式与原理' in content

def get_last_question_num(content):
    nums = re.findall(r'^### Q(\d+):', content, re.MULTILINE)
    return max(int(n) for n in nums) if nums else 0

# Core formulas and extra Q&A by synthesis file
ENHANCEMENTS = {
    # ==================== ADS ====================
    '广告CTR_CVR预估与校准.md': {
        'formulas': """## 📐 核心公式与原理

### 1. ESMM 全空间多任务
$$pCTCVR = pCTR \\times pCVR$$
- CTR 和 CVR 共享底层 embedding，CVR 在全曝光空间隐式训练，解决样本选择偏差

### 2. Platt Scaling 校准
$$p_{calibrated} = \\sigma(a \\cdot \\text{logit} + b)$$
- 用验证集拟合 a, b，使预测概率与实际概率对齐

### 3. FTRL 在线更新
$$w_{t+1} = \\arg\\min_w \\left( \\sum_{s=1}^t g_s \\cdot w + \\frac{1}{2}\\sum_{s=1}^t \\sigma_s \\|w - w_s\\|^2 + \\lambda_1 \\|w\\|_1 \\right)$$
- L1 正则产生稀疏解，适合十亿级特征的在线学习

### 4. DeepFM 预测公式
$$\\hat{y} = \\sigma(y_{FM} + y_{DNN})$$
- FM 层捕获二阶特征交叉，DNN 层捕获高阶非线性交互

### 5. Focal Loss
$$FL(p_t) = -\\alpha_t (1 - p_t)^\\gamma \\log(p_t)$$
- γ>0 时对易分样本降权，聚焦难分样本，缓解 CVR 正负样本不均衡""",
        'extra_qa': """
### Q7: DeepFM 和 Wide&Deep 的核心区别？
**30秒答案**：Wide&Deep 的 Wide 部分需要手工设计交叉特征，DeepFM 用 FM 层自动做二阶交叉，省去特征工程。两者的 Deep 部分相同。DeepFM 还做了 embedding 共享（FM 和 DNN 共用 embedding）。

### Q8: 如何检测模型是否需要重新校准？
**30秒答案**：监控 ECE（Expected Calibration Error），将预测概率分桶后计算每桶预测均值与实际正样本率的差异加权平均。ECE > 5% 通常需要重新校准。也可以看 calibration plot（理想是 y=x 对角线）。

### Q9: 多目标 CTR+CVR 联合建模的其他方案？
**30秒答案**：除 ESMM 外：①MMoE（Multi-gate Mixture-of-Experts）：多个 Expert 网络 + 每个任务独立 Gate；②PLE（Progressive Layered Extraction）：任务独有 Expert + 共享 Expert，逐层提取；③DBMTL（Deep Bayesian MTL）：贝叶斯方法建模任务间不确定性。

### Q10: 增量训练 vs 全量重训如何选择？
**30秒答案**：增量训练（小时级 FTRL）处理分布漂移快但容易遗忘，全量重训（天级）保持全局最优但延迟大。工业实践：天级全量重训作为 base model，小时级增量 fine-tune，两者互补。关键是监控 AUC 和校准度，劣化时触发全量重训。"""
    },
    '广告系统RTB架构全景.md': {
        'formulas': """## 📐 核心公式与原理

### 1. eCPM 排序公式
$$eCPM = pCTR \\times pCVR \\times bid_{CPA}$$
- 广告系统按 eCPM 排序决定展示优先级，平衡广告主价值和平台收入

### 2. GSP（广义第二价格）扣费
$$cost_i = \\frac{eCPM_{i+1}}{pCTR_i} + 0.01$$
- 获胜广告主按下一位的 eCPM 扣费，而非自身出价

### 3. VCG 机制社会福利
$$p_i = \\sum_{j \\neq i} v_j(a^*_{-i}) - \\sum_{j \\neq i} v_j(a^*)$$
- 每个广告主付的是「因为你参与导致其他人的福利损失」

### 4. Reserve Price
$$bid_{effective} = \\max(bid, r)$$
- 最低出价保护，防止竞争不充分时广告贱卖""",
        'extra_qa': """
### Q7: DSP 和 SSP 分别是什么？
**30秒答案**：DSP（Demand Side Platform）代表广告主，负责出价策略和预算管理；SSP（Supply Side Platform）代表媒体/出版商，负责广告位管理和收益最大化。两者通过 Ad Exchange 连接，在 100ms 内完成竞价。

### Q8: RTB 100ms 延迟预算如何分配？
**30秒答案**：典型分配：网络传输 20ms，Ad Exchange 处理 10ms，DSP 决策（特征提取+模型推理+出价计算）50ms，返回 20ms。模型推理是瓶颈，用模型蒸馏/量化降低延迟。

### Q9: 频次控制（Frequency Capping）的工程实现？
**30秒答案**：在 Redis/KV 存储中维护 user_id → {ad_id: count, last_time} 映射，每次展示前查询。挑战：分布式一致性（多个 DSP 实例并发更新）、TTL 管理（滑动窗口 vs 固定窗口）。

### Q10: 如何防止 Click Fraud（点击欺诈）？
**30秒答案**：①行为特征检测：点击间隔过短、IP 集中、设备指纹异常；②流量质量模型：训练 fraud detector 识别异常流量；③归因窗口：限制点击到转化的最长归因时间；④IP/设备黑名单。"""
    },
    '广告出价体系全景.md': {
        'formulas': """## 📐 核心公式与原理

### 1. CPC 出价公式
$$bid_{CPC} = target\\_CPA \\times pCVR$$
- 从目标 CPA 反推每次点击出价

### 2. oCPX 最优出价（PID 控制）
$$bid_t = bid_{base} \\cdot (1 + K_p \\cdot e_t + K_i \\cdot \\sum e + K_d \\cdot \\Delta e_t)$$
- PID 控制器根据消耗偏差动态调整出价，e_t = 目标消耗 - 实际消耗

### 3. 约束优化出价（Lagrangian）
$$\\max \\sum_i v_i \\cdot x_i \\quad s.t. \\sum_i c_i \\cdot x_i \\leq B$$
$$\\Rightarrow bid_i = v_i - \\lambda \\cdot c_i$$
- λ 是预算约束的拉格朗日乘子，反映预算紧张程度

### 4. Myerson 最优拍卖
$$\\phi_i(v_i) = v_i - \\frac{1 - F(v_i)}{f(v_i)}$$
- 虚拟价值函数，最优拍卖按虚拟价值分配""",
        'extra_qa': """
### Q7: PID 出价控制和 RL 出价的优劣对比？
**30秒答案**：PID 简单可解释、收敛快、无需训练，但只能做反馈控制；RL 能学习全局最优策略、处理复杂约束，但训练不稳定、冷启动慢。工业实践：PID 做 baseline + RL 做增量优化。

### Q8: First Price Auction vs Second Price Auction？
**30秒答案**：First Price（一价）：赢家按自身出价付费，需要 bid shading 避免多付；Second Price（二价）：赢家按第二高价付费，理论上鼓励真实出价。趋势：行业从二价转向一价（Google 2019），因为二价在 header bidding 场景下不再激励相容。

### Q9: 预算耗尽太快怎么办？
**30秒答案**：Budget Pacing 策略：①Throttling：随机丢弃一定比例竞价请求；②Bid Modification：按消耗进度调低出价乘数；③RL Pacing：强化学习优化每个时段的出价策略。核心指标：匀速消耗（平滑 delivery curve）。

### Q10: 多目标出价（ROI + 量）如何平衡？
**30秒答案**：Pareto 方法：构建 ROI 和 Volume 的 Pareto 前沿，让广告主在曲线上选点；约束方法：在 ROI≥目标 的约束下最大化 Volume；分层方法：先按 ROI 筛选，再按 Volume 排序。"""
    },
    '广告系统冷启动.md': {
        'formulas': """## 📐 核心公式与原理

### 1. Thompson Sampling 探索
$$\\theta_i \\sim Beta(\\alpha_i, \\beta_i), \\quad \\text{选择 } \\arg\\max_i \\theta_i$$
- 每次从 Beta 后验分布采样，自动平衡探索与利用

### 2. UCB（Upper Confidence Bound）
$$UCB_i = \\hat{\\mu}_i + c\\sqrt{\\frac{\\ln N}{n_i}}$$
- 置信上界策略，样本少的广告获得更多探索机会

### 3. IDProxy 冷启动
$$\\text{embedding}_{new} = f_{LLM}(\\text{title}, \\text{image}, \\text{category})$$
- 用多模态 LLM 从广告素材直接生成 embedding，跳过需要行为数据的 ID embedding""",
        'extra_qa': """
### Q7: Explore-Exploit 在广告冷启动中怎么做？
**30秒答案**：给新广告额外曝光预算（exploration budget），用 Thompson Sampling 或 ε-greedy 分配流量。核心：exploration 的 cost 要控制在广告主可接受范围内，通常限制总曝光的 5-10%。

### Q8: 新广告主的初始出价怎么设？
**30秒答案**：①行业均值法：按行业+品类的历史平均 CPC/CPA 推荐；②相似广告主法：找特征相似的成熟广告主作为参考；③阶梯探索法：从低价开始逐步提高，观察转化率。

### Q9: 冷启动阶段的模型预估偏差怎么处理？
**30秒答案**：①Bayesian Shrinkage：将新广告的预估向全局均值收缩；②Prior Injection：用先验知识（品类均值）作为模型初始值；③Multi-level Model：分层模型，新广告使用上层（品类级）参数，数据积累后切换到下层（广告级）。

### Q10: 素材冷启动 vs 用户冷启动 vs 广告主冷启动的区别？
**30秒答案**：素材冷启动（新创意）：依赖多模态特征，LLM 理解素材内容生成初始 embedding；用户冷启动：依赖实时行为序列，3-5 次交互后快速学习偏好；广告主冷启动：依赖行业先验和目标设置，需要更长的学习周期（通常 1-2 周）。"""
    },
    '广告系统多目标优化.md': {
        'formulas': """## 📐 核心公式与原理

### 1. eCPM 多目标排序
$$eCPM = pCTR \\times (w_1 \\cdot pCVR \\times CPA + w_2 \\cdot quality\\_score)$$
- 多目标加权排序，平衡收入和用户体验

### 2. Pareto 最优
$$\\text{解 } x^* \\text{ 是 Pareto 最优} \\iff \\nexists x: f_i(x) \\leq f_i(x^*) \\forall i, \\exists j: f_j(x) < f_j(x^*)$$
- 不存在另一个解在所有目标上都不差且至少一个更好

### 3. MMoE 门控
$$y_k = h_k\\left(\\sum_{i=1}^n g_k^i(x) \\cdot f_i(x)\\right)$$
- 每个任务有独立门控网络 gk 选择 Expert 的加权组合

### 4. Scalarization 方法
$$\\min_\\theta \\sum_k \\lambda_k \\cdot L_k(\\theta)$$
- 将多目标转为单目标，λk 控制任务权重，但难以覆盖非凸 Pareto 前沿""",
        'extra_qa': """
### Q7: MMoE 和 PLE 的核心区别？
**30秒答案**：MMoE 所有 Expert 共享，每个任务用 Gate 选择；PLE 增加了任务独有 Expert（Task-specific Expert），每层有共享 Expert + 独有 Expert，逐层渐进式提取（Progressive Layered Extraction）。PLE 解决了 MMoE 中任务冲突导致的 negative transfer 问题。

### Q8: 多目标优化中，如何动态调权？
**30秒答案**：①Uncertainty Weighting：按任务损失的不确定性自动调权；②GradNorm：根据梯度幅度归一化调权；③Nash Bargaining：将多任务看作博弈，求纳什均衡解；④Online A/B：固定几组权重，在线测试选最优。

### Q9: 广告多目标优化中的典型冲突有哪些？
**30秒答案**：①收入 vs 用户体验（高出价广告可能低质量）；②CTR vs CVR（点击诱饵高 CTR 但低 CVR）；③短期收入 vs 长期留存（过度广告导致用户流失）；④填充率 vs 质量（放松准入提高填充但降低整体质量）。

### Q10: 实际工程中多目标排序分数怎么融合？
**30秒答案**：①线性加权（简单但需要调权）；②乘法融合 eCPM = pCTR × bid × quality_boost（广告常用）；③级联排序（先按质量筛选，再按收入排序）；④Constrained Optimization（在约束下优化主目标，如 ROI≥阈值下最大化 GMV）。"""
    },
    '广告效果归因.md': {
        'formulas': """## 📐 核心公式与原理

### 1. Last-Click 归因
$$credit_i = \\begin{cases} 1 & \\text{if } i = \\text{last click} \\\\ 0 & \\text{otherwise} \\end{cases}$$
- 最简单但忽略上游触点贡献

### 2. Shapley Value 归因
$$\\phi_i = \\sum_{S \\subseteq N \\setminus \\{i\\}} \\frac{|S|!(|N|-|S|-1)!}{|N|!} [v(S \\cup \\{i\\}) - v(S)]$$
- 公平分配每个触点的边际贡献，但计算复杂度 O(2^n)

### 3. Uplift（增量因果效应）
$$\\tau(x) = E[Y(1) | X=x] - E[Y(0) | X=x]$$
- 广告真实效果 = 看到广告后的转化率 - 没看到广告的转化率""",
        'extra_qa': """
### Q7: 为什么 Last-Click 归因有问题？
**30秒答案**：用户可能看了 5 个广告才转化，Last-Click 只归功于最后一次点击，忽略了品牌展示广告的贡献。结果：展示类广告（如视频前贴片）被低估，搜索广告被高估。

### Q8: A/B 测试和增量归因的关系？
**30秒答案**：A/B 测试是增量归因的金标准。将用户随机分为实验组（看到广告）和对照组（不看广告），转化率之差就是广告的增量效果（Uplift）。但成本高，不能对每个广告都做 A/B。

### Q9: Multi-Touch Attribution（MTA）的主流方法？
**30秒答案**：①规则型（线性、时间衰减、U 型）：简单但不灵活；②数据驱动型（Shapley、Markov Chain）：更公平但计算复杂；③机器学习型（DNN 序列模型）：拟合能力强但解释性差。趋势：Shapley Value 成为业界默认。

### Q10: 跨设备归因怎么做？
**30秒答案**：①确定性匹配：用登录 ID 关联手机和 PC 设备；②概率性匹配：用 IP、浏览器指纹、行为模式推断同一用户；③设备图谱（Device Graph）：构建设备关联图，传播归因。挑战：隐私法规限制（GDPR、ATT）。"""
    },
    '广告创意优化.md': {
        'formulas': """## 📐 核心公式与原理

### 1. Multi-Armed Bandit 选素材
$$\\text{选择素材 } a^* = \\arg\\max_a \\hat{\\mu}_a + c\\sqrt{\\frac{\\ln t}{n_a}}$$
- UCB 策略在素材选择中平衡探索与利用

### 2. CTR 预估与素材质量
$$pCTR = f(user, context, creative)$$
- 素材特征（图片质量、文案情感、标题长度）直接影响 CTR 预估

### 3. AIGC 素材生成（Diffusion）
$$x_{t-1} = \\frac{1}{\\sqrt{\\alpha_t}}\\left(x_t - \\frac{1-\\alpha_t}{\\sqrt{1-\\bar{\\alpha}_t}} \\epsilon_\\theta(x_t, t)\\right) + \\sigma_t z$$
- 扩散模型逐步去噪生成广告图片""",
        'extra_qa': """
### Q7: LLM 在广告创意优化中的应用？
**30秒答案**：①文案生成：给定商品信息，LLM 生成多版本标题和描述；②A/B 文案：自动生成对比文案候选；③个性化文案：根据用户画像调整语气和卖点。关键：生成后仍需人工审核 + 合规检查。

### Q8: 动态创意优化（DCO）是什么？
**30秒答案**：Dynamic Creative Optimization：实时组合素材元素（标题×图片×CTA×背景色），根据用户特征选择最佳组合。例如 4 标题×3 图片×2 CTA = 24 种组合，用 Bandit 算法在线探索最优。

### Q9: 素材疲劳（Creative Fatigue）怎么检测和处理？
**30秒答案**：检测：监控同一素材的 CTR 随时间的衰减曲线，当 CTR 下降超过阈值（如 20%）时触发。处理：①自动换素材（从素材库轮换）；②降低展示频次；③触发新素材生成任务。

### Q10: 视频广告的自动裁剪怎么做？
**30秒答案**：①关键帧检测：用 CV 模型提取视频关键帧和场景切换点；②注意力热图：预测用户关注区域，确保裁剪后保留核心内容；③多比例适配：自动生成 16:9、9:16、1:1 等多种比例版本。"""
    },
    '广告预算Pacing算法全景.md': {
        'formulas': """## 📐 核心公式与原理

### 1. 匀速消耗公式
$$\\text{目标消耗率} = \\frac{\\text{总预算}}{\\text{总时间}} \\quad \\text{偏差} = \\frac{\\text{实际消耗}}{\\text{目标消耗}} - 1$$
- 理想状态是偏差始终为 0，即匀速消耗

### 2. PID Pacing Controller
$$u_t = K_p e_t + K_i \\sum_{s=0}^t e_s + K_d (e_t - e_{t-1})$$
- P 控制当前偏差，I 消除累积误差，D 预测趋势

### 3. Throttling 概率
$$p_{bid} = \\min\\left(1, \\frac{B_{remaining}}{\\hat{C}_{remaining}}\\right)$$
- 剩余预算/预估剩余花费，决定参与竞价的概率""",
        'extra_qa': """
### Q7: Pacing 为什么不能简单用「前半天花一半预算」？
**30秒答案**：流量分布不均匀（午高峰流量是凌晨的 10 倍），按时间均匀分配会在高峰期欠投、低谷期超投。好的 Pacing 要根据流量预测动态调整出价/参与率。

### Q8: Throttling vs Bid Modification 的区别？
**30秒答案**：Throttling 是「跳过部分竞价请求」（0/1 决策），Bid Modification 是「调整出价金额」（连续决策）。Throttling 更简单但粒度粗；Bid Modification 更精细但需要更复杂的控制逻辑。

### Q9: 预算快耗尽时应该怎么调整？
**30秒答案**：①急刹车：大幅降低出价乘数（如 0.3x）；②选择性参与：只竞标预估 ROI 最高的请求；③提前通知广告主续充。工程实现：设置 90%/95%/99% 预算消耗预警线，逐级收紧。

### Q10: 多广告计划共享预算池怎么分配？
**30秒答案**：①竞争分配：所有计划共同竞标，谁的 eCPM 高谁获得预算；②配额分配：按优先级预分配固定份额；③动态分配：实时计算每个计划的边际 ROI，将预算分给边际 ROI 最高的计划。"""
    },
    '广告出价体系_从手动规则到RL自动出价.md': {
        'formulas': """## 📐 核心公式与原理

### 1. 手动出价
$$bid = target\\_CPA \\times pCVR$$
- 最基础的出价策略，广告主设定目标 CPA

### 2. RL 自动出价（MDP）
$$\\pi^*(s) = \\arg\\max_a Q(s, a) = \\arg\\max_a \\left[ r(s,a) + \\gamma \\sum_{s'} P(s'|s,a) V(s') \\right]$$
- 状态 s = (剩余预算, 当前时段, 竞争情况)，动作 a = 出价乘数

### 3. 约束 RL 出价（Lagrangian Relaxation）
$$L(\\theta, \\lambda) = J(\\theta) - \\lambda (C(\\theta) - B)$$
- 将预算约束松弛为正则项，λ 自动调节预算约束强度""",
        'extra_qa': """
### Q7: RL 出价冷启动怎么解决？
**30秒答案**：①离线预训练：用历史竞价日志做 Offline RL（如 BCQ、CQL）；②规则 warmup：前期用 PID/规则出价，收集数据后切换 RL；③仿真环境：搭建竞价模拟器，RL agent 先在模拟器中训练。

### Q8: 实际工程中 RL 出价的主要挑战？
**30秒答案**：①延迟奖励（转化可能 7 天后才发生）；②环境非平稳（竞争对手也在调策略）；③安全约束（不能让 RL 疯狂出高价）；④可解释性差（出了问题难排查）。

### Q9: HALO 算法的核心思想？
**30秒答案**：Hindsight-Augmented Learning for Online Auto-bidding。用「后见之明」数据增强：在训练时将实际发生的竞价结果作为额外信息，帮助 agent 学习更好的出价策略。核心：将 offline 数据中的 hindsight 信息注入 online 学习过程。

### Q10: 多平台出价协调怎么做？
**30秒答案**：广告主同时在 Google、Facebook、TikTok 投放时，需要跨平台预算分配。方法：①集中式：一个中心 agent 统一决策所有平台出价；②分布式：每个平台独立 agent，通过预算约束协调；③层次化：上层分配平台预算，下层各平台独立优化。"""
    },
    '广告系统偏差治理三部曲.md': {
        'formulas': """## 📐 核心公式与原理

### 1. 位置偏差消除（IPW）
$$\\hat{R} = \\frac{1}{n} \\sum_{i=1}^n \\frac{r_i}{P(O=1|pos_i)}$$
- 逆倾向加权，按位置曝光概率的倒数加权

### 2. PAL（Position-Aware Learning）
$$P(click) = P(examine|position) \\times P(relevant|query, doc)$$
- 将点击概率分解为「位置导致的注意力」×「内容相关性」

### 3. 样本选择偏差校正
$$P(Y=1|X) = P(Y=1|X, S=1) \\times \\frac{P(S=1)}{P(S=1|X)}$$
- S 是选择变量（如是否点击），用 Heckman 方法校正选择偏差""",
        'extra_qa': """
### Q7: 位置偏差对广告排序的影响有多大？
**30秒答案**：研究表明第 1 位的 CTR 是第 5 位的 3-5 倍，但这并非全因内容质量。如果不消除位置偏差，模型会「强者恒强」——高位广告获得更多正样本，预估 CTR 更高，继续排在高位。

### Q8: Counterfactual Learning 在广告中怎么用？
**30秒答案**：核心思想：用反事实推理回答「如果换一个排序会怎样」。训练时用 IPS（Inverse Propensity Score）加权，将观察到的偏差数据纠正为无偏估计。关键是准确估计 propensity score。

### Q9: 全空间建模和去偏的关系？
**30秒答案**：全空间建模（如 ESMM）本质上是一种「隐式去偏」：CVR 模型在所有曝光样本上训练（而非只在点击样本），避免了样本选择偏差。但它无法处理位置偏差，需要和 PAL 等方法结合。

### Q10: 在线去偏 vs 离线去偏？
**30秒答案**：离线去偏（训练时纠正）：IPS 加权、Heckman 校正、双塔解耦位置因子。在线去偏（推理时消除）：推理时将位置设为固定值（如中位位置），消除位置影响。实践中两者结合效果最好。"""
    },
    '广告系统综合总结.md': {
        'formulas': """## 📐 核心公式与原理

### 1. 广告系统价值公式
$$value = pCTR \\times pCVR \\times bid - cost$$
- 广告系统的核心目标：最大化平台价值（广告收入 + 用户体验）

### 2. 注意力机制（DIN）
$$\\text{attention}(q, k, v) = \\sum_i \\frac{\\exp(f(q, k_i))}{\\sum_j \\exp(f(q, k_j))} \\cdot v_i$$
- DIN 用 target item 作为 query，对历史行为序列做 attention

### 3. 广告质量得分
$$quality = \\alpha \\cdot pCTR + \\beta \\cdot relevance + \\gamma \\cdot landing\\_page\\_score$$
- 综合质量得分影响排序和计费""",
        'extra_qa': """
### Q7: 广告系统和推荐系统的核心区别？
**30秒答案**：①商业模式：广告按效果计费（CPC/CPA），推荐按展示；②校准要求：广告需绝对概率准确（影响出价），推荐只需排序正确；③更新频率：广告需要更高频更新（小时级）；④约束：广告有预算/ROI 约束，推荐通常没有。

### Q8: 广告系统全链路延迟优化？
**30秒答案**：召回层 <10ms（倒排索引）→粗排 <20ms（轻量模型）→精排 <50ms（复杂模型）→机制层 <10ms（竞价扣费）。优化手段：模型蒸馏、特征缓存、异步预计算、分层架构。

### Q9: 广告主和平台的利益冲突如何平衡？
**30秒答案**：广告主要低成本高转化，平台要高收入。平衡方法：①二价机制（鼓励真实出价）；②质量得分（提高优质广告竞争力）；③ROI 保障（承诺最低 ROI，达不到退款）；④长期优化（留住广告主比短期收入更重要）。

### Q10: 未来广告系统的技术趋势？
**30秒答案**：①LLM 原生广告（在 AI 对话中自然植入）；②隐私计算（联邦学习+差分隐私替代三方 Cookie）；③端到端生成（从选品到出价到创意全自动）；④因果推断归因（替代相关性归因）。"""
    },
    '广告排序系统演进路线图.md': {
        'formulas': """## 📐 核心公式与原理

### 1. LR + 手工特征
$$P(y=1|x) = \\sigma(w^T x + b)$$
- 特征工程为王，十亿级特征维度

### 2. Wide & Deep
$$P(y=1|x) = \\sigma(w_{wide}^T [x, \\phi(x)] + a_{deep}^T h_L)$$
- Wide 记忆（手工交叉特征），Deep 泛化（DNN 学习高阶交互）

### 3. DIN 注意力
$$V_u = \\sum_{i=1}^N a(e_i, e_a) \\cdot e_i, \\quad a(e_i, e_a) = \\text{softmax}(\\text{MLP}(e_i, e_a))$$
- 用候选广告激活相关的历史行为，而非均匀聚合""",
        'extra_qa': """
### Q7: LR 到 DNN 的关键转折点是什么？
**30秒答案**：2016 年 Google 的 Wide&Deep 论文。之前 LR + 手工特征是主流（简单可解释），之后 DNN 自动学习特征交叉成为趋势。关键驱动力：GPU 算力提升 + 数据量增长使 DNN 效果超过 LR。

### Q8: Transformer 在广告排序中怎么用？
**30秒答案**：①DIEN/BST：对用户行为序列做 self-attention 建模；②CADET：用 Decoder-Only Transformer 做 CTR 预估（类 GPT 架构）；③Feature Interaction：用 multi-head attention 替代传统特征交叉。挑战：推理延迟，需要蒸馏或剪枝。

### Q9: 模型越大效果越好吗？
**30秒答案**：有 Scaling Law 但有天花板。CTR 模型的 AUC 提升随模型增大而递减，且大模型推理延迟高（广告要求 <50ms）。实践：大模型离线训练 → 蒸馏到小模型在线服务。EST 论文系统研究了 CTR 模型的 Scaling Law。

### Q10: 特征交叉的演进路线？
**30秒答案**：手工交叉（LR）→ FM/FFM（自动二阶）→ Deep Cross（自动高阶）→ AutoInt（self-attention 交叉）→ CAN（共现感知交叉）→ Transformer（全局注意力交叉）。趋势：从手工到自动，从低阶到高阶。"""
    },
    'AutoBidding技术演进_从规则到RL.md': {
        'formulas': """## 📐 核心公式与原理

### 1. 规则出价
$$bid = base\\_bid \\times \\text{time\\_factor} \\times \\text{audience\\_factor}$$
- 静态规则 + 人工调系数

### 2. PID 控制出价
$$\\Delta bid = K_p(e) + K_i\\int e\\,dt + K_d\\frac{de}{dt}$$
- 根据 KPI 偏差（如实际 CPA vs 目标 CPA）自动调整

### 3. RL 自动出价（DDPG）
$$Q(s,a) = r + \\gamma Q(s', \\mu(s')), \\quad \\mu(s) = \\arg\\max_a Q(s,a)$$
- Actor-Critic 框架，Actor 输出连续出价动作""",
        'extra_qa': """
### Q7: 从规则到 RL 的过渡为什么不能一步到位？
**30秒答案**：RL 需要大量数据和稳定环境。冷启动期没有数据训练 RL，且 RL 初期可能产生极端出价。渐进路线：规则→PID→约束优化→Offline RL→Online RL。每一步都在上一步的基础上改进。

### Q8: Offline RL 在出价中的应用？
**30秒答案**：用历史竞价日志训练 RL agent，不需要与真实环境交互。方法：BCQ（批约束 Q-learning）、CQL（保守 Q-learning）防止 Q 值高估。优势：安全、低成本。劣势：受限于历史数据分布。

### Q9: 自动出价系统的线上安全机制？
**30秒答案**：①出价上下限：限制单次最大/最小出价；②消耗速度监控：实时检测异常消耗；③回滚机制：效果劣化时自动切回规则出价；④A/B 测试：新策略灰度上线，逐步放量。

### Q10: 竞争对手也用 RL 出价，会出现什么问题？
**30秒答案**：多 Agent 博弈导致纳什均衡可能不稳定：互相抬价（出价螺旋）或互相降价（价格战）。解决方案：①Mean Field Game：假设对手策略为「平均场」近似；②Robust RL：对抗不确定性的鲁棒策略。"""
    },
    '广告系统混排演进路线.md': {
        'formulas': """## 📐 核心公式与原理

### 1. 硬规则混排
$$\\text{slot}[i] = \\begin{cases} \\text{ad} & \\text{if } i \\mod k = 0 \\\\ \\text{organic} & \\text{otherwise} \\end{cases}$$
- 固定广告位插入规则

### 2. DPP 多样性
$$P(Y=S) \\propto \\det(L_S)$$
- 行列式点过程，L 矩阵同时编码质量和多样性

### 3. RL 混排（MDP）
$$r_t = \\alpha \\cdot revenue_t + \\beta \\cdot engagement_t + \\gamma \\cdot diversity_t$$
- 多目标奖励函数，RL 学习广告和内容的最优混排策略""",
        'extra_qa': """
### Q7: 混排对用户体验的影响怎么衡量？
**30秒答案**：①整体时长/刷新次数（宏观）；②广告穿插后的 organic 内容 CTR（微观，看广告是否「吓走」用户）；③负反馈率（「不喜欢」按钮点击率）。A/B 测试对比不同混排策略的这些指标。

### Q8: 抖音/TikTok 的混排是怎么做的？
**30秒答案**：基于 RL 的实时混排：将 feed 流看作序列决策，每个位置决定放广告还是内容。状态包含用户近期浏览序列、疲劳度、广告库存。奖励同时优化广告收入和用户留存。核心论文：抖音 RL Mixing。

### Q9: DPP 在混排中怎么用？
**30秒答案**：DPP（Deterministic Point Process）通过行列式衡量子集的质量和多样性。在混排中：L_ij = quality_i × similarity(i,j) × quality_j。选择使行列式最大的子集，自动平衡「选高质量」和「选多样化」。

### Q10: 混排中的广告密度如何控制？
**30秒答案**：①全局限制：feed 中广告占比不超过 X%（如 15%）；②局部限制：连续 N 个位置最多 M 个广告；③动态调节：根据用户耐受度调整（高频用户可多放，新用户少放）；④分时段：流量低谷时减少广告密度。"""
    },
    '广告系统LLM集成框架.md': {
        'formulas': """## 📐 核心公式与原理

### 1. LLM Embedding for CTR
$$e_{item} = \\text{LLM}_{encode}(\\text{title} \\oplus \\text{desc} \\oplus \\text{attr})$$
- 用 LLM 编码广告文本特征，替代传统 ID embedding

### 2. LLM-Auction 定价
$$price = \\text{LLM}(query, ad\\_candidates, auction\\_rules)$$
- LLM 理解广告内容后直接生成出价建议

### 3. Knowledge Distillation
$$L_{KD} = \\alpha \\cdot L_{task} + (1-\\alpha) \\cdot KL(p_{teacher} \\| p_{student})$$
- 大 LLM 作为 teacher，蒸馏知识到小模型在线服务""",
        'extra_qa': """
### Q7: LLM 在广告系统的哪些环节最有价值？
**30秒答案**：①创意生成（文案/图片）：直接产出，ROI 最高；②Query 理解：理解搜索意图匹配广告；③冷启动特征：用 LLM 理解新广告内容生成特征向量；④辅助标注：生成训练数据（伪标签）。排序推理因延迟限制不适合直接用 LLM。

### Q8: LLM 广告的隐私和合规挑战？
**30秒答案**：①用户数据不能直接传给 LLM（隐私泄露）；②LLM 生成的广告文案可能违规（夸大宣传）；③广告标识：LLM 生成的广告必须标注为广告；④审核：需要后置审核系统过滤不当内容。

### Q9: 如何评估 LLM 在广告中的增量效果？
**30秒答案**：①A/B 测试：对比有/无 LLM 特征的模型效果差异；②Ablation Study：逐步去掉 LLM 各模块看效果下降；③离线评估：用 LLM embedding 替换 ID embedding 后的 AUC 变化。关键指标：CTR、CVR、RPM（千次展示收入）。

### Q10: LLM-native 广告是什么概念？
**30秒答案**：在 AI 对话（如 ChatGPT、Perplexity）中原生嵌入的广告形式。不是传统的 banner/搜索广告，而是 LLM 在回答中自然推荐产品。挑战：①如何保持回答质量；②如何标识广告；③如何计费（没有传统的展示/点击概念）。"""
    },
    
    # ==================== REC-SYS ====================
    '推荐系统排序范式演进.md': {
        'formulas': """## 📐 核心公式与原理

### 1. LR 排序
$$P(y=1|x) = \\sigma(w^T x)$$
- 手工特征交叉 + 大规模稀疏特征

### 2. FM 二阶交叉
$$\\hat{y} = w_0 + \\sum_i w_i x_i + \\sum_i \\sum_{j>i} \\langle v_i, v_j \\rangle x_i x_j$$
- 向量内积建模特征交叉，O(kn) 复杂度

### 3. DIN 注意力
$$V_u = \\sum_i \\alpha(e_i, e_{target}) \\cdot e_i$$
- 用 target item 激活历史行为中的相关部分

### 4. DCN-V2 交叉层
$$x_{l+1} = x_0 \\odot (W_l x_l + b_l) + x_l$$
- 显式高阶特征交叉，每层交叉阶数 +1

### 5. Transformer 排序
$$\\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$
- Self-attention 建模特征间全局交互""",
        'extra_qa': """
### Q7: FM 和 FFM 的区别？
**30秒答案**：FM 每个特征一个 embedding 向量，FFM（Field-aware FM）每个特征对每个 field 有不同的 embedding 向量。FFM 更精细但参数量大幅增加（O(nfk) vs O(nk)）。

### Q8: 精排模型推理延迟怎么优化？
**30秒答案**：①模型蒸馏（大模型→小模型）；②特征缓存（预计算 user embedding）；③量化（FP32→INT8）；④剪枝（去掉不重要的层/神经元）；⑤异步特征获取（并行请求特征服务）。

### Q9: 排序模型的评估指标有哪些？
**30秒答案**：离线：AUC、GAUC（分组 AUC）、LogLoss、NDCG。在线：CTR、CVR、用户时长、DAU 留存。注意：离线指标和在线效果可能不一致（推荐系统特有的 offline-online gap）。

### Q10: 为什么推荐排序用 pointwise 而不是 listwise？
**30秒答案**：推荐系统候选集大（几百～几千），listwise loss 计算成本高且难以有效学习。pointwise（预测 CTR/CVR）简单高效，且可以直接用于出价（bid = pCTR × value）。搜索排序因候选少（几十个）更适合 pairwise/listwise。"""
    },
    '推荐系统召回范式演进.md': {
        'formulas': """## 📐 核心公式与原理

### 1. 协同过滤
$$\\hat{r}_{ui} = \\mu + b_u + b_i + q_i^T p_u$$
- 矩阵分解：用户向量和物品向量的内积预测评分

### 2. 双塔召回
$$score(u, i) = \\cos(E_u(u), E_i(i)) = \\frac{E_u(u)^T E_i(i)}{\\|E_u(u)\\| \\|E_i(i)\\|}$$
- 用户塔和物品塔独立编码，内积/余弦相似度匹配

### 3. ANN 检索
$$NN(q) = \\arg\\min_{i \\in \\mathcal{I}} \\|q - e_i\\|_2$$
- 近似最近邻搜索（HNSW/IVF），毫秒级从百万候选中召回 Top-K

### 4. 生成式召回（Semantic ID）
$$P(\\text{item}) = \\prod_{t=1}^T P(c_t | c_{<t}, \\text{user\\_history})$$
- 自回归生成物品 Semantic ID token 序列""",
        'extra_qa': """
### Q7: 双塔模型的主要缺点？
**30秒答案**：用户和物品独立编码，无法建模细粒度交互（如「用户对物品某个属性的偏好」）。改进方向：①多向量双塔（ColBERT-style）；②交叉注意力蒸馏（训练时有交叉，推理时双塔）。

### Q8: 多路召回如何融合？
**30秒答案**：①简单去重+截断：各路 Top-K 取并集；②分数归一化+加权：将各路分数映射到 [0,1] 后加权排序；③精排兜底：直接将所有召回候选送精排，让精排做最终排序。工程实践以方案③最常见。

### Q9: Semantic ID 是什么？怎么构建？
**30秒答案**：将物品 ID 替换为有语义的 token 序列。构建方法：①RQ-VAE：将物品 embedding 做残差量化得到 codebook index 序列；②层次聚类：按类目→子类→品牌逐层编码。优势：天然支持新物品（基于内容生成 ID），支持生成式召回。

### Q10: 实时召回 vs 离线召回的区别？
**30秒答案**：离线召回：用户 embedding 定期更新（小时级），物品库变化时重建索引。实时召回：每次请求实时计算 user embedding（结合最新行为），然后 ANN 检索。实时召回效果更好但成本高，通常只对最新行为做实时更新。"""
    },
    '用户行为序列建模.md': {
        'formulas': """## 📐 核心公式与原理

### 1. DIN（Deep Interest Network）
$$V_u = \\sum_i a(e_i, e_{ad}) \\cdot e_i, \\quad a = \\text{softmax}(\\text{MLP}(e_i, e_{ad}, e_i - e_{ad}, e_i \\odot e_{ad}))$$
- 用候选 item 激活相关历史行为

### 2. DIEN（Deep Interest Evolution Network）
$$h_t' = \\text{AUGRU}(h_{t-1}', e_t, a_t), \\quad a_t = \\frac{\\exp(h_t^T e_{ad})}{\\sum_j \\exp(h_j^T e_{ad})}$$
- 兴趣演化：用 attention score 控制 GRU 的更新门

### 3. SIM（Search-based Interest Model）
$$\\text{GSU}: S_{hard} = \\{i : \\text{category}(i) = \\text{category}(target)\\}$$
$$\\text{ESU}: V_u = \\text{Transformer}(\\text{Embed}(S_{hard}), e_{target})$$
- 两阶段：先检索相关行为子序列，再精细建模

### 4. CLS（Compressed Long Sequence）
$$\\text{memory}_t = \\text{compress}(h_1, ..., h_t) \\in \\mathbb{R}^{m \\times d}, \\quad m \\ll t$$
- 压缩长序列为固定大小记忆，支持万级行为序列""",
        'extra_qa': """
### Q7: DIN 和 DIEN 的核心区别？
**30秒答案**：DIN 只做静态 attention（加权求和），不考虑行为的时间顺序；DIEN 用 GRU 建模兴趣演化过程，再用 attention 提取与候选相关的兴趣。DIEN 能捕捉「兴趣从 A 迁移到 B」的动态变化。

### Q8: 超长序列（万级行为）怎么处理？
**30秒答案**：①SIM：先检索再精排（GSU+ESU 两阶段）；②ETA：用 SimHash 做 O(1) 查询近似 attention；③HiSAC：层次稀疏激活压缩；④Linformer：线性复杂度 attention。核心思想：不对全序列做 full attention，而是先筛选再精细建模。

### Q9: 行为序列中的 side information 怎么利用？
**30秒答案**：除了 item ID，还可以编码：①行为类型（点击/收藏/购买/分享）；②时间间隔（距当前多久）；③场景信息（搜索/推荐/直播间）。用 multi-field embedding 拼接后送入序列模型。

### Q10: LLM 做序列推荐的优势和局限？
**30秒答案**：优势：①天然理解文本（item title/desc）；②Few-shot 能力强；③支持开放域推荐。局限：①推理延迟大（不适合在线精排）；②ID 建模弱（文本描述不能完全替代 ID embedding）；③幻觉问题。实践：LLM 做辅助（特征增强/数据增广），不直接做在线排序。"""
    },
    '推荐系统冷启动.md': {
        'formulas': """## 📐 核心公式与原理

### 1. Bandit 探索
$$a^* = \\arg\\max_a \\left( \\hat{\\mu}_a + \\sqrt{\\frac{2 \\ln t}{n_a}} \\right)$$
- UCB：置信上界，样本少的 item 获得更大探索奖励

### 2. Meta-Learning 冷启动
$$\\theta_i^* = \\theta_0 - \\alpha \\nabla_{\\theta_0} L_{\\mathcal{D}_i^{train}}(\\theta_0)$$
- MAML：通过少量样本快速适应新用户/新物品

### 3. Content-based Embedding
$$e_{new} = g(\\text{title}, \\text{image}, \\text{category}, \\text{price})$$
- 用内容特征为新物品生成初始 embedding""",
        'extra_qa': """
### Q7: 新用户冷启动的实际流程？
**30秒答案**：①注册时收集基础信息（年龄/性别/城市）；②引导页选兴趣标签（3-5 个）；③前 10 次交互用 Bandit 探索；④3 天后积累足够数据，切换到个性化模型。核心：快速收集信号，尽早过渡到个性化。

### Q8: 新物品冷启动和新用户冷启动哪个更难？
**30秒答案**：新物品更难。新用户至少能用人口统计特征做基本推荐；新物品需要从零开始获得行为反馈。特别是长尾物品（如新书、新歌），可能很长时间没有足够交互数据。

### Q9: DropoutNet 的原理？
**30秒答案**：训练时随机 dropout 掉 ID embedding（用零向量替代），强制模型学习从 content 特征预测偏好。推理时新物品没有 ID embedding，模型自然退化为 content-based 模式。简单有效的冷启动方法。

### Q10: 冷启动效果怎么评估？
**30秒答案**：①按用户/物品的交互次数分桶，分别看各桶的 AUC/NDCG；②新物品首日曝光到首次点击的转化率；③冷启动期（前 N 次交互）vs 热启动期的效果差距。目标：缩短冷启动期，减小冷热效果差距。"""
    },
    'Embedding学习_推荐系统表示基石.md': {
        'formulas': """## 📐 核心公式与原理

### 1. Word2Vec (Skip-gram)
$$\\max \\sum_{(w,c) \\in D} \\log \\sigma(v_c^T v_w) + \\sum_{(w,c') \\in D'} \\log \\sigma(-v_{c'}^T v_w)$$
- 正样本拉近，负样本推远

### 2. Item2Vec
$$P(i_j | i_k) = \\frac{\\exp(v_{i_j}^T v_{i_k})}{\\sum_l \\exp(v_{i_l}^T v_{i_k})}$$
- 将行为序列类比句子，item 类比词

### 3. 对比学习
$$L = -\\log \\frac{\\exp(\\text{sim}(z_i, z_j) / \\tau)}{\\sum_{k=1}^{2N} \\mathbf{1}_{[k \\neq i]} \\exp(\\text{sim}(z_i, z_k) / \\tau)}$$
- InfoNCE loss，正样本对 (i,j) 拉近，其余推远

### 4. Graph Embedding (Node2Vec)
$$\\max \\sum_{u \\in V} \\sum_{v \\in N(u)} \\log P(v | u)$$
- 随机游走生成序列，再用 Skip-gram 学习节点 embedding""",
        'extra_qa': """
### Q7: ID embedding vs Content embedding 的取舍？
**30秒答案**：ID embedding：记忆能力强但无法泛化到新 item；Content embedding：泛化好但丢失协同过滤信号。最佳实践：两者拼接使用，冷启动时依赖 content，热启动后 ID 主导。

### Q8: Embedding 维度怎么选？
**30秒答案**：经验法则：item 数量 N 的 log2 或 4次方根。通常 32-256 维。太小欠拟合，太大过拟合+推理慢。可以用自动搜索（NAS）或逐步增大观察 AUC 变化确定最优维度。

### Q9: 负采样策略对 embedding 质量的影响？
**30秒答案**：①均匀采样：简单但低效（大部分负样本太「容易」区分）；②频率采样（popular bias）：按流行度采样，更有挑战性；③Hard Negative：从 ANN 结果中挑选，最有效但可能导致 false negative（实际是正样本）。

### Q10: 多模态 embedding 融合方法？
**30秒答案**：①Early Fusion：特征拼接后统一编码；②Late Fusion：各模态独立编码再加权融合；③Cross-Modal Attention：模态间交叉注意力。实践中 Early Fusion 最常用（简单有效），Cross-Modal 在特定场景有优势。"""
    },
    '推荐系统特征工程体系.md': {
        'formulas': """## 📐 核心公式与原理

### 1. 特征交叉
$$\\phi(x_i, x_j) = x_i \\otimes x_j \\quad \\text{(Cartesian Product)}$$
- 手工交叉特征在 LR 时代是核心竞争力

### 2. 目标编码 (Target Encoding)
$$TE(c) = \\lambda \\cdot \\bar{y}_c + (1-\\lambda) \\cdot \\bar{y}_{global}$$
- 用类别变量对应的目标均值编码，λ 根据样本量 shrink

### 3. 特征哈希
$$h(x) = \\text{hash}(\\text{field\\_name} + \\text{value}) \\mod D$$
- 将高基数特征映射到固定维度的 embedding table""",
        'extra_qa': """
### Q7: 实时特征 vs 离线特征的区别和应用？
**30秒答案**：实时特征（毫秒级更新）：用户最近 N 次点击、当前 session 行为、实时统计量。离线特征（小时/天级）：用户画像、物品统计特征、交叉统计。实时特征对 CTR 提升显著但工程复杂度高。

### Q8: Feature Store 的核心价值？
**30秒答案**：①特征复用（同一特征多个模型共用）；②一致性（训练和推理用同一特征逻辑）；③版本管理（特征变更可追踪）；④低延迟（特征预计算+缓存）。主流工具：Feast、Tecton、Hopsworks。

### Q9: 连续特征的处理方法？
**30秒答案**：①分桶（等频/等宽/自定义）→ 类别化处理；②归一化（MinMax/Z-score）；③Log 变换（长尾分布）；④直接输入 DNN（让模型自动学习非线性变换）。DNN 时代主流是直接输入 + batch normalization。

### Q10: 特征重要性怎么评估？
**30秒答案**：①Ablation：去掉某特征看 AUC 变化（最可靠但成本高）；②SHAP Value：基于 Shapley 的特征贡献分析；③Permutation Importance：打乱某特征值看效果下降；④模型内置：GBDT 的 feature importance、DNN 的 attention weight。"""
    },
    '推荐系统全链路架构概览.md': {
        'formulas': """## 📐 核心公式与原理

### 1. 漏斗结构
$$\\text{全量} \\xrightarrow{\\text{召回}} \\text{千级} \\xrightarrow{\\text{粗排}} \\text{百级} \\xrightarrow{\\text{精排}} \\text{十级} \\xrightarrow{\\text{重排}} \\text{展示}$$
- 逐层过滤，平衡效果和计算成本

### 2. 精排 eCTR
$$eCTR = \\sigma(f_{DNN}(\\text{user\\_feat}, \\text{item\\_feat}, \\text{context\\_feat}))$$
- 深度模型预估点击概率

### 3. 重排多样性（MMR）
$$\\text{MMR} = \\arg\\max_{d \\in R \\setminus S} [\\lambda \\cdot \\text{Rel}(d) - (1-\\lambda) \\cdot \\max_{d' \\in S} \\text{Sim}(d, d')]$$
- 贪心选择：兼顾相关性和多样性""",
        'extra_qa': """
### Q7: 粗排和精排的区别是什么？
**30秒答案**：粗排处理千级候选（~1000→~200），模型轻量（双塔/浅层 DNN），延迟 <10ms；精排处理百级候选（~200→~50），模型复杂（DIN/Transformer），延迟 <50ms。粗排追求速度+大致排序，精排追求精准排序。

### Q8: 推荐系统的实时性如何保证？
**30秒答案**：①用户特征：实时计算（Flink 流处理最新行为）；②物品特征：定期更新（小时级）；③模型：天级全量重训 + 实时增量更新；④索引：物品上架/下架实时更新 ANN 索引。

### Q9: 推荐系统和搜索系统的核心区别？
**30秒答案**：①触发方式：推荐是被动（猜你喜欢）、搜索是主动（用户输入 query）；②候选空间：推荐全量物品、搜索局限于 query 相关；③评估指标：推荐看时长/留存、搜索看相关性/满意度；④排序目标：推荐多目标（CTR+时长+多样性）、搜索以相关性为主。

### Q10: 推荐系统面试中「设计 XXX 推荐系统」怎么答？
**30秒答案**：按层回答：①明确场景和指标 → ②召回策略（协同过滤+双塔+热门+地理位置）→ ③排序模型（DeepFM/DIN + 多目标）→ ④重排策略（多样性+去重）→ ⑤在线实验方案（A/B 测试）→ ⑥工程架构（特征服务+模型服务+日志闭环）。"""
    },
    '图神经网络在推荐中的应用.md': {
        'formulas': """## 📐 核心公式与原理

### 1. GCN 消息传递
$$h_v^{(l+1)} = \\sigma\\left(\\sum_{u \\in \\mathcal{N}(v)} \\frac{1}{\\sqrt{|\\mathcal{N}(u)||\\mathcal{N}(v)|}} W^{(l)} h_u^{(l)}\\right)$$
- 邻居特征聚合 + 对称归一化

### 2. PinSage（GraphSAGE 变体）
$$h_v^{(l+1)} = \\sigma(W \\cdot \\text{CONCAT}(h_v^{(l)}, \\text{AGG}(\\{h_u : u \\in \\mathcal{N}(v)\\})))$$
- 采样邻居 + 聚合，工业级图推荐

### 3. LightGCN
$$e_u^{(l+1)} = \\sum_{i \\in \\mathcal{N}_u} \\frac{1}{\\sqrt{|\\mathcal{N}_u||\\mathcal{N}_i|}} e_i^{(l)}$$
- 去掉特征变换和非线性激活，只保留邻居聚合""",
        'extra_qa': """
### Q7: 为什么 LightGCN 去掉了变换矩阵和激活函数？
**30秒答案**：在推荐场景中，输入是 one-hot ID（不像 NLP 有丰富语义），特征变换和非线性激活反而引入噪声。LightGCN 实验证明只做邻居聚合效果更好，且训练更稳定。

### Q8: GNN 推荐的可扩展性问题怎么解决？
**30秒答案**：①邻居采样（PinSage：随机游走采 top-k）；②Mini-batch 训练（子图采样）；③分布式图计算（AliGraph、DGL）；④蒸馏：GNN 训练后蒸馏到 MLP 做在线推理。

### Q9: 异构图在推荐中怎么用？
**30秒答案**：将 user-item 二部图扩展为多类型节点（user、item、author、brand、tag）和多类型边（点击、购买、收藏、评论）。用 RGCN（Relational GCN）或 HAN（Heterogeneous Attention Network）处理不同关系。

### Q10: GNN 推荐 vs 协同过滤 vs 深度排序的关系？
**30秒答案**：GNN 推荐是协同过滤的图泛化（显式建模高阶邻居关系），通常用于召回层生成 embedding。深度排序（DIN/DCN）用于精排层。三者互补：GNN 做图召回 → 深度模型精排，效果优于单用任何一种。"""
    },
    '推荐广告AB测试与在线实验.md': {
        'formulas': """## 📐 核心公式与原理

### 1. 假设检验
$$z = \\frac{\\bar{X}_T - \\bar{X}_C}{\\sqrt{\\frac{s_T^2}{n_T} + \\frac{s_C^2}{n_C}}}$$
- 检验实验组和对照组指标差异是否显著

### 2. 最小样本量
$$n = \\frac{(z_{\\alpha/2} + z_\\beta)^2 \\cdot 2\\sigma^2}{\\delta^2}$$
- δ 是最小可检测效应，σ 是指标标准差

### 3. DID（双重差分）
$$\\hat{\\tau} = (\\bar{Y}_{T,after} - \\bar{Y}_{T,before}) - (\\bar{Y}_{C,after} - \\bar{Y}_{C,before})$$
- 消除时间趋势，估计真正的处理效应""",
        'extra_qa': """
### Q7: A/B 测试中的常见陷阱？
**30秒答案**：①窥探效应（Peeking）：提前看结果导致误判；②辛普森悖论：总体显著但分组不显著（或反之）；③网络效应：用户间互相影响（社交推荐）；④新奇效应：新功能短期 CTR 高是因为好奇，不是真的更好。

### Q8: 推荐系统的 A/B 测试和普通 A/B 测试有什么不同？
**30秒答案**：①指标复杂：不只看 CTR，还看时长、留存、多样性；②长期效应：需要看 7-14 天（避免新奇效应）；③多目标权衡：可能 CTR 涨但时长降，需要综合判断；④溢出效应：实验组看到的推荐变了，可能影响对照组的物品池。

### Q9: Interleaving 实验是什么？
**30秒答案**：将两个排序模型的结果交叉混合展示给同一用户（如 A1,B1,A2,B2...），通过用户对哪个模型结果的点击更多来判断优劣。优势：灵敏度高（同一用户对比），劣势：只适合排序类实验。

### Q10: 如何处理 A/B 测试中的多重检验问题？
**30秒答案**：同时看 N 个指标时，5% 的显著性水平下期望有 0.05N 个假阳性。修正方法：①Bonferroni：α/N（过于保守）；②Benjamini-Hochberg：控制 FDR（假发现率）；③预注册主要指标：只对 1-2 个核心指标做正式检验。"""
    },
    '推荐系统重排与多样性.md': {
        'formulas': """## 📐 核心公式与原理

### 1. MMR（Maximal Marginal Relevance）
$$\\text{MMR} = \\arg\\max_{d_i \\in R \\setminus S} \\left[ \\lambda \\cdot \\text{Rel}(d_i) - (1-\\lambda) \\cdot \\max_{d_j \\in S} \\text{Sim}(d_i, d_j) \\right]$$
- 贪心选择：λ 平衡相关性和多样性

### 2. DPP（Deterministic Point Process）
$$P(Y = S) \\propto \\det(L_S), \\quad L_{ij} = q_i \\phi_i^T \\phi_j q_j$$
- 行列式衡量子集的质量×多样性，相似项不共存

### 3. 滑动窗口去重
$$\\text{if } \\text{category}(item_i) \\in \\{\\text{category}(item_j) : j \\in [i-w, i-1]\\}: \\text{skip}$$
- 相邻 w 个位置内不出现相同类目""",
        'extra_qa': """
### Q7: 多样性提升和 CTR 下降的矛盾怎么解决？
**30秒答案**：短期可能 CTR 下降（用户错过最相关的 top item），但长期多样性提升用户满意度和留存。方法：①设 diversity 下限而非最大化；②在 A/B 中看 7 天留存而非 1 天 CTR；③用户级别调节（探索型用户多样性高，精准型用户少）。

### Q8: DPP 的计算复杂度问题怎么解决？
**30秒答案**：精确 DPP 是 O(N³)（行列式计算），实际用：①Fast Greedy：贪心近似 O(NK²)，K 是选择数量；②Sliding Window DPP：只在窗口内做 DPP；③近似 DPP：低秩近似降低矩阵维度。

### Q9: 上下文感知重排（Context-aware Reranking）是什么？
**30秒答案**：考虑 item 之间的相互影响：①PRM（Personalized Re-ranking Model）：将候选列表作为序列输入 Transformer；②SetRank：直接建模集合中 item 的交互；③LLM Reranker：用 LLM 理解列表整体质量。

### Q10: 重排层还能做哪些事情？
**30秒答案**：①多样性保障（类目/来源/作者去重）；②流量调控（冷启动 item 加曝光）；③业务规则（置顶/运营插入）；④广告混排（广告和自然结果的混合排序）；⑤实时调控（大促/热点事件临时策略）。"""
    },
    '推荐系统综合总结.md': {
        'formulas': """## 📐 核心公式与原理

### 1. 矩阵分解
$$\\min_{P,Q} \\sum_{(u,i) \\in \\Omega} (r_{ui} - p_u^T q_i)^2 + \\lambda(\\|P\\|^2 + \\|Q\\|^2)$$
- 推荐系统的经典基石

### 2. BPR Loss
$$L_{BPR} = -\\sum_{(u,i,j)} \\ln \\sigma(\\hat{r}_{ui} - \\hat{r}_{uj})$$
- 正样本得分 > 负样本得分的 pairwise 优化

### 3. 多目标融合
$$score = w_1 \\cdot pCTR + w_2 \\cdot pCVR + w_3 \\cdot duration + w_4 \\cdot diversity$$
- 加权融合多个目标，权重需要精心调优""",
        'extra_qa': """
### Q7: 推荐系统面试最常问的 Top 3 问题？
**30秒答案**：①「设计一个推荐系统」（全链路架构）；②「CTR 预估模型的演进」（LR→FM→DNN→DIN→Transformer）；③「冷启动怎么做」（新用户/新物品/探索利用）。这三个覆盖了 80% 的推荐面试。

### Q8: 推荐系统的 position bias 怎么处理？
**30秒答案**：①训练时加 position feature 但推理时设为定值；②IPW（Inverse Propensity Weighting）：按位置曝光概率加权；③PAL：将 P(click) 分解为 P(examine|pos) × P(relevant|item)，训练后只用 P(relevant)。

### Q9: 工业界推荐系统和学术研究的最大差距？
**30秒答案**：①规模：工业是亿级 user × 亿级 item，学术是百万级；②指标：工业看留存/GMV/时长等商业指标，学术看 AUC/NDCG；③延迟：工业要求 <100ms 端到端，学术不关心；④迭代：工业靠 A/B 测试，学术靠离线评估。

### Q10: 2024-2025 推荐系统的技术趋势？
**30秒答案**：①生成式推荐（Semantic ID + 自回归召回）；②LLM 增强（特征/数据增广/知识蒸馏）；③Scaling Law（Meta Wukong 系统研究推荐的 scaling）；④端到端生成（OneRec 统一召回和排序）；⑤多模态推荐（视频/图文内容理解）。"""
    },
    '推荐系统LLM集成框架.md': {
        'formulas': """## 📐 核心公式与原理

### 1. LLM Feature Enhancement
$$e_{item}^{enhanced} = \\text{Concat}(e_{id}, \\text{LLM}_{encode}(\\text{text}))$$
- 用 LLM 编码物品文本特征，拼接 ID embedding

### 2. Knowledge Distillation
$$L = L_{task} + \\alpha \\cdot KL(p_{LLM} \\| p_{rec})$$
- 大 LLM 的知识蒸馏到轻量推荐模型

### 3. Prompt-based Recommendation
$$\\hat{r} = \\text{LLM}(\\text{"User liked [A,B,C]. Will they like [D]?"})$$
- 将推荐任务转化为 LLM 自然语言推理""",
        'extra_qa': """
### Q7: LLM 做推荐的三种范式？
**30秒答案**：①Feature Enhancement：LLM 作为特征提取器，生成文本 embedding 输入推荐模型；②Knowledge Distillation：LLM 生成训练信号（伪标签/数据增广）；③Direct Recommendation：LLM 直接做推荐决策（prompt-based）。工业中①最成熟，③延迟太大。

### Q8: HLLM 是什么？
**30秒答案**：Hierarchical LLM for Sequential Recommendation。两层结构：底层 LLM 理解每个 item 的文本内容，顶层 LLM 建模用户行为序列。通过层次化结构降低计算成本（不用把所有 item 文本拼在一起），同时保留深度语义理解。

### Q9: LLM 推荐中的幻觉问题？
**30秒答案**：LLM 可能推荐不存在的物品、编造物品属性、或给出不合理的推荐理由。解决方案：①Constrained Generation（限制输出在合法物品集合内）；②RAG（检索真实物品信息后再生成）；③Post-verification（推荐后验证物品存在性）。

### Q10: Wukong 的 Scaling Law 发现了什么？
**30秒答案**：Meta 在推荐系统中系统研究了 Scaling Law：模型参数量 ↑、训练数据 ↑、特征维度 ↑ 都能持续提升效果，且三者之间有最优配比。关键发现：推荐模型的 scaling 行为和 LLM 类似，但最优的 compute allocation 不同（推荐中 embedding 维度更重要）。"""
    },
    'SemanticID与生成式检索.md': {
        'formulas': """## 📐 核心公式与原理

### 1. RQ-VAE 量化
$$c_1, c_2, ..., c_L = \\text{RQ-VAE}(e_{item})$$
- 残差量化：逐层量化残差，得到多级 codebook index

### 2. 自回归生成
$$P(item | user) = \\prod_{l=1}^L P(c_l | c_{<l}, \\text{user\\_history})$$
- 逐个生成 Semantic ID 的 token

### 3. Beam Search 召回
$$\\text{Top-K items} = \\text{BeamSearch}(P(c_1|user), P(c_2|c_1, user), ...)$$
- 用 beam search 在 codebook 树上搜索最优 item""",
        'extra_qa': """
### Q7: Semantic ID vs 传统 Item ID 的核心优势？
**30秒答案**：①新物品天然支持（基于内容生成 ID，无需行为数据）；②ID 有语义结构（层次化 codebook 反映物品相似性）；③支持生成式召回（自回归模型直接生成候选）。劣势：codebook 训练复杂，量化信息损失。

### Q8: Variable-Length Semantic ID 解决了什么问题？
**30秒答案**：固定长度 Semantic ID 对热门物品编码冗余，对长尾物品编码不足。Variable-Length 根据物品复杂度自适应长度：热门物品用短 ID（高效），长尾物品用长 ID（精确）。类似 Huffman 编码的思想。

### Q9: 生成式召回 vs 向量检索（ANN）的对比？
**30秒答案**：向量检索：O(1) 编码 + ANN 搜索，速度快但只能做近似最近邻；生成式召回：O(L) 自回归生成，可以捕获复杂的用户-物品交互模式，但推理慢。趋势：两者互补，向量检索做主力，生成式召回做补充路。

### Q10: Spotify 的 Semantic ID 部署经验？
**30秒答案**：①大规模（百万级 Podcast）；②多级量化（4-8 层 codebook）；③离线生成 + 在线检索混合；④和传统双塔召回并行部署（A/B 测试验证增量效果）。核心经验：codebook 的训练质量是关键，需要大量数据和精心调参。"""
    },
    '生成式推荐范式对比.md': {
        'formulas': """## 📐 核心公式与原理

### 1. 自回归推荐
$$P(item | history) = \\prod_t P(token_t | token_{<t}, history)$$
- GPT-style：逐 token 生成推荐结果

### 2. 扩散模型推荐
$$p_\\theta(x_0) = \\int p(x_T) \\prod_{t=1}^T p_\\theta(x_{t-1}|x_t) dx_{1:T}$$
- 从噪声逐步去噪生成推荐 embedding

### 3. 对比：生成质量
$$\\text{AR: } \\text{exposure bias} \\quad \\text{vs} \\quad \\text{Diffusion: } \\text{mode collapse}$$
- 自回归有暴露偏差问题，扩散模型有模式坍塌风险""",
        'extra_qa': """
### Q7: 自回归 vs 扩散模型在推荐中各自适合什么场景？
**30秒答案**：自回归适合序列推荐（下一个点击/购买预测），因为天然建模序列依赖；扩散模型适合集合推荐（一次生成多个推荐），因为可以并行生成且考虑整体多样性。

### Q8: DiffGRM 是怎么做的？
**30秒答案**：用扩散模型在 embedding 空间做推荐：①正向过程：将 item embedding 逐步加噪；②反向过程：从噪声去噪生成推荐 embedding；③条件生成：以用户历史为条件引导去噪方向。优势：生成多样性好。

### Q9: 生成式推荐的评估难题？
**30秒答案**：传统指标（AUC/NDCG）不完全适用：①生成的物品可能不在候选集中（如何评估？）；②多样性和准确性的平衡难量化；③在线 A/B 是金标准但成本高。新指标：生成质量（生成的 ID 是否有效）、覆盖率、意外发现率。

### Q10: 2025 年生成式推荐的最新进展？
**30秒答案**：①OneRec（字节）：统一召回和排序；②ETEGRec：端到端可学习的 Semantic ID；③COBRA：融合稀疏和稠密检索；④MTGRBoost：大规模生成式推荐加速。趋势：从概念验证走向工业部署，Scaling 是核心挑战。"""
    },
    'SemanticID从论文到Spotify部署.md': {
        'formulas': """## 📐 核心公式与原理

### 1. Hierarchical Clustering ID
$$\\text{SID}(item) = [\\text{cluster}_1, \\text{cluster}_2, ..., \\text{cluster}_L]$$
- 层次聚类树的路径作为 Semantic ID

### 2. Contrastive Learning for Codebook
$$L = -\\log \\frac{\\exp(\\text{sim}(z_i, c_{pos})/\\tau)}{\\sum_k \\exp(\\text{sim}(z_i, c_k)/\\tau)}$$
- 对比学习训练高质量 codebook""",
        'extra_qa': """
### Q7: Semantic ID 的 codebook 大小怎么选？
**30秒答案**：通常每层 256-1024 个 code，4-8 层。总组合空间 = codebook_size^layers（如 512^6 ≈ 17万亿），远大于实际物品数。关键是避免 codebook 坍塌（大部分 code 没被使用）。

### Q8: Spotify 的 ULM（统一语言模型）怎么用 Semantic ID？
**30秒答案**：将推荐、搜索、推理统一到一个 LLM 中。物品用 Semantic ID 表示，和自然语言 token 混合输入。模型可以：给定历史推荐下一首歌（推荐），给定 query 检索播客（搜索），给定上下文回答问题（推理）。

### Q9: Semantic ID 的离线构建流程？
**30秒答案**：①物品 embedding 训练（双塔/多模态）→②RQ-VAE 量化（或层次聚类）→③codebook 质量验证（覆盖率、语义一致性）→④写入在线索引。定期重建（周级/月级），增量更新（新物品用预训练的量化器编码）。

### Q10: 生成式检索 vs 传统 ANN 检索的工程对比？
**30秒答案**：ANN：索引构建 O(N log N)，查询 O(log N)，支持十亿级；生成式：无需预建索引（模型即索引），查询 O(L×beam)，但 beam search 并行度有限。目前 ANN 更成熟，生成式作为互补路使用。"""
    },
    '生成式推荐完整技术图谱.md': {
        'formulas': """## 📐 核心公式与原理

### 1. Next Token Prediction (推荐版)
$$P(\\text{next\\_item} | \\text{history}) = \\text{softmax}(W \\cdot h_{\\text{last}})$$
- Transformer decoder 预测下一个交互物品

### 2. 多任务生成式推荐
$$L = L_{rec} + \\alpha L_{search} + \\beta L_{reasoning}$$
- 统一模型同时优化推荐、搜索和推理任务""",
        'extra_qa': """
### Q7: 生成式推荐的「端到端」是什么意思？
**30秒答案**：传统推荐是分层管道（召回→粗排→精排→重排），端到端生成式推荐是一个模型直接从用户历史生成最终推荐列表，省去中间环节。代表：OneRec（字节）统一 retrieve-and-rank。

### Q8: 大规模生成式推荐的推理效率问题？
**30秒答案**：自回归生成每个 token 需要一次前向传播，K 个推荐需要 K×L 次。加速方法：①Speculative Decoding；②KV Cache；③并行 beam search；④Tree-based 搜索（先粗后细）。

### Q9: 生成式推荐能完全替代传统推荐吗？
**30秒答案**：短期不能。原因：①推理效率差距大（ms vs s）；②传统系统有丰富的工程优化积累；③生成式模型的可控性差（难以精确控制多样性/新鲜度等业务约束）。长期趋势：生成式和传统混合架构。

### Q10: 生成式推荐对 Embedding 表的影响？
**30秒答案**：传统推荐需要巨大的 Embedding Table（亿级物品 × 数百维）；生成式推荐用 Semantic ID 后，只需要小的 Codebook（几千个 code × 数百维），大幅减少参数量和存储。这是 Scaling 的关键优势。"""
    },
    '推荐系统ScalingLaw_Wukong.md': {
        'formulas': """## 📐 核心公式与原理

### 1. Scaling Law (推荐版)
$$L(N, D, E) = \\frac{A}{N^\\alpha} + \\frac{B}{D^\\beta} + \\frac{C}{E^\\gamma} + L_\\infty$$
- 损失随模型大小 N、数据量 D、Embedding 维度 E 的幂律下降

### 2. Compute-Optimal Allocation
$$N^* \\propto C^a, \\quad D^* \\propto C^b$$
- 给定计算预算 C，最优的模型大小和数据量分配""",
        'extra_qa': """
### Q7: Wukong 和 LLM 的 Scaling Law 有什么不同？
**30秒答案**：LLM 的 Scaling Law 主要看模型参数量和数据量；Wukong 发现推荐还有第三个关键维度——Embedding 维度。且推荐中 Embedding 维度的 scaling 效率最高（同样增加计算量，扩大 embedding 比扩大 MLP 效果好）。

### Q8: Scaling Law 对工业推荐系统的实际指导？
**30秒答案**：①模型增大有持续收益，不必担心「够大了」；②投资 Embedding 维度比 MLP 层数更划算；③数据量也要同步增加（否则大模型过拟合）；④可以预测给定计算预算下的最优配置。

### Q9: 推荐系统为什么之前没有 Scaling Law 研究？
**30秒答案**：①学术数据集太小（MovieLens 百万级）；②工业界数据敏感不公开；③推荐模型结构多样（不像 LLM 统一为 Transformer）；④评估指标不统一。Wukong 在 Meta 内部的超大规模数据上实验才得以验证。

### Q10: Scaling 推荐模型的工程挑战？
**30秒答案**：①Embedding Table 是瓶颈（十亿 item × 256 维 = TB 级内存）；②分布式训练通信开销大（数据并行+模型并行）；③推理延迟约束（不能无限增大模型）；④特征实时性（大模型需要更多特征，特征服务压力大）。"""
    },

    # ==================== SEARCH ====================
    '搜索排序专项笔记.md': {
        'formulas': """## 📐 核心公式与原理

### 1. BM25
$$\\text{BM25}(q, d) = \\sum_{t \\in q} \\text{IDF}(t) \\cdot \\frac{tf(t,d) \\cdot (k_1 + 1)}{tf(t,d) + k_1 \\cdot (1 - b + b \\cdot \\frac{|d|}{\\text{avgdl}})}$$
- 经典稀疏检索，tf-idf 的改进版本

### 2. Dense Retrieval
$$score(q, d) = E_q(q)^T E_d(d)$$
- 双塔编码器独立编码 query 和 doc，内积匹配

### 3. Cross-Encoder
$$score(q, d) = \\text{MLP}(\\text{BERT}([q; \\text{SEP}; d]))$$
- query 和 doc 拼接后联合编码，交互更充分但推理慢""",
        'extra_qa': """
### Q7: BM25 为什么在 2025 年还没被完全替代？
**30秒答案**：①精确匹配能力强（专有名词、ID 查询）；②零成本（不需要 GPU 训练/推理）；③可解释性好（知道哪个词匹配了）；④和稠密检索互补（两者覆盖不同 query 类型）。

### Q8: Dense Retrieval 的训练数据怎么构造？
**30秒答案**：①人工标注（expensive but high quality）；②点击日志（query-clicked_doc 作为正样本）；③LLM 生成（用 LLM 为 doc 生成相关 query）。负样本：随机负样本 + BM25 负样本（hard negative）+ in-batch negative。

### Q9: 搜索排序的 L1/L2/L3 分层？
**30秒答案**：L0（召回）：BM25+ANN 从全量检索 ~1000 个；L1（粗排）：轻量双塔 ~1000→200；L2（精排）：Cross-Encoder 或复杂模型 ~200→20；L3（重排）：业务规则+多样性 ~20→展示。

### Q10: 搜索排序和推荐排序的核心区别？
**30秒答案**：①有 query 信号（搜索的核心是 query-doc 匹配）；②相关性优先（推荐可以牺牲相关性换多样性，搜索不行）；③Pairwise/Listwise loss 更常用（搜索候选少适合 learning to rank）；④可解释性要求高（用户能看到搜索结果和 query 的关系）。"""
    },
    '搜索系统LLM集成框架.md': {
        'formulas': """## 📐 核心公式与原理

### 1. LLM Query Rewriting
$$q' = \\text{LLM}(\\text{"Rewrite the query for better search: "} + q)$$
- LLM 理解用户意图后改写 query

### 2. LLM Reranking (Listwise)
$$\\text{ranking} = \\text{LLM}(\\text{"Rank these documents by relevance: "} + [d_1, ..., d_n])$$
- LLM 一次性对候选列表排序

### 3. RAG
$$answer = \\text{LLM}(q, \\text{retrieve}(q, \\text{corpus}))$$
- 先检索相关文档，再用 LLM 生成答案""",
        'extra_qa': """
### Q7: LLM Reranking 的三种方式？
**30秒答案**：①Pointwise：逐个打分 P(relevant|q,d)；②Pairwise：成对比较 P(d_a > d_b | q)；③Listwise：一次排整个列表。效果 Listwise > Pairwise > Pointwise，但成本也相应更高。RankLLM 系统对比了三种方式。

### Q8: LLM 做搜索的延迟问题怎么解决？
**30秒答案**：①只对 Top-K 结果做 LLM reranking（如 top 20）；②模型蒸馏（大 LLM → 小 Cross-Encoder）；③缓存常见 query 的结果；④异步重排（先展示初始结果，后台 LLM 重排后更新）。

### Q9: Query Understanding 包含哪些子任务？
**30秒答案**：①意图分类（导航/信息/交易）；②实体识别（品牌/品类/属性）；③Query 改写（同义词替换、缩写展开）；④Query 补全（输入联想）；⑤多意图识别（一个 query 多个意图）。LLM 可以统一解决这些子任务。

### Q10: GEO（Generative Engine Optimization）是什么？
**30秒答案**：类比 SEO，GEO 是针对 AI 搜索引擎（如 Perplexity、Google AI Overview）优化内容的方法。核心：让你的内容更容易被 LLM 引用为答案来源。方法：结构化内容、权威引用、直接回答常见问题。"""
    },
    '混合检索融合_多路召回实践.md': {
        'formulas': """## 📐 核心公式与原理

### 1. RRF（Reciprocal Rank Fusion）
$$RRF(d) = \\sum_{r \\in R} \\frac{1}{k + r(d)}$$
- 多路排名倒数融合，k=60 是常见默认值

### 2. 线性分数融合
$$score(d) = \\alpha \\cdot score_{sparse}(d) + (1-\\alpha) \\cdot score_{dense}(d)$$
- 归一化后加权，α 需要调优

### 3. Learned Fusion
$$score(d) = \\text{MLP}(\\text{features}_{sparse}(q,d), \\text{features}_{dense}(q,d))$$
- 学习型融合，自动学习最优组合""",
        'extra_qa': """
### Q7: RRF 为什么简单有效？
**30秒答案**：RRF 只依赖排名（不依赖分数绝对值），天然处理了不同检索器分数不可比的问题。且它对异常值不敏感（一个检索器排名极差不会拖累整体），实践中经常接近最优。

### Q8: 混合检索中各路的权重怎么确定？
**30秒答案**：①Grid Search：在验证集上搜索最优 α；②Bayesian Optimization：更高效的超参搜索；③Learned：训练小模型自动学权重；④Query-dependent：不同类型 query 用不同权重（精确匹配 query 偏向 sparse，语义 query 偏向 dense）。

### Q9: 多路召回结果的去重策略？
**30秒答案**：①文档级去重：相同 doc_id 取最高分；②语义去重：embedding 相似度超阈值的合并；③多样性去重：类似 MMR，在保持覆盖率的前提下去除冗余文档。

### Q10: 三路混合（BM25 + Dense + Late Interaction）的实践经验？
**30秒答案**：BM25 处理精确匹配，Dense 处理语义匹配，Late Interaction（ColBERT）处理细粒度交互。实践中两路（BM25+Dense）已经覆盖大部分场景，三路在高精度需求（如法律/医疗搜索）时额外收益明显。"""
    },
    'LearningToRank搜索排序三大范式.md': {
        'formulas': """## 📐 核心公式与原理

### 1. Pointwise (RankNet)
$$L = -y \\log(\\hat{y}) - (1-y) \\log(1-\\hat{y})$$
- 将排序转化为分类/回归问题

### 2. Pairwise (LambdaRank)
$$\\lambda_{ij} = \\frac{-\\sigma}{1 + e^{\\sigma(s_i - s_j)}} |\\Delta NDCG_{ij}|$$
- 文档对的梯度按 NDCG 变化加权

### 3. Listwise (ListNet)
$$L = -\\sum_\\pi P_y(\\pi) \\log P_s(\\pi)$$
- 直接优化排列概率分布的交叉熵

### 4. NDCG
$$NDCG@K = \\frac{DCG@K}{IDCG@K}, \\quad DCG@K = \\sum_{i=1}^K \\frac{2^{rel_i} - 1}{\\log_2(i+1)}$$
- 搜索排序的核心评估指标""",
        'extra_qa': """
### Q7: 三种范式在实际中怎么选？
**30秒答案**：Pointwise 最简单（可直接用 CTR 模型），适合大规模+多目标；Pairwise（LambdaMART）效果稳定，是搜索排序的经典选择；Listwise 理论最优但训练不稳定。工业搜索常用 LambdaMART（GBDT）或 Pointwise（DNN）。

### Q8: LambdaMART 为什么是搜索排序的标杆？
**30秒答案**：①GBDT 对 tabular 特征效果好（搜索特征多为统计特征）；②Lambda 梯度直接优化 NDCG（避免了 NDCG 不可导的问题）；③训练稳定（比 Listwise DNN）；④可解释性好。直到 2020+ 年 DNN 才开始超越。

### Q9: 搜索排序特征有哪些？
**30秒答案**：①Query-Doc 匹配特征（BM25 分数、embedding 相似度、TF-IDF）；②Doc 质量特征（PageRank、内容长度、freshness）；③用户特征（搜索历史、点击偏好）；④Context（设备、地理位置、时间）。

### Q10: Neural LTR vs Traditional LTR？
**30秒答案**：Traditional（LambdaMART）：手工特征 + GBDT，可解释+训练快；Neural（BERT/Cross-Encoder）：端到端学习语义匹配，效果更好但推理慢。趋势：Neural 做精排（top 20-50），Traditional 做粗排/线上兜底。"""
    },
    '搜索Query理解.md': {
        'formulas': """## 📐 核心公式与原理

### 1. Query 分类
$$P(intent | q) = \\text{softmax}(W \\cdot \\text{BERT}_{CLS}(q))$$
- 意图分类：导航型/信息型/交易型

### 2. NER（命名实体识别）
$$P(tag_i | q) = \\text{CRF}(\\text{BERT}(q))$$
- 序列标注：提取品牌/品类/属性实体

### 3. Query Expansion
$$q' = q + \\sum_{t \\in \\text{expanded}} w_t \\cdot t$$
- 同义词/相关词扩展，增加召回覆盖""",
        'extra_qa': """
### Q7: Query Rewriting 和 Query Expansion 的区别？
**30秒答案**：Rewriting 是改写 query（如「怎么减肥」→「减肥方法推荐」），改变了 query 结构；Expansion 是在原 query 基础上追加词（如「iPhone」→「iPhone 手机 苹果」），不改变原意。LLM 擅长 Rewriting，传统方法擅长 Expansion。

### Q8: 多意图 Query 怎么处理？
**30秒答案**：如「苹果」既可能是水果也可能是手机。方法：①意图分类 + 多路召回（每个意图召回一批结果）；②Diversified Ranking（结果中覆盖所有可能意图）；③用户上下文消歧（有搜索历史就能判断）。

### Q9: Query 自动补全（Auto-Complete）怎么做？
**30秒答案**：①前缀匹配：Trie 树快速查找；②流行度排序：按搜索频次排序；③个性化：结合用户历史偏好重排；④LLM 补全：理解语义后补全（如「如何」→「如何学好英语」）。延迟要求极低（<50ms），多级缓存很重要。

### Q10: 搜索 Query 理解的评估指标？
**30秒答案**：①意图分类：F1/Accuracy；②NER：实体 F1（严格匹配 vs 部分匹配）；③Query Rewriting：在线搜索满意度（重写后用户是否点击更多）；④端到端：用 NDCG/MRR 评估整个搜索质量的提升。"""
    },
    '搜索Reranker演进.md': {
        'formulas': """## 📐 核心公式与原理

### 1. Cross-Encoder 重排
$$score = \\sigma(W \\cdot \\text{BERT}_{CLS}([q; \\text{SEP}; d]))$$
- Query-Doc 联合编码，捕获细粒度交互

### 2. Scaling Law for Reranking
$$\\text{NDCG} \\propto \\log(\\text{model\\_size})$$
- Reranker 模型越大效果越好，遵循 Scaling Law

### 3. Distillation
$$L = \\alpha \\cdot L_{CE} + (1-\\alpha) \\cdot KL(p_{teacher} \\| p_{student})$$
- 大 reranker 蒸馏到小模型""",
        'extra_qa': """
### Q7: Reranker 和 Ranker 的区别？
**30秒答案**：Ranker（L2 精排）：处理几百个候选，可以是双塔或轻量交叉模型；Reranker（L3）：处理几十个候选，通常是重型 Cross-Encoder，可以用 LLM。Reranker 追求极致精度，Ranker 平衡精度和效率。

### Q8: LLM Reranker 的效果如何？
**30秒答案**：GPT-4/Claude 级别的 LLM 做 listwise reranking 在 BEIR 等 benchmark 上超过所有传统方法。但成本极高（每次 rerank 需要数千 token 输入）。实践：用 LLM 生成训练数据，蒸馏到小型 Cross-Encoder。

### Q9: Late Interaction (ColBERT) vs Cross-Encoder？
**30秒答案**：ColBERT：预计算 doc token embedding，查询时做 token-level 交互（MaxSim），推理快（可以用索引加速）；Cross-Encoder：query+doc 联合编码，效果更好但不能预计算。ColBERT 是两者的折中。

### Q10: Reranking 的输入文档数量怎么选？
**30秒答案**：太少（<10）：信息不足，可能错过好文档；太多（>100）：Cross-Encoder 推理成本高。经验值：20-50 个。也取决于 L1 召回质量：召回质量高可以少排一点，质量低需要多给 reranker 机会。"""
    },
    '检索三角形深析.md': {
        'formulas': """## 📐 核心公式与原理

### 1. Sparse (BM25/SPLADE)
$$score_{sparse} = \\sum_{t \\in q \\cap d} w_q(t) \\cdot w_d(t)$$
- 稀疏向量的点积，只在共现词上计算

### 2. Dense (DPR/E5)
$$score_{dense} = E_q(q)^T E_d(d)$$
- 稠密向量的内积/余弦相似度

### 3. Late Interaction (ColBERT)
$$score_{late} = \\sum_{i=1}^{|q|} \\max_{j=1}^{|d|} E_q^i \\cdot E_d^j$$
- MaxSim：每个 query token 找最相似的 doc token""",
        'extra_qa': """
### Q7: SPLADE 和 BM25 的区别？
**30秒答案**：BM25 的词权重基于 TF-IDF 统计；SPLADE 用 BERT 学习词权重（且可以激活原文中没有的词——learned expansion）。SPLADE 既保留了稀疏检索的效率（倒排索引），又学到了语义匹配能力。

### Q8: ColBERT v2/v3 的核心改进？
**30秒答案**：v2：残差压缩（将 doc token embedding 量化为残差+索引，大幅减小存储）；v3：更高效的训练策略（hard negative mining + distillation）+ 更好的量化方案。存储从 v1 的 100GB+ 降到 v3 的 ~10GB（同等数据集）。

### Q9: 什么时候用哪种检索方式？
**30秒答案**：①关键词搜索（精确匹配）→ BM25/SPLADE；②语义搜索（「如何减肥」找「瘦身方法」）→ Dense；③需要高精度 token-level 匹配 → ColBERT；④通用场景 → 混合（BM25+Dense，RRF 融合）。

### Q10: 检索三角形的效率-效果 trade-off？
**30秒答案**：效率：Sparse > Dense > Late Interaction（倒排索引 > ANN > multi-vector ANN）。效果：Late Interaction > Dense > Sparse（通常如此，但取决于数据）。存储：Sparse < Dense < Late Interaction。最优选择取决于场景的延迟