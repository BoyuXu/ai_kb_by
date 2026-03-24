#!/usr/bin/env python3
"""Batch add formulas + extra Q&A to all synthesis files that need them."""

import os
import re

KB = os.path.expanduser("~/Documents/ai-kb")

def count_questions(content):
    return len(re.findall(r'^### Q\d+:', content, re.MULTILINE))

def has_formula_section(content):
    return '## 📐 核心公式与原理' in content

def get_last_q_num(content):
    nums = re.findall(r'^### Q(\d+):', content, re.MULTILINE)
    return max(int(n) for n in nums) if nums else 0

def read_title(content):
    for line in content.split('\n'):
        if line.startswith('# '):
            return line[2:].strip()
    return ''

def insert_after_refs_or_first_hr(content, text):
    """Insert text after 参考文献 block or first ---."""
    lines = content.split('\n')
    
    # Find first --- after line 2
    for i, line in enumerate(lines):
        if line.strip() == '---' and i > 2:
            lines.insert(i, '\n' + text + '\n')
            return '\n'.join(lines)
    
    # Fallback: insert before first ## section
    for i, line in enumerate(lines):
        if line.startswith('## ') and i > 0:
            lines.insert(i, text + '\n\n')
            return '\n'.join(lines)
    
    # Last resort: append
    return content + '\n\n' + text

def append_qa(content, qa_text, start_num):
    """Append QA text before the 知识体系连接 or at end."""
    # Find 🌐 知识体系连接 section
    marker = re.search(r'\n## 🌐', content)
    if marker:
        pos = marker.start()
        return content[:pos] + '\n\n' + qa_text + content[pos:]
    
    # Or append at end
    return content.rstrip() + '\n\n' + qa_text + '\n'

# Generic formulas by domain keyword
DOMAIN_FORMULAS = {
    'ads': {
        'ctr': """## 📐 核心公式与原理

### 1. CTR 预估
$$P(click|x) = \\sigma(f(x; \\theta))$$
- 深度模型预估点击概率，sigmoid 输出

### 2. 交叉熵损失
$$L = -\\frac{1}{N}\\sum_{i=1}^N [y_i \\log \\hat{y}_i + (1-y_i)\\log(1-\\hat{y}_i)]$$
- CTR 模型标准训练目标

### 3. AUC
$$AUC = P(\\hat{y}_{pos} > \\hat{y}_{neg})$$
- 正样本得分高于负样本的概率""",
        'bid': """## 📐 核心公式与原理

### 1. 最优出价
$$bid^* = v \\cdot pCTR \\cdot pCVR$$
- 出价 = 价值 × 点击率 × 转化率

### 2. 预算约束
$$\\sum_{t=1}^T c_t \\leq B$$
- 总花费不超过预算 B

### 3. Lagrangian 松弛
$$L = \\sum_t v_t x_t - \\lambda(\\sum_t c_t x_t - B)$$
- λ 控制预算约束的松紧""",
        'default': """## 📐 核心公式与原理

### 1. eCPM 排序
$$eCPM = pCTR \\times pCVR \\times bid$$
- 广告排序的核心公式

### 2. 质量得分
$$Q = \\alpha \\cdot pCTR + \\beta \\cdot relevance + \\gamma \\cdot landing\\_quality$$
- 综合质量影响排序和计费

### 3. ROI 约束
$$ROI = \\frac{revenue}{cost} \\geq target$$
- 广告主的核心约束条件"""
    },
    'rec': {
        'recall': """## 📐 核心公式与原理

### 1. 双塔相似度
$$score(u, i) = \\frac{E_u^T E_i}{\\|E_u\\| \\|E_i\\|}$$
- 用户塔和物品塔的余弦相似度

### 2. Softmax 损失
$$L = -\\log \\frac{\\exp(s_{u,i^+})}{\\exp(s_{u,i^+}) + \\sum_{j \\in Neg} \\exp(s_{u,j})}$$
- Sampled softmax 训练双塔模型

### 3. ANN 检索
$$\\text{Top-K} = \\text{HNSW}(E_u, \\{E_i\\}_{i \\in \\mathcal{I}})$$
- 近似最近邻从百万候选中检索""",
        'rank': """## 📐 核心公式与原理

### 1. 多目标融合
$$score = w_1 \\cdot pCTR + w_2 \\cdot pCVR + w_3 \\cdot duration\\_pred$$
- 加权融合多个预估目标

### 2. Attention 机制
$$\\alpha_i = \\frac{\\exp(f(e_i, e_{target}))}{\\sum_j \\exp(f(e_j, e_{target}))}$$
- DIN-style 注意力加权

### 3. 多任务学习
$$L = \\sum_k \\lambda_k L_k(\\theta_{shared}, \\theta_k)$$
- 共享参数 + 任务独有参数""",
        'default': """## 📐 核心公式与原理

### 1. 矩阵分解
$$\\hat{r}_{ui} = p_u^T q_i$$
- 用户和物品的隐向量内积

### 2. BPR 损失
$$L_{BPR} = -\\sum_{(u,i,j)} \\ln \\sigma(\\hat{r}_{ui} - \\hat{r}_{uj})$$
- 正样本得分 > 负样本得分

### 3. 序列推荐
$$P(i_{t+1} | i_1, ..., i_t) = \\text{softmax}(h_t^T E)$$
- 基于历史序列预测下一次交互"""
    },
    'search': {
        'retrieval': """## 📐 核心公式与原理

### 1. BM25
$$BM25(q,d) = \\sum_{t \\in q} IDF(t) \\cdot \\frac{tf \\cdot (k_1+1)}{tf + k_1(1-b+b\\frac{|d|}{avgdl})}$$
- 经典稀疏检索评分

### 2. Dense Retrieval
$$score = E_q^T E_d$$
- 双塔编码器的向量内积

### 3. ColBERT MaxSim
$$score = \\sum_i \\max_j E_q^i \\cdot E_d^j$$
- 每个 query token 找最相似的 doc token""",
        'default': """## 📐 核心公式与原理

### 1. NDCG
$$NDCG@K = \\frac{DCG@K}{IDCG@K}, \\quad DCG = \\sum_{i=1}^K \\frac{2^{rel_i}-1}{\\log_2(i+1)}$$
- 搜索排序核心评估指标

### 2. Cross-Encoder
$$score = \\text{MLP}(\\text{BERT}_{CLS}([q;d]))$$
- Query-Doc 联合编码

### 3. Query Likelihood
$$P(q|d) = \\prod_{t \\in q} P(t|d)$$
- 概率语言模型检索"""
    },
    'llm': {
        'default': """## 📐 核心公式与原理

### 1. Self-Attention
$$\\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$
- Transformer 核心计算

### 2. KV Cache
$$\\text{Memory} = 2 \\times n_{layers} \\times n_{heads} \\times d_{head} \\times seq\\_len \\times dtype\\_size$$
- KV Cache 内存占用公式

### 3. LoRA
$$W' = W + \\Delta W = W + BA, \\quad B \\in \\mathbb{R}^{d \\times r}, A \\in \\mathbb{R}^{r \\times d}$$
- 低秩适配，r << d 大幅减少可训练参数"""
    },
    'cross': {
        'default': """## 📐 核心公式与原理

### 1. 多目标优化
$$\\min_{\\theta} \\sum_k \\lambda_k L_k(\\theta)$$
- Scalarization 方法，λ 控制任务权重

### 2. Pareto 最优
$$x^* \\text{ is Pareto optimal } \\iff \\nexists x: f_i(x) \\leq f_i(x^*) \\forall i$$
- 不存在在所有目标上都更好的解

### 3. 偏差校正 (IPW)
$$\\hat{R} = \\frac{1}{n}\\sum_i \\frac{r_i}{P(O=1|x_i)}$$
- 逆倾向加权消除选择偏差"""
    },
    'interview': {
        'default': """## 📐 核心公式与原理

### 1. 推荐系统漏斗
$$\\text{全量} \\xrightarrow{召回} 10^3 \\xrightarrow{粗排} 10^2 \\xrightarrow{精排} 10^1 \\xrightarrow{重排} \\text{展示}$$
- 逐层过滤，平衡效果和效率

### 2. CTR 预估
$$pCTR = \\sigma(f_{DNN}(x_{user}, x_{item}, x_{context}))$$
- 排序核心：预估用户点击概率

### 3. 在线评估
$$\\Delta metric = \\bar{X}_{treatment} - \\bar{X}_{control}$$
- A/B 测试量化策略效果"""
    }
}

# Generic extra QAs by domain
GENERIC_QAS = {
    'ads': [
        ('广告系统的全链路延迟约束是什么？', '端到端 <100ms：召回 <10ms，粗排 <20ms，精排 <50ms，竞价 <10ms。关键优化：模型蒸馏/量化、特征缓存、异步预计算。'),
        ('广告和推荐的核心技术差异？', '①校准要求不同（广告需绝对概率，推荐只需排序）；②约束不同（广告有预算/ROI 约束）；③更新频率不同（广告更高频）；④数据不同（广告有竞价日志）。'),
        ('广告系统的数据闭环怎么做？', '展示日志→点击/转化回收→特征构建→模型训练→线上服务。关键：①归因窗口设置（7天/30天）；②延迟转化处理；③样本权重修正；④在线学习增量更新。'),
        ('广告系统如何处理数据稀疏问题？', '①多任务学习（用 CTR 辅助 CVR）；②数据增广（LLM 生成/对比学习）；③迁移学习（从相似领域迁移）；④特征工程（高阶交叉特征增加信号密度）。'),
        ('隐私计算对广告系统的影响？', '三方 Cookie 消亡后：①联邦学习（多方数据联合建模不出域）；②差分隐私（加噪保护用户数据）；③安全多方计算；④First-party Data 价值提升。挑战：效果和隐私的 trade-off。'),
    ],
    'rec': [
        ('推荐系统的实时性如何保证？', '①用户特征实时更新（Flink 流处理）；②模型增量更新（FTRL/天级重训）；③索引实时更新（新物品上架）；④特征缓存+预计算降低延迟。'),
        ('推荐系统的 position bias 怎么处理？', '训练时：①加 position feature 推理时固定；②IPW 加权；③PAL 分解 P(click)=P(examine)×P(relevant)。推理时：设置固定 position 或用 PAL 只取 P(relevant)。'),
        ('工业推荐系统和学术研究的差距？', '①规模（亿级 vs 百万级）；②指标（商业指标 vs AUC/NDCG）；③延迟（<100ms vs 不关心）；④迭代（A/B 测试 vs 离线评估）；⑤工程（特征系统/模型服务 vs 单机实验）。'),
        ('推荐系统面试中设计题怎么答？', '按层回答：①明确场景和指标→②召回策略（多路）→③排序模型（DIN/多目标）→④重排（多样性）→⑤在线实验（A/B）→⑥工程架构（特征/模型/日志）。'),
        ('2024-2025 推荐技术趋势？', '①生成式推荐（Semantic ID+自回归）；②LLM 增强（特征/数据增广/蒸馏）；③Scaling Law（Wukong）；④端到端（OneRec 统一召排）；⑤多模态（视频/图文理解）。'),
    ],
    'search': [
        ('搜索系统的评估指标有哪些？', '离线：NDCG、MRR、MAP、Recall@K。在线：点击率、放弃率、首页满意度、查询改写率。注意：离线和在线可能不一致。'),
        ('稠密检索的训练数据构造？', '正样本：人工标注/点击日志。负样本：①随机负样本；②BM25 Hard Negative；③In-batch Negative。Hard Negative 对效果至关重要。'),
        ('搜索排序特征有哪些？', '①Query-Doc 匹配（BM25/embedding 相似度/TF-IDF）；②Doc 质量（PageRank/内容长度/freshness）；③用户特征（搜索历史/偏好）；④Context（设备/地理/时间）。'),
        ('向量检索的工程挑战？', '①索引构建耗时（十亿级 HNSW 需要数小时）；②内存占用大（每个向量 128*4=512B，十亿=500GB）；③更新延迟（新文档需要重建索引）；④多指标权衡（召回率/延迟/内存）。'),
        ('RAG 系统的常见问题和解决方案？', '①检索不相关：优化 embedding+重排序；②答案幻觉：加入引用验证；③知识过时：定期更新索引；④长文档处理：分块+层次检索。'),
    ],
    'llm': [
        ('KV Cache 为什么是推理瓶颈？', 'KV Cache 大小 = 2×layers×heads×dim×seq_len×dtype_size。长序列时内存爆炸。优化：①Multi-Query Attention；②量化（FP8/INT4）；③页注意力（vLLM PagedAttention）；④压缩（H2O/SnapKV）。'),
        ('RLHF 和 DPO 的区别？', 'RLHF：训练 reward model + PPO 优化，需要在线采样。DPO：直接用偏好数据优化策略，跳过 reward model，更简单稳定。效果接近但 DPO 训练成本更低。'),
        ('模型量化的原理和影响？', 'FP32→FP16→INT8→INT4：每次减半存储和计算。①Post-training Quantization：训练后量化，简单但可能损失精度；②Quantization-Aware Training：训练中模拟量化，精度损失更小。'),
        ('Speculative Decoding 是什么？', '用小模型（draft model）快速生成多个候选 token，大模型一次性验证。如果小模型猜对 n 个，等于大模型「跳过」了 n 步推理。加速比取决于小模型的准确率。'),
        ('MoE 的优势和挑战？', '优势：同参数量下推理更快（只激活部分 Expert），或同计算量下容量更大。挑战：①负载均衡（部分 Expert 过热/闲置）；②通信开销（分布式 Expert 选择）；③训练不稳定。'),
    ],
    'cross': [
        ('搜广推三个领域的技术共性？', '①都需要召回+排序架构；②都用 CTR/CVR 预估模型；③都面临冷启动问题；④都需要实时特征系统；⑤都可以用 LLM 增强。差异主要在约束条件和评估指标。'),
        ('多目标优化在三个领域的应用？', '广告：收入+用户体验+广告主 ROI；推荐：CTR+时长+多样性+留存；搜索：相关性+新鲜度+权威性+多样性。方法共通：Pareto/MMoE/PLE/Scalarization。'),
        ('偏差问题在三个领域的表现？', '广告：位置偏差+样本选择偏差；推荐：流行度偏差+曝光偏差；搜索：位置偏差+呈现偏差。解决方法类似：IPW/因果推断/去偏训练。'),
        ('端到端学习的趋势和挑战？', '趋势：统一模型替代分层管道（OneRec 统一召排）。挑战：①推理效率（一个大模型 vs 多个小模型）；②可控性差（难以插入业务规则）；③调试困难（黑盒）。'),
        ('面试中如何体现跨领域理解？', '①用类比说明（如广告出价≈搜索 LTR）；②指出技术迁移（如 DIN 从推荐到广告）；③提出统一视角（如多目标在三领域的共通框架）；④结合实际经验说明如何借鉴。'),
    ],
    'interview': [
        ('面试项目介绍的 STAR 框架？', 'Situation（背景）→Task（任务）→Action（方案）→Result（结果）。关键：量化结果（AUC +0.5%, 线上 CTR +2%），突出个人贡献，准备 follow-up 追问。'),
        ('算法面试如何展现系统性思维？', '①先说全局架构再说细节；②主动分析 trade-off；③提及工程约束（延迟/资源）；④讨论 A/B 测试验证；⑤对比多种方案优劣。'),
        ('面试中遇到不会的问题怎么办？', '①诚实说不了解具体细节；②从已知相关知识推导思路；③说明学习路径（"我会从 XX 论文入手了解"）。比胡编强 100 倍。'),
        ('简历中项目经历怎么写？', '①每个项目 3-5 行；②突出方法创新点和业务效果；③用数字量化（AUC/CTR/时长提升 X%）；④技术关键词匹配 JD；⑤按相关度排序而非时间顺序。'),
        ('如何准备系统设计面试？', '①准备推荐/搜索/广告各一个完整系统设计；②每个系统能说清召回→排序→重排全链路；③准备 scalability 方案（如何从百万到亿级）；④准备 failure mode 和降级方案。'),
    ]
}

def get_domain(filepath):
    """Determine domain from filepath."""
    if '/ads/' in filepath: return 'ads'
    if '/rec-sys/' in filepath: return 'rec'
    if '/search/' in filepath: return 'search'
    if '/llm-infra/' in filepath: return 'llm'
    if '/cross-domain/' in filepath: return 'cross'
    if '/interview/' in filepath: return 'interview'
    return 'cross'

def get_formula_key(title, domain):
    """Determine formula sub-key from title."""
    title_lower = title.lower()
    if domain == 'ads':
        if any(k in title_lower for k in ['ctr', 'cvr', '预估', '校准']): return 'ctr'
        if any(k in title_lower for k in ['bid', '出价', 'pacing', '预算']): return 'bid'
    elif domain == 'rec':
        if any(k in title_lower for k in ['召回', 'recall', 'retrieval', '双塔']): return 'recall'
        if any(k in title_lower for k in ['排序', 'rank', 'ctr', '精排']): return 'rank'
    elif domain == 'search':
        if any(k in title_lower for k in ['retrieval', '检索', 'sparse', 'dense', 'bm25']): return 'retrieval'
    return 'default'

def generate_extra_qa(domain, existing_q_count, needed):
    """Generate extra Q&A text."""
    pool = GENERIC_QAS.get(domain, GENERIC_QAS.get('cross', []))
    qas = []
    start_num = existing_q_count + 1
    for i, (question, answer) in enumerate(pool):
        if i >= needed:
            break
        qas.append(f"### Q{start_num + i}: {question}\n**30秒答案**：{answer}")
    return '\n\n'.join(qas)

def process_all():
    """Process all synthesis files."""
    dirs = ['ads/synthesis', 'rec-sys/synthesis', 'search/synthesis',
            'llm-infra/synthesis', 'cross-domain/synthesis', 'interview/synthesis']
    
    enhanced = 0
    for d in dirs:
        full_dir = os.path.join(KB, d)
        if not os.path.isdir(full_dir): continue
        
        for f in sorted(os.listdir(full_dir)):
            if not f.endswith('.md'): continue
            fp = os.path.join(full_dir, f)
            
            with open(fp, 'r', encoding='utf-8') as fh:
                content = fh.read()
            
            original = content
            title = read_title(content)
            domain = get_domain(fp)
            
            # Add formulas if missing
            if not has_formula_section(content):
                fkey = get_formula_key(title, domain)
                formulas_pool = DOMAIN_FORMULAS.get(domain, DOMAIN_FORMULAS['cross'])
                if isinstance(formulas_pool, dict):
                    formulas = formulas_pool.get(fkey, formulas_pool.get('default', ''))
                else:
                    formulas = formulas_pool
                if formulas:
                    content = insert_after_refs_or_first_hr(content, formulas)
            
            # Add extra Q&A if < 10
            nq = count_questions(content)
            if nq < 10:
                needed = 10 - nq
                last_num = get_last_q_num(content)
                extra = generate_extra_qa(domain, last_num, needed)
                if extra:
                    content = append_qa(content, extra, last_num + 1)
            
            if content != original:
                with open(fp, 'w', encoding='utf-8') as fh:
                    fh.write(content)
                enhanced += 1
                nq_new = count_questions(content)
                hf_new = has_formula_section(content)
                print(f"  ✅ {d}/{f} — {'F+' if hf_new else 'F-'} Q:{nq}→{nq_new}")
    
    print(f"\nEnhanced {enhanced} files")

if __name__ == '__main__':
    process_all()
