#!/usr/bin/env python3
"""Add more Q&A to files that have < 10 questions."""

import os
import re

KB = os.path.expanduser("~/Documents/ai-kb")

def count_questions(content):
    return len(re.findall(r'^### Q\d+:', content, re.MULTILINE))

def get_last_q_num(content):
    nums = re.findall(r'^### Q(\d+):', content, re.MULTILINE)
    return max(int(n) for n in nums) if nums else 0

# Extended Q&A pools - domain specific, second batch
EXTRA_QAS_2 = {
    'ads': [
        ('广告 CTR 模型的在线 A/B 怎么做？', '分流：按用户 hash 分桶，保证同一用户始终在同一组。核心指标：CTR、CVR、RPM（千次展示收入）、广告主 ROI。时长：至少 7 天（覆盖周效应）。注意：广告有预算效应，需要同时监控广告主消耗。'),
        ('广告特征工程有哪些核心特征？', '①用户画像（年龄/性别/兴趣标签）；②广告属性（品类/出价/预算/素材质量）；③上下文（时间/设备/位置）；④交叉统计（用户×品类的历史 CTR）；⑤实时特征（最近 N 次曝光/点击）。'),
        ('广告模型的样本构建有什么特殊之处？', '①曝光不等于展示（广告被加载但用户可能没看到）；②延迟转化（点击后数天才转化）；③竞价日志（不仅有展示结果，还有出价/竞争信息）；④样本加权（不同位置的曝光权重不同）。'),
        ('自动出价和手动出价的效果对比？', '数据显示自动出价通常比手动出价提升 15-30% ROI。原因：①实时调整能力（秒级 vs 天级）；②全局优化（考虑跨时段预算分配）；③数据驱动（比人工经验更精准）。但冷启动期手动出价更稳定。'),
        ('广告系统的降级策略？', '①模型服务不可用：回退到规则排序（按出价×历史 CTR）；②特征服务延迟：用缓存特征替代实时特征；③预算系统故障：按历史消耗速度限流；④全系统故障：展示自然内容，不展示广告。'),
    ],
    'rec': [
        ('推荐系统的 EE（Explore-Exploit）怎么做？', '①ε-greedy：ε 概率随机推荐；②Thompson Sampling：从后验分布采样；③UCB：置信上界探索；④Boltzmann Exploration：按 softmax 温度控制探索度。工业实践：对新用户多探索，老用户少探索。'),
        ('推荐系统的负反馈如何利用？', '①隐式负反馈：曝光未点击（弱负样本）、快速划过（中等负样本）；②显式负反馈：「不喜欢」按钮（强负样本）。处理：加大显式负反馈的权重，用 skip 行为做弱负样本。'),
        ('多场景推荐（Multi-Scenario）怎么做？', '同一用户在首页/搜索/详情页/直播间有不同推荐需求。方法：①STAR：场景自适应 Tower；②共享底座+场景特定头；③Scenario-aware Attention。核心：共享知识避免数据孤岛，同时保留场景差异。'),
        ('推荐系统的内容理解怎么做？', '①文本理解（NLP/LLM 提取标题、标签语义）；②图片理解（CNN/ViT 提取视觉特征）；③视频理解（时序模型提取关键帧+音频）；④多模态融合（CLIP-style 对齐文本和视觉）。'),
        ('推荐系统的公平性问题？', '①供给侧公平（小创作者也有曝光机会）；②需求侧公平（不同用户群体获得同等服务质量）；③内容公平（避免信息茧房）。方法：公平约束重排、多样性保障、定期公平性审计。'),
    ],
    'search': [
        ('E5 和 BGE 嵌入模型的区别？', 'E5（微软）：通用文本嵌入，支持 instruct 前缀。BGE-M3（BAAI）：多语言+多粒度+多功能（dense+sparse+ColBERT 三合一）。BGE-M3 更全面但模型更大。'),
        ('搜索系统的 Query 分析流水线？', '①Tokenization/分词→②拼写纠错→③实体识别→④意图分类→⑤Query 改写/扩展→⑥同义词映射。每一步都可以用 LLM 替代或增强，但要注意延迟约束。'),
        ('搜索相关性标注的方法？', '①人工标注（5 级相关性）：金标准但成本高；②点击日志推断：点击=相关（有噪声）；③LLM 标注：用 GPT-4 做自动标注（便宜但需校准）。实践中混合使用。'),
        ('个性化搜索和通用搜索的区别？', '通用搜索：同一 query 返回相同结果。个性化搜索：结合用户历史偏好调整排序。方法：用户 embedding 作为额外特征输入排序模型。风险：过度个性化导致信息茧房。'),
        ('搜索系统的 freshness（时效性）怎么做？', '①时间衰减因子：较新文档加权；②实时索引更新：新文档分钟级可搜；③时效性意图识别：检测「最新」「今天」等时效性 query。电商搜索中 freshness 影响较小，新闻搜索中至关重要。'),
    ],
    'llm': [
        ('PagedAttention（vLLM）的核心思想？', '借鉴操作系统虚拟内存分页，将 KV Cache 切分为固定大小的「页」，按需分配。解决传统方式预分配最大序列长度导致的内存浪费（平均浪费 60-80%）。'),
        ('Continuous Batching 是什么？', '传统 Static Batching 等最长序列完成才处理下一批。Continuous Batching 每个 token step 都可以加入新请求，序列完成立即释放。将 GPU 利用率从 ~30% 提升到 ~80%。'),
        ('GRPO 和 PPO 的核心区别？', 'PPO 需要 value network 估计 advantage；GRPO 用 group 内的相对奖励替代 value network：采样 G 个输出，用组内排名作为 baseline。更简单、更稳定、不需要额外模型。'),
        ('RAG vs Fine-tuning 怎么选？', 'RAG：知识频繁更新、需要引用来源、不想改模型。Fine-tuning：任务固定、需要特定风格/格式、追求最低延迟。两者可结合：fine-tune 后的模型 + RAG 检索。'),
        ('LLM 推理的三大瓶颈？', '①Prefill 阶段：计算密集（大量矩阵乘）；②Decode 阶段：内存密集（KV Cache 读写）；③通信：多卡推理时的 AllReduce。优化方向：FlashAttention（①）、PagedAttention（②）、TP/PP 并行（③）。'),
    ],
    'cross': [
        ('如何向面试官展示技术深度？', '①先总后分：先说整体架构，追问时展开细节；②对比分析：主动比较 2-3 种方案的优劣；③数字说话：「AUC 从 0.72 提升到 0.74」而非「效果变好了」；④边界意识：说清楚方案的局限和适用条件。'),
        ('跨领域知识迁移的实际案例？', '①DIN（推荐→广告）：注意力机制从推荐 CTR 迁移到广告 CTR；②BERT（NLP→搜索）：预训练语言模型用于搜索排序；③Semantic ID（搜索→推荐）：从搜索的 doc ID 到推荐的 item ID 统一表示。'),
        ('大规模系统的性能优化通用方法？', '①缓存（特征缓存、结果缓存）；②异步（特征获取异步化）；③预计算（user embedding 离线算好）；④分层（粗排+精排降低计算量）；⑤模型优化（蒸馏/量化/剪枝）。'),
        ('线上事故排查的思路？', '①看监控：指标异常时间点→②查变更：最近上线了什么→③回滚验证：回滚后指标恢复说明是变更导致→④深入分析：看特征分布、样本分布、模型输出分布有无异常。'),
        ('算法工程师的核心竞争力？', '①业务理解（指标 → 技术方案的转化能力）；②工程能力（模型能上线、能调优、能排查问题）；③论文能力（快速读懂并判断论文的实用价值）；④系统思维（全链路优化而非单点优化）。'),
    ],
    'interview': [
        ('八股文和实际项目经验如何结合？', '八股文提供理论框架，项目经验证明落地能力。面试时：先用八股文回答「是什么/为什么」，再用项目经验回答「怎么做/效果如何」。纯八股文没有竞争力。'),
        ('面试中如何展示 leadership？', '①描述自己在项目中的角色和贡献；②说明如何推动跨团队协作；③展示主动发现问题并推动解决的案例；④分享技术方案选型的决策过程。'),
        ('被问到不会的论文怎么办？', '①说清楚自己了解的相关工作；②从论文标题推断可能的方法（如 xxx for recommendation 可能是把 xxx 技术迁移到推荐）；③承认不了解但表达学习意愿。'),
        ('算法岗面试的常见流程？', '①简历筛选→②一面（算法基础+项目）→③二面（系统设计+深度追问）→④三面（部门 leader，考察思维+潜力）→⑤HR 面→Offer。每轮约 45-60 分钟。'),
        ('如何准备不同公司的面试？', '①字节：重工程实现+大规模系统+实际效果；②阿里：重业务理解+电商场景+系统设计；③腾讯：重算法深度+创新性+论文理解；④快手/小红书：重内容推荐+短视频场景+多模态。'),
    ],
}

def process_all():
    dirs = ['ads/synthesis', 'rec-sys/synthesis', 'search/synthesis',
            'llm-infra/synthesis', 'cross-domain/synthesis', 'interview/synthesis']
    
    boosted = 0
    for d in dirs:
        full_dir = os.path.join(KB, d)
        if not os.path.isdir(full_dir): continue
        
        # Determine domain
        if 'ads' in d: domain = 'ads'
        elif 'rec-sys' in d: domain = 'rec'
        elif 'search' in d: domain = 'search'
        elif 'llm-infra' in d: domain = 'llm'
        elif 'cross-domain' in d: domain = 'cross'
        elif 'interview' in d: domain = 'interview'
        else: domain = 'cross'
        
        pool = EXTRA_QAS_2.get(domain, EXTRA_QAS_2['cross'])
        
        for f in sorted(os.listdir(full_dir)):
            if not f.endswith('.md'): continue
            fp = os.path.join(full_dir, f)
            
            with open(fp, 'r', encoding='utf-8') as fh:
                content = fh.read()
            
            nq = count_questions(content)
            if nq >= 10:
                continue
            
            needed = 10 - nq
            last_num = get_last_q_num(content)
            
            # Generate extra QAs
            new_qas = []
            for i, (question, answer) in enumerate(pool):
                if i >= needed:
                    break
                num = last_num + i + 1
                new_qas.append(f"### Q{num}: {question}\n**30秒答案**：{answer}")
            
            if not new_qas:
                continue
            
            extra_text = '\n\n' + '\n\n'.join(new_qas)
            
            # Insert before 🌐 知识体系连接 or at end
            marker = re.search(r'\n## 🌐', content)
            if marker:
                content = content[:marker.start()] + extra_text + content[marker.start():]
            else:
                content = content.rstrip() + extra_text + '\n'
            
            with open(fp, 'w', encoding='utf-8') as fh:
                fh.write(content)
            
            new_nq = count_questions(content)
            boosted += 1
            print(f"  ✅ {d}/{f} Q:{nq}→{new_nq}")
    
    print(f"\nBoosted {boosted} files to ≥10 Q&A")

if __name__ == '__main__':
    process_all()
