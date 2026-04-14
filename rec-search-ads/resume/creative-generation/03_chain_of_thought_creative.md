# 03 思维链处理模糊广告诉求

> 核心论文：Chain-of-Thought (Wei et al., NeurIPS 2022)、Self-Consistency (Wang et al., 2022)、Tree of Thoughts (Yao et al., 2023)

---

## 一、广告创意生成的"模糊诉求"问题

### 1.1 典型的模糊输入

广告主往往不能清晰描述需求，给出的是高度模糊的自然语言：

```
广告主输入（现实中）：
  ✗ 「帮我写一个吸引年轻人的文案」
  ✗ 「要有冲击力，能让人下单那种」
  ✗ 「这个产品很好，你帮我写个好的」

广告主实际需要（系统需要挖掘的）：
  ✅ 产品：某品牌男士护肤套装
  ✅ 目标人群：18-28岁，一线城市，注重仪表感的年轻男性
  ✅ 平台：抖音（15秒短视频配套文案）
  ✅ 诉求：首购激励 + 社交认同感（同龄人都在用）
  ✅ 禁忌：不能提"便宜"（伤害品牌调性）
```

**模糊诉求的四个维度：**
1. **产品维度**：核心卖点不明确，LLM不知道写什么
2. **人群维度**：目标用户画像模糊，文案语气、词汇选择错误
3. **平台维度**：不同平台字数、风格要求差异巨大
4. **效果维度**：广告主说"要好"，但好的标准不同

### 1.2 直接生成的失败

```
直接Prompt（无CoT）：
  输入：「帮我写一个效果好的推广文案，产品是男士护肤套装」

  输出：
  「精选男士护肤套装，多重功效一步到位，
   让您的肌肤焕然一新，立即购买享受优惠！」

  问题：
  - 语言老气，不吸引年轻人
  - 没有具体卖点（"多重功效"是什么？）
  - "享受优惠"暗示低价，伤害中高端定位
  - 完全忽略平台特性
```

---

## 二、CoT的核心原理

### 2.1 Wei et al. 2022 详解

**标准Prompt vs CoT Prompt 对比：**

```
标准Prompt（直接回答）：
  Q: 小明买了3个苹果，又买了2个梨，还剩多少水果？
  A: 5个

CoT Prompt（展示推理过程）：
  Q: 小明买了3个苹果，又买了2个梨，还剩多少水果？
     让我们一步步思考...
  A: 小明买了3个苹果。
     然后又买了2个梨。
     苹果和梨都是水果，所以总共有3+2=5个水果。
     答案是5个水果。
```

**两种CoT模式：**

| 模式 | 触发方式 | 优点 | 缺点 |
|------|---------|------|------|
| **Few-shot CoT** | 提供带推理步骤的示例 | 质量高，示例引导明确 | 需要标注示例，token消耗大 |
| **Zero-shot CoT** | "Let's think step by step" | 无需示例，灵活 | 推理质量不如few-shot稳定 |

**为什么CoT有效？（三个角度）**

1. **计算量角度**：中间步骤增加了模型的有效计算深度，相当于隐式扩展了模型容量
2. **错误修正角度**：每个中间步骤都可以自我验证，早期错误被修正后不会传播到最终结果
3. **注意力引导角度**：中间步骤为后续生成提供了"锚点"，让注意力聚焦在相关信息上

**实验证据（Wei et al. 2022）：**
- 数学推理任务（GSM8K）：标准Prompt 17.9% → CoT 56.4%（+38.5%）
- 常识推理（CommonsenseQA）：标准Prompt 71.2% → CoT 78.4%
- **模型规模效应**：CoT在小模型（<100B）上几乎无效，在大模型（>100B）上效果显著

---

## 三、广告创意生成的CoT适配

### 3.1 多阶段拆解框架

**系统设计：将模糊诉求拆解为4个明确子问题**

```python
COT_SYSTEM_PROMPT = """你是专业的广告创意策略师。
收到广告需求后，必须按以下4个步骤思考，每步输出具体结论：

步骤1【理解产品】：
- 这是什么产品？
- 核心卖点和差异化优势是什么？
- 产品定位（价位、调性）？

步骤2【理解用户】：
- 目标用户画像（年龄、性别、职业、消费水平）？
- 用户核心痛点和需求？
- 用户在什么场景下看到这个广告？

步骤3【确定策略】：
- 情感诉求 vs 功效诉求 vs 社交认同，选哪个方向？
- 广告的钩子（Hook）是什么？（前3秒抓住注意力）
- 文案风格：直白/幽默/文艺/专业？

步骤4【生成创意】：
- 基于步骤1-3，生成5条不同风格的文案
- 每条标注：风格标签、适合人群、预估吸引力（1-10分）
"""
```

**完整CoT流程示例：**

```
原始诉求：「帮我写一个效果好的推广文案，产品是男士护肤套装，定价999元」

=== Step 1 - 理解产品 ===
产品：中高端男士护肤套装（水乳防晒三件套）
核心卖点：
  - 专为男性皮肤设计（控油、抗氧化）
  - 一套搞定护肤全流程（简单省时）
  - 科学配方，皮肤科医生背书
产品定位：中高端（999元），目标男性对自身形象有要求

=== Step 2 - 理解用户 ===
目标用户：25-35岁职场男性，城市白领
核心痛点：
  - 皮肤问题影响形象（油光、暗沉、粗糙）
  - 不懂护肤，选择困难（太多产品不知道用什么）
  - 没时间（护肤步骤太复杂）
场景：下班后刷手机，刷到短视频广告

=== Step 3 - 确定策略 ===
策略选择：社交认同 + 功效简化
理由：
  - 男性购买护肤品有"面子压力"，社交认同降低决策障碍
  - "简单"是男性护肤最大需求，功效简化是核心卖点
Hook方向：「一套三件，3分钟搞定一整天的面子工程」
风格：自信、简洁、有点幽默

=== Step 4 - 生成创意 ===
1. 【社交认同型】「10万+男生都在偷偷用的护肤秘密，你还不知道？」
   风格：悬念+好奇，适合年轻职场男性，吸引力8/10

2. 【效率型】「早上3分钟，一整天都是最在状态的你」
   风格：实用主义，强调省时，适合30+忙碌男性，吸引力7/10

3. 【结果导向型】「面试前一晚用上，第二天面试官看你的眼神都不一样」
   风格：场景化，有代入感，适合求职场景，吸引力9/10

4. 【产品专业型】「皮肤科医生同款套装，解决男性皮肤3大问题」
   风格：权威背书，适合注重品质的用户，吸引力7/10

5. 【价值锚定型】「1000元，但你的形象值3000」
   风格：价值对比，适合价格敏感但在意形象的用户，吸引力8/10
```

### 3.2 Python实现

```python
from dataclasses import dataclass
from typing import List
import json

@dataclass
class CotResult:
    product_analysis: str
    user_analysis: str
    strategy: str
    creatives: List[dict]
    raw_output: str

def generate_creative_with_cot(
    product_info: dict,
    user_profile: dict,
    platform: str,
    llm_client,
    reference_creatives: List[str] = None  # RAG检索到的参考创意
) -> CotResult:
    """
    使用CoT生成广告创意
    """
    # 构建参考创意上下文
    reference_context = ""
    if reference_creatives:
        reference_context = "\n参考历史高效创意：\n" + "\n".join(
            [f"- {c}" for c in reference_creatives[:5]]
        )
    
    prompt = f"""{COT_SYSTEM_PROMPT}

商品信息：
- 名称：{product_info['title']}
- 类别：{product_info['category']}
- 定价：{product_info['price']}元
- 卖点：{product_info.get('features', '未知')}

广告主诉求：{product_info.get('advertiser_brief', '效果好')}

目标平台：{platform}（{get_platform_constraints(platform)}）
{reference_context}

请按步骤1→2→3→4逐步分析，最终输出JSON格式的5条创意。
"""
    
    response = llm_client.generate(prompt, temperature=0.7)
    
    # 解析CoT输出
    return parse_cot_response(response)

def get_platform_constraints(platform: str) -> str:
    constraints = {
        "douyin": "文案 ≤ 35字，需要钩子（前5字吸引眼球）",
        "jd": "文案 ≤ 60字，强调促销和品质",
        "xiaohongshu": "文案 150-300字，种草风格，真实感强",
        "weixin": "文案 ≤ 80字，强调品牌信任"
    }
    return constraints.get(platform, "标准文案，≤ 50字")
```

---

## 四、Self-Consistency 在创意多样性中的应用

### 4.1 原理（Wang et al. 2022）

**核心思想：** 同一问题用不同推理路径（高温度采样）得到多个答案，通过"多数投票"选最一致的答案

```
传统CoT（贪婪解码）：
  问题 → 推理路径A → 答案X

Self-Consistency（多次采样）：
  问题 → 推理路径A → 答案X
       → 推理路径B → 答案X  （多数）
       → 推理路径C → 答案Y
  最终答案：X（3次中2次）
```

### 4.2 广告创意生成适配

**核心改动：用CTR预估模型代替"投票"机制**

```python
import asyncio
from typing import List

async def self_consistency_creative(
    product_info: dict,
    user_profile: dict,
    llm_client,
    ctr_predictor,
    n_samples: int = 5,    # 采样次数
    temperature: float = 1.2,  # 高温度增加多样性
    final_k: int = 3       # 最终选取条数
) -> List[dict]:
    """
    Self-Consistency创意生成：
    多次高温度采样 → CTR预估筛选（代替投票）
    """
    # 并行生成N条候选创意
    tasks = [
        generate_creative_with_cot(
            product_info, user_profile,
            platform="douyin",
            llm_client=llm_client
        )
        for _ in range(n_samples)
    ]
    all_results = await asyncio.gather(*tasks)
    
    # 收集所有候选创意
    all_creatives = []
    for result in all_results:
        all_creatives.extend(result.creatives)
    
    # 用CTR预估模型打分（代替投票）
    for creative in all_creatives:
        creative["predicted_ctr"] = ctr_predictor.predict(
            creative["text"], product_info, user_profile
        )
    
    # 去重 + 按CTR排序，取top-K
    deduplicated = deduplicate_by_similarity(all_creatives, threshold=0.85)
    final_creatives = sorted(
        deduplicated,
        key=lambda x: x["predicted_ctr"],
        reverse=True
    )[:final_k]
    
    return final_creatives

def deduplicate_by_similarity(
    creatives: List[dict],
    threshold: float = 0.85
) -> List[dict]:
    """
    基于语义相似度去重，避免返回同质化结果
    """
    selected = []
    embeddings = [get_embedding(c["text"]) for c in creatives]
    
    for i, creative in enumerate(creatives):
        is_duplicate = False
        for j, selected_creative in enumerate(selected):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > threshold:
                # 如果重复且分数更高，替换
                if creative["predicted_ctr"] > selected_creative["predicted_ctr"]:
                    selected[j] = creative
                is_duplicate = True
                break
        
        if not is_duplicate:
            selected.append(creative)
    
    return selected
```

**温度参数的影响：**
| 温度 | 输出特点 | 适用场景 |
|------|---------|---------|
| T=0.7 | 高质量但相似 | 保守创意（大品牌） |
| T=1.0 | 平衡 | 默认设置 |
| T=1.2 | 多样性高但质量波动 | Self-Consistency多次采样 |
| T=1.5+ | 过于随机 | 通常不推荐 |

---

## 五、Prompt工程细节

### 5.1 System Prompt结构设计

```python
SYSTEM_PROMPT_TEMPLATE = """
# 角色定义
你是{platform}平台的专业广告文案创意师，擅长{category}品类的文案创作。

# 输出格式要求
- 字数限制：{max_chars}字以内
- 风格要求：{style_requirements}
- 禁用词：{forbidden_words}

# 质量标准
- 文案必须有明确的行动号召（CTA）
- 避免绝对化用语（"最好"、"第一"等违规词）
- 每条创意的风格和切入角度必须不同

# CoT思考格式
请严格按步骤1→2→3→4输出，每步用【步骤X】标注。
"""

def build_system_prompt(
    platform: str,
    category: str,
    advertiser_preferences: dict
) -> str:
    forbidden = advertiser_preferences.get("forbidden_words", [])
    return SYSTEM_PROMPT_TEMPLATE.format(
        platform=platform,
        category=category,
        max_chars=get_platform_limits(platform),
        style_requirements=get_style_guide(platform),
        forbidden_words="、".join(forbidden) if forbidden else "无特殊要求"
    )
```

### 5.2 Few-shot示例选择策略

```python
def select_few_shot_examples(
    product_info: dict,
    example_pool: List[dict],   # 历史高CTR创意+CoT推理过程
    n_examples: int = 3
) -> List[dict]:
    """
    从历史CoT示例库中选择最相关的few-shot示例
    策略：
    1. 同品类优先
    2. 高CTR（> 均值1.5倍）
    3. 风格多样（覆盖情感/功效/社交3种方向）
    """
    # 按品类过滤
    same_category = [e for e in example_pool
                     if e["category"] == product_info["category"]]
    
    if len(same_category) < n_examples:
        # 品类不足时，补充相近品类
        same_category += get_similar_category_examples(product_info)
    
    # 按CTR排序，取高质量示例
    top_examples = sorted(same_category,
                          key=lambda x: x["ctr"],
                          reverse=True)[:10]
    
    # 从top-10中选3条，确保风格多样
    selected = select_diverse_examples(top_examples, n=n_examples)
    return selected
```

### 5.3 约束注入

```python
PLATFORM_CONSTRAINTS = {
    "douyin": {
        "max_chars": 35,
        "forbidden_patterns": ["最.*?款", "第一", "百分百"],
        "required_elements": ["行动号召"],
        "style": "短促、有冲击力、口语化"
    },
    "jd": {
        "max_chars": 60,
        "forbidden_patterns": ["最低价", "全网最"],
        "required_elements": ["卖点", "价格引导"],
        "style": "清晰、专业、强调品质"
    }
}
```

---

## 六、常见考点 Q&A

**Q1：CoT在广告创意生成中能带来多少提升？如何量化？**

A：实验数据表明，CoT对广告创意的提升主要体现在：(1) CTR预估分提升10~25%——因为文案更贴合用户痛点；(2) 广告主满意度提升30~40%——文案更符合其实际意图（即使他们自己没说清楚）；(3) 修改率降低50%——广告主需要反复修改的次数减少。量化方法：内部A/B测试，实验组用CoT框架生成，对照组直接Prompt，用CTR预估模型（离线）和真实CTR（在线）双重评估。

**Q2：Zero-shot CoT vs Few-shot CoT 在广告场景哪个更适用？**

A：广告场景更适合Few-shot CoT，原因有三：(1) 广告文案有平台规范和风格要求，Zero-shot CoT容易忽略这些隐性约束；(2) 广告场景的CoT步骤需要遵循固定框架（产品→用户→策略→创意），few-shot示例能强制对齐这个框架；(3) 广告主的预期有行业偏好（如电商文案的特定风格），需要示例来传递这些"业内共识"。Zero-shot CoT适合用在冷启动期（无示例库）或需要最大创意自由度的场景。

**Q3：如何防止CoT中间步骤产生幻觉？**

A：四个防幻觉措施：(1) 步骤验证——让LLM每完成一步后检查与商品信息的一致性，例如"Step 1产品分析中的卖点是否来自商品描述？"；(2) 事实锚定——System Prompt中明确"所有描述必须基于以下已知商品信息，不得编造"；(3) 可信度打分——每个中间步骤要求LLM给出置信度（1-10），低置信度的步骤触发人工审核；(4) 结构化约束——用JSON格式输出中间步骤，格式约束减少自由发挥空间。

**Q4：Self-Consistency的采样次数怎么设置（成本vs质量权衡）？**

A：边际效益递减定律适用。从1次到3次采样，质量提升最显著（约15~20%）；从3次到5次，质量再提升5~8%；5次以上提升接近0。广告场景推荐：对高价值广告主（月消耗 > 10万）用5次采样；普通广告主3次；小广告主1次（直接用），用预测CTR动态决定采样次数。成本估算：每次LLM调用含CoT约消耗2000 tokens，每条广告5次采样 = 1万tokens ≈ $0.05，10万广告/天 = $5000/天，需要谨慎控制。

**Q5：CoT会增加多少延迟？如何优化？**

A：标准CoT生成4步约增加2~4秒延迟（相比直接生成）。三种优化方式：(1) 异步预生成——广告主上传商品后，在后台预先完成CoT分析（step1~3），广告主看到"生成中"时实际已完成前3步，最后一步（step4文案生成）在用户点击时完成，用户感知延迟降到1秒内；(2) 流式输出——文案逐步显示，用户感知更快；(3) 步骤缓存——同品类的Step 1~2（产品/用户分析）结果缓存24小时，复用给同品类其他广告主。

**Q6：Tree of Thoughts（ToT）在创意生成中有没有应用价值？**

A：ToT理论上有价值但工程落地困难。ToT允许在每个思考步骤进行多路分支和回溯（类似树搜索），适合需要大量探索的复杂问题。广告创意生成的应用场景：当Step 3（策略方向）有多个可行路径时（情感型、功效型、故事型），ToT可以并行探索所有分支，每条路径完整生成一套文案后用CTR模型评估，选最优路径。主要缺点：API调用次数是CoT的3~5倍，成本和延迟不可接受。建议用Self-Consistency代替（效果相近，成本更低）。

**Q7：如何确保CoT的每一步都在广告平台约束内？**

A：在每个CoT步骤的输出后插入"约束检查"子步骤：(1) 字数约束——在Step 4生成完每条文案后，立即检查`len(text) <= max_chars`，超限则缩短；(2) 违禁词检查——用正则表达式匹配平台违禁词库，触发时要求LLM重写该条；(3) 步骤内置约束——在System Prompt中明确"Step 4每条文案必须附上自查结果：[字数: X/35] [违禁词: 无/XX] [CTA: 有/无]"，让LLM自我审查；(4) 后置过滤——CoT输出后统一过广告合规API（如平台提供的审核接口），不合规的返回null，触发重新生成。

---

*参考文献：*
- *Chain-of-Thought Prompting: Wei et al., NeurIPS 2022, arXiv:2201.11903*
- *Self-Consistency: Wang et al., ICLR 2023, arXiv:2203.11171*
- *Tree of Thoughts: Yao et al., NeurIPS 2023*
