# 分词方法全景对比 — 面试向深度总结

> 标签：#tokenization #BPE #WordPiece #SentencePiece #Unigram #NLP #面试

---

## 总对比表

| 方法 | 年份 | 提出者/论文 | 核心思想 | 词表大小(典型) | 代表模型 | 优点 | 缺点 |
|------|------|------------|---------|---------------|---------|------|------|
| Word-level | — | 传统 NLP | 按空格/规则切分为完整词 | 50K–200K | word2vec, GloVe | 语义完整、直觉 | OOV 严重、词表爆炸 |
| Character-level | — | 传统 NLP | 以单个字符为 token | 几百 | CharCNN (Zhang et al. 2015) | 零 OOV | 序列过长、语义密度低 |
| BPE | 2016 | Sennrich et al. | 自底向上合并最高频 bigram | 30K–50K | GPT-2, GPT-3, LLaMA, RoBERTa | 平衡词表与 OOV | 贪心合并、非概率 |
| Byte-level BPE | 2019 | Radford et al. (GPT-2) | BPE 以 byte 为基本单元，256 base vocab | 50K–100K | GPT-2, GPT-3, GPT-4, LLaMA | 真正零 OOV、多语言友好 | 常见词也需多步 merge |
| WordPiece | 2012/2016 | Schuster & Nakajima / Wu et al. | 用似然增益选 merge（非频率） | 30K | BERT, DistilBERT, ELECTRA | 概率最优 merge | 训练较慢 |
| Unigram LM | 2018 | Kudo (SentencePiece) | 自顶向下裁剪大词表（EM） | 32K | T5, ALBERT, mBART, XLNet | 概率化、多切分采样 | 实现复杂 |
| SentencePiece | 2018 | Kudo & Richardson | 工具库，封装 BPE + Unigram | — | LLaMA, T5, ALBERT | 语言无关、可逆 | 非新算法 |
| tiktoken | 2022 | OpenAI | Byte-level BPE 的 Rust 实现 | 100K–200K | GPT-4, ChatGPT, o1 | 极快编解码 | 闭源编码表 |

---

## 1. Word-level Tokenization（词级别）

### 算法原理

最朴素的分词方式：按空格和标点将文本切分为完整的词。

```
Input:  "The cat sat on the mat."
Output: ["The", "cat", "sat", "on", "the", "mat", "."]
```

英文中直接按空格拆分基本可用；中文等无空格语言则需依赖分词工具（jieba、pkuseg）。

**规则分词变体：**
- **空格分词**：最简单，`text.split()`
- **正则分词**：`\w+` 匹配，处理标点和缩写
- **语言学分词**：利用词典 + 规则（如 Moses tokenizer、Stanford tokenizer）

### OOV 问题

Word-level 的致命缺陷：

$$P(\text{OOV}) = 1 - \frac{|\text{vocab} \cap \text{test words}|}{|\text{test words}|}$$

- 训练词表 50K 时，新闻文本 OOV 率约 2-5%，社交媒体可达 10%+
- OOV 词统一映射到 `<UNK>`，丢失全部语义信息
- 长尾词（人名、专有名词、新造词）几乎不可能覆盖

### 文献溯源

- 空格分词是传统 NLP 的默认方式，无特定论文
- 规则分词器：Koehn et al. (2007) Moses, Manning et al. (2014) Stanford CoreNLP

### 代表模型

word2vec (Mikolov et al., 2013), GloVe (Pennington et al., 2014), ELMo (Peters et al., 2018)

### 优缺点

| 优点 | 缺点 |
|------|------|
| 语义完整，每个 token 有明确含义 | OOV 问题严重 |
| 实现简单，人类可读 | 词表巨大（>100K），embedding 矩阵占大量参数 |
| 与传统 NLP 工具链兼容 | 无法处理形态变化（run/running/runs 是三个独立 token） |

### 面试高频问题

**Q1：Word-level tokenization 的核心问题是什么？如何解决？**

> 核心问题是 OOV。解决方案有三个方向：(1) 增大词表 — 但参数量和稀疏性同步增长；(2) 使用 subword 方法（BPE/WordPiece/Unigram）在词和字符之间找平衡；(3) character-level 模型完全消除 OOV 但序列过长。工业界主流选择 subword。

**Q2：为什么现代 LLM 不用 word-level？**

> 两个原因：(1) 词表大小不可控——英文常见词就有 50K+，加上多语言和专业术语会爆炸到百万级，embedding 矩阵参数量不可接受；(2) 无法泛化到词的形态变化和新造词，而 subword 天然支持组合。

---

## 2. Character-level Tokenization（字符级别）

### 算法原理

将文本拆分为单个字符（或 Unicode code point）：

```
Input:  "lowest"
Output: ["l", "o", "w", "e", "s", "t"]
```

词表极小（英文仅 26 字母 + 数字 + 标点 ≈ 100-300 tokens），理论上零 OOV。

### 序列长度问题

一个关键的数量级估算：

$$\text{序列长度比} = \frac{\text{字符数}}{\text{词数}} \approx 4\text{-}5 \times \text{（英文）}$$

- "The cat sat on the mat" → word-level: 6 tokens, char-level: 22 tokens
- Transformer 的自注意力复杂度 $O(n^2)$，序列长 4x 意味着计算量 16x
- 信息密度低：单个字符几乎不携带语义，模型必须自己学会从字符组合中提取语义

### 文献溯源

- Zhang et al. (2015) "Character-level Convolutional Networks for Text Classification" — 首个证明字符级 CNN 可与词级模型媲美
- Kim et al. (2016) "Character-Aware Neural Language Models" — 字符级 CNN + 词级 LSTM 混合架构

### 代表模型

CharCNN (Zhang et al., 2015), ELMo 底层 CharCNN (Peters et al., 2018), ByT5 (Xue et al., 2022)

### 优缺点

| 优点 | 缺点 |
|------|------|
| 零 OOV，任何文本都能处理 | 序列长度膨胀 4-5 倍 |
| 词表极小，embedding 矩阵参数少 | 自注意力 $O(n^2)$ 计算量 16-25 倍增长 |
| 天然捕捉形态学信息（前缀、后缀） | 单字符语义信息密度极低 |
| 对拼写错误/噪声文本鲁棒 | 长距离依赖更难学习 |

### 面试高频问题

**Q1：ByT5 是如何使字符级模型可行的？**

> ByT5 (Xue et al., 2022) 用 byte 而非 Unicode 字符作为 token（词表固定 256），并通过不对称架构缓解长序列问题：encoder 深（12 层）而 decoder 浅（4 层），因为 encoder 处理长 byte 序列的负担更重。实验表明在小模型和噪声数据上 ByT5 优于 mT5。

---

## 3. BPE (Byte Pair Encoding)

### 算法原理

BPE 是自底向上的贪心合并算法：从字符级出发，反复合并语料中出现频率最高的相邻 token 对，直到达到目标词表大小。

**以 "lowest" 为例的 merge 过程：**

假设训练语料中各词频率为：`low(5), lowest(2), newer(6), wider(3)`

初始词表（字符级 + 词尾标记 `</w>`）：

```
l o w </w>    ×5
l o w e s t </w>  ×2
n e w e r </w>    ×6
w i d e r </w>    ×3
```

迭代 merge：

| 步骤 | 最高频 pair | 频率 | 合并后新 token |
|------|-----------|------|--------------|
| 1 | (e, r) | 6+3=9 | er |
| 2 | (er, \</w\>) | 6+3=9 | er\</w\> |
| 3 | (l, o) | 5+2=7 | lo |
| 4 | (lo, w) | 5+2=7 | low |
| 5 | (n, e) | 6 | ne |
| 6 | (ne, w) | 6 | new |
| 7 | (new, er\</w\>) | 6 | newer\</w\> |
| ... | ... | ... | ... |

### 训练伪代码

```python
def train_bpe(corpus, num_merges):
    """BPE 训练算法"""
    # Step 1: 初始化 — 将所有词拆成字符序列
    vocab = defaultdict(int)
    for word, freq in word_freq(corpus).items():
        vocab[' '.join(list(word)) + ' </w>'] = freq

    merge_rules = []
    for i in range(num_merges):
        # Step 2: 统计所有相邻 pair 的频率
        pairs = get_pair_stats(vocab)
        if not pairs:
            break

        # Step 3: 选频率最高的 pair
        best_pair = max(pairs, key=pairs.get)
        merge_rules.append(best_pair)

        # Step 4: 在所有词中执行 merge
        vocab = merge_vocab(best_pair, vocab)

    return merge_rules

def apply_bpe(word, merge_rules):
    """BPE 编码（推理时）"""
    tokens = list(word) + ['</w>']
    for pair in merge_rules:  # 按训练时的顺序逐条应用
        i = 0
        while i < len(tokens) - 1:
            if (tokens[i], tokens[i+1]) == pair:
                tokens[i:i+2] = [tokens[i] + tokens[i+1]]
            else:
                i += 1
    return tokens
```

**关键细节：** 推理时必须按训练时的 merge 顺序逐条应用规则，不能直接查表。这保证了确定性编码。

### 编码示例

训练好的 BPE（GPT-2 风格）对 "lowest" 的切分：

```
"lowest" → ["low", "est"]
```

因为 "low" 和 "est" 都是高频 subword，在训练时被合并成了完整 token。而罕见词如 "lowest-priced" 可能被切为 `["low", "est", "-", "pr", "iced"]`。

### 文献溯源

- 原始 BPE 算法：Gage (1994) "A New Algorithm for Data Compression" — 用于数据压缩
- 引入 NLP：Sennrich et al. (2016) "Neural Machine Translation of Rare Words with Subword Units" — 首次将 BPE 用于 NMT，彻底改变了分词范式

### 代表模型

GPT (Radford et al., 2018), GPT-2 (Radford et al., 2019), GPT-3 (Brown et al., 2020), RoBERTa (Liu et al., 2019), LLaMA (Touvron et al., 2023)

### 优缺点

| 优点 | 缺点 |
|------|------|
| 有效平衡词表大小与 OOV | 贪心合并，非全局最优 |
| 高频词保持完整，低频词拆为 subword | 同一词在不同上下文中切分相同（确定性，无法概率采样） |
| 算法简单、训练高效 | 对语料统计高度依赖 |
| 工业界验证充分 | merge 顺序固定，推理不够灵活 |

### 面试高频问题

**Q1：BPE 的时间复杂度是多少？如何加速？**

> 朴素 BPE 训练：每轮需遍历所有词统计 pair 频率，$O(N \cdot V)$（$N$=词数，$V$=当前平均 token 数/词）。总复杂度 $O(M \cdot N \cdot V)$（$M$=merge 次数）。加速方法：(1) 用 priority queue 维护 pair 频率，增量更新；(2) 并行化统计；(3) tiktoken 用 Rust 实现编码加速。

**Q2：BPE 能处理新语言吗？如果训练语料主要是英文？**

> BPE 完全依赖训练语料的统计。如果语料以英文为主，其他语言的 subword 粒度会很粗（可能退化到字符级），导致序列过长、效率低。解决方案：(1) 多语言均衡语料训练；(2) 用 Byte-level BPE 保证至少字节级覆盖；(3) 为特定语言增量扩展词表。

---

## 4. Byte-level BPE

### 算法原理

与标准 BPE 的唯一区别：**基本单元从 Unicode 字符变为 byte（0x00-0xFF）**。

```
标准 BPE:  基本词表 = Unicode 字符集（可能数万个）
Byte BPE:  基本词表 = 256 个 byte + 特殊 token
```

任何文本（任何语言、emoji、代码、二进制）都能被表示为 byte 序列，因此 **真正实现零 OOV**。

### 与 BPE 的关键区别

| 维度 | 标准 BPE | Byte-level BPE |
|------|---------|---------------|
| 基本单元 | Unicode 字符 | byte (0-255) |
| 初始词表大小 | 数千（取决于语言） | 固定 256 |
| OOV 可能性 | 理论上存在（未见过的 Unicode） | 完全不可能 |
| 多语言支持 | 需要覆盖各语言字符 | 天然支持一切语言 |
| 中文处理 | 一个汉字 = 1 个初始 token | 一个汉字 = 3 个 byte（UTF-8） |

### GPT-2 的处理方式

GPT-2 首次采用 Byte-level BPE，但为了可读性做了一个巧妙映射：将 256 个 byte 映射到可打印的 Unicode 字符。例如空格 (0x20) 被映射为 `Ġ`。

```python
# GPT-2 的 byte-to-unicode 映射（简化）
# 0x20 (space) → Ġ
# 0x0A (newline) → Ċ
# 这样 tokenizer 的 merge 操作仍然在 "字符" 上进行
# 但底层实际是 byte 序列
```

### 编码示例

```
"Hello 你好"
→ UTF-8 bytes: [72, 101, 108, 108, 111, 32, 228, 189, 160, 229, 165, 189]
→ 经 BPE merge 后: ["Hello", " ä½ ", "好"]  (实际切分取决于训练数据)
```

### 文献溯源

- Radford et al. (2019) "Language Models are Unsupervised Multitask Learners"（GPT-2 论文）首次在大规模 LM 中采用 Byte-level BPE
- Wang et al. (2020) 进一步分析了 byte-level 方法在多语言场景的优势

### 代表模型

GPT-2, GPT-3, GPT-4, LLaMA, LLaMA-2, CodeLlama, Mistral, Falcon

### 优缺点

| 优点 | 缺点 |
|------|------|
| 真正零 OOV | 非 ASCII 文本（中日韩）的 token 效率较低 |
| 固定 256 base vocab，无需预处理 | 同一语义单元可能跨多个 byte token |
| 代码、多语言、emoji 统一处理 | merge 后的 token 可能不对应语义边界 |
| 可逆编解码，无信息损失 | 训练语料的语言分布仍然影响效率 |

### 面试高频问题

**Q1：Byte-level BPE 中，中文一个字需要多少 token？**

> UTF-8 编码下，一个中文字占 3 个 byte。如果训练语料中有足够的中文，BPE 的 merge 操作会将这 3 个 byte 合并为 1 个 token（常见字）或 2 个 token（罕见字）。但如果中文语料极少（如 GPT-2 原始模型），一个中文字可能需要 2-3 个 token，导致中文推理成本是英文的 2-3 倍。这就是为什么 LLaMA-2 中文版需要扩展词表。

---

## 5. WordPiece

### 算法原理

WordPiece 与 BPE 同为自底向上 merge，但 **选择 merge pair 的标准不同**：

- **BPE**：选频率最高的 pair
- **WordPiece**：选使语言模型似然增益最大的 pair

似然增益公式：

$$\text{score}(x, y) = \frac{P(xy)}{P(x) \cdot P(y)}$$

即合并后的 bigram 概率除以两个 unigram 概率的乘积。这等价于逐点互信息（PMI）：

$$\text{PMI}(x, y) = \log \frac{P(xy)}{P(x) \cdot P(y)}$$

**直觉：** 频率高但各自也很常见的 pair（如 "t" + "h"）PMI 不一定高；而频率中等但高度共现的 pair（如 "##ing"）PMI 更高。WordPiece 偏好后者。

### `##` 前缀标记

WordPiece 用 `##` 标记一个 subword 是某词的 **continuation**（非首 token）：

```
"unbelievable" → ["un", "##believ", "##able"]
"lowest"       → ["low", "##est"]
```

这使得解码时可以无歧义地重建原始词：连续的 `##` token 属于同一个词。

### 编码示例（对比 BPE）

```
BPE:       "unhappiness" → ["un", "happiness"]  (如果 happiness 是完整 token)
WordPiece: "unhappiness" → ["un", "##happ", "##iness"]  (##标记 subword continuation)
```

### 文献溯源

- Schuster & Nakajima (2012) "Japanese and Korean Voice Search" — 首次提出 WordPiece 用于语音搜索
- Wu et al. (2016) "Google's Neural Machine Translation System" — 将 WordPiece 扩展到 NMT，奠定现代用法

### 代表模型

BERT (Devlin et al., 2019), DistilBERT (Sanh et al., 2019), ELECTRA (Clark et al., 2020), MobileBERT

### 优缺点

| 优点 | 缺点 |
|------|------|
| PMI 选择 merge，比频率更优 | 训练需要计算似然，比 BPE 慢 |
| `##` 标记使得 word boundary 明确 | 似然计算依赖语言模型估计 |
| BERT 验证了其在理解任务上的效果 | 实现比 BPE 复杂 |

### 面试高频问题

**Q1：BPE 和 WordPiece 的核心区别是什么？**

> 一句话：BPE 按频率合并，WordPiece 按似然增益（PMI）合并。BPE 倾向合并出现次数最多的 pair，即使它们独立出现也很频繁；WordPiece 倾向合并那些共现远超独立概率之积的 pair，即真正具有内聚性的 subword。实际效果上两者差异不大，但 WordPiece 在理论上更优。

**Q2：BERT 的词表是怎么构建的？**

> BERT 使用 WordPiece，词表大小 30,522。训练数据是 BookCorpus + English Wikipedia。特殊 token 包括 `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`。词表构建后固定不变。由于训练数据以英文为主，multilingual BERT 的中文/日文等覆盖较粗，导致非英文文本的 token 效率低。

---

## 6. Unigram Language Model

### 算法原理

与 BPE/WordPiece 的 **根本区别**：

| 维度 | BPE / WordPiece | Unigram LM |
|------|----------------|------------|
| 方向 | 自底向上（从字符开始合并） | 自顶向下（从大词表开始裁剪） |
| 初始状态 | 字符级词表 | 巨大的候选词表（数百万） |
| 操作 | 添加新 token（merge） | 删除 token（prune） |
| 概率模型 | 无（BPE）/ 近似（WordPiece） | 完整的 unigram LM |
| 切分方式 | 确定性（唯一切分） | 概率性（可采样多种切分） |

### EM 算法训练过程

Unigram 假设每个 token 独立出现，文本的概率为各 token 概率之积：

$$P(\mathbf{x}) = \prod_{i=1}^{n} P(x_i)$$

对一个句子 $S$，其最优切分为：

$$\mathbf{x}^* = \arg\max_{\mathbf{x} \in \mathcal{S}(S)} P(\mathbf{x}) = \arg\max_{\mathbf{x}} \sum_{i} \log P(x_i)$$

其中 $\mathcal{S}(S)$ 是 $S$ 的所有可能切分。用 Viterbi 算法高效求解。

**训练算法（EM）：**

```python
def train_unigram(corpus, initial_vocab_size, target_vocab_size):
    """Unigram Language Model 训练"""
    # Step 1: 初始化 — 生成大量候选 subword
    # 用 suffix array 或 BPE 预训练获取候选词表
    vocab = initialize_large_vocab(corpus, initial_vocab_size)  # ~1M tokens

    while len(vocab) > target_vocab_size:
        # E-step: 用 Viterbi 算法，基于当前 vocab 概率
        #         找出每个句子的最优切分
        for sentence in corpus:
            best_segmentation = viterbi(sentence, vocab)

        # M-step: 基于最优切分，重新估计每个 token 的概率
        #         P(x_i) = count(x_i) / total_tokens
        vocab = update_probabilities(vocab, corpus)

        # Prune: 计算每个 token 的 loss 贡献
        #         如果移除某 token 后总 loss 增量最小，则移除它
        losses = {}
        for token in vocab:
            losses[token] = compute_loss_increase_if_removed(token, vocab, corpus)

        # 移除 loss 增量最小的 20% token（保留字符级 token）
        vocab = prune_bottom_20_percent(vocab, losses)

    return vocab
```

### 概率性切分的优势

Unigram 的一个独特能力是 **subword regularization**：训练时对同一句子采样不同切分，作为数据增强。

```
"international" 的多种切分：
P=0.4:  ["inter", "national"]
P=0.3:  ["in", "terna", "tion", "al"]
P=0.2:  ["inter", "na", "tion", "al"]
P=0.1:  ["international"]
```

训练时随机采样切分，等效于正则化，提升模型鲁棒性。这是 BPE 做不到的。

### 文献溯源

- Kudo (2018) "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates" — 提出 Unigram LM 和 subword regularization

### 代表模型

T5 (Raffel et al., 2020), ALBERT (Lan et al., 2020), mBART (Liu et al., 2020), XLNet (Yang et al., 2019), mT5

### 优缺点

| 优点 | 缺点 |
|------|------|
| 概率模型，理论优雅 | 训练比 BPE 慢（EM 迭代） |
| 支持多种切分采样（subword regularization） | 初始候选词表构建需要额外步骤 |
| 裁剪法可以更全局地优化词表 | 实现比 BPE 复杂得多 |
| 多语言效果好 | 工业界用 BPE 的更多（惯性） |

### 面试高频问题

**Q1：Unigram LM 为什么是自顶向下？有什么好处？**

> BPE 从空开始自底向上添加 token，每步贪心地选最优 merge，但前期决策会影响后期——一旦合并就无法撤销。Unigram 从一个庞大候选集开始，逐步裁剪价值最低的 token，相当于在更大搜索空间中优化，不容易陷入贪心局部最优。此外，自顶向下保留了概率模型，使得 subword regularization 成为可能。

**Q2：什么是 subword regularization？有什么用？**

> 在训练时，对同一个词/句子按概率采样不同的 subword 切分方式（而非固定用 Viterbi 最优切分）。这相当于数据增强——模型在训练中看到同一个词的多种 subword 表示，学到更鲁棒的 subword embedding。Kudo (2018) 实验表明 subword regularization 在低资源翻译任务上提升 1-2 BLEU。

---

## 7. SentencePiece

### 定位：工具库，非新算法

**SentencePiece 不是一种新的分词算法**，而是一个将 BPE 和 Unigram 封装在一起的开源工具库。

```
SentencePiece = {BPE, Unigram} × {语言无关} × {可逆} × {直接处理 raw text}
```

### 关键特性

**1. 语言无关（Language-agnostic）**

不需要预分词（pre-tokenization）。传统 BPE 实现（如 subword-nmt）先按空格切词再做 BPE；SentencePiece 直接在 raw byte 流上操作，对中日韩等无空格语言天然友好。

```python
# 传统 BPE（需要预分词）
text = "I love NLP"
words = text.split()  # ["I", "love", "NLP"] — 依赖空格
bpe_tokens = [apply_bpe(w) for w in words]

# SentencePiece（无需预分词）
text = "I love NLP"
tokens = sp.encode(text)  # 直接处理，空格被视为特殊字符 ▁
# ["▁I", "▁love", "▁N", "LP"]
```

**2. 可逆编解码**

使用特殊字符 `▁`（U+2581，LOWER ONE EIGHTH BLOCK）表示空格/词边界：

```
原文:  "New York is big"
编码:  ["▁New", "▁York", "▁is", "▁big"]
解码:  将 ▁ 替换为空格 → "New York is big"  ✓ 完全可逆
```

**3. 直接处理 raw text**

输入可以是任意原始文本文件，不需要任何预处理（小写化、Unicode 归一化等）。SentencePiece 在内部处理一切。

### 使用方式

```python
import sentencepiece as spm

# 训练
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='my_model',
    vocab_size=32000,
    model_type='unigram'  # 或 'bpe'
)

# 加载和使用
sp = spm.SentencePieceProcessor(model_file='my_model.model')
tokens = sp.encode('Hello world', out_type=str)
# ['▁Hello', '▁world']

ids = sp.encode('Hello world', out_type=int)
# [8774, 1362]

text = sp.decode(ids)
# 'Hello world'  — 完全可逆
```

### 文献溯源

- Kudo & Richardson (2018) "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing"

### 代表模型

LLaMA (Touvron et al., 2023), T5 (Raffel et al., 2020), ALBERT (Lan et al., 2020), mBART, XLNet, Gemma

### 面试高频问题

**Q1：SentencePiece 和 Hugging Face tokenizers 库的区别？**

> SentencePiece 是 C++ 实现的独立工具，支持 BPE 和 Unigram，强调语言无关性和可逆性。Hugging Face tokenizers 是 Rust 实现的通用框架，支持更多算法（BPE, WordPiece, Unigram），但需要 pre-tokenizer 配置。LLaMA 和 T5 用 SentencePiece；BERT 用 WordPiece（Hugging Face 实现）。

---

## 8. tiktoken (OpenAI)

### 定位

tiktoken 是 OpenAI 开源的 **Byte-level BPE 编码器的 Rust 实现**，核心特点是极快的编解码速度。

### 核心特点

```python
import tiktoken

# 加载编码器
enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 使用的编码
tokens = enc.encode("Hello, world! 你好世界")
# [9906, 11, 1917, 0, 220, 57668, 53901, 3574, 244, 98220]

text = enc.decode(tokens)
# "Hello, world! 你好世界"  — 完全可逆
```

### 编码方案演进

| 编码名称 | 词表大小 | 使用模型 | 发布时间 |
|---------|---------|---------|---------|
| r50k_base | 50,257 | GPT-2, early GPT-3 | 2019 |
| p50k_base | 50,281 | code-davinci-002 | 2022 |
| cl100k_base | 100,256 | GPT-4, GPT-3.5, Embeddings v3 | 2023 |
| o200k_base | 200,019 | GPT-4o, o1, o3 | 2024 |

**词表扩大的趋势：** 从 50K→100K→200K，原因是：
1. 更大词表意味着更多常见词/短语被编码为单一 token → 序列更短 → 推理更快
2. 多语言覆盖需要更多 token（中日韩等 3-byte UTF-8 字符）
3. 代码场景需要更多 token（缩进、常见函数名/关键字）

### 性能对比

tiktoken vs SentencePiece vs Hugging Face tokenizers：

| 维度 | tiktoken | SentencePiece | HF tokenizers |
|------|---------|--------------|---------------|
| 实现语言 | Rust + Python binding | C++ + Python binding | Rust + Python binding |
| 编码速度 | 极快（~3-5x SentencePiece） | 中等 | 快（~2x SentencePiece） |
| 算法 | Byte-level BPE only | BPE + Unigram | BPE + WordPiece + Unigram |
| 训练支持 | 不支持（仅编解码） | 支持 | 支持 |
| 词表来源 | OpenAI 预定义 | 用户训练 | 用户训练或预定义 |

### 文献溯源

- tiktoken 无独立论文，是 OpenAI 的开源工程实现
- 底层算法仍是 Byte-level BPE (Radford et al., 2019)

### 代表模型

GPT-3.5, GPT-4, GPT-4o, o1, o3, ChatGPT 全系列, OpenAI Embeddings

### 面试高频问题

**Q1：为什么 GPT-4o 的词表扩大到 200K？这有什么影响？**

> 词表从 100K (cl100k_base) 扩大到 200K (o200k_base) 主要是为了提升多语言和代码场景的 token 效率。更大词表意味着更多常见片段被编码为单一 token，序列长度缩短，推理 KV cache 占用减少。代价是 embedding 矩阵参数量翻倍（200K × hidden_dim），但在大模型中这个增量可忽略（GPT-4 参数量远大于 embedding 层）。

---

## 9. BPE vs WordPiece vs Unigram 三方对比

### 核心差异

| 维度 | BPE | WordPiece | Unigram LM |
|------|-----|-----------|------------|
| **训练方向** | 自底向上（合并） | 自底向上（合并） | 自顶向下（裁剪） |
| **Merge 策略** | 频率最高的 pair | PMI 最高的 pair | N/A（不 merge，而是 prune） |
| **概率模型** | 无 | 近似（似然增益） | 完整 unigram LM |
| **切分方式** | 确定性 | 确定性 | 概率性（可采样） |
| **训练效率** | 最快 | 中等 | 最慢（EM 迭代） |
| **理论优雅度** | 低（纯贪心） | 中 | 高（有完整概率框架） |
| **工业使用率** | 最广（GPT 系列） | 中（BERT 系列） | 较少（T5 系列） |

### 面试一句话总结

> BPE 是频率驱动的贪心合并，WordPiece 是似然驱动的贪心合并，Unigram 是概率模型驱动的自顶向下裁剪。三者效果差异通常 <1%，但 Unigram 独有 subword regularization 能力。

### 选哪个？

```
需要简单快速 → BPE（大多数场景）
需要配合 BERT → WordPiece
需要 subword regularization / 多语言 → Unigram
需要语言无关 + 可逆 → SentencePiece (BPE 或 Unigram)
需要极速编解码 + OpenAI 兼容 → tiktoken
```

---

## 10. 中文分词的特殊性

### 字 vs 词的选择

中文没有天然的词边界，"分词"本身就是一个任务：

```
"我喜欢自然语言处理"
字级别: ["我", "喜", "欢", "自", "然", "语", "言", "处", "理"]
词级别: ["我", "喜欢", "自然语言处理"]
Subword: ["我", "喜欢", "自然", "语言", "处理"]  (取决于词表)
```

### 中文分词工具

| 工具 | 方法 | 特点 |
|------|------|------|
| jieba | 基于前缀词典 + HMM | 最流行、速度快、支持自定义词典 |
| pkuseg | 基于 CRF | 多领域模型、准确率更高 |
| LTP | 基于深度学习 | 功能全面（分词+词性+NER+...） |
| THULAC | 基于结构化感知机 | 清华出品、准确率高 |

### 现代 LLM 的中文分词

现代 LLM **不使用传统中文分词工具**，而是让 BPE/Unigram 直接学习：

- **BERT-Chinese**：字级别（每个汉字一个 token），简单粗暴但有效
- **LLaMA + 中文**：Byte-level BPE，每个汉字 2-3 个 token（效率低）
- **ChatGLM**：SentencePiece，中文训练语料充足，常见词合并为单一 token
- **Qwen**：BPE，中文词表大幅扩展，中文 token 效率高

### 面试高频问题

**Q1：为什么 BERT-Chinese 用字级别而非词级别？**

> 三个原因：(1) 中文分词本身就有歧义和错误，分词错误会直接传播到模型；(2) 字级别词表小（约 21K 汉字），embedding 矩阵参数可控；(3) BERT 的 MLM 预训练在字级别上也能学到词级语义（通过 attention 隐式建模词边界）。Google 的实验表明 whole word masking 比 char masking 效果好，说明模型确实需要词级信息，但可以通过训练策略弥补，不必在分词层解决。

---

## 11. 多语言分词挑战

### 核心难题

**1. Code-switching（语码转换）**

```
"今天的meeting被cancel了，我们reschedule到下周"
→ 中英混合，BPE 需要同时覆盖中文和英文 subword
```

**2. 混合脚本（Mixed scripts）**

```
"BERT模型在NER任务上F1=92.3%"
→ 拉丁字母、汉字、数字、标点混合
```

**3. 词表分配不均**

如果训练语料以英文为主，其他语言的 token 粒度粗糙：

| 语言 | "Hello world" 等价表达 | GPT-2 token 数 | GPT-4 token 数 |
|------|----------------------|----------------|----------------|
| English | Hello world | 2 | 2 |
| 中文 | 你好世界 | 5-6 | 2-3 |
| 日文 | こんにちは世界 | 6-8 | 3-4 |
| 阿拉伯文 | مرحبا بالعالم | 10+ | 4-5 |

### 解决方案

1. **均衡多语言训练语料**：mBERT, XLM-R 按语言比例采样
2. **扩大词表**：GPT-4o 的 200K 词表显著改善非英文效率
3. **语言自适应词表扩展**：为特定语言增量训练新 token，如 Chinese-LLaMA
4. **Byte-level 兜底**：Byte-level BPE 保证任何语言至少 byte 级可表示

---

## 12. 词表大小选择的 Trade-off

### 核心权衡

$$\text{词表大小} \uparrow \implies \begin{cases} \text{序列长度} \downarrow & \text{（推理更快、KV cache 更小）} \\ \text{embedding 参数} \uparrow & \text{（模型更大）} \\ \text{token 语义密度} \uparrow & \text{（每个 token 含义更丰富）} \\ \text{低频 token 训练不充分} \uparrow & \text{（embedding 质量下降）} \end{cases}$$

### 经验值

| 模型规模 | 推荐词表大小 | 理由 |
|---------|-----------|------|
| 小模型 (<1B) | 30K-50K | embedding 层占比大，词表不宜太大 |
| 中模型 (1B-10B) | 50K-100K | 平衡效率和覆盖 |
| 大模型 (>10B) | 100K-200K | embedding 占比小，可放大词表提升效率 |
| 多语言模型 | 100K-250K | 需覆盖多种语言的 subword |

### 数学分析

embedding 层参数量 = $V \times d$（$V$=词表大小，$d$=hidden dim）。对于 LLaMA-7B（$d=4096$）：

- $V=32K$：$32000 \times 4096 = 131M$（占总参数 1.9%）
- $V=100K$：$100000 \times 4096 = 410M$（占总参数 5.9%）
- $V=200K$：$200000 \times 4096 = 819M$（占总参数 11.7%）

对大模型而言，词表翻倍带来的参数增量远小于 Transformer 层的参数量，而序列长度的缩短带来的推理加速是直接的。

---

## 13. 最新进展：无分词趋势

### 传统分词的根本局限

所有 subword 方法都有一个共同问题：**token 边界是预先固定的**，模型无法根据上下文动态调整粒度。

### MegaByte (Yu et al., 2023)

Meta 提出的 **byte-level Transformer**，核心思想是分层处理：

```
Global Transformer:  处理 patch 级别的表示（每个 patch = P 个 byte）
Local Transformer:   在每个 patch 内部处理 byte 级别的生成
```

- 将长 byte 序列分成固定大小的 patch（如 P=8），global model 处理 patch 间关系（$O((N/P)^2)$），local model 处理 patch 内 byte 生成（$O(P^2)$）
- 总复杂度从 $O(N^2)$ 降到 $O(N^2/P)$，当 $P=8$ 时计算量减少 8 倍
- 无需 tokenizer，直接处理 raw bytes

### Patch-level Tokenization

类似思路的变体：

- **BLT (Byte Latent Transformer)** (Meta, 2024)：动态 patch 大小，根据 entropy 自适应决定 patch 边界
- **SpaceByte** (2024)：在空格处分 patch，结合语言先验

### Token-free 的优势与挑战

| 优势 | 挑战 |
|------|------|
| 无需训练 tokenizer，端到端 | 序列长度大幅增加 |
| 对任何语言/模态统一处理 | 需要更复杂的架构设计 |
| 无 tokenizer 引入的信息损失 | 目前尚未在大规模 LLM 上验证 |
| 动态粒度适应 | 训练和推理效率仍需优化 |

### 面试高频问题

**Q1：你认为未来会完全抛弃 tokenizer 吗？**

> 短期（2-3 年）不会。BPE/Unigram 在工程上已经非常成熟，token-free 方法（MegaByte、BLT）在大规模模型上还没有充分验证。但长期趋势是向更灵活的粒度发展——固定的 tokenizer 限制了模型对不同语言和模态的适应性。一个可能的中间路线是可学习的 tokenizer（如 BLT 的动态 patching），而非完全取消 tokenizer。

---

## 附录：常用工具速查

```python
# === tiktoken (OpenAI) ===
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode("Hello world")
text = enc.decode(tokens)
print(f"Token count: {len(tokens)}")

# === SentencePiece ===
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file='model.model')
tokens = sp.encode('Hello world', out_type=str)

# === Hugging Face tokenizers ===
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("Hello world")
ids = tokenizer.encode("Hello world")

# === 快速计算 token 数（命令行） ===
# pip install tiktoken
# python -c "import tiktoken; e=tiktoken.get_encoding('cl100k_base'); print(len(e.encode(open('file.txt').read())))"
```
