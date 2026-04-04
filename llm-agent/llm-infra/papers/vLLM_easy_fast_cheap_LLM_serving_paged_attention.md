# vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention

> 来源：SOSP 2023 | 领域：llm-infra | 学习日期：20260404

## 问题定义

LLM 推理服务的核心瓶颈：**KV Cache 内存管理低效**

- Transformer 推理需要缓存所有历史 token 的 Key/Value（KV Cache）
- 传统实现：每个请求预分配最大序列长度的连续内存
- 问题：
  1. **内存碎片化**：不同长度请求造成大量内碎片（平均浪费 60-80%）
  2. **无法共享**：批内不同请求的 KV Cache 无法复用
  3. **低吞吐**：内存限制导致 batch size 小，GPU 利用率低

$$\text{Waste} = \frac{\text{Allocated} - \text{Used}}{\text{Allocated}} \approx 60\%$$

## 核心方法与创新点

**PagedAttention**：借鉴操作系统虚拟内存分页思想：

1. **KV Cache 分页（Paging）**：
   - KV Cache 划分为固定大小的 Block（Page），每 Block 存 B 个 token
   - 逻辑 Block（Sequence 视角） → 物理 Block（实际内存）
   - Block Table 维护映射关系
   
```
Logical:  [Block 0][Block 1][Block 2]...
Physical: [Page 7 ][Page 2 ][Page 9 ]...（非连续物理内存）
```

2. **动态内存分配**：
   - 请求仅按需分配新 Block（不预分配最大长度）
   - 内存碎片仅存在于 Block 内（最多浪费 Block_size-1 个 token 内存）

3. **KV Cache 共享（Prefix Sharing）**：
   - 相同 prefix（系统 prompt）的请求共享物理 Block（COW，Copy-on-Write）
   - 大量节省 System Prompt 场景的内存

4. **连续批处理（Continuous Batching）**：
   - 请求级别的细粒度调度（完成一个请求立即加入新请求，不等待整批完成）
   - GPU 利用率大幅提升

## 实验结论

- 吞吐量 vs HuggingFace Transformers: **24x**
- 吞吐量 vs FasterTransformer: **3.5x**
- 内存利用率: **96%**（vs 传统 ~40%）
- Prefix Sharing 场景内存节省: **55-80%**

## 工程落地要点

- Block Size 推荐 16-32（小 Block 灵活但 Table 开销大，大 Block 内部碎片多）
- Continuous Batching：需要调度器控制请求队列
- GPU 型号差异：A100 HBM 80GB 可同时服务更多请求
- 多 GPU（Tensor Parallel）：每个 GPU 维护 partial KV Cache

## 面试考点

1. **Q**: PagedAttention 的核心思想是什么？  
   **A**: 借鉴 OS 虚拟内存分页：KV Cache 不需要连续内存，用 Block + Block Table 映射，按需分配，支持多请求间内存共享（Prefix Sharing）。

2. **Q**: Continuous Batching vs Static Batching 的区别？  
   **A**: Static Batching：等一批请求全部完成才开始下一批（GPU 空闲等待短请求完成后的长请求）。Continuous Batching：每个请求完成立即替换，GPU 持续高利用率。

3. **Q**: KV Cache 为什么这么占内存？  
   **A**: 每个 token 需要存 $2 \times L \times H \times d_h$ 个 float16（L 层，H 个头，$d_h$ 头维度）。13B 模型，序列长 4096，约需 **3.2GB** KV Cache/请求。
