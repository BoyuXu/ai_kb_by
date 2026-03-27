# MelonEggLearn：大厂高频算法题 + 编程基础

> 整理自 AIGC-Interview-Book，聚焦推荐算法岗高频考点

---

## 一、算法题（推荐算法岗高频）

### 1. 双指针/滑动窗口类题型

**应用场景**：召回/排序理解、子数组/子串问题

#### 例题1：两数之和（有序数组）

```python
def two_sum(numbers, target):
    """
    给定升序数组，找出两个数之和等于target
    时间复杂度：O(n)
    """
    left, right = 0, len(numbers) - 1
    
    while left < right:
        current_sum = numbers[left] + numbers[right]
        
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
            
    return []

# 示例
two_sum([1, 2, 4, 6, 10], 8)  # 输出: [1, 3]
```

#### 例题2：三数之和

```python
def three_sum(nums):
    """
    找出所有和为0的不重复三元组
    时间复杂度：O(n^2)
    """
    nums.sort()
    result = []
    n = len(nums)
    
    for i in range(n):
        if nums[i] > 0:
            break
        if i > 0 and nums[i] == nums[i-1]:
            continue
        
        left, right = i + 1, n - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
    
    return result
```

#### 例题3：滑动窗口最小子串（高频面试题）

```python
def min_window(s: str, t: str) -> str:
    """
    在s中找到包含t所有字符的最小子串
    时间复杂度：O(n)
    """
    from collections import Counter
    
    need = Counter(t)  # 需要的字符及其次数
    window = Counter()  # 窗口内字符计数
    
    left, right = 0, 0
    valid = 0  # 窗口中满足need条件的字符个数
    
    start, length = 0, float('inf')
    
    while right < len(s):
        c = s[right]
        right += 1
        
        if c in need:
            window[c] += 1
            if window[c] == need[c]:
                valid += 1
        
        # 收缩窗口
        while valid == len(need):
            if right - left < length:
                start = left
                length = right - left
            
            d = s[left]
            left += 1
            
            if d in need:
                if window[d] == need[d]:
                    valid -= 1
                window[d] -= 1
    
    return "" if length == float('inf') else s[start:start+length]
```

**推荐场景关联**：
- 滑动窗口用于实时计算用户最近N个行为的特征
- 双指针用于高效匹配用户-物品相似度计算

---

### 2. 堆/优先队列（TopK问题）

**应用场景**：召回阶段TopK物品筛选、热门物品排序

```python
import heapq

def find_top_k(nums, k):
    """
    找数组中第K大的元素
    时间复杂度：O(n log k)
    """
    min_heap = []
    for num in nums:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)
    return min_heap[0]

def find_top_k_frequent(nums, k):
    """
    找出现频率前K高的元素
    """
    from collections import Counter
    
    count = Counter(nums)
    # 使用堆，存储 (-频率, 元素)
    heap = [(-freq, num) for num, freq in count.items()]
    heapq.heapify(heap)
    
    return [heapq.heappop(heap)[1] for _ in range(k)]

class KthLargest:
    """
    数据流中的第K大元素
    """
    def __init__(self, k: int, nums: list):
        self.k = k
        self.min_heap = nums
        heapq.heapify(self.min_heap)
        while len(self.min_heap) > k:
            heapq.heappop(self.min_heap)
    
    def add(self, val: int) -> int:
        heapq.heappush(self.min_heap, val)
        if len(self.min_heap) > self.k:
            heapq.heappop(self.min_heap)
        return self.min_heap[0]
```

**推荐场景关联**：
- TopK召回：从百万级物品中快速选出用户最可能感兴趣的TopK
- 实时热门榜：维护动态变化的TopK热门物品

---

### 3. 动态规划（序列DP）

**应用场景**：序列建模、用户行为预测

#### 例题1：最长递增子序列（LIS）

```python
def length_of_lis(nums):
    """
    最长递增子序列
    dp[i]表示以第i个元素结尾的最长递增子序列长度
    时间复杂度：O(n^2)
    """
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n
    
    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

def length_of_lis_binary(nums):
    """
    使用二分优化的LIS，O(n log n)
    """
    tails = []
    for num in nums:
        # 二分查找第一个 >= num 的位置
        left, right = 0, len(tails)
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < num:
                left = mid + 1
            else:
                right = mid
        if left == len(tails):
            tails.append(num)
        else:
            tails[left] = num
    return len(tails)
```

#### 例题2：编辑距离

```python
def min_distance(word1: str, word2: str) -> int:
    """
    将word1转换为word2的最少操作数
    操作：插入、删除、替换
    dp[i][j]表示word1前i个字符转换到word2前j个字符的最小操作数
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 初始化
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j],     # 删除
                    dp[i][j-1],     # 插入
                    dp[i-1][j-1]    # 替换
                ) + 1
    
    return dp[m][n]
```

#### 例题3：最长公共子序列（LCS）

```python
def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    两个字符串的最长公共子序列长度
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

**推荐场景关联**：
- 序列DP是DIN、DIEN等深度兴趣网络的基础
- 编辑距离用于文本相似度计算（搜索推荐）

---

### 4. 图算法基础（GNN前置知识）

**应用场景**：图神经网络、社交关系推荐

```python
# 图的表示
class Graph:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]
    
    def add_edge(self, u, v):
        self.adj[u].append(v)
        self.adj[v].append(u)  # 无向图

# BFS广度优先遍历
def bfs(graph, start):
    from collections import deque
    
    visited = [False] * graph.n
    queue = deque([start])
    visited[start] = True
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph.adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    
    return result

# DFS深度优先遍历
def dfs(graph, start):
    visited = [False] * graph.n
    result = []
    
    def dfs_helper(node):
        visited[node] = True
        result.append(node)
        for neighbor in graph.adj[node]:
            if not visited[neighbor]:
                dfs_helper(neighbor)
    
    dfs_helper(start)
    return result

# 岛屿数量（经典图题）
def num_islands(grid):
    """
    计算网格中岛屿数量
    1表示陆地，0表示水
    """
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    count = 0
    
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
            return
        grid[r][c] = '0'  # 标记已访问
        dfs(r+1, c)
        dfs(r-1, c)
        dfs(r, c+1)
        dfs(r, c-1)
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                dfs(i, j)
                count += 1
    
    return count
```

**推荐场景关联**：
- GraphSAGE、PinSAGE等图神经网络用于社交推荐
- 用户-物品交互图建模

---

### 5. 其他高频算法

#### 快速排序

```python
def quick_sort(arr):
    """快速排序（递归版）"""
    def partition(low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
    
    def quick_sort_recursive(low, high):
        if low < high:
            pi = partition(low, high)
            quick_sort_recursive(low, pi - 1)
            quick_sort_recursive(pi + 1, high)
    
    quick_sort_recursive(0, len(arr) - 1)
    return arr
```

#### 二分查找

```python
def binary_search(nums, target):
    """二分查找（闭区间写法）"""
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

---

## 二、Python 编程

### 1. numpy/pandas 核心操作

#### NumPy核心操作

```python
import numpy as np

# 数组创建
arr = np.array([[1, 2, 3], [4, 5, 6]])
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
random_arr = np.random.randn(3, 3)

# 形状操作
arr.reshape(-1, 1)  # 变为一列
arr.flatten()       # 展平
arr.T               # 转置

# 索引与切片
arr[0, :]      # 第一行
arr[:, 1]      # 第二列
arr[arr > 3]   # 布尔索引

# 广播机制
arr + 5              # 标量广播
arr + np.array([1, 2, 3])  # 行向量广播

# 常用运算
np.dot(a, b)         # 矩阵乘法
np.matmul(a, b)      # 矩阵乘法
a * b                # 元素乘法
np.sum(arr, axis=0)  # 按列求和
np.mean(arr, axis=1) # 按行求平均
np.argmax(arr, axis=1)  # 最大值的索引

# 合并与分割
np.concatenate([a, b], axis=0)  # 拼接
np.vstack([a, b])    # 垂直拼接
np.hstack([a, b])    # 水平拼接
np.split(arr, 3)     # 分割
```

#### Pandas核心操作

```python
import pandas as pd

# 数据读取
df = pd.read_csv('data.csv')
df = pd.read_sql(query, connection)

# 数据查看
df.head(10)
df.info()
df.describe()
df.shape

# 选择与过滤
df['column']         # 单列
df[['col1', 'col2']] # 多列
df.loc[0:10, 'col']  # 标签索引
df.iloc[0:10, 0:2]   # 位置索引
df[df['age'] > 18]   # 条件过滤

# 数据处理
df.dropna()          # 删除缺失值
df.fillna(0)         # 填充缺失值
df.drop_duplicates() # 去重
df.sort_values('col', ascending=False)  # 排序

# 分组聚合
df.groupby('category')['value'].mean()
df.groupby('category').agg({'value': 'sum', 'count': 'count'})

# 特征工程常用
df['new_col'] = df['col1'].apply(lambda x: x**2)
df['category'] = df['category'].astype('category')
df = pd.get_dummies(df, columns=['category'])  # one-hot编码

# 合并操作
pd.merge(df1, df2, on='key', how='inner')
pd.concat([df1, df2], axis=0)
```

---

### 2. 多进程/多线程 vs 协程

#### GIL与并发选择

| 方式 | 适用场景 | 特点 |
|------|----------|------|
| 多线程 | I/O密集型 | 受GIL限制，无法真正并行 |
| 多进程 | CPU密集型 | 突破GIL，充分利用多核 |
| 协程 | 高并发I/O | 轻量级，单线程内高并发 |

#### 多进程（推荐用于特征处理）

```python
from multiprocessing import Process, Pool, Queue
import os

# 基础多进程
def worker(name):
    print(f'Worker {name}, PID: {os.getpid()}')

processes = []
for i in range(4):
    p = Process(target=worker, args=(i,))
    processes.append(p)
    p.start()

for p in processes:
    p.join()

# 进程池（更常用）
def process_data(data):
    # 处理单个数据块
    return data ** 2

if __name__ == '__main__':
    data = list(range(100))
    with Pool(processes=4) as pool:
        results = pool.map(process_data, data)
        
    # 使用ProcessPoolExecutor
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_data, data))
```

#### 多线程

```python
from threading import Thread
import threading

# 基础多线程
def thread_task(n):
    print(f'Thread {n} running')

threads = []
for i in range(4):
    t = Thread(target=thread_task, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

# 线程池
from concurrent.futures import ThreadPoolExecutor

def fetch_url(url):
    import requests
    return requests.get(url).text

urls = ['http://example.com'] * 10
with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(fetch_url, urls))
```

#### 协程（asyncio）

```python
import asyncio

async def async_task(n):
    print(f'Task {n} started')
    await asyncio.sleep(1)  # 模拟异步I/O
    print(f'Task {n} finished')
    return n

async def main():
    # 创建多个任务并并发执行
    tasks = [async_task(i) for i in range(5)]
    results = await asyncio.gather(*tasks)
    print(results)

# 运行协程
asyncio.run(main())

# 实际应用：异步并发请求
async def fetch_all(urls):
    import aiohttp
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, url) for url in urls]
        return await asyncio.gather(*tasks)

async def fetch_one(session, url):
    async with session.get(url) as response:
        return await response.text()
```

#### fork vs spawn模式

```python
from multiprocessing import set_start_method

# Unix系统默认fork，Windows/Mac默认spawn
# fork: 复制父进程资源，启动快
# spawn: 创建全新进程，启动慢但更干净

set_start_method("spawn")  # 或 "fork"
```

---

### 3. Python内存优化技巧

```python
# 1. 使用生成器代替列表（惰性计算）
def read_large_file(file_path):
    """逐行读取大文件，不占内存"""
    with open(file_path, 'r') as f:
        for line in f:
            yield line.strip()

# 2. 列表推导式 vs 生成器表达式
# 列表推导式（占用内存）
squares_list = [x**2 for x in range(1000000)]
# 生成器表达式（惰性计算）
squares_gen = (x**2 for x in range(1000000))

# 3. 使用__slots__减少对象内存占用
class Point:
    __slots__ = ['x', 'y']  # 不使用__dict__，节省内存
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 4. 使用数组array代替list存储数字
from array import array
# 'i'表示整数
arr = array('i', [1, 2, 3, 4, 5])  # 比list节省内存

# 5. 使用lru_cache缓存结果
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 6. 及时删除大对象，强制垃圾回收
import gc

def process_large_data():
    large_data = load_large_data()
    result = process(large_data)
    del large_data  # 显式删除
    gc.collect()    # 强制垃圾回收
    return result

# 7. 使用内存映射处理大文件
import mmap

def process_large_file(file_path):
    with open(file_path, 'r+b') as f:
        with mmap.mmap(f.fileno(), 0) as mm:
            # 像操作内存一样操作文件
            return mm.readline()

# 8. 使用pandas的category类型
import pandas as pd
df['category_col'] = df['category_col'].astype('category')

# 9. 批量处理数据
def process_in_batches(data, batch_size=1000):
    """分批处理，控制内存使用"""
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        process(batch)
```

---

## 三、计算机基础

### 1. 分布式训练基础

#### PS架构（Parameter Server）

```
┌─────────────────────────────────────────┐
│            Parameter Server              │
│        (存储和更新全局模型参数)            │
└─────────────────────────────────────────┘
                    ↑↓
    ┌───────────────┼───────────────┐
    ↓               ↓               ↓
┌───────┐      ┌───────┐      ┌───────┐
│Worker1│      │Worker2│      │Worker3│
│计算梯度│      │计算梯度│      │计算梯度│
└───────┘      └───────┘      └───────┘
```

**特点**：
- PS节点存储模型参数
- Worker节点计算梯度，推送到PS
- PS聚合梯度，更新参数
- 适合大规模稀疏特征（推荐系统）

#### AllReduce架构

```
┌───────┐      ┌───────┐      ┌───────┐      ┌───────┐
│ GPU 0 │←────→│ GPU 1 │←────→│ GPU 2 │←────→│ GPU 3 │
│       │      │       │      │       │      │       │
└───────┘      └───────┘      └───────┘      └───────┘
   Ring AllReduce: 每个GPU只与相邻GPU通信
```

**特点**：
- 无中心节点，各节点对等
- Ring-AllReduce算法高效聚合梯度
- 适合稠密模型（CV、NLP大模型）
- NCCL库优化GPU间通信

#### 对比

| 特性 | PS架构 | AllReduce |
|------|--------|-----------|
| 适用场景 | 稀疏特征、大规模参数 | 稠密模型、大batch训练 |
| 扩展性 | 参数规模可水平扩展 | 受限于单机GPU数量 |
| 通信开销 | 高频小数据推送 | 梯度全量同步 |
| 代表框架 | TensorFlow PS | PyTorch DDP, Horovod |

```python
# PyTorch DDP示例
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup():
    dist.init_process_group("nccl")
    
def main():
    setup()
    model = MyModel().to(device)
    ddp_model = DDP(model, device_ids=[local_rank])
    # 训练时自动进行梯度AllReduce
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
```

---

### 2. 大规模特征存储

#### Redis

```python
import redis

# 连接Redis
r = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True
)

# 基本操作
r.set('user:1001:age', 25)
age = r.get('user:1001:age')

# 哈希存储（适合存储用户特征）
r.hset('user:1001:features', mapping={
    'age': 25,
    'gender': 'male',
    'city': 'beijing'
})
features = r.hgetall('user:1001:features')

# 批量获取（Pipeline优化）
pipe = r.pipeline()
for user_id in user_ids:
    pipe.hgetall(f'user:{user_id}:features')
results = pipe.execute()

# 设置过期时间（TTL）
r.setex('session:token', 3600, 'user_data')

# 布隆过滤器（去重判断）
r.bf().add('user:bloom', 'user_1001')
exists = r.bf().exists('user:bloom', 'user_1001')
```

#### KV Store选型对比

| 系统 | 特点 | 适用场景 |
|------|------|----------|
| Redis | 内存级，支持丰富数据结构 | 实时特征、缓存 |
| Memcached | 纯内存，简单KV | 高并发缓存 |
| RocksDB | 磁盘级，LSM树 | 大规模持久化特征 |
| LevelDB | 轻量级LSM | 嵌入式存储 |
| TiKV | 分布式事务KV | 海量特征存储 |

#### 推荐系统特征存储实践

```python
class FeatureStore:
    """
    推荐系统特征存储示例
    """
    def __init__(self):
        self.redis = redis.Redis()
        self.local_cache = {}  # L1缓存
        
    def get_user_features(self, user_id):
        # L1缓存检查
        if user_id in self.local_cache:
            return self.local_cache[user_id]
        
        # Redis查询
        key = f"user:{user_id}:features"
        features = self.redis.hgetall(key)
        
        if features:
            # 更新L1缓存
            self.local_cache[user_id] = features
        return features
    
    def get_item_features_batch(self, item_ids):
        """批量获取物品特征"""
        pipe = self.redis.pipeline()
        for item_id in item_ids:
            pipe.hgetall(f"item:{item_id}:features")
        return pipe.execute()
```

---

## 四、快速参考

### 算法题速查

| 题型 | 核心思路 | 时间复杂度 |
|------|----------|------------|
| 双指针 | 左右指针向中间移动 | O(n) |
| 滑动窗口 | 动态维护满足条件的区间 | O(n) |
| 堆TopK | 维护大小为K的堆 | O(n log k) |
| 序列DP | 定义以i结尾的最优状态 | O(n) or O(n²) |
| 二分 | 确定循环不变量 | O(log n) |
| 图遍历 | BFS/DFS标记访问 | O(V+E) |

### Python并发选择决策

```
任务类型判断
    ↓
I/O密集型? ──Yes──→ 协程(asyncio) ──→ 需要多核? ──Yes──→ 多进程+协程
    ↓No                              ↓No
CPU密集型? ──Yes──→ 多进程(ProcessPoolExecutor)
    ↓No
多线程(ThreadPoolExecutor)
```

### 内存优化检查清单

- [ ] 使用生成器代替列表推导式
- [ ] 大对象使用后及时del
- [ ] 使用__slots__定义小对象
- [ ] pandas使用category类型
- [ ] 使用lru_cache缓存计算结果
- [ ] 大文件使用mmap内存映射
- [ ] 批量处理控制内存峰值

---

*Generated by MelonEggLearn | 最后更新: 2026-03-11*

---

## 五、推荐/广告工程数据结构实战

> 针对算法岗面试，聚焦工业界高频使用的数据结构

---

### 1. 哈希表 (Hash Table)

#### 核心考点
- **碰撞处理**: 链地址法 vs 开放寻址法
- **一致性哈希**: 分布式系统中的数据分片与负载均衡
- **负载因子**: 动态扩容时机与 rehash 成本

#### Python实现

```python
class HashTable:
    """
    哈希表实现（链地址法解决冲突）
    """
    def __init__(self, capacity=16):
        self.capacity = capacity
        self.size = 0
        self.buckets = [[] for _ in range(capacity)]
        self.load_factor_threshold = 0.75
    
    def _hash(self, key):
        """哈希函数"""
        return hash(key) % self.capacity
    
    def put(self, key, value):
        """插入/更新"""
        index = self._hash(key)
        bucket = self.buckets[index]
        
        # 检查是否已存在
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        
        # 新元素
        bucket.append((key, value))
        self.size += 1
        
        # 扩容检查
        if self.size / self.capacity > self.load_factor_threshold:
            self._resize()
    
    def get(self, key):
        """查找"""
        index = self._hash(key)
        bucket = self.buckets[index]
        for k, v in bucket:
            if k == key:
                return v
        return None
    
    def _resize(self):
        """扩容并rehash"""
        old_buckets = self.buckets
        self.capacity *= 2
        self.size = 0
        self.buckets = [[] for _ in range(self.capacity)]
        
        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)

class ConsistentHash:
    """
    一致性哈希（用于分布式特征存储）
    解决：节点增删时最小化数据迁移
    """
    def __init__(self, replicas=150):
        self.replicas = replicas  # 虚拟节点数，平衡负载
        self.ring = {}  # 哈希环: {hash_val: node}
        self.sorted_keys = []  # 排序后的哈希值
        self.nodes = set()
    
    def _hash(self, key):
        """MD5哈希"""
        import hashlib
        return int(hashlib.md5(str(key).encode()).hexdigest(), 16)
    
    def add_node(self, node):
        """添加节点（服务器）"""
        self.nodes.add(node)
        for i in range(self.replicas):
            virtual_key = f"{node}:{i}"
            hash_val = self._hash(virtual_key)
            self.ring[hash_val] = node
            self.sorted_keys.append(hash_val)
        self.sorted_keys.sort()
    
    def remove_node(self, node):
        """移除节点"""
        self.nodes.discard(node)
        for i in range(self.replicas):
            virtual_key = f"{node}:{i}"
            hash_val = self._hash(virtual_key)
            del self.ring[hash_val]
            self.sorted_keys.remove(hash_val)
    
    def get_node(self, key):
        """获取key应该存储的节点"""
        if not self.ring:
            return None
        
        hash_val = self._hash(key)
        
        # 二分查找第一个 >= hash_val 的节点
        idx = self._bisect_right(self.sorted_keys, hash_val)
        if idx == len(self.sorted_keys):
            idx = 0
        
        return self.ring[self.sorted_keys[idx]]
    
    def _bisect_right(self, arr, target):
        """二分查找"""
        left, right = 0, len(arr)
        while left < right:
            mid = (left + right) // 2
            if arr[mid] <= target:
                left = mid + 1
            else:
                right = mid
        return left

# ============ 推荐系统实战场景 ============

class FeatureStorageRouter:
    """
    【推荐系统实战】分布式特征存储路由
    使用一致性哈希将用户/物品特征分散到不同Redis节点
    """
    def __init__(self, redis_nodes):
        self.ch = ConsistentHash(replicas=150)
        self.redis_clients = {}
        
        for node in redis_nodes:
            self.ch.add_node(node['name'])
            self.redis_clients[node['name']] = redis.Redis(
                host=node['host'], port=node['port']
            )
    
    def get_user_features(self, user_id):
        """获取用户特征（自动路由到对应Redis节点）"""
        node = self.ch.get_node(f"user:{user_id}")
        redis_client = self.redis_clients[node]
        return redis_client.hgetall(f"user:{user_id}:features")
    
    def set_user_features(self, user_id, features):
        """存储用户特征"""
        node = self.ch.get_node(f"user:{user_id}")
        redis_client = self.redis_clients[node]
        redis_client.hset(f"user:{user_id}:features", mapping=features)

"""
面试要点：
1. 为什么用一致性哈希？
   - 普通哈希: hash(key) % N，节点变化时大量数据需要迁移
   - 一致性哈希: 只影响相邻节点间的数据，迁移量降为 1/N

2. 虚拟节点作用？
   - 解决数据倾斜问题，让每台物理服务器承担更均匀的负载
   - 虚拟节点越多，负载越均匀，但元数据开销增大

3. 工程实践：
   - Redis Cluster、Memcached 都使用一致性哈希
   - 特征存储中，用户ID/物品ID作为key，自动分片到不同节点
"""
```

---

### 2. 堆 / 优先队列 (Heap / Priority Queue)

#### 核心考点
- **堆的数组表示**: 父节点i的左右子节点为 2i+1, 2i+2
- **上浮/下沉操作**: 插入和删除的时间复杂度 O(log n)
- **TopK问题**: 维护大小为K的堆，处理海量数据

#### Python实现

```python
import heapq
from typing import List, Tuple

class TopKSelector:
    """
    【推荐系统实战】多路召回结果合并TopK
    场景：从多个召回源（协同过滤、向量召回、热门召回）合并TopK
    """
    def __init__(self, k: int):
        self.k = k
        # 小顶堆：存储当前最大的K个元素（堆顶是第K大的）
        self.min_heap = []
    
    def add(self, item_id: str, score: float, source: str):
        """
        添加召回结果
        :param item_id: 物品ID
        :param score: 排序分数
        :param source: 召回源（用于去重和统计）
        """
        if len(self.min_heap) < self.k:
            heapq.heappush(self.min_heap, (score, item_id, source))
        elif score > self.min_heap[0][0]:
            # 新分数比堆顶大，替换
            heapq.heapreplace(self.min_heap, (score, item_id, source))
    
    def get_top_k(self) -> List[Tuple]:
        """获取TopK结果（按分数降序）"""
        return sorted(self.min_heap, key=lambda x: -x[0])
    
    def merge_recall_results(self, recall_sources: dict) -> List[Tuple]:
        """
        合并多路召回结果
        recall_sources: {'cf': [(item1, 0.9), ...], 'vector': [...], 'hot': [...]}
        """
        seen_items = set()
        
        for source, items in recall_sources.items():
            for item_id, score in items:
                # 去重：同一物品只保留最高分的来源
                if item_id not in seen_items:
                    seen_items.add(item_id)
                    self.add(item_id, score, source)
        
        return self.get_top_k()

class RealtimeLeaderboard:
    """
    【推荐系统实战】实时热门物品榜单（滑动窗口TopK）
    场景：实时统计过去1小时点击量Top100的物品
    """
    def __init__(self, k: int = 100, window_seconds: int = 3600):
        self.k = k
        self.window = window_seconds
        from collections import deque
        self.clicks = deque()  # (timestamp, item_id)
        self.item_counts = {}  # item_id -> count
        # 堆：(-count, item_id) 实现大顶堆效果
        self.heap = []
        self.item_in_heap = set()
    
    def click(self, item_id: str, timestamp: int):
        """记录点击事件"""
        self.clicks.append((timestamp, item_id))
        self.item_counts[item_id] = self.item_counts.get(item_id, 0) + 1
        
        # 清理过期数据
        self._cleanup(timestamp)
        
        # 更新堆
        if item_id not in self.item_in_heap:
            heapq.heappush(self.heap, (-self.item_counts[item_id], item_id))
            self.item_in_heap.add(item_id)
    
    def _cleanup(self, current_time: int):
        """清理滑动窗口外的数据"""
        cutoff = current_time - self.window
        while self.clicks and self.clicks[0][0] < cutoff:
            _, item_id = self.clicks.popleft()
            self.item_counts[item_id] -= 1
            if self.item_counts[item_id] == 0:
                del self.item_counts[item_id]
    
    def get_top_k(self) -> List[Tuple[str, int]]:
        """获取当前TopK热门物品"""
        # 重建堆确保数据最新（或者使用懒删除）
        self.heap = [(-count, item_id) for item_id, count in self.item_counts.items()]
        heapq.heapify(self.heap)
        
        result = []
        temp_heap = self.heap.copy()
        for _ in range(min(self.k, len(temp_heap))):
            neg_count, item_id = heapq.heappop(temp_heap)
            result.append((item_id, -neg_count))
        return result

class MedianFinder:
    """
    【推荐系统实战】数据流中位数（A/B测试指标监控）
    场景：实时监控CTR分布的中位数
    """
    def __init__(self):
        # 大顶堆：存储较小一半的数据
        self.small = []
        # 小顶堆：存储较大一半的数据
        self.large = []
    
    def add_num(self, num: float):
        """添加数字"""
        if not self.small or num <= -self.small[0]:
            heapq.heappush(self.small, -num)
        else:
            heapq.heappush(self.large, num)
        
        # 平衡两个堆的大小
        if len(self.small) > len(self.large) + 1:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        elif len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))
    
    def find_median(self) -> float:
        """获取中位数"""
        if len(self.small) == len(self.large):
            return (-self.small[0] + self.large[0]) / 2.0
        return -self.small[0]

"""
面试要点：
1. TopK问题的时间复杂度分析：
   - 全排序: O(n log n)
   - 堆方法: O(n log k)，k远小于n时优势明显
   - 快速选择: O(n)平均，但无法处理动态数据流

2. 工程实践：
   - 召回阶段：多路召回合并，每路几百个结果合并成最终Top100
   - 排序阶段：精排模型打分后，用堆取TopK返回给前端
   - 实时监控：点击率、曝光量的TopK统计

3. 堆 vs 排序选择：
   - 数据量固定：排序更直观
   - 数据流动态到达：堆更合适
   - n >> k（如从亿级取Top100）：必须用堆
"""
```

---

### 3. 树结构 (Tree)

#### 核心考点
- **B+树**: 多路平衡搜索树，磁盘IO优化，数据库索引核心
- **Trie树**: 前缀树，高效前缀匹配和自动补全
- **AVL/红黑树**: 自平衡二叉搜索树

#### Python实现

```python
class BPlusTreeNode:
    """
    B+树节点
    - 内部节点：存储键和子节点指针
    - 叶子节点：存储键和数据指针，且叶子节点形成链表
    """
    def __init__(self, leaf=False):
        self.leaf = leaf
        self.keys = []  # 键列表（有序）
        self.children = []  # 子节点指针（内部节点使用）
        self.next = None  # 下一个叶子节点（叶子节点使用）
        self.data = []  # 数据指针（叶子节点使用）

class BPlusTree:
    """
    【推荐系统实战】B+树实现特征索引
    场景：用户ID范围查询、物品ID有序遍历
    """
    def __init__(self, degree=4):
        self.degree = degree  # 节点的最小度数
        self.root = BPlusTreeNode(leaf=True)
    
    def insert(self, key, value):
        """插入键值对"""
        root = self.root
        
        if len(root.keys) == (2 * self.degree) - 1:
            # 根节点满了，需要分裂
            new_root = BPlusTreeNode(leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0, self.root)
            self.root = new_root
        
        self._insert_non_full(self.root, key, value)
    
    def _split_child(self, parent, index, child):
        """分裂子节点"""
        degree = self.degree
        new_node = BPlusTreeNode(leaf=child.leaf)
        
        # 分裂键
        mid = degree - 1
        parent.keys.insert(index, child.keys[mid])
        parent.children.insert(index + 1, new_node)
        
        # 分配键
        new_node.keys = child.keys[mid + 1:]
        child.keys = child.keys[:mid]
        
        if child.leaf:
            new_node.data = child.data[mid + 1:]
            child.data = child.data[:mid + 1]
            new_node.next = child.next
            child.next = new_node
        else:
            new_node.children = child.children[mid + 1:]
            child.children = child.children[:mid + 1]
    
    def _insert_non_full(self, node, key, value):
        """向非满节点插入"""
        if node.leaf:
            # 叶子节点直接插入
            idx = self._find_index(node.keys, key)
            node.keys.insert(idx, key)
            node.data.insert(idx, value)
        else:
            # 内部节点，找到合适的子节点
            idx = self._find_index(node.keys, key)
            child = node.children[idx]
            
            if len(child.keys) == (2 * self.degree) - 1:
                self._split_child(node, idx, child)
                if key > node.keys[idx]:
                    idx += 1
            
            self._insert_non_full(node.children[idx], key, value)
    
    def _find_index(self, keys, key):
        """二分查找插入位置"""
        left, right = 0, len(keys)
        while left < right:
            mid = (left + right) // 2
            if keys[mid] < key:
                left = mid + 1
            else:
                right = mid
        return left
    
    def search(self, key):
        """精确查询"""
        node = self.root
        while not node.leaf:
            idx = self._find_index(node.keys, key)
            node = node.children[idx]
        
        # 在叶子节点中查找
        for i, k in enumerate(node.keys):
            if k == key:
                return node.data[i]
        return None
    
    def range_query(self, start_key, end_key):
        """
        【关键特性】范围查询 - B+树相比B树的优势
        场景：查询用户ID在[1000, 2000]之间的所有特征
        """
        results = []
        node = self.root
        
        # 定位到起始叶子节点
        while not node.leaf:
            idx = self._find_index(node.keys, start_key)
            node = node.children[idx]
        
        # 遍历叶子节点链表
        while node:
            for i, key in enumerate(node.keys):
                if start_key <= key <= end_key:
                    results.append((key, node.data[i]))
                elif key > end_key:
                    return results
            node = node.next
        
        return results

class TrieNode:
    """Trie树节点"""
    def __init__(self):
        self.children = {}  # 字符 -> TrieNode
        self.is_end = False  # 是否是完整单词结尾
        self.data = None  # 存储关联数据
        self.count = 0  # 经过该节点的单词数（用于前缀统计）

class Trie:
    """
    【推荐系统实战】Trie树实现搜索前缀匹配
    场景：搜索框自动补全、查询词前缀匹配
    """
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str, data=None):
        """插入单词"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1
        node.is_end = True
        node.data = data
    
    def search(self, word: str) -> bool:
        """精确匹配"""
        node = self._find_node(word)
        return node is not None and node.is_end
    
    def starts_with(self, prefix: str) -> bool:
        """前缀是否存在"""
        return self._find_node(prefix) is not None
    
    def _find_node(self, prefix: str) -> TrieNode:
        """找到前缀对应的节点"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
    def get_words_with_prefix(self, prefix: str, limit: int = 10) -> List[Tuple[str, any]]:
        """
        【核心功能】获取指定前缀的所有单词
        场景：用户输入"ipho"，返回["iphone", "iphone15", "iphonex", ...]
        """
        node = self._find_node(prefix)
        if not node:
            return []
        
        results = []
        self._dfs(node, prefix, results, limit)
        return results
    
    def _dfs(self, node: TrieNode, prefix: str, results: List, limit: int):
        """DFS遍历收集单词"""
        if len(results) >= limit:
            return
        
        if node.is_end:
            results.append((prefix, node.data))
        
        for char, child in node.children.items():
            self._dfs(child, prefix + char, results, limit)
    
    def get_top_k_with_prefix(self, prefix: str, k: int = 5) -> List[str]:
        """
        【推荐系统实战】热门搜索词自动补全
        返回前缀匹配下，count最高的K个词
        """
        node = self._find_node(prefix)
        if not node:
            return []
        
        # 收集所有匹配的单词及其count
        candidates = []
        self._collect_with_count(node, prefix, candidates)
        
        # 按count排序取TopK
        candidates.sort(key=lambda x: -x[1])
        return [word for word, _ in candidates[:k]]
    
    def _collect_with_count(self, node: TrieNode, prefix: str, results: List):
        """收集所有单词及其count"""
        if node.is_end:
            results.append((prefix, node.count))
        for char, child in node.children.items():
            self._collect_with_count(child, prefix + char, results)

# ============ 推荐系统实战场景 ============

class SearchAutocomplete:
    """
    【推荐系统实战】搜索框自动补全服务
    """
    def __init__(self):
        self.trie = Trie()
        self.query_freq = {}  # 查询词频率统计
    
    def record_query(self, query: str):
        """记录用户搜索词（用于更新热度）"""
        self.query_freq[query] = self.query_freq.get(query, 0) + 1
        self.trie.insert(query, self.query_freq[query])
    
    def suggest(self, prefix: str, k: int = 5) -> List[str]:
        """
        根据前缀返回搜索建议
        策略：结合词频和个性化（简化版只考虑词频）
        """
        return self.trie.get_top_k_with_prefix(prefix, k)

class FeatureIndex:
    """
    【推荐系统实战】基于B+树的用户特征索引
    支持高效的范围查询（如：查询最近7天活跃用户）
    """
    def __init__(self):
        self.tree = BPlusTree(degree=10)
    
    def index_user(self, user_id: int, last_active_time: int, features: dict):
        """
        索引用户
        key: last_active_time (用于按活跃时间范围查询)
        value: (user_id, features)
        """
        self.tree.insert(last_active_time, (user_id, features))
    
    def get_active_users(self, start_time: int, end_time: int) -> List:
        """获取指定时间范围内活跃的用户"""
        return self.tree.range_query(start_time, end_time)

"""
面试要点：
1. B树 vs B+树区别：
   - B树：数据存储在所有节点，范围查询需要中序遍历
   - B+树：数据只存储在叶子节点，叶子形成链表，范围查询只需遍历叶子
   - B+树更适合磁盘存储（MySQL InnoDB使用B+树）

2. 为什么MySQL用B+树不用哈希？
   - 哈希只支持精确查询，不支持范围查询和排序
   - B+树支持范围查询、排序、模糊查询

3. Trie树应用场景：
   - 搜索自动补全、拼写检查、IP路由最长前缀匹配
   - 时间复杂度：插入和查询都是 O(m)，m为单词长度

4. 推荐系统中：
   - 用户ID/物品ID索引：B+树支持高效范围查询
   - 搜索自动补全：Trie树支持高效前缀匹配
"""
```

---

### 4. 图 (Graph)

#### 核心考点
- **BFS/DFS**: 图的遍历，连通性检测
- **最短路径**: Dijkstra、Bellman-Ford、Floyd-Warshall
- **拓扑排序**: DAG的线性化

#### Python实现

```python
from collections import deque, defaultdict
import heapq
from typing import List, Dict, Tuple, Set

class Graph:
    """
    图的邻接表表示
    支持有向图和无向图
    """
    def __init__(self, directed=False):
        self.directed = directed
        self.adj = defaultdict(list)  # 邻接表
        self.edges = []  # 边列表 (u, v, weight)
    
    def add_edge(self, u, v, weight=1):
        """添加边"""
        self.adj[u].append((v, weight))
        self.edges.append((u, v, weight))
        if not self.directed:
            self.adj[v].append((u, weight))
    
    def bfs(self, start) -> List:
        """
        BFS广度优先搜索
        应用：最短路径（无权图）、层次遍历
        """
        visited = {start}
        queue = deque([start])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for neighbor, _ in self.adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return result
    
    def dfs(self, start) -> List:
        """
        DFS深度优先搜索
        应用：连通分量、拓扑排序、环路检测
        """
        visited = set()
        result = []
        
        def dfs_helper(node):
            visited.add(node)
            result.append(node)
            for neighbor, _ in self.adj[node]:
                if neighbor not in visited:
                    dfs_helper(neighbor)
        
        dfs_helper(start)
        return result
    
    def dijkstra(self, start) -> Dict:
        """
        Dijkstra单源最短路径
        适用：非负权边
        时间复杂度：O((V+E)logV)
        """
        dist = {node: float('inf') for node in self.adj}
        dist[start] = 0
        pq = [(0, start)]  # (距离, 节点)
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if d > dist[u]:
                continue
            
            for v, weight in self.adj[u]:
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    heapq.heappush(pq, (dist[v], v))
        
        return dist
    
    def topological_sort(self) -> List:
        """
        拓扑排序（Kahn算法）
        适用：有向无环图（DAG）
        应用：任务调度、依赖解析
        """
        # 计算入度
        in_degree = defaultdict(int)
        for u in self.adj:
            if u not in in_degree:
                in_degree[u] = 0
            for v, _ in self.adj[u]:
                in_degree[v] += 1
        
        # 入度为0的节点入队
        queue = deque([u for u in in_degree if in_degree[u] == 0])
        result = []
        
        while queue:
            u = queue.popleft()
            result.append(u)
            
            for v, _ in self.adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        # 检测环
        if len(result) != len(in_degree):
            return []  # 存在环
        
        return result

# ============ 推荐系统实战场景 ============

class UserItemGraph:
    """
    【推荐系统实战】用户-物品交互图
    GNN（图神经网络）的前置基础
    """
    def __init__(self):
        # 二分图：用户和物品是两类节点
        self.user_adj = defaultdict(list)  # 用户 -> [(物品, 交互分数)]
        self.item_adj = defaultdict(list)  # 物品 -> [(用户, 交互分数)]
        self.users = set()
        self.items = set()
    
    def add_interaction(self, user_id, item_id, score=1.0):
        """添加用户-物品交互"""
        self.user_adj[user_id].append((item_id, score))
        self.item_adj[item_id].append((user_id, score))
        self.users.add(user_id)
        self.items.add(item_id)
    
    def get_user_neighbors(self, user_id) -> List[Tuple]:
        """获取用户的邻居物品（已交互的物品）"""
        return self.user_adj[user_id]
    
    def get_item_neighbors(self, item_id) -> List[Tuple]:
        """获取物品的邻居用户（交互过的用户）"""
        return self.item_adj[item_id]
    
    def bfs_sample_neighbors(self, start_node, depth=2, sample_size=10):
        """
        【GraphSAGE采样】BFS邻居采样
        GNN中常用的邻居采样策略
        """
        visited = {start_node}
        queue = deque([(start_node, 0)])  # (节点, 深度)
        layers = {0: [start_node]}
        
        while queue:
            node, d = queue.popleft()
            if d >= depth:
                continue
            
            # 获取邻居
            if node in self.users:
                neighbors = [item for item, _ in self.user_adj[node]]
            else:
                neighbors = [user for user, _ in self.item_adj[node]]
            
            # 采样（避免邻居过多）
            if len(neighbors) > sample_size:
                import random
                neighbors = random.sample(neighbors, sample_size)
            
            layer_nodes = []
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, d + 1))
                    layer_nodes.append(neighbor)
            
            if layer_nodes:
                layers[d + 1] = layer_nodes
        
        return layers

class SocialGraph:
    """
    【推荐系统实战】社交关系图
    场景：基于社交关系的推荐（好友在看、社交传播）
    """
    def __init__(self):
        self.graph = Graph(directed=False)
        self.user_features = {}  # 用户特征
    
    def add_friendship(self, user1, user2, strength=1.0):
        """添加好友关系"""
        self.graph.add_edge(user1, user2, strength)
    
    def find_influencers(self, k=10) -> List[Tuple]:
        """
        【社交推荐】发现KOL（关键意见领袖）
        基于度中心性简单实现
        """
        degrees = [(node, len(neighbors)) for node, neighbors in self.graph.adj.items()]
        degrees.sort(key=lambda x: -x[1])
        return degrees[:k]
    
    def shortest_path_recommend(self, user_a, user_b) -> List:
        """
        【社交推荐】找出用户A和用户B的共同好友路径
        用于"你可能认识"推荐
        """
        # BFS找最短路径
        if user_a not in self.graph.adj or user_b not in self.graph.adj:
            return []
        
        queue = deque([(user_a, [user_a])])
        visited = {user_a}
        
        while queue:
            node, path = queue.popleft()
            
            for neighbor, _ in self.graph.adj[node]:
                if neighbor == user_b:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []
    
    def common_neighbors(self, user_a, user_b) -> Set:
        """
        【好友推荐】共同邻居数量（CN指标）
        推荐好友的经典算法
        """
        neighbors_a = set(n for n, _ in self.graph.adj.get(user_a, []))
        neighbors_b = set(n for n, _ in self.graph.adj.get(user_b, []))
        return neighbors_a & neighbors_b

class KnowledgeGraph:
    """
    【推荐系统实战】知识图谱
    场景：基于知识图谱的推荐，增强物品语义理解
    """
    def __init__(self):
        self.triples = []  # (head, relation, tail)
        self.entity_adj = defaultdict(list)  # 实体 -> [(关系, 尾实体)]
        self.relations = set()
    
    def add_triple(self, head, relation, tail):
        """添加知识三元组"""
        self.triples.append((head, relation, tail))
        self.entity_adj[head].append((relation, tail))
        self.relations.add(relation)
    
    def find_paths(self, start_entity, end_entity, max_depth=3) -> List[List]:
        """
        【知识图谱推理】查找两实体间的路径
        用于解释推荐结果（可解释性推荐）
        """
        paths = []
        queue = deque([(start_entity, [start_entity])])
        
        while queue:
            entity, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            for relation, tail in self.entity_adj[entity]:
                new_path = path + [relation, tail]
                
                if tail == end_entity:
                    paths.append(new_path)
                else:
                    queue.append((tail, new_path))
        
        return paths

"""
面试要点：
1. BFS vs DFS选择：
   - 最短路径（无权图）：BFS
   - 连通分量、环路检测：DFS
   - 拓扑排序：DFS或Kahn算法（BFS）

2. 最短路径算法对比：
   - Dijkstra: 单源、非负权、O((V+E)logV)
   - Bellman-Ford: 单源、可负权、O(VE)
   - Floyd: 全源、可负权、O(V³)
   - SPFA: Bellman-Ford的队列优化

3. GNN中的图算法：
   - GraphSAGE: BFS邻居采样 + 聚合
   - PinSAGE: 基于随机游走的采样
   - LightGCN: 简化图卷积，去除特征变换和非线性激活

4. 知识图谱在推荐中的作用：
   - 增强物品语义：电影 <- 主演 <- 演员 <- 出演 <- 其他电影
   - 可解释推荐：提供推荐理由（路径）
   - 解决冷启动：新物品通过KG链接到已有物品
"""
```

---

### 5. 链表 / 双端队列 (Linked List / Deque)

#### 核心考点
- **LRU Cache**: 哈希表 + 双向链表实现 O(1) 访问和淘汰
- **双端队列**: 滑动窗口最大值、单调队列

#### Python实现

```python
class DLinkedNode:
    """双向链表节点"""
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    """
    【推荐系统实战】LRU缓存实现
    核心：哈希表 + 双向链表，保证O(1)读写和淘汰
    场景：用户特征缓存、模型结果缓存
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> DLinkedNode
        self.size = 0
        
        # 伪头部和伪尾部节点，简化边界处理
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_to_head(self, node):
        """将节点添加到头部（最近使用）"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node):
        """移除节点"""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _move_to_head(self, node):
        """将节点移动到头部"""
        self._remove_node(node)
        self._add_to_head(node)
    
    def _pop_tail(self):
        """移除尾部节点（最久未使用）"""
        node = self.tail.prev
        self._remove_node(node)
        return node
    
    def get(self, key: int) -> int:
        """获取值，同时更新访问时间"""
        if key not in self.cache:
            return -1
        
        node = self.cache[key]
        self._move_to_head(node)  # 更新为最近使用
        return node.value
    
    def put(self, key: int, value: int):
        """插入或更新"""
        if key not in self.cache:
            # 新建节点
            node = DLinkedNode(key, value)
            self.cache[key] = node
            self._add_to_head(node)
            self.size += 1
            
            # 超出容量，淘汰最久未使用
            if self.size > self.capacity:
                removed = self._pop_tail()
                del self.cache[removed.key]
                self.size -= 1
        else:
            # 更新值
            node = self.cache[key]
            node.value = value
            self._move_to_head(node)

class LFUCache:
    """
    【进阶】LFU缓存（Least Frequently Used）
    相比LRU，考虑访问频率，更适合某些推荐场景
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.min_freq = 0
        self.key_table = {}  # key -> (value, freq)
        self.freq_table = defaultdict(OrderedDict)  # freq -> {key: None}
    
    def get(self, key: int) -> int:
        if key not in self.key_table:
            return -1
        
        value, freq = self.key_table[key]
        # 从原频率列表移除
        del self.freq_table[freq][key]
        if not self.freq_table[freq]:
            del self.freq_table[freq]
            if self.min_freq == freq:
                self.min_freq += 1
        
        # 加入新频率列表
        self.freq_table[freq + 1][key] = None
        self.key_table[key] = (value, freq + 1)
        
        return value
    
    def put(self, key: int, value: int):
        if self.capacity <= 0:
            return
        
        if key in self.key_table:
            self.key_table[key] = (value, self.key_table[key][1])
            self.get(key)  # 更新频率
            return
        
        if len(self.key_table) >= self.capacity:
            # 淘汰最小频率的最旧节点
            evict_key, _ = self.freq_table[self.min_freq].popitem(last=False)
            del self.key_table[evict_key]
        
        # 插入新节点，频率为1
        self.key_table[key] = (value, 1)
        self.freq_table[1][key] = None
        self.min_freq = 1

from collections import deque

class MonotonicQueue:
    """
    【推荐系统实战】单调队列
    场景：实时计算滑动窗口内的最大值（如最近N个行为的最高CTR）
    """
    def __init__(self):
        # 双端队列，存储 (index, value)，按value递减
        self.queue = deque()
        self.index = 0
    
    def push(self, value):
        """入队，保持单调递减"""
        while self.queue and self.queue[-1][1] <= value:
            self.queue.pop()
        self.queue.append((self.index, value))
        self.index += 1
    
    def pop(self):
        """出队（如果队首是当前要移除的元素）"""
        if self.queue:
            self.queue.popleft()
    
    def max(self):
        """获取当前最大值"""
        return self.queue[0][1] if self.queue else None

# ============ 推荐系统实战场景 ============

class FeatureCache:
    """
    【推荐系统实战】用户特征缓存系统
    解决：高频访问的用户特征（如热门用户）缓存在内存，减少Redis查询
    """
    def __init__(self, capacity=10000):
        self.lru = LRUCache(capacity)
        self.hit_count = 0
        self.miss_count = 0
    
    def get_features(self, user_id: str, fetch_from_db_func):
        """
        获取用户特征
        :param user_id: 用户ID
        :param fetch_from_db_func: 从数据库获取特征的函数
        """
        result = self.lru.get(user_id)
        if result != -1:
            self.hit_count += 1
            return result
        
        self.miss_count += 1
        # 从数据库获取
        features = fetch_from_db_func(user_id)
        self.lru.put(user_id, features)
        return features
    
    def get_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0

class SlidingWindowStats:
    """
    【推荐系统实战】滑动窗口统计（实时特征）
    场景：统计用户最近30分钟内的平均点击率
    """
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.queue = deque()  # (timestamp, value)
    
    def add(self, timestamp: int, value: float):
        """添加数据点"""
        # 移除过期数据
        cutoff = timestamp - self.window_size
        while self.queue and self.queue[0][0] < cutoff:
            self.queue.popleft()
        
        self.queue.append((timestamp, value))
    
    def get_average(self) -> float:
        """获取窗口内平均值"""
        if not self.queue:
            return 0.0
        total = sum(v for _, v in self.queue)
        return total / len(self.queue)
    
    def get_max(self) -> float:
        """获取窗口内最大值（使用单调队列优化）"""
        if not self.queue:
            return 0.0
        return max(v for _, v in self.queue)

class SessionSequence:
    """
    【推荐系统实战】用户会话序列管理（双端队列）
    场景：存储用户当前会话内的行为序列，用于实时推荐
    """
    def __init__(self, max_length=50):
        self.max_length = max_length
        self.sequence = deque(maxlen=max_length)  # 固定长度队列
    
    def add_action(self, item_id: str, action_type: str, timestamp: int):
        """添加用户行为"""
        self.sequence.append({
            'item_id': item_id,
            'action': action_type,  # 'click', 'view', 'purchase'
            'time': timestamp
        })
    
    def get_recent_actions(self, k: int = 10) -> List[dict]:
        """获取最近的K个行为"""
        return list(self.sequence)[-k:]
    
    def get_item_sequence(self) -> List[str]:
        """获取物品ID序列（用于序列模型输入）"""
        return [action['item_id'] for action in self.sequence]

"""
面试要点：
1. LRU实现关键点：
   - 哈希表提供O(1)查找
   - 双向链表提供O(1)的插入和删除
   - 为什么是双向链表？删除节点时需要知道前驱节点

2. LRU vs LFU：
   - LRU：最近最少使用，适合时间局部性好的场景
   - LFU：最少使用频率，适合频率差异明显的场景
   - 混合策略：LRU-K、ARC等

3. 推荐系统中的缓存策略：
   - 用户特征缓存：LRU，热门用户常驻内存
   - 模型结果缓存：对相同请求直接返回缓存结果
   - 召回结果缓存：减少重复召回计算

4. 双端队列应用场景：
   - 滑动窗口问题（最大值、平均值）
   - 用户行为序列存储（固定长度，自动淘汰老数据）
   - 单调队列优化DP

5. 工程实践：
   - Python可用 OrderedDict 简化LRU实现
   - 实际生产使用 Redis + LRU本地缓存 二级缓存架构
   - 缓存穿透、击穿、雪崩的处理策略
"""
```

---

## 六、数据结构选择决策树

```
问题场景分析
    │
    ├── 需要快速查找？
    │       ├── 精确查找 → 哈希表 O(1)
    │       ├── 范围查询 → B+树 O(log n + k)
    │       └── 前缀匹配 → Trie树 O(m)
    │
    ├── 需要维护有序性？
    │       ├── 动态TopK → 堆 O(log k)
    │       ├── 全序排列 → 平衡树 O(log n)
    │       └── 滑动窗口最值 → 单调队列 O(1)
    │
    ├── 有图关系？
    │       ├── 连通性/最短路径 → BFS/DFS
    │       ├── 拓扑排序 → Kahn算法/DFS
    │       └── 推荐场景 → 用户-物品二分图
    │
    └── 需要淘汰策略？
            ├── 时间局部性 → LRU
            ├── 频率局部性 → LFU
            └── 混合策略 → LRU-K
```

---

## 七、推荐系统数据结构设计题精选

### 题目1：设计一个实时热门商品排行榜
**要求**：
- 支持商品被点击时更新分数
- 能获取当前Top100热门商品
- 查询时间复杂度 O(1) 或 O(log n)

**思路**：
- 堆 + 哈希表：堆维护TopK，哈希表记录商品当前分数
- 或者跳表（Sorted Set）：Redis ZSET的实现方式

### 题目2：设计用户最近浏览记录
**要求**：
- 记录用户最近浏览的100个商品
- 去重：同一商品多次浏览只保留最近一次
- 支持快速获取浏览列表

**思路**：
- 哈希表 + 双向链表（LRU的简化版）
- 或者直接用 OrderedDict

### 题目3：设计分布式ID生成器
**要求**：
- 全局唯一
- 趋势递增（有利于B+树索引）
- 高可用，低延迟

**思路**：
- Snowflake算法：时间戳 + 机器ID + 序列号

---

*Generated by MelonEggLearn | 数据结构实战篇 | 最后更新: 2026-03-11*
