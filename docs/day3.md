# Day 3: Kernel 3 - Shared Memory Tiling (2026-02-17)

## ✅ 完成任务
- [x] 运行Kernel 3并记录性能
- [x] 详细理解Kernel 3源码的每一行
- [x] 理解Shared Memory的作用和声明
- [x] 理解指针预移动的含义
- [x] 理解Tiling主循环的执行流程
- [x] 理解两次__syncthreads()的必要性
- [x] 分析Shared Memory带来的数据复用
- [x] 分析为什么提升是1.4x而不是更高

---

## 📊 性能数据

### RTX 3070 Laptop - 全矩阵大小对比

| 矩阵大小 | Naive | Coalescing | **Shared Mem** | vs K2 |
|---------|-------|-----------|---------------|-------|
| 128³    | 45.3  | 246.1     | **331.1**     | 1.35x |
| 256³    | 85.6  | 651.9     | **910.7**     | 1.40x |
| 512³    | 107.1 | 735.2     | **1008.2**    | 1.37x |
| 1024³   | 140.4 | 868.0     | **1117.2**    | 1.29x |
| 2048³   | 143.9 | 992.5     | **1345.1**    | 1.36x |
| 4096³   | 140.7 | 982.1     | **1363.6**    | 1.39x |

### 4096×4096 里程碑

| Kernel | 时间(ms) | GFLOPS | vs cuBLAS | vs Naive |
|--------|---------|---------|-----------|----------|
| 0 - cuBLAS | 17.88 | 7686.3 | 100% | 54.6x |
| 1 - Naive | 976.99 | 140.7 | 1.8% | 1.0x |
| 2 - Coalescing | 139.95 | 982.1 | 12.8% | 6.98x |
| **3 - Shared Mem** | **100.79** | **1363.6** | **17.7%** | **9.69x** |

### 关键发现
- 🚀 vs Kernel 2提升**39%**（982 → 1363 GFLOPS）
- 📈 vs Naive累计提升**9.7倍**
- ⏱️ 计算时间减少**28%**（140ms → 101ms）
- 🎯 达到cuBLAS的**17.7%**

---

## 💻 Kernel 3源码分析

### 完整代码
```cpp
template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  A += cRow * BLOCKSIZE * K;
  B += cCol * BLOCKSIZE;
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
    __syncthreads();

    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    __syncthreads();
  }
  C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}
```

---

## 🔍 逐段详解

### Part 1: Block和线程索引
```cpp
const uint cRow = blockIdx.x;  // 当前block负责C的第几行块
const uint cCol = blockIdx.y;  // 当前block负责C的第几列块

const uint threadCol = threadIdx.x % BLOCKSIZE;  // 块内列偏移（0-31）
const uint threadRow = threadIdx.x / BLOCKSIZE;  // 块内行偏移（0-31）
```

**注意**：这里cRow/cCol不乘以BLOCKSIZE，在后面的指针预移动中处理

**例子（BLOCKSIZE=32）**：
```
Block(1,2)：
  cRow=1 → 负责C的第32-63行
  cCol=2 → 负责C的第64-95列

Block内线程（1024个）：
  threadIdx.x=0:    threadRow=0, threadCol=0
  threadIdx.x=31:   threadRow=0, threadCol=31
  threadIdx.x=32:   threadRow=1, threadCol=0  ← 换行
  threadIdx.x=1023: threadRow=31, threadCol=31
```

---

### Part 2: Shared Memory声明
```cpp
__shared__ float As[BLOCKSIZE * BLOCKSIZE];  // 32×32 = 1024个float = 4KB
__shared__ float Bs[BLOCKSIZE * BLOCKSIZE];  // 32×32 = 1024个float = 4KB
```

**关键特性**：

| 特性 | 全局内存 | Shared Memory |
|------|---------|--------------|
| 位置 | 显卡DRAM | GPU芯片上 |
| 速度 | 慢（~500GB/s） | 快（~20-100倍） |
| 大小 | 8GB | 约48KB/SM |
| 作用域 | 所有线程 | 同一block内 |
| 生命周期 | 整个程序 | 与block同生共死 |

**虽然声明为1D，但存储2D tile**：
```
As[1024] 逻辑上是：
  As[0-31]  → 第0行
  As[32-63] → 第1行
  ...
  As[992-1023] → 第31行

访问As[row][col] = As[row * 32 + col]
```

---

### Part 3: 指针预移动
```cpp
A += cRow * BLOCKSIZE * K;                    // 移到当前block的起始行
B += cCol * BLOCKSIZE;                        // 移到当前block的起始列
C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // 移到当前block的起始位置
```

**以Block(1,2)为例（BLOCKSIZE=32, K=64, N=128）**：
```
A的移动：
  A += 1 * 32 * 64 = 2048
  A从A[0][0]移到A[32][0]
  → Block(1,2)需要A的第32-63行

B的移动：
  B += 2 * 32 = 64
  B从B[0][0]移到B[0][64]
  → Block(1,2)需要B的第64-95列

C的移动：
  C += 1*32*128 + 2*32 = 4096 + 64 = 4160
  C从C[0][0]移到C[32][64]
  → Block(1,2)负责C[32-63][64-95]
```

**可视化**：
```
A矩阵：
       列0-31 | 列32-63
行0   [      |       ]
...
行31  [      |       ]
行32  [◄─────A指针    ]  ← Block(1,?)的起始行
...
行63  [      |       ]

B矩阵：
       列0-63 | 列64-95 | 列96-127
行0   [       |◄──B指针|       ]  ← Block(?,2)的起始列
...

C矩阵：
       列0-63 | 列64-95 | 列96-127
行0   [       |        |       ]
...
行32  [       |◄──C指针|       ]  ← Block(1,2)负责此区域
...
row63 [       |▓▓▓▓▓▓▓|       ]
```

---

### Part 4: Tiling主循环

**核心思想**：K维度太长，分成多个小块（tile）处理
```cpp
for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // K=4096, BLOCKSIZE=32 → 循环128次
    // K=64,   BLOCKSIZE=32 → 循环2次
}
```

**为什么要分块？**
```
计算C的一个32×32块需要：
  A的32行 × K列
  B的K行 × 32列

K可能有4096，一次装不进Shared Memory
→ 把K分成 4096/32 = 128 个tile分批处理
```

---

### Part 5: 加载数据到Shared Memory
```cpp
// 每个线程加载A和B各一个元素
As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
```

**1024个线程分工**：
```
线程(threadRow=0, threadCol=0):
  As[0] = A[0]        // A的第0行第0列
  Bs[0] = B[0]        // B的第0行第0列

线程(threadRow=0, threadCol=1):
  As[1] = A[1]        // A的第0行第1列
  Bs[1] = B[1]        // B的第0行第1列

...（每个线程负责1个元素）

1024个线程 = 32×32个元素
→ 刚好填满As和Bs这两个tile！
```

**内存访问合并**（关键）：
```
同一warp（threadCol=0-31，threadRow固定）:

访问全局内存A：
  线程0:  A[threadRow*K + 0]
  线程1:  A[threadRow*K + 1]  ← 连续！
  ...
  线程31: A[threadRow*K + 31]
  ✅ 合并访问，1个内存事务

访问全局内存B：
  线程0:  B[threadRow*N + 0]
  线程1:  B[threadRow*N + 1]  ← 连续！
  ...
  线程31: B[threadRow*N + 31]
  ✅ 合并访问，1个内存事务
```

---

### Part 6: 两次同步及其原因
```cpp
__syncthreads();  // 第1次：加载后
// ...计算...
__syncthreads();  // 第2次：计算后
```

**第1次同步（加载后）**：
```
问题：1024个线程并发加载，速度不一
  线程0:   已加载完 → 想开始计算
  线程500: 还在加载 → As[500]还没数据！

如果线程0直接计算：
  可能读到还未写入的As[500] → 结果错误！

__syncthreads()：等所有线程加载完，才允许计算
```

**第2次同步（计算后）**：
```
问题：下一次循环会覆盖As和Bs
  线程0:   已计算完 → 开始加载下一个tile覆盖As
  线程500: 还在计算 → 还需要读As的数据！

如果线程0直接覆盖：
  线程500读到错误数据 → 结果错误！

__syncthreads()：等所有线程计算完，才允许加载新数据
```

**类比**：
```
就像在教室考试：
  第1次同步：等所有人拿到试卷，才能开始答题
  第2次同步：等所有人答完，才能收卷发新卷子
```

---

### Part 7: 指针移动（在计算之前）
```cpp
A += BLOCKSIZE;      // A向右移一个tile（32列）
B += BLOCKSIZE * N;  // B向下移一个tile（32行）
```

**为什么在计算前移动？**
```
现在的顺序：加载 → 同步 → 移动指针 → 计算
你想到的：  加载 → 同步 → 计算 → 移动指针

两种都正确！但现在的顺序更优：
  - 移动指针（整数加法）与计算（浮点运算）可以并行
  - GPU可以同时执行地址计算和浮点运算
  - 相当于"做饭的同时准备下一道菜的食材"

关键：计算用的是As和Bs（共享内存）
     不是A和B（全局内存指针）
     所以指针提前移动不影响计算结果！
```

**数据流向**：
```
循环1（bkIdx=0）:
  A指针: A[0][0] → 加载到As → A指针移到A[0][32] → 用As计算

循环2（bkIdx=32）:
  A指针: A[0][32] → 加载到As → A指针移到A[0][64] → 用As计算
```

---

### Part 8: 内层循环计算
```cpp
for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
  tmp += As[threadRow * BLOCKSIZE + dotIdx] *
         Bs[dotIdx * BLOCKSIZE + threadCol];
}
```

**理解**：
```
线程(threadRow=2, threadCol=5)执行：

tmp += As[2*32 + 0]  * Bs[0*32 + 5]   // As第2行第0个 × Bs第0行第5个
tmp += As[2*32 + 1]  * Bs[1*32 + 5]   // As第2行第1个 × Bs第1行第5个
...
tmp += As[2*32 + 31] * Bs[31*32 + 5]  // As第2行第31个 × Bs第31行第5个

这就是：A的第2行 点积 B的第5列（在tile范围内）
```

**数据复用（关键优势）**：
```
同一warp（32个线程，threadCol=0-31，threadRow=2）:
  所有线程读取 As[2*32 + dotIdx]（A的同一行）
  只需从Shared Memory读1次 → 32个线程都用
  
  不像全局内存，每次访问都有高延迟
  Shared Memory：读1次，用32次 ✅
```

---

## 🆚 Kernel 2 vs Kernel 3 对比总结

### 内存访问次数对比

计算一个32×32的C块（K=4096）：

**Kernel 2**：
```
从全局内存读A：32 × 4096 = 131,072次
从全局内存读B：4096 × 32 = 131,072次
总计：262,144次全局内存读取
每次读取都是全局内存（慢）
```

**Kernel 3**：
```
循环128次，每次：
  从全局内存加载A tile：32×32 = 1,024次（合并）
  从全局内存加载B tile：32×32 = 1,024次（合并）
  
  然后在Shared Memory上计算32次
  每次计算从Shared Memory读32+32个元素

全局内存读取次数：128 × 2048 = 262,144次（相同）
但后续计算从Shared Memory读 → 快很多！
```

---

### 执行流程对比

**Kernel 2**：
```
每个线程：
  for i in K:
    从全局内存读A[row][i]    ← 每次都是全局内存
    从全局内存读B[i][col]    ← 每次都是全局内存
    tmp += A * B
```

**Kernel 3**：
```
block内所有线程协作：
  for tile in K/32:
    一起加载 32×32的A tile → Shared Memory  ← 合并，快
    一起加载 32×32的B tile → Shared Memory  ← 合并，快
    同步
    for i in 32:
      从Shared Memory读As   ← 快！
      从Shared Memory读Bs   ← 快！
      tmp += As * Bs
    同步
```

---

## 💡 为什么提升是1.4x而不是更高？

### 理论预期 vs 实际结果
```
理论：Shared Memory比全局内存快20-100倍
     应该有很大提升

实际：只有1.4倍

原因分析：
```

**原因1：Kernel 2已经很高效**
```
Kernel 2的合并访问让内存带宽利用率已经很高
→ 进一步优化内存的收益递减
```

**原因2：瓶颈转移**
```
Kernel 2: Memory-bound（等内存）
Kernel 3: 开始向Compute-bound转变（等计算）

优化了内存后，计算延迟开始显现
→ 单纯优化内存收益减小
```

**原因3：同步开销**
```
K=4096, BLOCKSIZE=32：
  循环128次 × 每次2次同步 = 256次同步
每次同步都有等待开销
```

**原因4：Occupancy降低**
```
每个Block使用：
  As: 4KB + Bs: 4KB = 8KB Shared Memory

RTX 3070的每个SM有48KB Shared Memory
→ 每个SM最多6个Block同时运行
→ 并发度可能下降 → 延迟隐藏能力降低
```

---

## 📈 性能进化路线图（更新）
```
Naive          140 GFLOPS  (1.8% of cuBLAS)
    ↓ 内存合并访问 (7x)
Coalescing     982 GFLOPS  (12.8% of cuBLAS)
    ↓ Shared Memory + Tiling (1.4x)
Shared Mem    1364 GFLOPS  (17.7% of cuBLAS) ← 我们在这里
    ↓ 1D/2D Blocktiling
  ~2000 GFLOPS (~26%)
    ↓ Vectorization + Bank Conflict消除
  ~4000 GFLOPS (~52%)
    ↓ Warptiling + Double Buffering
  ~6000+ GFLOPS (~78%)
    ↓
cuBLAS        7686 GFLOPS  (100%)
```

---

## 🔖 关键概念整理

### Shared Memory
- GPU片上内存，比全局内存快约20-100倍
- 同一block内的所有线程共享
- `__shared__` 关键字声明
- 容量有限（约48KB/SM）

### Tiling（分块）
- 把大矩阵分成小块（tile）处理
- 每次加载一个tile到Shared Memory
- 在tile内部复用数据
- 循环次数：K / BLOCKSIZE

### __syncthreads()
- 同步block内所有线程
- 第1次：确保加载完成再计算
- 第2次：确保计算完成再加载新数据
- 两次都必不可少！

### 数据复用
- 一个tile加载1次（全局内存）
- 在Shared Memory上使用32次（内层循环）
- 每个元素复用32倍

---

## ❓ 遗留疑问

- [ ] 如何用Nsight验证Shared Memory的命中率？
- [ ] Bank Conflict是什么？这个kernel有吗？
- [ ] 如何进一步提高Occupancy？
- [ ] 什么是1D/2D Blocktiling？

---

## 🎯 明天计划 (Day 4)

- [ ] 学习1D Blocktiling（Kernel 4）
  - 每个线程计算多个C元素（不只是1个）
  - 增加寄存器复用
- [ ] 运行Kernel 4，记录性能
- [ ] 理解为什么增加每线程工作量能提升性能
- [ ] 预期目标：突破2000 GFLOPS

---

## 📚 参考资料

- Simon的博客：[How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)
- 项目源码：`src/kernels/3_kernel_shared_mem_block.cuh`
- CUDA编程指南：Shared Memory章节

---

**学习时长**：约2小时  
**完成日期**：2026-02-17  
